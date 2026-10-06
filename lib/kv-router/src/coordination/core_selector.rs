// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A [`StageSelector`] over one [`SelectionCore`] partition.
//!
//! Previews run as advisory selections: the scheduler reports its choice and
//! that worker's load without queue admission or a booking. Admissions book
//! through the core in one of two modes:
//!
//! - [`CoreAdmissionMode::Lease`] returns the scheduler's booking handle inside
//!   the reservation. The host owns it; dropping it frees the booking.
//! - [`CoreAdmissionMode::Book`] installs the reservation in the core's index
//!   under [`SelectionInput::reservation_id`], so a host that observes
//!   responses out of process (the EPP) can drive `prefill_complete` and
//!   `free_reservation` by id.

use std::any::Any;
use std::collections::HashSet;
use std::sync::Arc;

use async_trait::async_trait;
use futures_util::future::BoxFuture;

use crate::identity::RoutingPartitionId;
use crate::protocols::{WorkerId, WorkerWithDpRank};
use crate::scheduling::KvSchedulerError;
use crate::scheduling::queue::BookingHandle;
use crate::services::selection::{
    PromptView, Selected, SelectionAdmission, SelectionCore, SelectionError, SelectionOperation,
    SelectionOutcome, SessionBinding, WorkerCatalogRecord, WorkerLifecycle,
};

use super::error::CoordinationError;
use super::ids::StageId;
use super::selector::{
    AdmissionTarget, PrefillLoadSignal, Preview, ReservationLease, ReservationOwner,
    SelectedTarget, SelectionInput, SelectionSignals, StageReservation, StageSelector, WorkerFacts,
};
use super::stage::{PoolRef, WorkAccounting};

/// How a [`CoreStageSelector`] books an admitted worker.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoreAdmissionMode {
    /// The core records the reservation by id; release goes through
    /// [`SelectionCore::free_reservation`].
    Book,
    /// The reservation owns the scheduler booking handle directly.
    Lease,
}

/// Selects workers for one stage from one selection-core partition.
pub struct CoreStageSelector {
    core: Arc<SelectionCore>,
    partition: RoutingPartitionId,
    pool: PoolRef,
    mode: CoreAdmissionMode,
}

impl CoreStageSelector {
    pub fn new(
        core: Arc<SelectionCore>,
        partition: RoutingPartitionId,
        pool: PoolRef,
        mode: CoreAdmissionMode,
    ) -> Self {
        Self {
            core,
            partition,
            pool,
            mode,
        }
    }

    pub fn core(&self) -> &Arc<SelectionCore> {
        &self.core
    }

    pub fn partition(&self) -> &RoutingPartitionId {
        &self.partition
    }

    pub fn pool(&self) -> &PoolRef {
        &self.pool
    }

    pub fn mode(&self) -> CoreAdmissionMode {
        self.mode
    }

    fn schedulable_workers(&self) -> HashSet<WorkerId> {
        self.core
            .list_workers(
                Some(&self.partition.model_name),
                Some(&self.partition.routing_group),
            )
            .into_iter()
            .filter(|record| record.lifecycle == WorkerLifecycle::Schedulable)
            .map(|record| record.worker_id)
            .collect()
    }

    fn facts_for(&self, worker_id: WorkerId) -> Arc<WorkerFacts> {
        Arc::new(
            self.core
                .worker_record(worker_id)
                .map(facts_from_record)
                .unwrap_or_default(),
        )
    }

    fn operation<'a>(
        &self,
        input: &SelectionInput<'a>,
        admission: SelectionAdmission,
        pinned_worker: Option<WorkerWithDpRank>,
    ) -> SelectionOperation<'a> {
        let restrictions = input.restrictions;
        let settings = input.settings;
        let steerable = pinned_worker.is_none() && restrictions.affinity_target.is_none();
        let session = match (&settings.session_context, &admission) {
            (Some(context), SelectionAdmission::Book { .. }) if steerable => {
                SessionBinding::Managed {
                    session_id: context.session_id().to_string(),
                }
            }
            (Some(context), _) if steerable => SessionBinding::Query {
                session_id: context.session_id().to_string(),
            },
            _ => SessionBinding::None,
        };
        SelectionOperation {
            key: self.partition.clone(),
            prompt: PromptView {
                token_ids: Some(input.prompt.token_ids),
                mm_routing_info: None,
                block_mm_infos: input.prompt.block_mm_infos,
                block_hashes: None,
                sequence_hashes: None,
                isl_tokens: None,
                lora_name: input.prompt.lora_name,
                cache_namespace: input.prompt.cache_namespace,
                is_eagle: None,
            },
            router_config_override: input.router_config_override(),
            expected_output_tokens: input.expected_output_tokens(),
            priority_jump: settings.priority_jump,
            strict_priority: settings.strict_priority,
            policy_class: settings.policy_class.clone(),
            session_context: settings.session_context.clone(),
            session,
            affinity_target: restrictions.affinity_target,
            pinned_worker,
            allowed_worker_ids: restrictions
                .effective_allowed_worker_ids(|| self.schedulable_workers()),
            routing_constraints: restrictions.routing_constraints.clone(),
            admission,
            track_active_blocks: input.profile.work != WorkAccounting::None,
            return_routing_hashes: false,
            replay_id: None,
        }
    }

    async fn run(
        &self,
        input: &SelectionInput<'_>,
        admission: SelectionAdmission,
        pinned_worker: Option<WorkerWithDpRank>,
    ) -> Result<Selected, CoordinationError> {
        let operation = self.operation(input, admission, pinned_worker);
        match self.core.run_selection(operation).await.result {
            Ok(SelectionOutcome::Selected(selected)) => Ok(selected),
            Ok(SelectionOutcome::QueueRejected { rejection }) => {
                Err(CoordinationError::AdmissionRejected {
                    stage: input.stage.clone(),
                    reason: rejection.to_string(),
                })
            }
            Err(error) => Err(map_selection_error(input.stage, error)),
        }
    }

    fn validate_preview(
        &self,
        input: &SelectionInput<'_>,
        preview: &Preview,
    ) -> Result<(), CoordinationError> {
        let stage = input.stage.clone();
        if preview.target.stage != *input.stage {
            return Err(CoordinationError::StalePreview {
                stage,
                reason: format!("preview belongs to stage {}", preview.target.stage),
            });
        }
        if preview.target.pool != self.pool {
            return Err(CoordinationError::StalePreview {
                stage,
                reason: format!(
                    "preview selected from pool {} but the stage now draws from {}",
                    preview.target.pool, self.pool
                ),
            });
        }
        let worker = preview.target.worker;
        if let Some(pinned) = input.restrictions.pinned_worker
            && pinned != worker
        {
            return Err(CoordinationError::ConflictingRestrictions {
                stage,
                reason: format!(
                    "previewed worker {} dp_rank {} conflicts with pinned worker {} dp_rank {}",
                    worker.worker_id, worker.dp_rank, pinned.worker_id, pinned.dp_rank
                ),
            });
        }
        if !input.restrictions.permits(worker.worker_id) {
            return Err(CoordinationError::StalePreview {
                stage,
                reason: format!(
                    "previewed worker {} is no longer permitted",
                    worker.worker_id
                ),
            });
        }
        Ok(())
    }

    fn signals(selected: &Selected, facts: &WorkerFacts) -> SelectionSignals {
        let advisory_load = selected.advisory_load;
        SelectionSignals {
            overlap_blocks: selected.response.effective_overlap_blocks.round() as u32,
            cached_tokens: selected.response.cached_tokens,
            potential_decode_blocks: selected.response.potential_decode_blocks as u64,
            total_kv_blocks: selected
                .total_kv_blocks
                .or_else(|| advisory_load.and_then(|load| load.total_kv_blocks.map(|b| b as u64)))
                .or(facts.total_kv_blocks),
            prefill_load: advisory_load.map(|load| PrefillLoadSignal {
                active_prefill_tokens: load.active_prefill_tokens,
                prefill_token_capacity: load.prefill_token_capacity,
            }),
        }
    }

    fn target(&self, input: &SelectionInput<'_>, worker: WorkerWithDpRank) -> SelectedTarget {
        SelectedTarget {
            invocation: input.invocation,
            attempt: input.attempt,
            stage: input.stage.clone(),
            pool: self.pool.clone(),
            worker,
            facts: self.facts_for(worker.worker_id),
        }
    }
}

#[async_trait]
impl StageSelector for CoreStageSelector {
    async fn preview(&self, input: SelectionInput<'_>) -> Result<Preview, CoordinationError> {
        let admission = SelectionAdmission::Advisory {
            request_id: Some(input.request_id.to_string()),
        };
        let selected = self
            .run(&input, admission, input.restrictions.pinned_worker)
            .await?;
        let target = self.target(&input, selected.response.best_worker);
        let signals = Self::signals(&selected, &target.facts);
        Ok(Preview { target, signals })
    }

    async fn admit(
        &self,
        input: SelectionInput<'_>,
        target: AdmissionTarget,
    ) -> Result<StageReservation, CoordinationError> {
        let pinned_worker = match &target {
            AdmissionTarget::AnyEligible => input.restrictions.pinned_worker,
            AdmissionTarget::FromPreview(preview) => {
                self.validate_preview(&input, preview)?;
                Some(preview.target.worker)
            }
        };
        let selection_id = input.reservation_id();
        let admission = match self.mode {
            CoreAdmissionMode::Book => SelectionAdmission::Book {
                selection_id: selection_id.clone(),
            },
            CoreAdmissionMode::Lease => SelectionAdmission::Lease {
                request_id: selection_id.clone(),
            },
        };
        let mut selected = self.run(&input, admission, pinned_worker).await?;
        let worker = selected.response.best_worker;
        let lease = match self.mode {
            CoreAdmissionMode::Lease => ReservationLease::new(LeasedBooking {
                booking: selected.booking.take(),
                selection_id,
            }),
            CoreAdmissionMode::Book => ReservationLease::new(CoreBooking {
                core: Arc::clone(&self.core),
                selection_id,
                released: false,
            }),
        };
        if let Some(pinned) = pinned_worker
            && worker != pinned
        {
            // The scheduler honours pins, so this is a contract violation, not
            // a routing outcome. Release before reporting it.
            lease.release().await?;
            return Err(CoordinationError::StalePreview {
                stage: input.stage.clone(),
                reason: format!(
                    "scheduler admitted worker {} dp_rank {} instead of pinned worker {} dp_rank {}",
                    worker.worker_id, worker.dp_rank, pinned.worker_id, pinned.dp_rank
                ),
            });
        }
        let target = self.target(&input, worker);
        let signals = Self::signals(&selected, &target.facts);
        Ok(StageReservation::new(target, signals, lease))
    }
}

fn facts_from_record(record: WorkerCatalogRecord) -> WorkerFacts {
    let dp_ranks = record.dp_ranks();
    WorkerFacts {
        data_parallel_start_rank: dp_ranks.start,
        data_parallel_size: dp_ranks.end.saturating_sub(dp_ranks.start),
        total_kv_blocks: record.total_kv_blocks,
        stable_routing_id: record.stable_routing_id,
        taints: record.taints,
        topology_domains: record.topology_domains,
        kv_transfer_domain: record.kv_transfer_domain,
        kv_transfer_enforcement: record.kv_transfer_enforcement,
        kv_transfer_preferred_weight: record.kv_transfer_preferred_weight,
    }
}

fn map_selection_error(stage: &StageId, error: SelectionError) -> CoordinationError {
    let stage = stage.clone();
    match error {
        SelectionError::NotReady(reason) => CoordinationError::NoEligibleWorkers { stage, reason },
        SelectionError::Scheduler(error) => match error {
            KvSchedulerError::NoEndpoints
            | KvSchedulerError::AllEligibleWorkersFiltered
            | KvSchedulerError::PinnedWorkerNotAllowed { .. } => {
                CoordinationError::NoEligibleWorkers {
                    stage,
                    reason: error.to_string(),
                }
            }
            KvSchedulerError::QueueRejected(_)
            | KvSchedulerError::AllEligibleWorkersOverloaded
            | KvSchedulerError::PinnedWorkerOverloaded { .. }
            | KvSchedulerError::DeadlineExceeded => CoordinationError::AdmissionRejected {
                stage,
                reason: error.to_string(),
            },
            other => CoordinationError::Selector(anyhow::Error::new(other)),
        },
        other => CoordinationError::Selector(anyhow::Error::new(other)),
    }
}

/// A `Lease` admission: the scheduler booking handle itself.
struct LeasedBooking {
    booking: Option<BookingHandle>,
    selection_id: String,
}

impl ReservationOwner for LeasedBooking {
    fn release(mut self: Box<Self>) -> BoxFuture<'static, Result<(), CoordinationError>> {
        let booking = self.booking.take();
        Box::pin(async move {
            match booking {
                Some(booking) => booking
                    .release()
                    .await
                    .map_err(|error| CoordinationError::Release(error.to_string())),
                None => Ok(()),
            }
        })
    }

    fn into_any(self: Box<Self>) -> Box<dyn Any + Send> {
        self
    }

    fn describe(&self) -> String {
        format!("leased booking {}", self.selection_id)
    }
}

/// A `Book` admission: a reservation the core indexes by id.
///
/// Hosts that observe responses by id downcast to this through
/// [`ReservationLease::into_owner`] to drive `prefill_complete`.
pub struct CoreBooking {
    core: Arc<SelectionCore>,
    selection_id: String,
    released: bool,
}

impl CoreBooking {
    pub fn selection_id(&self) -> &str {
        &self.selection_id
    }

    /// Record that the worker finished prefill for this booking. A booking
    /// that was already freed is not an error.
    pub async fn prefill_complete(&self) -> Result<(), CoordinationError> {
        match self.core.prefill_complete(&self.selection_id).await {
            Ok(()) | Err(SelectionError::NotFound(_)) => Ok(()),
            Err(error) => Err(CoordinationError::Selector(anyhow::Error::new(error))),
        }
    }

    /// Record one more generated output block for this booking.
    pub fn add_output_block(&self, decay_fraction: Option<f64>) -> Result<(), CoordinationError> {
        match self
            .core
            .add_output_block(&self.selection_id, decay_fraction)
        {
            Ok(()) | Err(SelectionError::NotFound(_)) => Ok(()),
            Err(error) => Err(CoordinationError::Selector(anyhow::Error::new(error))),
        }
    }
}

impl ReservationOwner for CoreBooking {
    fn release(mut self: Box<Self>) -> BoxFuture<'static, Result<(), CoordinationError>> {
        self.released = true;
        let core = Arc::clone(&self.core);
        let selection_id = self.selection_id.clone();
        Box::pin(async move {
            match core.free_reservation(&selection_id).await {
                Ok(()) | Err(SelectionError::NotFound(_)) => Ok(()),
                Err(error) => Err(CoordinationError::Release(error.to_string())),
            }
        })
    }

    fn into_any(self: Box<Self>) -> Box<dyn Any + Send> {
        self
    }

    fn describe(&self) -> String {
        format!("core booking {}", self.selection_id)
    }
}

impl Drop for CoreBooking {
    fn drop(&mut self) {
        if self.released {
            return;
        }
        let core = Arc::clone(&self.core);
        let selection_id = std::mem::take(&mut self.selection_id);
        match tokio::runtime::Handle::try_current() {
            Ok(handle) => {
                handle.spawn(async move {
                    if let Err(error) = core.free_reservation(&selection_id).await
                        && !matches!(error, SelectionError::NotFound(_))
                    {
                        tracing::warn!(
                            selection_id,
                            %error,
                            "failed to free a dropped core booking"
                        );
                    }
                });
            }
            Err(_) => {
                tracing::warn!(
                    selection_id,
                    "core booking dropped outside a runtime; the reservation will expire"
                );
            }
        }
    }
}
