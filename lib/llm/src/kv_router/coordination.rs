// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Frontend adapter for the stage-based routing coordinator.
//!
//! [`HostStageSelector`] implements the coordinator's `StageSelector` over a
//! [`RoutingHost`]: a preview is the host's advisory KV route preview, an
//! admission is a `RoutePlan` the host later dispatches. The host keeps
//! dispatch, response streams, retries, cancellation, and buffer cleanup; the
//! coordinator decides selection order, branch, profiles, and placement.
//!
//! Selection runs against a fork of the host's request that shares its
//! cancellation controller and metadata, with the coordinator's per-stage
//! router override, placement constraints, and allowlist rendered onto the
//! copy. The real request is dispatched untouched.

use std::collections::HashSet;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::anyhow;
use async_trait::async_trait;
use futures::future::BoxFuture;
use parking_lot::Mutex;

use dynamo_kv_router::conditional_disagg::{
    ConditionalDisaggDecisionInput, ConditionalDisaggPolicy,
};
use dynamo_kv_router::coordination::{
    AdmissionTarget, AttemptId, CoordinationError, InvocationId, PoolRef, PrefillLoadSignal,
    Preview, PromptInput, RequestFacts, RequestSettings, ReservationLease, ReservationOwner,
    RoutingRequest, SelectedTarget, SelectionInput, SelectionSignals, StageId, StageReservation,
    StageSelector, WorkerFacts,
};
use dynamo_kv_router::protocols::{WorkerId, WorkerWithDpRank};
use dynamo_kv_router::scheduling::{KvSchedulerError, queue::BookingHandle};
use dynamo_runtime::error::{ErrorType, match_error_chain};
use dynamo_runtime::pipeline::{Context, SingleIn};

use crate::kv_router::prefill_router::PrefillError;
use crate::kv_router::request_lease::RequestAttemptLease;
use crate::kv_router::routing_host::{RoutePlan, RoutePlanSignals, RoutePreview, RoutingHost};
use crate::kv_router::to_worker_selection_session_context;
use crate::kv_router::{FindBestMatchOutcome, KvRouter, PrefillRouter};
use crate::preprocessor::PreprocessedRequest;
use crate::protocols::common::timing::RequestPhase;

/// Translate a host request into the coordinator's request shape.
///
/// Prompt tokens are shared, not copied. Caller pins stay on the host request
/// (the routing host applies them); the facts only record which stages the
/// caller decided so policies can skip their own decision.
pub(crate) fn routing_request_for(
    request: &PreprocessedRequest,
    request_id: &str,
    policy_class: Option<String>,
) -> RoutingRequest {
    let routing = request.routing.as_ref();
    let (routing_tokens, block_mm_infos) = request.block_mm_routing_info();
    let token_ids = if std::ptr::eq(routing_tokens.as_ptr(), request.token_ids.as_ptr()) {
        Arc::clone(&request.token_ids)
    } else {
        Arc::new(routing_tokens.to_vec())
    };
    let prompt = PromptInput {
        token_ids,
        block_mm_infos: block_mm_infos.map(<[_]>::to_vec),
        lora_name: routing.and_then(|routing| routing.lora_name.clone()),
        cache_namespace: routing.and_then(|routing| routing.cache_namespace.clone()),
    };
    let settings = RequestSettings {
        priority_jump: routing
            .and_then(|routing| routing.priority_jump)
            .unwrap_or(0.0),
        strict_priority: routing
            .and_then(|routing| routing.strict_priority)
            .unwrap_or(0),
        policy_class,
        session_context: request
            .agent_context
            .as_ref()
            .map(to_worker_selection_session_context),
        expected_output_tokens: routing.and_then(|routing| routing.expected_output_tokens),
        router_config_override: request.router_config_override.clone(),
    };
    let mut facts = RequestFacts {
        prompt_tokens: routing_tokens.len(),
        requires_encode: false,
        pinned_stages: HashSet::new(),
    };
    if routing.is_some_and(|routing| {
        routing.prefill_worker_id.is_some() || routing.backend_instance_id.is_some()
    }) {
        facts.pinned_stages.insert(StageId::PREFILL);
    }
    if routing.is_some_and(|routing| {
        routing.decode_worker_id.is_some() || routing.backend_instance_id.is_some()
    }) {
        facts.pinned_stages.insert(StageId::DECODE);
    }
    let mut routing_request = RoutingRequest::new(request_id, prompt).with_settings(settings);
    routing_request.facts = facts;
    routing_request
}

/// A coordinator stage selector over one routing host and one request.
pub(crate) struct HostStageSelector {
    host: Arc<RoutingHost>,
    phase: RequestPhase,
    pool: PoolRef,
    /// The request to select with: shares the dispatch request's controller.
    base: Context<PreprocessedRequest>,
    /// Whether a remote prefill has staged KV for this request's decode leg,
    /// which keeps decode selection going through a client disconnect.
    staged_kv_cleanup: AtomicBool,
    /// The host preview behind the coordinator's last `Preview`, so an
    /// admission from that preview continues its cleanup budget.
    pending_preview: Mutex<Option<(InvocationId, AttemptId, RoutePreview)>>,
}

impl HostStageSelector {
    pub(crate) fn new(
        host: Arc<RoutingHost>,
        phase: RequestPhase,
        pool: PoolRef,
        request: &SingleIn<PreprocessedRequest>,
    ) -> Self {
        Self {
            host,
            phase,
            pool,
            base: request.fork(request.content().clone()),
            staged_kv_cleanup: AtomicBool::new(request.content().staged_kv_cleanup),
            pending_preview: Mutex::new(None),
        }
    }

    /// Mark that a prefill worker holds KV staged for this request's decode.
    pub(crate) fn set_staged_kv_cleanup(&self, staged: bool) {
        self.staged_kv_cleanup.store(staged, Ordering::Release);
    }

    fn worker_ids(&self) -> HashSet<WorkerId> {
        self.host
            .kv_router()
            .workers_with_configs
            .borrow()
            .keys()
            .copied()
            .collect()
    }

    fn facts_for(&self, worker_id: WorkerId) -> Arc<WorkerFacts> {
        Arc::new(
            self.host
                .kv_router()
                .workers_with_configs
                .borrow()
                .get(&worker_id)
                .map(WorkerFacts::from_config)
                .unwrap_or_default(),
        )
    }

    /// The selection request: the base request with this stage's override,
    /// constraints, allowlist, projected output, and pins applied.
    pub(crate) fn render(&self, input: &SelectionInput<'_>) -> Context<PreprocessedRequest> {
        let mut body = self.base.content().clone();
        body.staged_kv_cleanup = self.staged_kv_cleanup.load(Ordering::Acquire);
        // Layer the stage profile over the caller's override, whichever side
        // carries it.
        let caller_override = input
            .settings
            .router_config_override
            .as_ref()
            .or(body.router_config_override.as_ref());
        if let Some(config_override) = input
            .profile
            .resolve_router_config_override(caller_override)
        {
            body.router_config_override = Some(config_override);
        }
        let restrictions = input.restrictions;
        if !restrictions.routing_constraints.is_empty() {
            let constraints = body
                .routing_mut()
                .routing_constraints
                .get_or_insert_with(Default::default);
            constraints.required_taints.extend(
                restrictions
                    .routing_constraints
                    .required_taints
                    .iter()
                    .cloned(),
            );
            for (taint, weight) in &restrictions.routing_constraints.preferred_taints {
                *constraints
                    .preferred_taints
                    .entry(taint.clone())
                    .or_insert(0.0) += weight;
            }
        }
        if let Some(allowed) = restrictions.effective_allowed_worker_ids(|| self.worker_ids()) {
            let routing = body.routing_mut();
            routing.allowed_worker_ids = Some(match routing.allowed_worker_ids.take() {
                Some(existing) => existing.intersection(&allowed).copied().collect(),
                None => allowed,
            });
        }
        if let Some(expected) = input.expected_output_tokens() {
            body.routing_mut().expected_output_tokens = Some(expected);
        }
        if let Some(pinned) = restrictions.pinned_worker {
            let routing = body.routing_mut();
            match self.phase {
                RequestPhase::Prefill => {
                    routing.prefill_worker_id = Some(pinned.worker_id);
                    routing.prefill_dp_rank = Some(pinned.dp_rank);
                }
                RequestPhase::Decode | RequestPhase::Aggregated => {
                    routing.decode_worker_id = Some(pinned.worker_id);
                    routing.dp_rank = Some(pinned.dp_rank);
                }
            }
        }
        self.base.fork(body)
    }

    fn target(&self, input: &SelectionInput<'_>, signals: RoutePlanSignals) -> SelectedTarget {
        SelectedTarget {
            invocation: input.invocation,
            attempt: input.attempt,
            stage: input.stage.clone(),
            pool: self.pool.clone(),
            worker: signals.worker,
            facts: self.facts_for(signals.worker.worker_id),
        }
    }

    fn signals(signals: RoutePlanSignals) -> SelectionSignals {
        SelectionSignals {
            overlap_blocks: signals.overlap_blocks,
            cached_tokens: signals.cached_tokens,
            potential_decode_blocks: signals.potential_decode_blocks,
            total_kv_blocks: signals.total_kv_blocks,
            prefill_load: signals.prefill_load.map(|load| PrefillLoadSignal {
                active_prefill_tokens: load.active_prefill_tokens,
                prefill_token_capacity: load.prefill_token_capacity,
            }),
        }
    }

    fn validate_preview(
        &self,
        input: &SelectionInput<'_>,
        preview: &Preview,
    ) -> Result<(), CoordinationError> {
        if preview.target.stage != *input.stage {
            return Err(CoordinationError::StalePreview {
                stage: input.stage.clone(),
                reason: format!("preview belongs to stage {}", preview.target.stage),
            });
        }
        if preview.target.pool != self.pool {
            return Err(CoordinationError::StalePreview {
                stage: input.stage.clone(),
                reason: format!(
                    "preview selected from pool {} but the stage now draws from {}",
                    preview.target.pool, self.pool
                ),
            });
        }
        if !input.restrictions.permits(preview.target.worker.worker_id) {
            return Err(CoordinationError::StalePreview {
                stage: input.stage.clone(),
                reason: format!(
                    "previewed worker {} is no longer permitted",
                    preview.target.worker.worker_id
                ),
            });
        }
        Ok(())
    }
}

#[async_trait]
impl StageSelector for HostStageSelector {
    async fn preview(&self, input: SelectionInput<'_>) -> Result<Preview, CoordinationError> {
        let request = self.render(&input);
        let route_preview = self
            .host
            .preview_kv_route(&request, self.phase)
            .await
            .map_err(CoordinationError::Selector)?;
        let signals = route_preview.signals;
        let target = self.target(&input, signals);
        *self.pending_preview.lock() = Some((input.invocation, input.attempt, route_preview));
        Ok(Preview {
            target,
            signals: Self::signals(signals),
        })
    }

    async fn admit(
        &self,
        input: SelectionInput<'_>,
        target: AdmissionTarget,
    ) -> Result<StageReservation, CoordinationError> {
        let request = self.render(&input);
        let plan = match target {
            AdmissionTarget::AnyEligible => self
                .host
                .admit_kv_route(&request, self.phase, None)
                .await
                .map_err(CoordinationError::Selector)?,
            AdmissionTarget::FromPreview(preview) => {
                self.validate_preview(&input, &preview)?;
                let stashed = self.pending_preview.lock().take().filter(
                    |(invocation, attempt, route_preview)| {
                        *invocation == input.invocation
                            && *attempt == input.attempt
                            && route_preview.signals.worker == preview.target.worker
                    },
                );
                match stashed {
                    // Continue the preview's cleanup budget.
                    Some((_, _, route_preview)) => self
                        .host
                        .plan_kv_route_from_preview(&request, route_preview)
                        .await
                        .map_err(CoordinationError::Selector)?,
                    None => self
                        .host
                        .admit_kv_route(&request, self.phase, Some(preview.target.worker))
                        .await
                        .map_err(CoordinationError::Selector)?,
                }
            }
        };
        let signals = plan.signals;
        let target = self.target(&input, signals);
        Ok(StageReservation::new(
            target,
            Self::signals(signals),
            ReservationLease::new(HostRoutePlan { plan: Some(plan) }),
        ))
    }
}

/// The frontend's reservation owner: an admitted route awaiting dispatch.
pub(crate) struct HostRoutePlan {
    plan: Option<RoutePlan>,
}

impl HostRoutePlan {
    /// Take the route plan out of a coordinator reservation for dispatch.
    pub(crate) fn take(reservation: StageReservation) -> anyhow::Result<RoutePlan> {
        let (target, _, lease) = reservation.into_parts();
        let mut owner = lease.into_owner::<HostRoutePlan>().map_err(|_| {
            anyhow!(
                "stage {} reservation was not admitted by the frontend routing host",
                target.stage
            )
        })?;
        owner
            .plan
            .take()
            .ok_or_else(|| anyhow!("stage {} route plan was already taken", target.stage))
    }
}

impl ReservationOwner for HostRoutePlan {
    fn release(mut self: Box<Self>) -> BoxFuture<'static, Result<(), CoordinationError>> {
        let plan = self.plan.take();
        Box::pin(async move {
            if let Some(plan) = plan {
                plan.abort().await;
            }
            Ok(())
        })
    }

    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
        self
    }

    fn describe(&self) -> String {
        match &self.plan {
            Some(plan) => format!(
                "frontend route plan for worker {} dp_rank {}",
                plan.signals.worker.worker_id, plan.signals.worker.dp_rank
            ),
            None => "frontend route plan (taken)".to_string(),
        }
    }
}

/// Lets one conditional-disaggregation decision policy serve many per-request
/// coordination policies.
#[derive(Clone)]
pub(crate) struct SharedConditionalPolicy(pub(crate) Arc<dyn ConditionalDisaggPolicy>);

#[async_trait]
impl ConditionalDisaggPolicy for SharedConditionalPolicy {
    fn is_enabled(&self) -> bool {
        self.0.is_enabled()
    }

    async fn should_bypass_remote_prefill(&self, input: ConditionalDisaggDecisionInput) -> bool {
        self.0.should_bypass_remote_prefill(input).await
    }

    fn needs_prefill_worker_busy(&self) -> bool {
        self.0.needs_prefill_worker_busy()
    }
}

/// The host-side error behind a coordination failure, for callers that match
/// on typed Dynamo errors (cancellation, invalid arguments).
pub(crate) fn host_error(error: &anyhow::Error) -> &anyhow::Error {
    match error.downcast_ref::<CoordinationError>() {
        Some(CoordinationError::Selector(inner)) => inner,
        _ => error,
    }
}

/// Translate a routing-host selection failure into the coordinator's
/// categories: scheduler rejections are retryable admissions, an empty or
/// filtered pool is "no eligible workers", anything else is host-specific.
fn map_router_error(stage: &StageId, error: anyhow::Error) -> CoordinationError {
    if let Some(scheduler) = error
        .chain()
        .find_map(|cause| cause.downcast_ref::<KvSchedulerError>())
    {
        return match scheduler {
            KvSchedulerError::NoEndpoints
            | KvSchedulerError::AllEligibleWorkersFiltered
            | KvSchedulerError::PinnedWorkerNotAllowed { .. } => {
                CoordinationError::NoEligibleWorkers {
                    stage: stage.clone(),
                    reason: scheduler.to_string(),
                }
            }
            KvSchedulerError::QueueRejected(_)
            | KvSchedulerError::AllEligibleWorkersOverloaded
            | KvSchedulerError::PinnedWorkerOverloaded { .. }
            | KvSchedulerError::DeadlineExceeded => CoordinationError::AdmissionRejected {
                stage: stage.clone(),
                reason: scheduler.to_string(),
            },
            _ => CoordinationError::Selector(error),
        };
    }
    if match_error_chain(
        error.as_ref(),
        &[
            ErrorType::ResourceExhausted,
            ErrorType::WorkerOverloaded,
            ErrorType::DeadlineExceeded,
        ],
        &[],
    ) {
        return CoordinationError::AdmissionRejected {
            stage: stage.clone(),
            reason: error.to_string(),
        };
    }
    if match_error_chain(error.as_ref(), &[ErrorType::Unavailable], &[]) {
        return CoordinationError::NoEligibleWorkers {
            stage: stage.clone(),
            reason: error.to_string(),
        };
    }
    CoordinationError::Selector(error)
}

fn worker_facts_from_router(router: &KvRouter, worker_id: WorkerId) -> Arc<WorkerFacts> {
    Arc::new(
        router
            .workers_with_configs
            .borrow()
            .get(&worker_id)
            .map(WorkerFacts::from_config)
            .unwrap_or_default(),
    )
}

fn validate_foreign_preview(
    pool: &PoolRef,
    input: &SelectionInput<'_>,
    preview: &Preview,
) -> Result<(), CoordinationError> {
    if preview.target.stage != *input.stage {
        return Err(CoordinationError::StalePreview {
            stage: input.stage.clone(),
            reason: format!("preview belongs to stage {}", preview.target.stage),
        });
    }
    if preview.target.pool != *pool {
        return Err(CoordinationError::StalePreview {
            stage: input.stage.clone(),
            reason: format!(
                "preview selected from pool {} but the stage now draws from {pool}",
                preview.target.pool
            ),
        });
    }
    if !input.restrictions.permits(preview.target.worker.worker_id) {
        return Err(CoordinationError::StalePreview {
            stage: input.stage.clone(),
            reason: format!(
                "previewed worker {} is no longer permitted",
                preview.target.worker.worker_id
            ),
        });
    }
    Ok(())
}

/// A coordinator stage selector over a [`KvRouter`] for hosts that do not
/// dispatch through a `RoutingHost` (the EPP): previews are advisory
/// selections, admissions book through the scheduler and return a
/// [`KvRouterReservation`] the host drives by response observation.
pub struct KvRouterStageSelector {
    router: Arc<KvRouter>,
    pool: PoolRef,
}

impl KvRouterStageSelector {
    pub fn new(router: Arc<KvRouter>, pool: PoolRef) -> Self {
        Self { router, pool }
    }

    pub fn router(&self) -> &Arc<KvRouter> {
        &self.router
    }

    pub fn pool(&self) -> &PoolRef {
        &self.pool
    }

    fn worker_ids(&self) -> HashSet<WorkerId> {
        self.router
            .workers_with_configs
            .borrow()
            .keys()
            .copied()
            .collect()
    }

    fn target(&self, input: &SelectionInput<'_>, worker: WorkerWithDpRank) -> SelectedTarget {
        SelectedTarget {
            invocation: input.invocation,
            attempt: input.attempt,
            stage: input.stage.clone(),
            pool: self.pool.clone(),
            worker,
            facts: worker_facts_from_router(&self.router, worker.worker_id),
        }
    }
}

#[async_trait]
impl StageSelector for KvRouterStageSelector {
    async fn preview(&self, input: SelectionInput<'_>) -> Result<Preview, CoordinationError> {
        let settings = input.settings;
        let restrictions = input.restrictions;
        let admitted = self
            .router
            .preview_best_match_details_with_policy_class(
                Some(input.request_id),
                input.prompt.token_ids,
                input.prompt.block_mm_infos,
                input.router_config_override().as_ref(),
                input.prompt.lora_name.map(str::to_string),
                input.prompt.cache_namespace.map(str::to_string),
                settings.priority_jump,
                settings.strict_priority,
                settings.policy_class.clone(),
                settings.session_context.clone(),
                input.expected_output_tokens(),
                restrictions.pinned_worker,
                restrictions.effective_allowed_worker_ids(|| self.worker_ids()),
                restrictions.routing_constraints.clone(),
            )
            .await
            .map_err(|error| map_router_error(input.stage, error))?;
        let advisory_load = admitted.advisory_load;
        match admitted.outcome {
            FindBestMatchOutcome::Routed {
                worker,
                overlap_blocks,
                cached_tokens,
                potential_decode_blocks,
                ..
            } => {
                let target = self.target(&input, worker);
                let signals = SelectionSignals {
                    overlap_blocks,
                    cached_tokens,
                    potential_decode_blocks,
                    total_kv_blocks: advisory_load
                        .and_then(|load| load.total_kv_blocks.map(|blocks| blocks as u64))
                        .or(target.facts.total_kv_blocks),
                    prefill_load: advisory_load.map(|load| PrefillLoadSignal {
                        active_prefill_tokens: load.active_prefill_tokens,
                        prefill_token_capacity: load.prefill_token_capacity,
                    }),
                };
                Ok(Preview { target, signals })
            }
            FindBestMatchOutcome::QueueRejected { rejection } => {
                Err(CoordinationError::AdmissionRejected {
                    stage: input.stage.clone(),
                    reason: rejection.to_string(),
                })
            }
        }
    }

    async fn admit(
        &self,
        input: SelectionInput<'_>,
        target: AdmissionTarget,
    ) -> Result<StageReservation, CoordinationError> {
        let pinned_worker = match &target {
            AdmissionTarget::AnyEligible => input.restrictions.pinned_worker,
            AdmissionTarget::FromPreview(preview) => {
                validate_foreign_preview(&self.pool, &input, preview)?;
                Some(preview.target.worker)
            }
        };
        let settings = input.settings;
        let restrictions = input.restrictions;
        let reservation_id = input.reservation_id();
        let admitted = self
            .router
            .find_best_match_details_with_policy_class_admitted(
                Some(&reservation_id),
                input.prompt.token_ids,
                input.prompt.block_mm_infos,
                input.router_config_override().as_ref(),
                true,
                false,
                input.prompt.lora_name.map(str::to_string),
                input.prompt.cache_namespace.map(str::to_string),
                settings.priority_jump,
                settings.strict_priority,
                settings.policy_class.clone(),
                settings.session_context.clone(),
                input.expected_output_tokens(),
                pinned_worker,
                restrictions.effective_allowed_worker_ids(|| self.worker_ids()),
                restrictions.routing_constraints.clone(),
            )
            .await
            .map_err(|error| map_router_error(input.stage, error))?;
        let (outcome, booking) = admitted.into_parts();
        match outcome {
            FindBestMatchOutcome::Routed {
                worker,
                overlap_blocks,
                cached_tokens,
                potential_decode_blocks,
                ..
            } => {
                let target = self.target(&input, worker);
                let signals = SelectionSignals {
                    overlap_blocks,
                    cached_tokens,
                    potential_decode_blocks,
                    total_kv_blocks: target.facts.total_kv_blocks,
                    prefill_load: None,
                };
                let reservation =
                    KvRouterReservation::new(Arc::clone(&self.router), worker, booking);
                Ok(StageReservation::new(
                    target,
                    signals,
                    ReservationLease::new(reservation),
                ))
            }
            FindBestMatchOutcome::QueueRejected { rejection } => {
                // Nothing was booked; a handle, if any, frees itself on drop.
                drop(booking);
                Err(CoordinationError::AdmissionRejected {
                    stage: input.stage.clone(),
                    reason: rejection.to_string(),
                })
            }
        }
    }
}

/// A booked KV-router selection a host drives from observed responses.
///
/// The booking lives in the router's request-lease manager, so a host that
/// never finishes it is reaped on expiry like any other lease.
pub struct KvRouterReservation {
    router: Arc<KvRouter>,
    worker: WorkerWithDpRank,
    lease: Option<RequestAttemptLease>,
}

impl KvRouterReservation {
    fn new(
        router: Arc<KvRouter>,
        worker: WorkerWithDpRank,
        booking: Option<BookingHandle>,
    ) -> Self {
        // Nothing awaits between taking the booking over and registering it.
        let lease = booking.map(|booking| {
            router
                .request_lease_manager()
                .register_local(booking.commit(), None)
        });
        Self {
            router,
            worker,
            lease,
        }
    }

    pub fn worker(&self) -> WorkerWithDpRank {
        self.worker
    }

    /// Whether the scheduler still tracks this booking.
    pub fn is_active(&self) -> bool {
        self.lease
            .as_ref()
            .is_some_and(RequestAttemptLease::is_active)
    }

    /// Record that the worker finished prefill for this request.
    pub async fn prefill_complete(&self) -> Result<(), CoordinationError> {
        let Some(lease) = self.lease.as_ref().filter(|lease| lease.is_active()) else {
            return Ok(());
        };
        lease.touch();
        self.router
            .mark_prefill_completed_if_booking(lease.booking())
            .await
            .map_err(|error| CoordinationError::Selector(error.into()))
    }

    /// Record generated output blocks for this request.
    pub async fn add_output_blocks(
        &self,
        num_blocks: usize,
        decay_fraction: Option<f64>,
    ) -> Result<(), CoordinationError> {
        let Some(lease) = self.lease.as_ref().filter(|lease| lease.is_active()) else {
            return Ok(());
        };
        lease.touch();
        self.router
            .add_output_blocks_if_booking(lease.booking(), num_blocks, decay_fraction)
            .await
            .map_err(|error| CoordinationError::Selector(error.into()))
    }

    /// Release the booking and wait for the scheduler. Idempotent.
    pub async fn finish(&self) {
        if let Some(lease) = &self.lease {
            lease.finish().await;
        }
    }
}

impl ReservationOwner for KvRouterReservation {
    fn release(self: Box<Self>) -> BoxFuture<'static, Result<(), CoordinationError>> {
        Box::pin(async move {
            self.finish().await;
            Ok(())
        })
    }

    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
        self
    }

    fn describe(&self) -> String {
        format!(
            "kv router booking on worker {} dp_rank {}",
            self.worker.worker_id, self.worker.dp_rank
        )
    }
}

/// A coordinator stage selector over a [`PrefillRouter`]'s reservation API,
/// for hosts that route prefill by header (the EPP). Previews are not
/// supported; admissions return the router's own [`PrefillReservation`].
pub struct PrefillRouterStageSelector {
    prefill_router: Arc<PrefillRouter>,
    pool: PoolRef,
}

impl PrefillRouterStageSelector {
    pub fn new(prefill_router: Arc<PrefillRouter>, pool: PoolRef) -> Self {
        Self {
            prefill_router,
            pool,
        }
    }

    pub fn pool(&self) -> &PoolRef {
        &self.pool
    }
}

fn map_prefill_error(stage: &StageId, error: anyhow::Error) -> CoordinationError {
    if error.chain().any(|cause| {
        matches!(
            cause.downcast_ref::<PrefillError>(),
            Some(PrefillError::NotActivated)
        )
    }) {
        return CoordinationError::NoEligibleWorkers {
            stage: stage.clone(),
            reason: "no prefill workers are active".to_string(),
        };
    }
    let message = error.to_string();
    if message.contains("queue rejection") {
        return CoordinationError::AdmissionRejected {
            stage: stage.clone(),
            reason: message,
        };
    }
    if message.contains("No workers available") {
        return CoordinationError::NoEligibleWorkers {
            stage: stage.clone(),
            reason: message,
        };
    }
    map_router_error(stage, error)
}

#[async_trait]
impl StageSelector for PrefillRouterStageSelector {
    async fn preview(&self, input: SelectionInput<'_>) -> Result<Preview, CoordinationError> {
        Err(CoordinationError::PreviewUnsupported {
            stage: input.stage.clone(),
        })
    }

    async fn admit(
        &self,
        input: SelectionInput<'_>,
        target: AdmissionTarget,
    ) -> Result<StageReservation, CoordinationError> {
        if let AdmissionTarget::FromPreview(_) = target {
            return Err(CoordinationError::PreviewUnsupported {
                stage: input.stage.clone(),
            });
        }
        let settings = input.settings;
        let restrictions = input.restrictions;
        let reservation_id = input.reservation_id();
        let reservation = self
            .prefill_router
            .reserve_prefill_worker(
                &reservation_id,
                input.prompt.token_ids,
                input.prompt.block_mm_infos,
                input.prompt.lora_name.map(str::to_string),
                input.prompt.cache_namespace.map(str::to_string),
                settings.priority_jump,
                settings.strict_priority,
                settings.policy_class.clone(),
                restrictions.allowed_worker_ids.clone(),
                restrictions.routing_constraints.clone(),
            )
            .await
            .map_err(|error| map_prefill_error(input.stage, error))?;
        let worker =
            WorkerWithDpRank::new(reservation.worker_id(), reservation.dp_rank().unwrap_or(0));
        if !restrictions.permits(worker.worker_id) {
            // Exclusions are not part of the reservation API; enforce them here.
            reservation
                .release()
                .await
                .map_err(|error| CoordinationError::Release(error.to_string()))?;
            return Err(CoordinationError::NoEligibleWorkers {
                stage: input.stage.clone(),
                reason: format!("prefill selected excluded worker {}", worker.worker_id),
            });
        }
        let facts = Arc::new(
            self.prefill_router
                .prefill_worker_facts(worker.worker_id)
                .unwrap_or_default(),
        );
        let target = SelectedTarget {
            invocation: input.invocation,
            attempt: input.attempt,
            stage: input.stage.clone(),
            pool: self.pool.clone(),
            worker,
            facts,
        };
        Ok(StageReservation::new(
            target,
            SelectionSignals::default(),
            ReservationLease::new(reservation),
        ))
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use dynamo_kv_router::protocols::RoutingConstraints;

    use super::*;
    use crate::protocols::common::preprocessor::RoutingHints;

    fn request() -> PreprocessedRequest {
        PreprocessedRequest::builder()
            .model("test".to_string())
            .token_ids(vec![1, 2, 3, 4])
            .stop_conditions(Default::default())
            .sampling_options(Default::default())
            .output_options(Default::default())
            .build()
            .unwrap()
    }

    #[test]
    fn routing_request_shares_tokens_and_records_pinned_stages() {
        let mut body = request();
        body.routing = Some(RoutingHints {
            prefill_worker_id: Some(7),
            expected_output_tokens: Some(32),
            priority_jump: Some(1.5),
            lora_name: Some("adapter".to_string()),
            ..Default::default()
        });
        let routing_request = routing_request_for(&body, "req-1", Some("batch".to_string()));
        assert!(Arc::ptr_eq(
            &routing_request.prompt.token_ids,
            &body.token_ids
        ));
        assert_eq!(routing_request.facts.prompt_tokens, 4);
        assert!(
            routing_request
                .facts
                .pinned_stages
                .contains(&StageId::PREFILL)
        );
        assert!(
            !routing_request
                .facts
                .pinned_stages
                .contains(&StageId::DECODE)
        );
        assert_eq!(routing_request.settings.expected_output_tokens, Some(32));
        assert_eq!(routing_request.settings.priority_jump, 1.5);
        assert_eq!(
            routing_request.settings.policy_class.as_deref(),
            Some("batch")
        );
        assert_eq!(routing_request.prompt.lora_name.as_deref(), Some("adapter"));

        let mut both = request();
        both.routing = Some(RoutingHints {
            backend_instance_id: Some(3),
            ..Default::default()
        });
        let routing_request = routing_request_for(&both, "req-2", None);
        assert!(
            routing_request
                .facts
                .pinned_stages
                .contains(&StageId::PREFILL)
        );
        assert!(
            routing_request
                .facts
                .pinned_stages
                .contains(&StageId::DECODE)
        );
    }

    #[test]
    fn host_error_unwraps_selector_failures() {
        let inner = anyhow!("worker rejected");
        let wrapped = anyhow::Error::from(CoordinationError::Selector(inner));
        assert_eq!(host_error(&wrapped).to_string(), "worker rejected");
        let other = anyhow::Error::from(CoordinationError::Finished);
        assert!(host_error(&other).to_string().contains("finished"));
        let plain = anyhow!("plain");
        assert_eq!(host_error(&plain).to_string(), "plain");
    }

    #[test]
    fn merged_constraints_add_preferred_weights() {
        // The render path merges coordinator constraints into the request's
        // own; exercise the arithmetic it relies on.
        let mut constraints = RoutingConstraints {
            required_taints: HashSet::from(["a".to_string()]),
            preferred_taints: HashMap::from([("zone".to_string(), 0.5)]),
        };
        let derived = RoutingConstraints {
            required_taints: HashSet::from(["b".to_string()]),
            preferred_taints: HashMap::from([("zone".to_string(), 0.25)]),
        };
        constraints
            .required_taints
            .extend(derived.required_taints.iter().cloned());
        for (taint, weight) in &derived.preferred_taints {
            *constraints
                .preferred_taints
                .entry(taint.clone())
                .or_insert(0.0) += weight;
        }
        assert_eq!(constraints.required_taints.len(), 2);
        assert_eq!(constraints.preferred_taints["zone"], 0.75);
    }
}
