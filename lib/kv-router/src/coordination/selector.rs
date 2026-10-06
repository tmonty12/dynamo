// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The per-stage selection contract: preview a candidate, or admit and reserve
//! one, and own that reservation until exactly one owner releases it.
//!
//! # Ownership
//!
//! A [`StageReservation`] is linear: it is not `Clone`, it is `#[must_use]`,
//! and its [`ReservationLease`] releases the exact attempt it admitted when
//! dropped. Moving a reservation from the coordinator's session to a host
//! guard is a Rust move, so at every await point there is one owner that can
//! release it.

use std::any::Any;
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;
use futures_util::future::BoxFuture;

use crate::protocols::{
    BlockExtraInfo, KvTransferEnforcement, RoutingConstraints, WorkerAffinityTarget,
    WorkerConfigLike, WorkerId, WorkerWithDpRank,
};
use crate::scheduling::SessionContext;
use crate::scheduling::config::RouterConfigOverride;

use super::error::CoordinationError;
use super::ids::{AttemptId, InvocationId, StageId};
use super::stage::{PoolRef, StageProfile};

/// Metadata for one worker instance, captured when a stage selects it.
///
/// Placement rules read these facts to constrain later stages, so a selection
/// retains them even after the worker leaves discovery.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct WorkerFacts {
    pub data_parallel_start_rank: u32,
    pub data_parallel_size: u32,
    pub total_kv_blocks: Option<u64>,
    pub stable_routing_id: Option<String>,
    pub taints: HashSet<String>,
    pub topology_domains: HashMap<String, String>,
    pub kv_transfer_domain: Option<String>,
    pub kv_transfer_enforcement: Option<KvTransferEnforcement>,
    pub kv_transfer_preferred_weight: Option<f32>,
}

impl WorkerFacts {
    pub fn from_config<C: WorkerConfigLike + ?Sized>(config: &C) -> Self {
        Self {
            data_parallel_start_rank: config.data_parallel_start_rank(),
            data_parallel_size: config.data_parallel_size(),
            total_kv_blocks: config.total_kv_blocks(),
            stable_routing_id: config.stable_routing_id().map(str::to_string),
            taints: config.taints().clone(),
            topology_domains: config.topology_domains().cloned().unwrap_or_default(),
            kv_transfer_domain: config.kv_transfer_domain().map(str::to_string),
            kv_transfer_enforcement: config.kv_transfer_enforcement(),
            kv_transfer_preferred_weight: config.kv_transfer_preferred_weight(),
        }
    }

    /// The worker's value for one topology domain, such as `zone`.
    pub fn topology_value(&self, domain: &str) -> Option<&str> {
        self.topology_domains.get(domain).map(String::as_str)
    }
}

/// A worker chosen for one stage invocation.
#[derive(Debug, Clone)]
pub struct SelectedTarget {
    /// Logical stage invocation, stable across retries.
    pub invocation: InvocationId,
    /// The attempt this target belongs to.
    pub attempt: AttemptId,
    pub stage: StageId,
    /// Retains the selected binding generation.
    pub pool: PoolRef,
    pub worker: WorkerWithDpRank,
    /// Metadata for this worker instance.
    pub facts: Arc<WorkerFacts>,
}

/// The selected worker's prefill queue at selection time.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct PrefillLoadSignal {
    pub active_prefill_tokens: usize,
    pub prefill_token_capacity: usize,
}

impl PrefillLoadSignal {
    pub fn exceeds(&self, threshold: f64) -> bool {
        self.active_prefill_tokens as f64 > threshold * self.prefill_token_capacity as f64
    }
}

/// Cache and load signals a selector reports for its chosen worker.
///
/// Previews carry these so a policy can decide without reserving; admitted
/// selections carry them for accounting and logging.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct SelectionSignals {
    /// Device-local prefix blocks the worker already holds for this prompt.
    pub overlap_blocks: u32,
    /// Weighted cache credit on the worker, in tokens.
    pub cached_tokens: usize,
    /// Projected KV blocks on the worker once this request decodes.
    pub potential_decode_blocks: u64,
    pub total_kv_blocks: Option<u64>,
    /// Present for previews, which read the scheduler's load snapshot.
    pub prefill_load: Option<PrefillLoadSignal>,
}

impl SelectionSignals {
    /// Whether the projected decode occupancy exceeds `threshold` of capacity.
    /// `None` when capacity is unknown.
    pub fn decode_load_exceeds(&self, threshold: f64) -> Option<bool> {
        let total_kv_blocks = self.total_kv_blocks?;
        Some(self.potential_decode_blocks as f64 > threshold * total_kv_blocks as f64)
    }

    /// Whether the worker's prefill queue exceeds `threshold` of capacity.
    /// `None` when the signal was not reported.
    pub fn prefill_load_exceeds(&self, threshold: f64) -> Option<bool> {
        self.prefill_load.map(|load| load.exceeds(threshold))
    }
}

/// A candidate and its signals, chosen without reserving it.
#[derive(Debug, Clone)]
pub struct Preview {
    pub target: SelectedTarget,
    pub signals: SelectionSignals,
}

/// The release side of one admitted attempt.
///
/// Implementations must release on drop as well as on [`release`]: a lease
/// dropped at a cancellation point must not leak its booking.
///
/// [`release`]: ReservationOwner::release
pub trait ReservationOwner: Send + 'static {
    /// Release the exact attempt this owner admitted and wait for the
    /// scheduler to acknowledge it. Called at most once.
    fn release(self: Box<Self>) -> BoxFuture<'static, Result<(), CoordinationError>>;

    /// Hand the owner to a host that knows its concrete type.
    fn into_any(self: Box<Self>) -> Box<dyn Any + Send>;

    /// A short description for logs.
    fn describe(&self) -> String {
        "reservation".to_string()
    }
}

/// Owns cleanup of exactly one admitted attempt.
///
/// Dropping the lease releases the reservation through its owner. An untracked
/// lease (an advisory selection with nothing to release) is also valid.
#[must_use = "dropping a lease releases its reservation; call `release` to wait for the scheduler"]
pub struct ReservationLease {
    owner: Option<Box<dyn ReservationOwner>>,
}

impl ReservationLease {
    pub fn new(owner: impl ReservationOwner) -> Self {
        Self {
            owner: Some(Box::new(owner)),
        }
    }

    /// A lease with nothing to release.
    pub fn untracked() -> Self {
        Self { owner: None }
    }

    pub fn is_tracked(&self) -> bool {
        self.owner.is_some()
    }

    /// Release now and wait for the scheduler to acknowledge it.
    pub async fn release(mut self) -> Result<(), CoordinationError> {
        match self.owner.take() {
            Some(owner) => owner.release().await,
            None => Ok(()),
        }
    }

    /// Take the owner as its concrete type, for hosts that dispatch through
    /// state the owner carries. Returns the lease unchanged when the owner is
    /// a different type or the lease is untracked.
    pub fn into_owner<T: ReservationOwner>(mut self) -> Result<Box<T>, Self> {
        let Some(owner) = self.owner.take() else {
            return Err(self);
        };
        match owner.into_any().downcast::<T>() {
            Ok(owner) => Ok(owner),
            Err(any) => {
                // Restore the owner: `Box<dyn Any>` can not be turned back into
                // `Box<dyn ReservationOwner>`, so wrap it in an owner that
                // releases through the original on drop.
                self.owner = Some(Box::new(OpaqueOwner(Some(any))));
                Err(self)
            }
        }
    }
}

/// Retains a downcast miss so the original owner still releases on drop.
struct OpaqueOwner(Option<Box<dyn Any + Send>>);

impl ReservationOwner for OpaqueOwner {
    fn release(mut self: Box<Self>) -> BoxFuture<'static, Result<(), CoordinationError>> {
        // The boxed `Any` drops here, running the original owner's drop
        // release. Its acknowledgement can not be awaited through `Any`.
        drop(self.0.take());
        Box::pin(async { Ok(()) })
    }

    fn into_any(self: Box<Self>) -> Box<dyn Any + Send> {
        self
    }

    fn describe(&self) -> String {
        "reservation (opaque)".to_string()
    }
}

impl fmt::Debug for ReservationLease {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.owner {
            Some(owner) => formatter
                .debug_struct("ReservationLease")
                .field("owner", &owner.describe())
                .finish(),
            None => formatter.write_str("ReservationLease(untracked)"),
        }
    }
}

/// An admitted stage selection together with the lease that owns its cleanup.
///
/// Not `Clone`: ownership moves from the coordinator's session to a host guard
/// before dispatch or export, and whoever holds it last releases it.
#[must_use = "dropping a reservation releases it; move it to its owner or call `release`"]
#[derive(Debug)]
pub struct StageReservation {
    pub target: SelectedTarget,
    pub signals: SelectionSignals,
    lease: ReservationLease,
}

impl StageReservation {
    pub fn new(target: SelectedTarget, signals: SelectionSignals, lease: ReservationLease) -> Self {
        Self {
            target,
            signals,
            lease,
        }
    }

    pub fn lease(&self) -> &ReservationLease {
        &self.lease
    }

    pub fn into_parts(self) -> (SelectedTarget, SelectionSignals, ReservationLease) {
        (self.target, self.signals, self.lease)
    }

    /// Release the reservation now and wait for the scheduler.
    pub async fn release(self) -> Result<(), CoordinationError> {
        self.lease.release().await
    }
}

/// What an admission may choose.
#[derive(Debug)]
pub enum AdmissionTarget {
    /// Any worker the stage's restrictions allow.
    AnyEligible,
    /// Reserve exactly the previewed worker and rank, or fail.
    FromPreview(Preview),
}

/// Caller and cross-stage restrictions on which workers a stage may select.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SelectionRestrictions {
    /// Candidates are limited to this set when present.
    pub allowed_worker_ids: Option<HashSet<WorkerId>>,
    /// Workers earlier attempts failed on; never selected again for this invocation.
    pub excluded_worker_ids: HashSet<WorkerId>,
    /// An exact worker and rank the selection must use.
    pub pinned_worker: Option<WorkerWithDpRank>,
    /// A worker the selection should prefer.
    pub affinity_target: Option<WorkerAffinityTarget>,
    pub routing_constraints: RoutingConstraints,
}

impl SelectionRestrictions {
    /// Combine two restriction sets for the same stage.
    ///
    /// Allowlists intersect, exclusions and required taints union, preferred
    /// taint weights add, and a pin is preserved only when both sides agree.
    pub fn merge(&self, other: &Self, stage: &StageId) -> Result<Self, CoordinationError> {
        let pinned_worker = match (self.pinned_worker, other.pinned_worker) {
            (Some(left), Some(right)) if left != right => {
                return Err(CoordinationError::ConflictingRestrictions {
                    stage: stage.clone(),
                    reason: format!(
                        "pinned to worker {} dp_rank {} and worker {} dp_rank {}",
                        left.worker_id, left.dp_rank, right.worker_id, right.dp_rank
                    ),
                });
            }
            (left, right) => left.or(right),
        };
        let affinity_target = match (self.affinity_target, other.affinity_target) {
            (Some(left), Some(right)) if left != right => {
                return Err(CoordinationError::ConflictingRestrictions {
                    stage: stage.clone(),
                    reason: format!(
                        "affinity to worker {} and worker {}",
                        left.worker_id, right.worker_id
                    ),
                });
            }
            (left, right) => left.or(right),
        };
        let allowed_worker_ids = match (&self.allowed_worker_ids, &other.allowed_worker_ids) {
            (Some(left), Some(right)) => Some(left.intersection(right).copied().collect()),
            (Some(only), None) | (None, Some(only)) => Some(only.clone()),
            (None, None) => None,
        };
        let mut excluded_worker_ids = self.excluded_worker_ids.clone();
        excluded_worker_ids.extend(other.excluded_worker_ids.iter().copied());

        let mut routing_constraints = self.routing_constraints.clone();
        routing_constraints
            .required_taints
            .extend(other.routing_constraints.required_taints.iter().cloned());
        for (taint, weight) in &other.routing_constraints.preferred_taints {
            *routing_constraints
                .preferred_taints
                .entry(taint.clone())
                .or_insert(0.0) += weight;
        }

        if let Some(pinned) = pinned_worker
            && excluded_worker_ids.contains(&pinned.worker_id)
        {
            return Err(CoordinationError::ConflictingRestrictions {
                stage: stage.clone(),
                reason: format!("pinned worker {} is excluded", pinned.worker_id),
            });
        }
        if let (Some(pinned), Some(allowed)) = (pinned_worker, &allowed_worker_ids)
            && !allowed.contains(&pinned.worker_id)
        {
            return Err(CoordinationError::ConflictingRestrictions {
                stage: stage.clone(),
                reason: format!("pinned worker {} is not in the allowlist", pinned.worker_id),
            });
        }

        Ok(Self {
            allowed_worker_ids,
            excluded_worker_ids,
            pinned_worker,
            affinity_target,
            routing_constraints,
        })
    }

    /// Whether these restrictions let `worker_id` be selected, ignoring taints.
    pub fn permits(&self, worker_id: WorkerId) -> bool {
        if self.excluded_worker_ids.contains(&worker_id) {
            return false;
        }
        if let Some(pinned) = self.pinned_worker
            && pinned.worker_id != worker_id
        {
            return false;
        }
        self.allowed_worker_ids
            .as_ref()
            .is_none_or(|allowed| allowed.contains(&worker_id))
    }

    /// The allowlist a selector should pass to the scheduler, with exclusions
    /// applied. `universe` supplies the candidate set when there is no
    /// allowlist but there are exclusions.
    pub fn effective_allowed_worker_ids(
        &self,
        universe: impl FnOnce() -> HashSet<WorkerId>,
    ) -> Option<HashSet<WorkerId>> {
        if self.excluded_worker_ids.is_empty() {
            return self.allowed_worker_ids.clone();
        }
        let mut allowed = self.allowed_worker_ids.clone().unwrap_or_else(universe);
        allowed.retain(|worker_id| !self.excluded_worker_ids.contains(worker_id));
        Some(allowed)
    }
}

/// Request-wide settings every stage shares.
#[derive(Debug, Clone, Default)]
pub struct RequestSettings {
    pub priority_jump: f64,
    pub strict_priority: u32,
    pub policy_class: Option<String>,
    pub session_context: Option<SessionContext>,
    pub expected_output_tokens: Option<u32>,
    /// The caller's per-request router override; profiles layer over it.
    pub router_config_override: Option<RouterConfigOverride>,
}

/// The prompt as a stage selector consumes it. Shared across stages.
#[derive(Debug, Clone, Copy)]
pub struct PromptInputView<'a> {
    pub token_ids: &'a [u32],
    pub block_mm_infos: Option<&'a [Option<BlockExtraInfo>]>,
    pub lora_name: Option<&'a str>,
    pub cache_namespace: Option<&'a str>,
}

/// Everything a selector needs for one preview or admission.
#[derive(Debug, Clone, Copy)]
pub struct SelectionInput<'a> {
    pub request_id: &'a str,
    pub stage: &'a StageId,
    pub invocation: InvocationId,
    pub attempt: AttemptId,
    pub prompt: PromptInputView<'a>,
    pub profile: &'a StageProfile,
    /// Caller restrictions merged with the cross-stage constraints derived
    /// from earlier selections.
    pub restrictions: &'a SelectionRestrictions,
    pub settings: &'a RequestSettings,
}

/// The scheduler-facing id for one stage attempt's booking: unique per
/// request, stage, and attempt, so a retry never collides with the booking it
/// replaces and a host can address lifecycle calls without keeping a map.
pub fn reservation_id(request_id: &str, stage: &StageId, attempt: AttemptId) -> String {
    format!("{request_id}/{stage}/{attempt}")
}

impl SelectionInput<'_> {
    /// The scheduler-facing id for this attempt's booking; see [`reservation_id`].
    pub fn reservation_id(&self) -> String {
        reservation_id(self.request_id, self.stage, self.attempt)
    }

    /// The router override to apply: the caller's, layered with the profile.
    pub fn router_config_override(&self) -> Option<RouterConfigOverride> {
        self.profile
            .resolve_router_config_override(self.settings.router_config_override.as_ref())
    }

    pub fn expected_output_tokens(&self) -> Option<u32> {
        self.profile
            .resolve_expected_output_tokens(self.settings.expected_output_tokens)
    }
}

/// Preview or admit a worker for one stage.
#[async_trait]
pub trait StageSelector: Send + Sync {
    /// Inspect a candidate and its cache and load signals without reserving
    /// it or creating an affinity binding.
    async fn preview(&self, input: SelectionInput<'_>) -> Result<Preview, CoordinationError>;

    /// Select and reserve a worker, or reserve the exact worker and rank from a
    /// preview after rechecking its eligibility and capacity.
    async fn admit(
        &self,
        input: SelectionInput<'_>,
        target: AdmissionTarget,
    ) -> Result<StageReservation, CoordinationError>;
}
