// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Stage-based routing coordination (DEP #15457).
//!
//! A request runs through one or more named stages (aggregated, prefill,
//! decode, encode). Each stage is bound to a worker pool through a
//! [`StageSelector`] that can preview a candidate or admit and reserve one.
//! A [`Topology`] declares the execution paths, their handoff dependencies,
//! and the placement rules that connect stages. A [`CoordinationPolicy`]
//! chooses the selection order, branch, and profiles for one request, and the
//! [`RoutingCoordinator`] runs it: it enforces constraints, bounds admission
//! waits and reservation holds, and hands each admitted stage to the host once
//! its inputs are ready, or exports a complete [`DelegatedPlan`].
//!
//! The contracts have no runtime or transport dependencies. Host adapters
//! (the frontend in `dynamo-llm`, the EPP in `dynamo-ext-proc`) implement
//! [`StageSelector`] over their own selection state; [`CoreStageSelector`]
//! implements it over a [`SelectionCore`](crate::services::selection::SelectionCore)
//! partition for hosts that embed one.

mod builtin_policies;
mod coordinator;
mod error;
mod ids;
mod placement;
mod policy;
mod registry;
mod selector;
mod session;
mod stage;
mod topology;

#[cfg(feature = "standalone-selection")]
mod core_selector;

#[cfg(any(test, feature = "testing"))]
pub mod test_support;

#[cfg(test)]
mod coordinator_tests;
#[cfg(test)]
mod tests;

pub use builtin_policies::{
    AggregatedPolicy, ConditionalDisaggThresholds, ConditionalDisaggregationPolicy,
    EncodePrefillDecodePolicy, PrefillDecodePolicy, ProgressivePrefillDecodePolicy, SelectionOrder,
    decode_gate_allows_bypass,
};
pub use coordinator::{
    CoordinationLimits, DelegatedPlan, HostAction, HostEvent, PlannedStage, ReadyStage,
    RoutingCoordinator,
};
pub use error::CoordinationError;
pub use ids::{AttemptId, BranchId, InvocationId, PoolId, ProfileName, StageId};
pub use placement::{TOPOLOGY_TAINT_PREFIX, derived_restrictions, topology_taint, validate_pair};
pub use policy::{
    AdmissionIntent, AdmissionSource, CoordinationOp, CoordinationPolicy,
    CoordinationPolicyFactory, CoordinationView, PlanningMode, RequestFacts, SelectionIntent,
    StageStatus, StageView,
};
pub use registry::{
    AGGREGATED_POLICY, CONDITIONAL_DISAGGREGATION_POLICY, CoordinationPolicyRegistry,
    CoordinationPolicyRegistryError, DECODE_FIRST_POLICY, ENCODE_PREFILL_DECODE_POLICY,
    PREFILL_FIRST_POLICY, PROGRESSIVE_PREFILL_DECODE_POLICY,
};
pub use selector::{
    AdmissionTarget, PrefillLoadSignal, Preview, PromptInputView, RequestSettings,
    ReservationLease, ReservationOwner, SelectedTarget, SelectionInput, SelectionRestrictions,
    SelectionSignals, StageReservation, StageSelector, WorkerFacts,
};
pub use session::{PromptInput, RouteSession, RoutingRequest, StageState};
pub use stage::{
    PoolRef, ScoringMode, StageBinding, StageCapabilities, StageProfile, StageProfiles,
    WorkAccounting,
};
pub use topology::{
    Branch, DEFAULT_BRANCH, ENCODE_PREFILL_DECODE_BRANCH, Handoff, LOCAL_PREFILL_DECODE_BRANCH,
    PREFILL_DECODE_BRANCH, PlacementMode, PlacementRule, REMOTE_PREFILL_DECODE_BRANCH, Topology,
};

#[cfg(feature = "standalone-selection")]
pub use core_selector::{CoreAdmissionMode, CoreBooking, CoreStageSelector};
