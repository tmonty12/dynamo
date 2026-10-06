// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Stage-based routing coordination (DEP #15457).
//!
//! A request runs through one or more named stages (aggregated, prefill,
//! decode, encode). Each stage is bound to a worker pool through a
//! [`StageSelector`] that can preview a candidate or admit and reserve one.
//! This module holds those contracts: the identifiers, the stage
//! configuration, the selection input and signals, and the linear ownership
//! of a [`StageReservation`] from admission to release.
//!
//! The contracts have no runtime or transport dependencies. Host adapters
//! (the frontend in `dynamo-llm`, the EPP in `dynamo-ext-proc`) implement
//! [`StageSelector`] over their own selection state; [`CoreStageSelector`]
//! implements it over a [`SelectionCore`](crate::services::selection::SelectionCore)
//! partition for hosts that embed one.

mod error;
mod ids;
mod selector;
mod stage;

#[cfg(feature = "standalone-selection")]
mod core_selector;

#[cfg(any(test, feature = "testing"))]
pub mod test_support;

#[cfg(test)]
mod tests;

pub use error::CoordinationError;
pub use ids::{AttemptId, BranchId, InvocationId, PoolId, ProfileName, StageId};
pub use selector::{
    AdmissionTarget, PrefillLoadSignal, Preview, PromptInputView, RequestSettings,
    ReservationLease, ReservationOwner, SelectedTarget, SelectionInput, SelectionRestrictions,
    SelectionSignals, StageReservation, StageSelector, WorkerFacts,
};
pub use stage::{
    PoolRef, ScoringMode, StageBinding, StageCapabilities, StageProfile, StageProfiles,
    WorkAccounting,
};

#[cfg(feature = "standalone-selection")]
pub use core_selector::{CoreAdmissionMode, CoreBooking, CoreStageSelector};
