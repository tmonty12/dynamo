// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::time::Duration;

use super::ids::{BranchId, ProfileName, StageId};

/// Why a coordination step could not complete.
#[derive(Debug, thiserror::Error)]
pub enum CoordinationError {
    #[error("stage {stage} is not bound in this topology")]
    UnknownStage { stage: StageId },

    #[error("branch {branch} is not declared by this topology")]
    UnknownBranch { branch: BranchId },

    #[error("stage {stage} has no selection profile {profile}")]
    UnknownProfile {
        stage: StageId,
        profile: ProfileName,
    },

    #[error("stage {stage} does not support previews")]
    PreviewUnsupported { stage: StageId },

    #[error("preview for stage {stage} is stale: {reason}")]
    StalePreview { stage: StageId, reason: String },

    #[error("no eligible workers for stage {stage}: {reason}")]
    NoEligibleWorkers { stage: StageId, reason: String },

    /// The scheduler declined to admit the request now (queue limits,
    /// overload, or an unavailable pinned worker).
    #[error("admission for stage {stage} rejected: {reason}")]
    AdmissionRejected { stage: StageId, reason: String },

    #[error("placement rule violated between stages {first} and {second}: {reason}")]
    Placement {
        first: StageId,
        second: StageId,
        reason: String,
    },

    #[error("caller restrictions for stage {stage} conflict: {reason}")]
    ConflictingRestrictions { stage: StageId, reason: String },

    #[error("planning deadline exceeded after {elapsed:?}")]
    DeadlineExceeded { elapsed: Duration },

    #[error("routing session already finished")]
    Finished,

    #[error("coordination policy returned an invalid operation: {0}")]
    InvalidPolicyOperation(String),

    #[error("reservation release failed: {0}")]
    Release(String),

    /// A host-specific selector failure that is not one of the categories above.
    #[error(transparent)]
    Selector(#[from] anyhow::Error),
}

impl CoordinationError {
    /// Whether another admission attempt for the same stage could succeed
    /// without changing the request.
    pub fn is_retryable(&self) -> bool {
        matches!(self, Self::AdmissionRejected { .. })
    }
}
