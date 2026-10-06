// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The coordination policy contract: a compiled policy chooses the next
//! selection operation for one request from a read-only view of the session.
//!
//! Policies decide *which* stage to preview or admit, with which profile, and
//! which branch to take. The coordinator decides *when* an admitted stage may
//! run, from the topology's handoff dependencies; a policy can neither dispatch
//! work nor change dependencies.
//!
//! Built-in policies derive their next operation from the view rather than
//! from private counters, so a retry that returns a stage to `Pending` is
//! handled by the same code path as the first attempt.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use async_trait::async_trait;

use super::error::CoordinationError;
use super::ids::{AttemptId, BranchId, ProfileName, StageId};
use super::selector::{Preview, SelectedTarget, SelectionSignals};
use super::stage::StageCapabilities;
use super::topology::{Branch, Topology};

/// What the coordinator knows about a request before any selection.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RequestFacts {
    /// Prompt length in tokens.
    pub prompt_tokens: usize,
    /// The request carries input an encoder stage must process first.
    pub requires_encode: bool,
    /// Stages the caller pinned to an exact worker; a policy treats these as
    /// decided (for example, a gateway that already chose the prefill worker).
    pub pinned_stages: HashSet<StageId>,
}

/// How a session is driven.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlanningMode {
    /// Stages are handed to the host as their inputs become ready; the policy
    /// may wait for execution results between selections.
    Progressive,
    /// Every stage is selected before anything runs; a policy that waits for
    /// execution results is rejected.
    Upfront,
}

/// Where a stage stands in its current attempt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StageStatus {
    /// Not yet selected for this attempt.
    Pending,
    /// A preview exists; nothing is reserved.
    Previewed,
    /// Reserved and owned by the session.
    Admitted,
    /// The reservation moved to the host for dispatch.
    Executing,
    /// The host reported dispatch.
    Dispatched,
    /// The host reported successful completion.
    Completed,
    /// The host reported a failure that will not be retried.
    Failed,
}

impl StageStatus {
    /// Whether a reservation exists for the current attempt, in either owner.
    pub fn is_admitted(self) -> bool {
        matches!(
            self,
            Self::Admitted | Self::Executing | Self::Dispatched | Self::Completed
        )
    }

    /// Whether the host holds the current attempt.
    pub fn is_host_owned(self) -> bool {
        matches!(self, Self::Executing | Self::Dispatched | Self::Completed)
    }
}

/// A stage to preview.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SelectionIntent {
    pub stage: StageId,
    /// `None` uses the stage's default profile.
    pub profile: Option<ProfileName>,
}

impl SelectionIntent {
    pub fn new(stage: StageId) -> Self {
        Self {
            stage,
            profile: None,
        }
    }

    pub fn with_profile(mut self, profile: ProfileName) -> Self {
        self.profile = Some(profile);
        self
    }
}

/// Which worker an admission may reserve.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AdmissionSource {
    /// Any worker the stage's restrictions allow.
    AnyEligible,
    /// Exactly the worker the session previewed for this stage.
    FromPreview,
}

/// A stage to admit and reserve.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdmissionIntent {
    pub stage: StageId,
    /// `None` uses the stage's default profile.
    pub profile: Option<ProfileName>,
    pub source: AdmissionSource,
}

impl AdmissionIntent {
    pub fn new(stage: StageId) -> Self {
        Self {
            stage,
            profile: None,
            source: AdmissionSource::AnyEligible,
        }
    }

    pub fn with_profile(mut self, profile: ProfileName) -> Self {
        self.profile = Some(profile);
        self
    }

    pub fn from_preview(mut self) -> Self {
        self.source = AdmissionSource::FromPreview;
        self
    }
}

/// The next thing a policy wants the coordinator to do.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CoordinationOp {
    /// Inspect a candidate and its signals without reserving it.
    Preview(SelectionIntent),
    /// Select and reserve a worker.
    Admit(AdmissionIntent),
    /// Choose a configured execution path.
    SelectBranch(BranchId),
    /// Wait for routing input the policy needs; stages already ready may run.
    Wait,
    /// No further selection decisions. Execution and cleanup may continue.
    Finish,
}

/// What a policy may read about one stage.
#[derive(Debug, Clone)]
pub struct StageView {
    pub stage: StageId,
    pub capabilities: StageCapabilities,
    pub status: StageStatus,
    pub attempt: AttemptId,
    pub preview: Option<Preview>,
    pub target: Option<SelectedTarget>,
    pub signals: Option<SelectionSignals>,
    pub handoff_ready: bool,
}

/// A read-only snapshot of one request's session for the policy.
///
/// Taken before each policy step so the policy never aliases live session
/// state; previews and targets are cheap to clone (shared worker facts).
pub struct CoordinationView<'a> {
    pub(super) facts: RequestFacts,
    pub(super) mode: PlanningMode,
    pub(super) topology: &'a Topology,
    pub(super) chosen_branch: Option<BranchId>,
    pub(super) cancelled: bool,
    pub(super) stages: HashMap<StageId, StageView>,
}

impl<'a> CoordinationView<'a> {
    pub fn facts(&self) -> &RequestFacts {
        &self.facts
    }

    pub fn mode(&self) -> PlanningMode {
        self.mode
    }

    pub fn topology(&self) -> &'a Topology {
        self.topology
    }

    /// The branch the policy selected, if it has.
    pub fn chosen_branch(&self) -> Option<&BranchId> {
        self.chosen_branch.as_ref()
    }

    /// The branch in effect: the chosen one, or the topology default.
    pub fn branch(&self) -> Option<&'a Branch> {
        self.chosen_branch
            .as_ref()
            .and_then(|id| self.topology.branch(id))
            .or_else(|| self.topology.default_branch())
    }

    /// Whether the client cancelled; remaining selections serve cleanup only.
    pub fn cancelled(&self) -> bool {
        self.cancelled
    }

    pub fn stage(&self, stage: &StageId) -> Option<&StageView> {
        self.stages.get(stage)
    }

    pub fn status(&self, stage: &StageId) -> StageStatus {
        self.stages
            .get(stage)
            .map(|view| view.status)
            .unwrap_or(StageStatus::Pending)
    }

    pub fn is_admitted(&self, stage: &StageId) -> bool {
        self.status(stage).is_admitted()
    }

    pub fn attempt(&self, stage: &StageId) -> AttemptId {
        self.stages
            .get(stage)
            .map(|view| view.attempt)
            .unwrap_or_default()
    }

    pub fn preview(&self, stage: &StageId) -> Option<&Preview> {
        self.stages.get(stage)?.preview.as_ref()
    }

    pub fn target(&self, stage: &StageId) -> Option<&SelectedTarget> {
        self.stages.get(stage)?.target.as_ref()
    }

    pub fn signals(&self, stage: &StageId) -> Option<SelectionSignals> {
        self.stages.get(stage)?.signals
    }

    pub fn handoff_ready(&self, stage: &StageId) -> bool {
        self.stages
            .get(stage)
            .is_some_and(|view| view.handoff_ready)
    }

    pub fn supports_preview(&self, stage: &StageId) -> bool {
        self.stages
            .get(stage)
            .is_some_and(|view| view.capabilities.supports_preview)
    }
}

/// Chooses selection order, execution path, and selection profiles for one
/// request. One instance per request; the coordinator calls `next` until it
/// returns `Finish`.
#[async_trait]
pub trait CoordinationPolicy: Send {
    async fn next(
        &mut self,
        view: &CoordinationView<'_>,
    ) -> Result<CoordinationOp, CoordinationError>;

    /// Whether the policy can finish selection without execution results.
    /// Required for `plan_all`.
    fn supports_upfront_planning(&self) -> bool {
        true
    }
}

/// Builds one policy instance per request.
pub type CoordinationPolicyFactory =
    Arc<dyn Fn(&RequestFacts) -> Box<dyn CoordinationPolicy> + Send + Sync>;
