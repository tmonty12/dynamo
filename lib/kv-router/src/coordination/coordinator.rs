// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The routing coordinator: runs a policy against a session, enforces
//! topology and placement constraints, bounds admission waits and reservation
//! holds, and hands each admitted stage to the host once its inputs are ready.
//!
//! # Ownership and fencing
//!
//! A reservation has one owner at a time: the session until the coordinator
//! returns [`HostAction::Execute`] (or a [`DelegatedPlan`] exports it), then
//! the host. Every host event names the stage attempt it concerns; events for
//! a superseded attempt are ignored, and terminal transitions are idempotent,
//! so a late duplicate cannot double-release or disturb a replacement attempt.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use tokio::time::Instant;

use super::error::CoordinationError;
use super::ids::{AttemptId, BranchId, InvocationId, StageId};
use super::placement::{derived_restrictions, validate_pair};
use super::policy::{
    AdmissionSource, CoordinationOp, CoordinationPolicyFactory, CoordinationView, PlanningMode,
    StageStatus, StageView,
};
use super::selector::{
    AdmissionTarget, SelectedTarget, SelectionInput, SelectionRestrictions, SelectionSignals,
    StageReservation,
};
use super::session::{RouteSession, RoutingRequest};
use super::stage::StageBinding;
use super::topology::{Branch, Topology};

/// Bounds the coordinator applies to every session.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CoordinationLimits {
    /// How long one preview or admission may wait in the selector.
    pub admission_timeout: Duration,
    /// Total time `plan_all` may spend selecting before it releases
    /// everything and fails.
    pub planning_deadline: Duration,
    /// How long the session may hold an admitted reservation before the host
    /// takes it. Exceeding it aborts the session rather than hoarding.
    pub max_reservation_hold: Duration,
    /// Attempts per stage invocation, counting the first.
    pub max_attempts_per_stage: u32,
}

impl Default for CoordinationLimits {
    fn default() -> Self {
        Self {
            admission_timeout: Duration::from_secs(30),
            planning_deadline: Duration::from_secs(30),
            max_reservation_hold: Duration::from_secs(120),
            max_attempts_per_stage: 3,
        }
    }
}

/// What the host tells the coordinator about execution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HostEvent {
    /// No new information; produce the next action.
    Continue,
    /// The host sent the stage's attempt to its worker.
    Dispatched { stage: StageId, attempt: AttemptId },
    /// The stage produced what its dependents need to start.
    HandoffReady { stage: StageId, attempt: AttemptId },
    /// The stage's attempt finished successfully.
    Completed { stage: StageId, attempt: AttemptId },
    /// The stage's attempt failed. With `retry`, the coordinator re-admits
    /// it (and any dependent it still owns) on a new attempt.
    Failed {
        stage: StageId,
        attempt: AttemptId,
        retry: bool,
    },
    /// The client cancelled. Stages a dispatched producer still needs for
    /// handoff cleanup continue; everything else is released.
    Cancelled,
}

/// A stage whose inputs are ready, with its reservation moved to the host.
#[derive(Debug)]
pub struct ReadyStage {
    pub stage: StageId,
    pub invocation: InvocationId,
    pub attempt: AttemptId,
    pub branch: BranchId,
    pub target: SelectedTarget,
    pub signals: SelectionSignals,
    pub reservation: StageReservation,
    /// Stages whose handoffs this stage consumes.
    pub inputs: Vec<StageId>,
}

/// What the host should do next.
#[derive(Debug)]
pub enum HostAction {
    /// Dispatch this stage. The host now owns its reservation.
    Execute(Box<ReadyStage>),
    /// Nothing is ready; report the next host event.
    Wait,
    /// Routing is complete: every reservation left the session. Execution
    /// and cleanup may continue in the host.
    Complete,
}

/// One stage of a complete plan, with its reservation.
#[derive(Debug)]
pub struct PlannedStage {
    pub stage: StageId,
    pub invocation: InvocationId,
    pub attempt: AttemptId,
    pub target: SelectedTarget,
    pub signals: SelectionSignals,
    pub reservation: StageReservation,
    /// Stages whose handoffs this stage consumes, in path order.
    pub depends_on: Vec<StageId>,
}

/// Every stage on the chosen path, selected and reserved before execution.
///
/// The plan owns its reservations; dropping it releases them all.
#[derive(Debug)]
pub struct DelegatedPlan {
    pub request_id: String,
    pub branch: BranchId,
    /// In execution order.
    pub stages: Vec<PlannedStage>,
}

impl DelegatedPlan {
    pub fn stage(&self, stage: &StageId) -> Option<&PlannedStage> {
        self.stages.iter().find(|planned| planned.stage == *stage)
    }

    /// Release every reservation and wait for each acknowledgement.
    pub async fn release_all(self) -> Result<(), CoordinationError> {
        let mut first_error = None;
        for planned in self.stages {
            if let Err(error) = planned.reservation.release().await
                && first_error.is_none()
            {
                first_error = Some(error);
            }
        }
        first_error.map_or(Ok(()), Err)
    }
}

/// Selects workers for every stage of a request, enforces constraints, and
/// returns stages ready to run.
pub struct RoutingCoordinator {
    topology: Arc<Topology>,
    bindings: HashMap<StageId, StageBinding>,
    policy_factory: CoordinationPolicyFactory,
    limits: CoordinationLimits,
}

impl RoutingCoordinator {
    /// Build a coordinator; every stage the topology names must be bound.
    pub fn new(
        topology: Topology,
        bindings: impl IntoIterator<Item = StageBinding>,
        policy_factory: CoordinationPolicyFactory,
    ) -> Result<Self, CoordinationError> {
        let bindings: HashMap<StageId, StageBinding> = bindings
            .into_iter()
            .map(|binding| (binding.id.clone(), binding))
            .collect();
        for stage in topology.stages() {
            if !bindings.contains_key(stage) {
                return Err(CoordinationError::UnknownStage {
                    stage: stage.clone(),
                });
            }
        }
        Ok(Self {
            topology: Arc::new(topology),
            bindings,
            policy_factory,
            limits: CoordinationLimits::default(),
        })
    }

    pub fn with_limits(mut self, limits: CoordinationLimits) -> Self {
        self.limits = limits;
        self
    }

    pub fn topology(&self) -> &Topology {
        &self.topology
    }

    pub fn limits(&self) -> CoordinationLimits {
        self.limits
    }

    pub fn binding(&self, stage: &StageId) -> Option<&StageBinding> {
        self.bindings.get(stage)
    }

    fn bound(&self, stage: &StageId) -> Result<&StageBinding, CoordinationError> {
        self.bindings
            .get(stage)
            .ok_or_else(|| CoordinationError::UnknownStage {
                stage: stage.clone(),
            })
    }

    /// Open a session. `Upfront` requires a policy that can finish without
    /// execution results.
    pub fn start(
        &self,
        request: RoutingRequest,
        mode: PlanningMode,
    ) -> Result<RouteSession, CoordinationError> {
        let policy = (self.policy_factory)(&request.facts);
        if mode == PlanningMode::Upfront && !policy.supports_upfront_planning() {
            return Err(CoordinationError::InvalidPolicyOperation(
                "policy needs execution results and cannot plan every stage upfront".to_string(),
            ));
        }
        Ok(RouteSession::new(
            request,
            mode,
            policy,
            self.topology.stages(),
        ))
    }

    /// Report a host event and get the next action. On error the session is
    /// aborted: every reservation it still owned is released and it is closed.
    pub async fn advance(
        &self,
        session: &mut RouteSession,
        event: HostEvent,
    ) -> Result<HostAction, CoordinationError> {
        if session.closed {
            return Err(CoordinationError::Finished);
        }
        match self.advance_inner(session, event).await {
            Ok(action) => Ok(action),
            Err(error) => {
                self.abort(session).await;
                Err(error)
            }
        }
    }

    /// Select and reserve every stage of the request's path without
    /// dispatching anything.
    pub async fn plan_all(
        &self,
        request: RoutingRequest,
    ) -> Result<DelegatedPlan, CoordinationError> {
        let started = Instant::now();
        let request_id = request.request_id.clone();
        let mut session = self.start(request, PlanningMode::Upfront)?;
        let mut planned: Vec<ReadyStage> = Vec::new();
        let result = async {
            loop {
                if started.elapsed() >= self.limits.planning_deadline {
                    return Err(CoordinationError::DeadlineExceeded {
                        elapsed: started.elapsed(),
                    });
                }
                match self.advance(&mut session, HostEvent::Continue).await? {
                    HostAction::Execute(ready) => planned.push(*ready),
                    HostAction::Wait => {
                        return Err(CoordinationError::InvalidPolicyOperation(
                            "policy waited for execution input during upfront planning".to_string(),
                        ));
                    }
                    HostAction::Complete => return Ok(()),
                }
            }
        }
        .await;
        if let Err(error) = result {
            // `advance` released what the session owned; release what it
            // already handed to this plan, so no partial plan escapes.
            for ready in planned {
                if let Err(release_error) = ready.reservation.release().await {
                    tracing::warn!(%request_id, %release_error, "failed to release a planned stage");
                }
            }
            return Err(error);
        }
        let branch = self
            .resolved_branch(&session)
            .ok_or_else(|| {
                CoordinationError::InvalidPolicyOperation(
                    "planning finished without a branch".to_string(),
                )
            })?
            .clone();
        let mut stages: Vec<PlannedStage> = planned
            .into_iter()
            .map(|ready| PlannedStage {
                depends_on: branch.inputs_of(&ready.stage).cloned().collect(),
                stage: ready.stage,
                invocation: ready.invocation,
                attempt: ready.attempt,
                target: ready.target,
                signals: ready.signals,
                reservation: ready.reservation,
            })
            .collect();
        stages.sort_by_key(|planned| {
            branch
                .path
                .iter()
                .position(|stage| *stage == planned.stage)
                .unwrap_or(usize::MAX)
        });
        Ok(DelegatedPlan {
            request_id,
            branch: branch.id.clone(),
            stages,
        })
    }

    async fn advance_inner(
        &self,
        session: &mut RouteSession,
        event: HostEvent,
    ) -> Result<HostAction, CoordinationError> {
        if let Some(action) = self.apply_event(session, event).await? {
            return Ok(action);
        }
        loop {
            self.check_reservation_holds(session)?;
            if let Some(ready) = self.next_ready(session)? {
                return Ok(HostAction::Execute(Box::new(ready)));
            }
            if session.finished {
                return Ok(self.settle(session));
            }
            let view = self.view(session);
            let op = session.policy.next(&view).await?;
            match op {
                CoordinationOp::Preview(intent) => {
                    if session.cancelled && !self.needed_after_cancel(session, &intent.stage) {
                        self.abort(session).await;
                        return Ok(HostAction::Complete);
                    }
                    self.run_preview(session, &intent.stage, intent.profile.as_ref())
                        .await?;
                }
                CoordinationOp::Admit(intent) => {
                    if session.cancelled && !self.needed_after_cancel(session, &intent.stage) {
                        self.abort(session).await;
                        return Ok(HostAction::Complete);
                    }
                    self.run_admit(
                        session,
                        &intent.stage,
                        intent.profile.as_ref(),
                        intent.source,
                    )
                    .await?;
                }
                CoordinationOp::SelectBranch(branch) => {
                    self.select_branch(session, branch)?;
                }
                CoordinationOp::Wait => {
                    if session.mode() == PlanningMode::Upfront {
                        return Err(CoordinationError::InvalidPolicyOperation(
                            "policy waited for execution input during upfront planning".to_string(),
                        ));
                    }
                    return Ok(HostAction::Wait);
                }
                CoordinationOp::Finish => {
                    self.finish(session)?;
                }
            }
        }
    }

    /// Once selection is over: complete when nothing is owned, else wait for
    /// the host events that make the owned stages ready.
    fn settle(&self, session: &mut RouteSession) -> HostAction {
        if session.owns_any_reservation() {
            HostAction::Wait
        } else {
            session.closed = true;
            HostAction::Complete
        }
    }

    fn view<'a>(&'a self, session: &RouteSession) -> CoordinationView<'a> {
        let stages = self
            .topology
            .stages()
            .iter()
            .filter_map(|stage| {
                let state = session.stage_state(stage)?;
                let binding = self.bindings.get(stage)?;
                Some((
                    stage.clone(),
                    StageView {
                        stage: stage.clone(),
                        capabilities: binding.capabilities,
                        status: state.status(),
                        attempt: state.attempt(),
                        preview: state.preview().cloned(),
                        target: state.target().cloned(),
                        signals: state.signals(),
                        handoff_ready: state.handoff_ready(),
                    },
                ))
            })
            .collect();
        CoordinationView {
            facts: session.facts().clone(),
            mode: session.mode(),
            topology: &self.topology,
            chosen_branch: session.branch().cloned(),
            cancelled: session.cancelled(),
            stages,
        }
    }

    fn resolved_branch<'a>(&'a self, session: &RouteSession) -> Option<&'a Branch> {
        session
            .branch()
            .and_then(|id| self.topology.branch(id))
            .or_else(|| self.topology.default_branch())
    }

    /// The first admitted stage on the path whose inputs are ready, with its
    /// reservation moved out of the session.
    fn next_ready(
        &self,
        session: &mut RouteSession,
    ) -> Result<Option<ReadyStage>, CoordinationError> {
        let Some(branch) = self.resolved_branch(session) else {
            return Ok(None);
        };
        let upfront = session.mode() == PlanningMode::Upfront;
        let mut ready_stage = None;
        for stage in &branch.path {
            let Some(state) = session.stage_state(stage) else {
                continue;
            };
            if !state.owns_reservation() {
                continue;
            }
            let inputs: Vec<StageId> = branch.inputs_of(stage).cloned().collect();
            let inputs_ready = upfront
                || inputs.iter().all(|input| {
                    session
                        .stage_state(input)
                        .is_some_and(|input_state| input_state.handoff_ready())
                });
            if inputs_ready {
                ready_stage = Some((stage.clone(), inputs));
                break;
            }
        }
        let Some((stage, inputs)) = ready_stage else {
            return Ok(None);
        };
        let branch_id = branch.id.clone();
        let state = session
            .stage_state_mut(&stage)
            .expect("stage state exists for a ready stage");
        let invocation = state.invocation();
        let attempt = state.attempt();
        let reservation = state
            .take_reservation()
            .expect("ready stage owns its reservation");
        Ok(Some(ReadyStage {
            stage,
            invocation,
            attempt,
            branch: branch_id,
            target: reservation.target.clone(),
            signals: reservation.signals,
            reservation,
            inputs,
        }))
    }

    fn check_reservation_holds(&self, session: &RouteSession) -> Result<(), CoordinationError> {
        for stage in self.topology.stages() {
            let Some(state) = session.stage_state(stage) else {
                continue;
            };
            if !state.owns_reservation() {
                continue;
            }
            if let Some(admitted_at) = state.admitted_at()
                && admitted_at.elapsed() > self.limits.max_reservation_hold
            {
                return Err(CoordinationError::DeadlineExceeded {
                    elapsed: admitted_at.elapsed(),
                });
            }
        }
        Ok(())
    }

    /// Caller restrictions for `stage`, merged with the constraints the
    /// admitted stages impose through placement rules.
    fn restrictions_for(
        &self,
        session: &RouteSession,
        stage: &StageId,
    ) -> Result<SelectionRestrictions, CoordinationError> {
        let caller = session.caller_restrictions(stage);
        let derived =
            derived_restrictions(self.topology.rules(), session.admitted_targets(), stage)?;
        caller.merge(&derived, stage)
    }

    async fn run_preview(
        &self,
        session: &mut RouteSession,
        stage: &StageId,
        profile: Option<&super::ids::ProfileName>,
    ) -> Result<(), CoordinationError> {
        let binding = self.bound(stage)?;
        if !binding.capabilities.supports_preview {
            return Err(CoordinationError::PreviewUnsupported {
                stage: stage.clone(),
            });
        }
        let profile = binding.profile(profile)?;
        let restrictions = self.restrictions_for(session, stage)?;
        let (invocation, attempt) = {
            let state =
                session
                    .stage_state(stage)
                    .ok_or_else(|| CoordinationError::UnknownStage {
                        stage: stage.clone(),
                    })?;
            (state.invocation(), state.attempt())
        };
        let request_id = session.request_id().to_string();
        let prompt = Arc::clone(&session.request().prompt);
        let settings = session.request().settings.clone();
        let input = SelectionInput {
            request_id: &request_id,
            stage,
            invocation,
            attempt,
            prompt: prompt.view(),
            profile,
            restrictions: &restrictions,
            settings: &settings,
        };
        let preview = self.bounded(binding.selector.preview(input)).await??;
        let generation = session.selection_generation();
        if let Some(state) = session.stage_state_mut(stage) {
            state.set_preview(preview, generation);
        }
        Ok(())
    }

    async fn run_admit(
        &self,
        session: &mut RouteSession,
        stage: &StageId,
        profile: Option<&super::ids::ProfileName>,
        source: AdmissionSource,
    ) -> Result<(), CoordinationError> {
        let binding = self.bound(stage)?;
        let profile = binding.profile(profile)?;
        let restrictions = self.restrictions_for(session, stage)?;
        let (invocation, attempt, target) = {
            let state =
                session
                    .stage_state(stage)
                    .ok_or_else(|| CoordinationError::UnknownStage {
                        stage: stage.clone(),
                    })?;
            if state.status().is_admitted() {
                return Err(CoordinationError::InvalidPolicyOperation(format!(
                    "stage {stage} is already admitted"
                )));
            }
            let target = match source {
                AdmissionSource::AnyEligible => AdmissionTarget::AnyEligible,
                AdmissionSource::FromPreview => {
                    let preview = state.preview().cloned().ok_or_else(|| {
                        CoordinationError::StalePreview {
                            stage: stage.clone(),
                            reason: "no preview exists for this attempt".to_string(),
                        }
                    })?;
                    if !self.topology.rules().is_empty()
                        && state.preview_generation() != session.selection_generation()
                    {
                        return Err(CoordinationError::StalePreview {
                            stage: stage.clone(),
                            reason: "other stages were selected after the preview".to_string(),
                        });
                    }
                    AdmissionTarget::FromPreview(preview)
                }
            };
            (state.invocation(), state.attempt(), target)
        };
        let request_id = session.request_id().to_string();
        let prompt = Arc::clone(&session.request().prompt);
        let settings = session.request().settings.clone();
        let input = SelectionInput {
            request_id: &request_id,
            stage,
            invocation,
            attempt,
            prompt: prompt.view(),
            profile,
            restrictions: &restrictions,
            settings: &settings,
        };
        let reservation = self
            .bounded(binding.selector.admit(input, target))
            .await??;
        for other in session.admitted_targets() {
            if let Err(error) = validate_pair(self.topology.rules(), &reservation.target, other) {
                if let Err(release_error) = reservation.release().await {
                    tracing::warn!(%request_id, %release_error, "failed to release an incompatible selection");
                }
                return Err(error);
            }
        }
        if let Some(state) = session.stage_state_mut(stage) {
            state.set_admitted(reservation);
        }
        session.bump_selection_generation();
        Ok(())
    }

    fn select_branch(
        &self,
        session: &mut RouteSession,
        branch: BranchId,
    ) -> Result<(), CoordinationError> {
        let path = &self
            .topology
            .branch(&branch)
            .ok_or_else(|| CoordinationError::UnknownBranch {
                branch: branch.clone(),
            })?
            .path;
        if let Some(current) = session.branch()
            && *current != branch
        {
            let admitted_off_path = session
                .admitted_targets()
                .any(|target| !path.contains(&target.stage));
            if admitted_off_path {
                return Err(CoordinationError::InvalidPolicyOperation(format!(
                    "cannot switch from branch {current} to {branch} after admitting a stage off its path"
                )));
            }
        }
        // Placement constraints derive from admitted workers only, so a branch
        // choice does not stale previews; only admissions bump the generation.
        session.set_branch(branch);
        Ok(())
    }

    fn finish(&self, session: &mut RouteSession) -> Result<(), CoordinationError> {
        let branch = self.resolved_branch(session).ok_or_else(|| {
            CoordinationError::InvalidPolicyOperation(
                "policy finished without selecting a branch".to_string(),
            )
        })?;
        for stage in &branch.path {
            let status = session.status(stage);
            if status.is_admitted() {
                continue;
            }
            if session.cancelled && !self.needed_after_cancel(session, stage) {
                continue;
            }
            return Err(CoordinationError::InvalidPolicyOperation(format!(
                "policy finished with stage {stage} not admitted"
            )));
        }
        session.finished = true;
        Ok(())
    }

    /// After cancellation, a stage still matters only if a producer it
    /// depends on already reached a worker: its handoff must be consumed.
    fn needed_after_cancel(&self, session: &RouteSession, stage: &StageId) -> bool {
        let Some(branch) = self.resolved_branch(session) else {
            return false;
        };
        branch.inputs_of(stage).any(|input| {
            session
                .stage_state(input)
                .is_some_and(|state| state.status().is_host_owned())
        })
    }

    /// Apply one host event. Returns an action only when the event ends
    /// routing on its own.
    async fn apply_event(
        &self,
        session: &mut RouteSession,
        event: HostEvent,
    ) -> Result<Option<HostAction>, CoordinationError> {
        match event {
            HostEvent::Continue => Ok(None),
            HostEvent::Dispatched { stage, attempt } => {
                if let Some(state) = fenced(session, &stage, attempt) {
                    state.mark_dispatched();
                }
                Ok(None)
            }
            HostEvent::HandoffReady { stage, attempt } => {
                if let Some(state) = fenced(session, &stage, attempt)
                    && state.status() != StageStatus::Failed
                {
                    state.mark_handoff_ready();
                }
                Ok(None)
            }
            HostEvent::Completed { stage, attempt } => {
                if let Some(state) = fenced(session, &stage, attempt)
                    && state.status() != StageStatus::Failed
                {
                    state.mark_completed();
                }
                Ok(None)
            }
            HostEvent::Failed {
                stage,
                attempt,
                retry,
            } => self.apply_failure(session, stage, attempt, retry).await,
            HostEvent::Cancelled => {
                session.cancelled = true;
                let mut to_release = Vec::new();
                for stage in self.topology.stages() {
                    let owns = session
                        .stage_state(stage)
                        .is_some_and(|state| state.owns_reservation());
                    if owns
                        && !self.needed_after_cancel(session, stage)
                        && let Some(released) = session
                            .stage_state_mut(stage)
                            .and_then(|state| state.reset())
                    {
                        to_release.push(released);
                    }
                }
                self.release_all(session.request_id(), to_release).await;
                Ok(None)
            }
        }
    }

    async fn apply_failure(
        &self,
        session: &mut RouteSession,
        stage: StageId,
        attempt: AttemptId,
        retry: bool,
    ) -> Result<Option<HostAction>, CoordinationError> {
        let (failed_worker, already_terminal) = {
            let Some(state) = fenced(session, &stage, attempt) else {
                return Ok(None);
            };
            (
                state.target().map(|target| target.worker.worker_id),
                state.is_terminal(),
            )
        };
        if already_terminal {
            return Ok(None);
        }
        let dependents: Vec<StageId> = self
            .resolved_branch(session)
            .map(|branch| transitive_dependents(branch, &stage))
            .unwrap_or_default();
        let can_retry = retry && attempt.get() + 1 < self.limits.max_attempts_per_stage;
        let mut to_release = Vec::new();
        if can_retry {
            if let Some(state) = session.stage_state_mut(&stage)
                && let Some(released) = state.begin_retry(failed_worker)
            {
                to_release.push(released);
            }
            // Dependents the session still holds were chosen against the
            // failed worker; select them again after the retry.
            for dependent in &dependents {
                if let Some(state) = session.stage_state_mut(dependent)
                    && state.owns_reservation()
                    && let Some(released) = state.begin_retry(None)
                {
                    to_release.push(released);
                }
            }
            // Selection is open again.
            session.finished = false;
            self.release_all(session.request_id(), to_release).await;
            return Ok(None);
        }
        if let Some(state) = session.stage_state_mut(&stage) {
            if let Some(released) = state.reset() {
                to_release.push(released);
            }
            state.mark_failed();
        }
        self.release_all(session.request_id(), to_release).await;
        // The request cannot complete; release everything still owned.
        self.abort(session).await;
        Ok(Some(HostAction::Complete))
    }

    async fn bounded<T>(
        &self,
        operation: impl std::future::Future<Output = T>,
    ) -> Result<T, CoordinationError> {
        let started = Instant::now();
        tokio::time::timeout(self.limits.admission_timeout, operation)
            .await
            .map_err(|_| CoordinationError::DeadlineExceeded {
                elapsed: started.elapsed(),
            })
    }

    async fn release_all(&self, request_id: &str, reservations: Vec<StageReservation>) {
        for reservation in reservations {
            let stage = reservation.target.stage.clone();
            if let Err(error) = reservation.release().await {
                tracing::warn!(%request_id, %stage, %error, "failed to release a coordinator-owned reservation");
            }
        }
    }

    /// Release everything the session still owns and close it.
    async fn abort(&self, session: &mut RouteSession) {
        let owned = session.take_owned_reservations();
        self.release_all(session.request_id(), owned).await;
        session.closed = true;
    }
}

/// The stage state for `attempt`, or `None` when the event is for another
/// attempt or an unknown stage.
fn fenced<'a>(
    session: &'a mut RouteSession,
    stage: &StageId,
    attempt: AttemptId,
) -> Option<&'a mut super::session::StageState> {
    let state = session.stage_state_mut(stage)?;
    if state.attempt() != attempt {
        tracing::debug!(
            %stage,
            event_attempt = %attempt,
            current_attempt = %state.attempt(),
            "ignoring host event for a superseded attempt"
        );
        return None;
    }
    Some(state)
}

fn transitive_dependents(branch: &Branch, stage: &StageId) -> Vec<StageId> {
    let mut found: Vec<StageId> = Vec::new();
    let mut frontier = vec![stage.clone()];
    while let Some(current) = frontier.pop() {
        for dependent in branch.dependents_of(&current) {
            if !found.contains(dependent) {
                found.push(dependent.clone());
                frontier.push(dependent.clone());
            }
        }
    }
    found
}
