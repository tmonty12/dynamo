// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Per-request routing state: previews, selections, reservations the session
//! still owns, stage readiness, and the attempt fence for host events.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use tokio::time::Instant;

use crate::protocols::WorkerId;

use super::ids::{AttemptId, BranchId, InvocationId, StageId};
use super::policy::{CoordinationPolicy, PlanningMode, RequestFacts, StageStatus};
use super::selector::{
    Preview, PromptInputView, RequestSettings, SelectedTarget, SelectionRestrictions,
    SelectionSignals, StageReservation,
};
use crate::protocols::BlockExtraInfo;

/// The prompt shared by every stage of one request.
#[derive(Debug, Clone, Default)]
pub struct PromptInput {
    /// Shared with the host's request so a long prompt is not copied per stage.
    pub token_ids: Arc<Vec<u32>>,
    pub block_mm_infos: Option<Vec<Option<BlockExtraInfo>>>,
    pub lora_name: Option<String>,
    pub cache_namespace: Option<String>,
}

impl PromptInput {
    pub fn from_tokens(token_ids: Vec<u32>) -> Self {
        Self::from_shared_tokens(Arc::new(token_ids))
    }

    pub fn from_shared_tokens(token_ids: Arc<Vec<u32>>) -> Self {
        Self {
            token_ids,
            ..Self::default()
        }
    }

    pub fn view(&self) -> PromptInputView<'_> {
        PromptInputView {
            token_ids: &self.token_ids,
            block_mm_infos: self.block_mm_infos.as_deref(),
            lora_name: self.lora_name.as_deref(),
            cache_namespace: self.cache_namespace.as_deref(),
        }
    }
}

/// One request as the coordinator receives it from a host.
#[derive(Debug, Clone)]
pub struct RoutingRequest {
    pub request_id: String,
    pub prompt: Arc<PromptInput>,
    pub settings: RequestSettings,
    /// Caller restrictions, applied to the stage they name only.
    pub restrictions: HashMap<StageId, SelectionRestrictions>,
    pub facts: RequestFacts,
}

impl RoutingRequest {
    pub fn new(request_id: impl Into<String>, prompt: PromptInput) -> Self {
        let facts = RequestFacts {
            prompt_tokens: prompt.token_ids.len(),
            ..RequestFacts::default()
        };
        Self {
            request_id: request_id.into(),
            prompt: Arc::new(prompt),
            settings: RequestSettings::default(),
            restrictions: HashMap::new(),
            facts,
        }
    }

    pub fn with_settings(mut self, settings: RequestSettings) -> Self {
        self.settings = settings;
        self
    }

    /// Restrict one stage. A pinned worker also marks the stage as decided by
    /// the caller in [`RequestFacts::pinned_stages`].
    pub fn with_stage_restrictions(
        mut self,
        stage: StageId,
        restrictions: SelectionRestrictions,
    ) -> Self {
        if restrictions.pinned_worker.is_some() {
            self.facts.pinned_stages.insert(stage.clone());
        }
        self.restrictions.insert(stage, restrictions);
        self
    }

    pub fn with_encode_input(mut self, requires_encode: bool) -> Self {
        self.facts.requires_encode = requires_encode;
        self
    }
}

/// A reservation the session holds for one stage attempt.
pub(super) struct AdmittedStage {
    pub(super) target: SelectedTarget,
    pub(super) signals: SelectionSignals,
    /// `None` once the reservation moved to the host.
    pub(super) reservation: Option<StageReservation>,
    pub(super) admitted_at: Instant,
}

/// Per-stage state for the current attempt.
pub struct StageState {
    invocation: InvocationId,
    attempt: AttemptId,
    status: StageStatus,
    preview: Option<Preview>,
    /// The selection generation the preview was taken at; a preview from an
    /// older generation is stale once placement rules apply.
    preview_generation: u64,
    pub(super) admitted: Option<AdmittedStage>,
    handoff_ready: bool,
    /// Workers failed by earlier attempts of this invocation.
    excluded: HashSet<WorkerId>,
}

impl StageState {
    fn new(invocation: InvocationId) -> Self {
        Self {
            invocation,
            attempt: AttemptId::FIRST,
            status: StageStatus::Pending,
            preview: None,
            preview_generation: 0,
            admitted: None,
            handoff_ready: false,
            excluded: HashSet::new(),
        }
    }

    pub fn invocation(&self) -> InvocationId {
        self.invocation
    }

    pub fn attempt(&self) -> AttemptId {
        self.attempt
    }

    pub fn status(&self) -> StageStatus {
        self.status
    }

    pub fn preview(&self) -> Option<&Preview> {
        self.preview.as_ref()
    }

    pub(super) fn preview_generation(&self) -> u64 {
        self.preview_generation
    }

    pub fn target(&self) -> Option<&SelectedTarget> {
        self.admitted.as_ref().map(|admitted| &admitted.target)
    }

    pub fn signals(&self) -> Option<SelectionSignals> {
        self.admitted.as_ref().map(|admitted| admitted.signals)
    }

    pub fn handoff_ready(&self) -> bool {
        self.handoff_ready
    }

    pub fn excluded_workers(&self) -> &HashSet<WorkerId> {
        &self.excluded
    }

    /// Whether the session still owns this attempt's reservation.
    pub(super) fn owns_reservation(&self) -> bool {
        self.admitted
            .as_ref()
            .is_some_and(|admitted| admitted.reservation.is_some())
    }

    pub(super) fn set_preview(&mut self, preview: Preview, generation: u64) {
        self.preview = Some(preview);
        self.preview_generation = generation;
        if self.status == StageStatus::Pending {
            self.status = StageStatus::Previewed;
        }
    }

    pub(super) fn set_admitted(&mut self, reservation: StageReservation) {
        let target = reservation.target.clone();
        let signals = reservation.signals;
        self.admitted = Some(AdmittedStage {
            target,
            signals,
            reservation: Some(reservation),
            admitted_at: Instant::now(),
        });
        self.preview = None;
        self.status = StageStatus::Admitted;
    }

    /// Move the reservation to the host.
    pub(super) fn take_reservation(&mut self) -> Option<StageReservation> {
        let reservation = self.admitted.as_mut()?.reservation.take()?;
        self.status = StageStatus::Executing;
        Some(reservation)
    }

    pub(super) fn admitted_at(&self) -> Option<Instant> {
        self.admitted.as_ref().map(|admitted| admitted.admitted_at)
    }

    pub(super) fn mark_dispatched(&mut self) {
        if self.status == StageStatus::Executing {
            self.status = StageStatus::Dispatched;
        }
    }

    pub(super) fn mark_handoff_ready(&mut self) {
        self.handoff_ready = true;
    }

    pub(super) fn mark_completed(&mut self) {
        self.status = StageStatus::Completed;
        self.handoff_ready = true;
    }

    pub(super) fn mark_failed(&mut self) {
        self.status = StageStatus::Failed;
    }

    /// Release the session's reservation (if any) and start the next attempt
    /// without the failed worker. Returns the released reservation so the
    /// caller can await its cleanup.
    pub(super) fn begin_retry(
        &mut self,
        failed_worker: Option<WorkerId>,
    ) -> Option<StageReservation> {
        let released = self
            .admitted
            .take()
            .and_then(|admitted| admitted.reservation);
        if let Some(worker_id) = failed_worker {
            self.excluded.insert(worker_id);
        }
        self.attempt = self.attempt.next();
        self.status = StageStatus::Pending;
        self.preview = None;
        self.handoff_ready = false;
        released
    }

    /// Take the reservation the session still owns, if any, without touching
    /// the attempt. An `Admitted` stage returns to `Pending`; terminal and
    /// host-owned statuses are kept.
    pub(super) fn release_owned(&mut self) -> Option<StageReservation> {
        let released = self
            .admitted
            .as_mut()
            .and_then(|admitted| admitted.reservation.take());
        if released.is_some() && self.status == StageStatus::Admitted {
            self.status = StageStatus::Pending;
        }
        released
    }

    /// Drop everything about the current attempt, keeping exclusions. Returns
    /// the released reservation, if the session still owned it.
    pub(super) fn reset(&mut self) -> Option<StageReservation> {
        let released = self
            .admitted
            .take()
            .and_then(|admitted| admitted.reservation);
        self.status = StageStatus::Pending;
        self.preview = None;
        self.handoff_ready = false;
        released
    }

    /// Whether the current attempt reached a terminal host state.
    pub(super) fn is_terminal(&self) -> bool {
        matches!(self.status, StageStatus::Completed | StageStatus::Failed)
    }
}

/// Tracks previews, selections, reservations, and stage readiness for one
/// request. Owned by the host, driven by the coordinator.
pub struct RouteSession {
    request: RoutingRequest,
    mode: PlanningMode,
    pub(super) policy: Box<dyn CoordinationPolicy>,
    branch: Option<BranchId>,
    stages: HashMap<StageId, StageState>,
    /// Bumped on every admission and branch choice; previews record it.
    selection_generation: u64,
    started: Instant,
    /// The policy returned `Finish`.
    pub(super) finished: bool,
    /// Routing is over: every reservation left the session, or it was aborted.
    pub(super) closed: bool,
    pub(super) cancelled: bool,
}

impl RouteSession {
    pub(super) fn new(
        request: RoutingRequest,
        mode: PlanningMode,
        policy: Box<dyn CoordinationPolicy>,
        stages: &[StageId],
    ) -> Self {
        let stages = stages
            .iter()
            .enumerate()
            .map(|(index, stage)| {
                (
                    stage.clone(),
                    StageState::new(InvocationId::new(index as u64 + 1)),
                )
            })
            .collect();
        Self {
            request,
            mode,
            policy,
            branch: None,
            stages,
            selection_generation: 0,
            started: Instant::now(),
            finished: false,
            closed: false,
            cancelled: false,
        }
    }

    pub fn request(&self) -> &RoutingRequest {
        &self.request
    }

    pub fn request_id(&self) -> &str {
        &self.request.request_id
    }

    pub fn facts(&self) -> &RequestFacts {
        &self.request.facts
    }

    pub fn mode(&self) -> PlanningMode {
        self.mode
    }

    pub fn branch(&self) -> Option<&BranchId> {
        self.branch.as_ref()
    }

    pub fn is_closed(&self) -> bool {
        self.closed
    }

    pub fn is_finished(&self) -> bool {
        self.finished
    }

    pub fn cancelled(&self) -> bool {
        self.cancelled
    }

    pub fn started(&self) -> Instant {
        self.started
    }

    pub fn stage_state(&self, stage: &StageId) -> Option<&StageState> {
        self.stages.get(stage)
    }

    pub fn status(&self, stage: &StageId) -> StageStatus {
        self.stages
            .get(stage)
            .map(StageState::status)
            .unwrap_or(StageStatus::Pending)
    }

    pub fn attempt(&self, stage: &StageId) -> AttemptId {
        self.stages
            .get(stage)
            .map(StageState::attempt)
            .unwrap_or_default()
    }

    pub(super) fn stage_state_mut(&mut self, stage: &StageId) -> Option<&mut StageState> {
        self.stages.get_mut(stage)
    }

    pub(super) fn selection_generation(&self) -> u64 {
        self.selection_generation
    }

    pub(super) fn bump_selection_generation(&mut self) -> u64 {
        self.selection_generation += 1;
        self.selection_generation
    }

    pub(super) fn set_branch(&mut self, branch: BranchId) {
        self.branch = Some(branch);
    }

    /// Every target the session has admitted, in either owner.
    pub(super) fn admitted_targets(&self) -> impl Iterator<Item = &SelectedTarget> {
        self.stages
            .values()
            .filter_map(|state| state.admitted.as_ref().map(|admitted| &admitted.target))
    }

    /// Caller restrictions for one stage, with this invocation's exclusions.
    pub(super) fn caller_restrictions(&self, stage: &StageId) -> SelectionRestrictions {
        let mut restrictions = self
            .request
            .restrictions
            .get(stage)
            .cloned()
            .unwrap_or_default();
        if let Some(state) = self.stages.get(stage) {
            restrictions
                .excluded_worker_ids
                .extend(state.excluded.iter().copied());
        }
        restrictions
    }

    /// Take every reservation the session still owns, leaving stage statuses
    /// otherwise intact.
    pub(super) fn take_owned_reservations(&mut self) -> Vec<StageReservation> {
        self.stages
            .values_mut()
            .filter_map(|state| state.release_owned())
            .collect()
    }

    pub(super) fn owns_any_reservation(&self) -> bool {
        self.stages.values().any(StageState::owns_reservation)
    }
}
