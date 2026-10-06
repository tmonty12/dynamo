// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Built-in coordination policies: aggregated, prefill-first, decode-first,
//! progressive prefill/decode, conditional disaggregation, and
//! encode/prefill/decode.
//!
//! Each policy reads the session view and returns the next operation. None
//! keeps private progress state, so a stage returned to `Pending` by a retry
//! is simply admitted again.

use async_trait::async_trait;

use crate::conditional_disagg::{ConditionalDisaggDecisionInput, ConditionalDisaggPolicy};
use crate::config::KvRouterConfig;

use super::error::CoordinationError;
use super::ids::{ProfileName, StageId};
use super::policy::{
    AdmissionIntent, CoordinationOp, CoordinationPolicy, CoordinationView, PlanningMode,
    SelectionIntent, StageStatus,
};
use super::topology::{
    ENCODE_PREFILL_DECODE_BRANCH, LOCAL_PREFILL_DECODE_BRANCH, PREFILL_DECODE_BRANCH,
    REMOTE_PREFILL_DECODE_BRANCH,
};

/// Admit the single aggregated stage, then finish.
#[derive(Debug, Clone)]
pub struct AggregatedPolicy {
    stage: StageId,
}

impl Default for AggregatedPolicy {
    fn default() -> Self {
        Self::new()
    }
}

impl AggregatedPolicy {
    pub fn new() -> Self {
        Self {
            stage: StageId::AGGREGATED,
        }
    }

    pub fn for_stage(stage: StageId) -> Self {
        Self { stage }
    }
}

#[async_trait]
impl CoordinationPolicy for AggregatedPolicy {
    async fn next(
        &mut self,
        view: &CoordinationView<'_>,
    ) -> Result<CoordinationOp, CoordinationError> {
        if !view.is_admitted(&self.stage) {
            return Ok(CoordinationOp::Admit(AdmissionIntent::new(
                self.stage.clone(),
            )));
        }
        Ok(CoordinationOp::Finish)
    }
}

/// The order two stages are selected in when both are selected before any
/// execution result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SelectionOrder {
    PrefillFirst,
    DecodeFirst,
}

/// Select prefill and decode in a fixed order, both before execution. Decode
/// still waits for prefill's handoff to run; this only fixes selection order.
#[derive(Debug, Clone)]
pub struct PrefillDecodePolicy {
    order: SelectionOrder,
    prefill: StageId,
    decode: StageId,
}

impl PrefillDecodePolicy {
    pub fn prefill_first() -> Self {
        Self::new(SelectionOrder::PrefillFirst)
    }

    pub fn decode_first() -> Self {
        Self::new(SelectionOrder::DecodeFirst)
    }

    pub fn new(order: SelectionOrder) -> Self {
        Self {
            order,
            prefill: StageId::PREFILL,
            decode: StageId::DECODE,
        }
    }

    fn admit_decode(&self) -> CoordinationOp {
        CoordinationOp::Admit(
            AdmissionIntent::new(self.decode.clone()).with_profile(ProfileName::DECODE_ONLY),
        )
    }
}

#[async_trait]
impl CoordinationPolicy for PrefillDecodePolicy {
    async fn next(
        &mut self,
        view: &CoordinationView<'_>,
    ) -> Result<CoordinationOp, CoordinationError> {
        let prefill_admitted = view.is_admitted(&self.prefill);
        let decode_admitted = view.is_admitted(&self.decode);
        let (first, first_admitted, second, second_admitted) = match self.order {
            SelectionOrder::PrefillFirst => (
                &self.prefill,
                prefill_admitted,
                &self.decode,
                decode_admitted,
            ),
            SelectionOrder::DecodeFirst => (
                &self.decode,
                decode_admitted,
                &self.prefill,
                prefill_admitted,
            ),
        };
        if !first_admitted {
            return Ok(if *first == self.decode {
                self.admit_decode()
            } else {
                CoordinationOp::Admit(AdmissionIntent::new(first.clone()))
            });
        }
        if !second_admitted {
            return Ok(if *second == self.decode {
                self.admit_decode()
            } else {
                CoordinationOp::Admit(AdmissionIntent::new(second.clone()))
            });
        }
        Ok(CoordinationOp::Finish)
    }
}

/// Select prefill, wait for its handoff, then select decode. Decode is chosen
/// with the freshest load picture, at the cost of needing execution results.
#[derive(Debug, Clone)]
pub struct ProgressivePrefillDecodePolicy {
    prefill: StageId,
    decode: StageId,
}

impl ProgressivePrefillDecodePolicy {
    pub fn new() -> Self {
        Self {
            prefill: StageId::PREFILL,
            decode: StageId::DECODE,
        }
    }
}

impl Default for ProgressivePrefillDecodePolicy {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl CoordinationPolicy for ProgressivePrefillDecodePolicy {
    async fn next(
        &mut self,
        view: &CoordinationView<'_>,
    ) -> Result<CoordinationOp, CoordinationError> {
        if !view.is_admitted(&self.prefill) {
            return Ok(CoordinationOp::Admit(AdmissionIntent::new(
                self.prefill.clone(),
            )));
        }
        if view.is_admitted(&self.decode) {
            return Ok(CoordinationOp::Finish);
        }
        // Upfront planning has no handoffs to wait for; select decode now.
        if view.mode() == PlanningMode::Progressive && !view.handoff_ready(&self.prefill) {
            return Ok(CoordinationOp::Wait);
        }
        Ok(CoordinationOp::Admit(
            AdmissionIntent::new(self.decode.clone()).with_profile(ProfileName::DECODE_ONLY),
        ))
    }

    fn supports_upfront_planning(&self) -> bool {
        // Selecting decode before prefill's handoff is a valid plan; the policy
        // only prefers to wait when it can.
        true
    }
}

/// Thresholds the conditional policy reads from the router configuration.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct ConditionalDisaggThresholds {
    /// Prefill-busy line; `None` disables the prefill-load signal.
    pub prefill_busy: Option<f64>,
    /// Decode-busy guard; `None` disables the gate.
    pub decode_busy: Option<f64>,
}

impl ConditionalDisaggThresholds {
    /// The same resolution the frontend applies: a dedicated prefill-busy
    /// threshold, else the queue threshold.
    pub fn from_config(config: &KvRouterConfig) -> Self {
        Self {
            prefill_busy: config
                .conditional_disagg_prefill_busy_threshold
                .or(config.router_queue_threshold),
            decode_busy: config.conditional_disagg_decode_busy_threshold,
        }
    }
}

/// The decode gate: a decision to bypass stands only when the gate is
/// disabled or the chosen decode worker is known not to be busy.
pub fn decode_gate_allows_bypass(
    policy_says_bypass: bool,
    decode_gate_configured: bool,
    decode_busy: Option<bool>,
) -> bool {
    policy_says_bypass && (!decode_gate_configured || matches!(decode_busy, Some(false)))
}

/// Preview decode before deciding whether remote prefill is needed, then run
/// either the local branch (admit the previewed decode worker) or the remote
/// branch under `remote`.
pub struct ConditionalDisaggregationPolicy {
    decision: Box<dyn ConditionalDisaggPolicy>,
    thresholds: ConditionalDisaggThresholds,
    remote: Box<dyn CoordinationPolicy>,
    prefill: StageId,
    decode: StageId,
}

impl ConditionalDisaggregationPolicy {
    pub fn new(
        decision: Box<dyn ConditionalDisaggPolicy>,
        thresholds: ConditionalDisaggThresholds,
        remote: Box<dyn CoordinationPolicy>,
    ) -> Self {
        Self {
            decision,
            thresholds,
            remote,
            prefill: StageId::PREFILL,
            decode: StageId::DECODE,
        }
    }

    fn choose_remote(&self) -> CoordinationOp {
        CoordinationOp::SelectBranch(REMOTE_PREFILL_DECODE_BRANCH)
    }
}

/// Preview-driven decision between local and remote prefill. A free function
/// so the policy's `&mut self` borrow (which holds a non-`Sync` inner policy)
/// is not held across the decision's await.
async fn decide_conditional(
    decision: &dyn ConditionalDisaggPolicy,
    thresholds: ConditionalDisaggThresholds,
    prefill: &StageId,
    decode: &StageId,
    view: &CoordinationView<'_>,
) -> Result<CoordinationOp, CoordinationError> {
    let Some(decode_preview) = view.preview(decode) else {
        return Ok(CoordinationOp::Preview(
            SelectionIntent::new(decode.clone()).with_profile(ProfileName::LOCAL_PREFILL_DECODE),
        ));
    };
    let needs_prefill_busy =
        decision.needs_prefill_worker_busy() && thresholds.prefill_busy.is_some();
    let prefill_busy = if needs_prefill_busy {
        match view.preview(prefill) {
            Some(preview) => preview
                .signals
                .prefill_load_exceeds(thresholds.prefill_busy.unwrap_or_default()),
            None if view.supports_preview(prefill) => {
                return Ok(CoordinationOp::Preview(SelectionIntent::new(
                    prefill.clone(),
                )));
            }
            None => None,
        }
    } else {
        None
    };

    let signals = decode_preview.signals;
    let input =
        ConditionalDisaggDecisionInput::new(view.facts().prompt_tokens, signals.cached_tokens)
            .with_prefill_chosen_worker_busy(prefill_busy);
    let policy_says_bypass = decision.should_bypass_remote_prefill(input).await;
    let decode_gate_configured = thresholds.decode_busy.is_some();
    let decode_busy = if policy_says_bypass {
        thresholds
            .decode_busy
            .and_then(|threshold| signals.decode_load_exceeds(threshold))
    } else {
        None
    };
    let bypass = decode_gate_allows_bypass(policy_says_bypass, decode_gate_configured, decode_busy);
    tracing::debug!(
        request_tokens = view.facts().prompt_tokens,
        worker_id = decode_preview.target.worker.worker_id,
        dp_rank = decode_preview.target.worker.dp_rank,
        cached_tokens = signals.cached_tokens,
        net_new_tokens = input.net_new_tokens(),
        prefill_chosen_worker_busy = ?prefill_busy,
        decode_chosen_worker_busy = ?decode_busy,
        bypass,
        "Conditional disagg decision"
    );
    Ok(if bypass {
        CoordinationOp::SelectBranch(LOCAL_PREFILL_DECODE_BRANCH)
    } else {
        CoordinationOp::SelectBranch(REMOTE_PREFILL_DECODE_BRANCH)
    })
}

#[async_trait]
impl CoordinationPolicy for ConditionalDisaggregationPolicy {
    async fn next(
        &mut self,
        view: &CoordinationView<'_>,
    ) -> Result<CoordinationOp, CoordinationError> {
        let chosen = match view.chosen_branch().cloned() {
            Some(branch) => branch,
            None => {
                // A caller-pinned prefill worker, a disabled policy, or a decode
                // stage that cannot preview all mean remote prefill.
                if !self.decision.is_enabled()
                    || view.facts().pinned_stages.contains(&self.prefill)
                    || !view.supports_preview(&self.decode)
                {
                    return Ok(self.choose_remote());
                }
                return decide_conditional(
                    self.decision.as_ref(),
                    self.thresholds,
                    &self.prefill,
                    &self.decode,
                    view,
                )
                .await;
            }
        };
        if chosen == LOCAL_PREFILL_DECODE_BRANCH {
            return Ok(match view.status(&self.decode) {
                status if status.is_admitted() => CoordinationOp::Finish,
                _ if view.preview(&self.decode).is_some() => CoordinationOp::Admit(
                    AdmissionIntent::new(self.decode.clone())
                        .with_profile(ProfileName::LOCAL_PREFILL_DECODE)
                        .from_preview(),
                ),
                // The preview was invalidated (for example by a retry); select afresh.
                _ => CoordinationOp::Admit(
                    AdmissionIntent::new(self.decode.clone())
                        .with_profile(ProfileName::LOCAL_PREFILL_DECODE),
                ),
            });
        }
        self.remote.next(view).await
    }

    fn supports_upfront_planning(&self) -> bool {
        self.remote.supports_upfront_planning()
    }
}

/// Encode, then prefill, then decode for multimodal requests; prefill and
/// decode otherwise. `progressive` waits for each handoff before selecting
/// the next stage.
#[derive(Debug, Clone)]
pub struct EncodePrefillDecodePolicy {
    progressive: bool,
    encode: StageId,
    prefill: StageId,
    decode: StageId,
}

impl EncodePrefillDecodePolicy {
    pub fn new(progressive: bool) -> Self {
        Self {
            progressive,
            encode: StageId::ENCODE,
            prefill: StageId::PREFILL,
            decode: StageId::DECODE,
        }
    }

    fn should_wait_for(&self, view: &CoordinationView<'_>, producer: &StageId) -> bool {
        self.progressive
            && view.mode() == PlanningMode::Progressive
            && !view.handoff_ready(producer)
    }
}

#[async_trait]
impl CoordinationPolicy for EncodePrefillDecodePolicy {
    async fn next(
        &mut self,
        view: &CoordinationView<'_>,
    ) -> Result<CoordinationOp, CoordinationError> {
        let Some(branch) = view.branch() else {
            return Ok(CoordinationOp::SelectBranch(
                if view.facts().requires_encode {
                    ENCODE_PREFILL_DECODE_BRANCH
                } else {
                    PREFILL_DECODE_BRANCH
                },
            ));
        };
        let encodes = branch.contains(&self.encode);
        if encodes {
            if !view.is_admitted(&self.encode) {
                return Ok(CoordinationOp::Admit(AdmissionIntent::new(
                    self.encode.clone(),
                )));
            }
            if view.status(&self.encode) == StageStatus::Failed {
                return Ok(CoordinationOp::Finish);
            }
        }
        if !view.is_admitted(&self.prefill) {
            if encodes && self.should_wait_for(view, &self.encode) {
                return Ok(CoordinationOp::Wait);
            }
            return Ok(CoordinationOp::Admit(AdmissionIntent::new(
                self.prefill.clone(),
            )));
        }
        if !view.is_admitted(&self.decode) {
            if self.should_wait_for(view, &self.prefill) {
                return Ok(CoordinationOp::Wait);
            }
            return Ok(CoordinationOp::Admit(
                AdmissionIntent::new(self.decode.clone()).with_profile(ProfileName::DECODE_ONLY),
            ));
        }
        Ok(CoordinationOp::Finish)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_gate_matches_the_frontend_rules() {
        assert!(decode_gate_allows_bypass(true, true, Some(false)));
        assert!(!decode_gate_allows_bypass(true, true, Some(true)));
        assert!(!decode_gate_allows_bypass(true, true, None));
        assert!(decode_gate_allows_bypass(true, false, None));
        assert!(decode_gate_allows_bypass(true, false, Some(true)));
        assert!(!decode_gate_allows_bypass(false, false, Some(false)));
    }

    #[test]
    fn thresholds_fall_back_to_the_queue_threshold() {
        let config = KvRouterConfig {
            router_queue_threshold: Some(0.6),
            conditional_disagg_prefill_busy_threshold: None,
            conditional_disagg_decode_busy_threshold: Some(0.9),
            ..Default::default()
        };
        let thresholds = ConditionalDisaggThresholds::from_config(&config);
        assert_eq!(thresholds.prefill_busy, Some(0.6));
        assert_eq!(thresholds.decode_busy, Some(0.9));

        let dedicated = KvRouterConfig {
            router_queue_threshold: Some(0.6),
            conditional_disagg_prefill_busy_threshold: Some(0.3),
            ..Default::default()
        };
        assert_eq!(
            ConditionalDisaggThresholds::from_config(&dedicated).prefill_busy,
            Some(0.3)
        );
    }
}
