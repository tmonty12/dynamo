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
use dynamo_kv_router::protocols::WorkerId;
use dynamo_runtime::pipeline::{Context, SingleIn};

use crate::kv_router::routing_host::{RoutePlan, RoutePlanSignals, RoutePreview, RoutingHost};
use crate::kv_router::to_worker_selection_session_context;
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
