// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The coordinated prefill/decode path: a per-request `RoutingCoordinator`
//! chooses the branch (local or remote prefill), the selection order, and
//! the stage profiles; this operator dispatches each stage the coordinator
//! hands it and reports dispatch and handoff back.
//!
//! Dispatch, response streams, the prefill background task, phase tracking,
//! and cleanup are unchanged from the legacy path; only the selection
//! decisions moved into the coordinator.

use std::collections::VecDeque;
use std::sync::{Arc, OnceLock};

use anyhow::{Result, anyhow};
use dynamo_kv_router::WorkerType;
use dynamo_kv_router::coordination::{
    ConditionalDisaggThresholds, ConditionalDisaggregationPolicy, CoordinationError,
    CoordinationPolicyFactory, HostAction, HostEvent, LOCAL_PREFILL_DECODE_BRANCH, PlacementRule,
    PlanningMode, PoolRef, ProfileName, ProgressivePrefillDecodePolicy, RoutingCoordinator,
    ScoringMode, StageBinding, StageId, StageProfile, StageProfiles, Topology, WorkAccounting,
};
use dynamo_runtime::{
    error::{ErrorType, match_error_chain},
    pipeline::{AsyncEngineContextProvider, Context, ManyOut, ResponseStream, SingleIn},
    protocols::annotated::Annotated,
};
use futures::stream::{self, StreamExt};

use super::{
    BYPASS_REMOTE_PREFILL_ANNOTATION, PrefillBinding, PrefillCompletion, PrefillOutcome,
    PrefillRouter, build_decode_router_override, extract_bootstrap_info,
    independent_prefill_context, into_decode_request, strip_terminal_disaggregated_params,
};
use crate::kv_router::RoutingHost;
use crate::kv_router::coordination::{
    HostRoutePlan, HostStageSelector, SharedConditionalPolicy, host_error, routing_request_for,
};
use crate::protocols::common::{
    extensions::{SESSION_AFFINITY_CONTEXT_KEY, SessionAffinityId},
    llm_backend::{LLMEngineOutput, PreprocessedRequest},
    preprocessor::PrefillResult,
    timing::{RequestPhase, RequestTracker},
};
use crate::session_affinity::AffinityTarget;

/// `DYN_ROUTER_STAGE_COORDINATOR=0` falls back to the legacy prefill/decode
/// path for one release while the coordinator settles in.
// TODO(v1.7): Remove the switch and the legacy KV prefill/decode path.
pub(crate) fn stage_coordinator_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| match std::env::var("DYN_ROUTER_STAGE_COORDINATOR") {
        Ok(value) => !matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "0" | "false" | "off" | "no"
        ),
        Err(_) => true,
    })
}

/// A decision the coordinator made that the host must still carry out.
enum Step {
    Execute(Box<dynamo_kv_router::coordination::ReadyStage>),
    Done,
}

impl PrefillRouter {
    /// The decode host to coordinate with, when both hops are KV-routed and
    /// the coordinated path is enabled.
    pub(super) fn coordinated_decode_host(
        &self,
        binding: &PrefillBinding,
    ) -> Option<Arc<RoutingHost>> {
        if !stage_coordinator_enabled() || !binding.prefill_router_mode.is_kv_routing() {
            return None;
        }
        binding.router.kv_router_if_enabled()?;
        let decode_host = self.decode_routing_host.get()?;
        decode_host.kv_router_if_enabled()?;
        Some(Arc::clone(decode_host))
    }

    /// Route one request's prefill and decode through the coordinator.
    pub(super) async fn generate_coordinated(
        &self,
        mut req: PreprocessedRequest,
        context: Context<()>,
        binding: Arc<PrefillBinding>,
        decode_host: Arc<RoutingHost>,
    ) -> Result<ManyOut<Annotated<LLMEngineOutput>>> {
        let request_id = context.id().to_string();
        let original_max_tokens = req.stop_conditions.max_tokens;
        if req.tracker.is_none() {
            req.tracker = Some(Arc::new(RequestTracker::new()));
        }
        let tracker = req.tracker.clone().expect("tracker set above");
        let session_affinity = context
            .get_optional::<SessionAffinityId>(SESSION_AFFINITY_CONTEXT_KEY)
            .map_err(|message| anyhow!("invalid session affinity context: {message}"))?;
        let policy_class = context.metadata().get("policy-class").cloned();

        let decode_request: SingleIn<PreprocessedRequest> = context.map(|_| req);
        let mut prefill_body = decode_request.content().clone();
        prefill_body.stop_conditions.max_tokens = Some(1);
        let mut prefill_request = independent_prefill_context(prefill_body, &decode_request)?;
        if let Some(session_affinity) = session_affinity {
            prefill_request.insert(
                SESSION_AFFINITY_CONTEXT_KEY,
                session_affinity.as_ref().clone(),
            );
        }

        let mut flow = CoordinatedFlow {
            router: self,
            binding: &binding,
            decode_host: &decode_host,
            tracker,
            request_id,
            policy_class,
            original_max_tokens,
            prefill_request: Some(prefill_request),
            decode_request: Some(decode_request),
        };

        let conditional = self.conditional_disagg_policy.is_enabled();
        match flow.run(conditional).await {
            Ok(response) => Ok(response),
            // A failed conditional decision falls back to remote prefill, as
            // before, unless the request itself was cancelled or invalid.
            Err(error)
                if conditional
                    && flow.nothing_dispatched()
                    && !match_error_chain(
                        host_error(&error).as_ref(),
                        &[ErrorType::Cancelled, ErrorType::InvalidArgument],
                        &[],
                    ) =>
            {
                tracing::warn!(
                    request_id = %flow.request_id,
                    error = %error,
                    "Conditional disagg decision failed; falling back to remote prefill"
                );
                flow.run(false).await
            }
            Err(error) => Err(error),
        }
    }
}

struct CoordinatedFlow<'a> {
    router: &'a PrefillRouter,
    binding: &'a Arc<PrefillBinding>,
    decode_host: &'a Arc<RoutingHost>,
    tracker: Arc<RequestTracker>,
    request_id: String,
    policy_class: Option<String>,
    original_max_tokens: Option<u32>,
    /// Consumed when prefill is dispatched.
    prefill_request: Option<SingleIn<PreprocessedRequest>>,
    /// Consumed when decode is dispatched.
    decode_request: Option<SingleIn<PreprocessedRequest>>,
}

impl CoordinatedFlow<'_> {
    fn nothing_dispatched(&self) -> bool {
        self.prefill_request.is_some() && self.decode_request.is_some()
    }

    fn coordinator(
        &self,
        conditional: bool,
    ) -> Result<(RoutingCoordinator, Arc<HostStageSelector>)> {
        let prefill_request = self
            .prefill_request
            .as_ref()
            .ok_or_else(|| anyhow!("prefill request already dispatched"))?;
        let decode_request = self
            .decode_request
            .as_ref()
            .ok_or_else(|| anyhow!("decode request already dispatched"))?;
        let prefill_pool = PoolRef::new("prefill", self.binding.generation);
        // The decode host is fixed for this router's lifetime.
        let decode_pool = PoolRef::new("decode", 0);
        let prefill_selector = Arc::new(HostStageSelector::new(
            Arc::clone(&self.binding.router),
            RequestPhase::Prefill,
            prefill_pool.clone(),
            prefill_request,
        ));
        let decode_selector = Arc::new(HostStageSelector::new(
            Arc::clone(self.decode_host),
            RequestPhase::Decode,
            decode_pool.clone(),
            decode_request,
        ));
        // Remote decode after prefill accounts decode work only. Conditional
        // disagg keeps the base router's overlap credit for that selection;
        // plain disagg forces load-only scoring, as before.
        let remote_decode_scoring = if conditional {
            ScoringMode::CacheAware
        } else {
            ScoringMode::LoadOnly
        };
        let decode_profiles = StageProfiles::single(StageProfile::new(
            ProfileName::DECODE_ONLY,
            WorkAccounting::DecodeOnly,
            remote_decode_scoring,
        ))
        .with_profile(StageProfile::new(
            ProfileName::LOCAL_PREFILL_DECODE,
            WorkAccounting::PrefillAndDecode,
            ScoringMode::CacheAware,
        ));
        let bindings = vec![
            StageBinding::new(
                StageId::PREFILL,
                prefill_pool,
                WorkerType::Prefill,
                prefill_selector,
            ),
            StageBinding::new(
                StageId::DECODE,
                decode_pool,
                WorkerType::Decode,
                Arc::clone(&decode_selector)
                    as Arc<dyn dynamo_kv_router::coordination::StageSelector>,
            )
            .with_profiles(decode_profiles),
        ];
        let topology = if conditional {
            Topology::conditional_prefill_decode()
        } else {
            Topology::prefill_decode()
        }
        .with_rule(PlacementRule::TransferCompatible);
        let policy_factory: CoordinationPolicyFactory = if conditional {
            let decision =
                SharedConditionalPolicy(Arc::clone(&self.router.conditional_disagg_policy));
            let thresholds = ConditionalDisaggThresholds {
                prefill_busy: self.router.conditional_disagg_prefill_busy_threshold,
                decode_busy: self.router.conditional_disagg_decode_busy_threshold,
            };
            Arc::new(move |_| {
                Box::new(ConditionalDisaggregationPolicy::new(
                    Box::new(decision.clone()),
                    thresholds,
                    Box::new(ProgressivePrefillDecodePolicy::new()),
                ))
            })
        } else {
            Arc::new(|_| Box::new(ProgressivePrefillDecodePolicy::new()))
        };
        Ok((
            RoutingCoordinator::new(topology, bindings, policy_factory)?,
            decode_selector,
        ))
    }

    async fn run(&mut self, conditional: bool) -> Result<ManyOut<Annotated<LLMEngineOutput>>> {
        let (coordinator, decode_selector) = self.coordinator(conditional)?;
        let routing_request = routing_request_for(
            self.decode_request
                .as_ref()
                .expect("decode request present before routing")
                .content(),
            &self.request_id,
            self.policy_class.clone(),
        );
        let mut session = coordinator.start(routing_request, PlanningMode::Progressive)?;
        let engine_ctx = self
            .decode_request
            .as_ref()
            .expect("decode request present before routing")
            .context();

        let mut pending: VecDeque<HostEvent> = VecDeque::new();
        let mut outcome: Option<PrefillOutcome> = None;
        let mut prefill_completion = None;
        loop {
            let event = pending.pop_front().unwrap_or(HostEvent::Continue);
            let step = match coordinator.advance(&mut session, event).await {
                Ok(HostAction::Execute(ready)) => Step::Execute(ready),
                Ok(HostAction::Wait) => {
                    if pending.is_empty() {
                        return Err(anyhow!(
                            "routing coordinator waited for a host event none is pending"
                        ));
                    }
                    continue;
                }
                Ok(HostAction::Complete) => Step::Done,
                Err(error) => return Err(self.routing_error(error)),
            };
            let ready = match step {
                Step::Execute(ready) => *ready,
                Step::Done => {
                    return Err(anyhow!("routing completed without dispatching decode"));
                }
            };
            if ready.stage == StageId::PREFILL {
                let plan = HostRoutePlan::take(ready.reservation)?;
                let mut prefill_request = self
                    .prefill_request
                    .take()
                    .ok_or_else(|| anyhow!("prefill stage executed twice"))?;
                let phase_barrier = self.tracker.set_phase(RequestPhase::Prefill).await;
                let target = AffinityTarget::new(
                    ready.target.worker.worker_id,
                    Some(ready.target.worker.dp_rank),
                );
                let prepared = self.router.prepare_prefill_dispatch(
                    &mut prefill_request,
                    target,
                    &self.binding.endpoint_id,
                    false,
                )?;
                let prefill_stream = match self
                    .binding
                    .router
                    .dispatch_kv_plan(prefill_request, plan)
                    .await
                {
                    Ok(stream) => stream,
                    Err(error) => return Err(self.prefill_error(error)),
                };
                pending.push_back(HostEvent::Dispatched {
                    stage: StageId::PREFILL,
                    attempt: ready.attempt,
                });
                // Decode must now be selected even if the client disconnects:
                // only its worker's KV-transfer path frees what prefill staged.
                decode_selector.set_staged_kv_cleanup(true);

                let result = if let Some(bootstrap_info) = prepared.bootstrap_info {
                    prefill_completion = Some(self.router.spawn_prefill_task(
                        prefill_stream,
                        Some(Arc::clone(&self.tracker)),
                        phase_barrier,
                    ));
                    PrefillOutcome::Bootstrap {
                        bootstrap_info,
                        worker_id: prepared.worker_id,
                        prefill_dp_rank: prepared.prefill_dp_rank,
                    }
                } else {
                    drop(phase_barrier);
                    let completion = PrefillRouter::consume_prefill_stream(
                        prefill_stream,
                        Some(Arc::clone(&self.tracker)),
                        self.router.task_guard.clone(),
                    )
                    .await
                    .map_err(|error| self.prefill_error(error.into()))?;
                    match completion {
                        PrefillCompletion::Handoff {
                            result,
                            worker_link,
                            completion,
                        } => {
                            prefill_completion = completion;
                            handoff_outcome(result, worker_link, &prepared)
                        }
                        PrefillCompletion::Terminal { output } => {
                            // The context step finished the request; nothing to
                            // hand off and no decode to select.
                            let output = strip_terminal_disaggregated_params(*output);
                            return Ok(ResponseStream::new(
                                Box::pin(stream::once(async move { output })),
                                engine_ctx,
                            ));
                        }
                    }
                };
                outcome = Some(result);
                pending.push_back(HostEvent::HandoffReady {
                    stage: StageId::PREFILL,
                    attempt: ready.attempt,
                });
                continue;
            }

            if ready.stage == StageId::DECODE {
                let plan = HostRoutePlan::take(ready.reservation)?;
                let decode_request = self
                    .decode_request
                    .take()
                    .ok_or_else(|| anyhow!("decode stage executed twice"))?;
                let local_prefill = ready.branch == LOCAL_PREFILL_DECODE_BRANCH;
                // In the bootstrap path this waits until the spawned prefill
                // task releases its phase barrier, keeping worker attribution
                // correct.
                let _decode_permit = self.tracker.set_phase(RequestPhase::Decode).await;
                let decode_request = if local_prefill {
                    tracing::info!(
                        request_id = %self.request_id,
                        worker_id = ready.target.worker.worker_id,
                        dp_rank = ready.target.worker.dp_rank,
                        cached_tokens = ready.signals.cached_tokens,
                        "Conditional disagg routing to decode worker"
                    );
                    decode_request.map(|mut request| {
                        request
                            .annotations
                            .push(BYPASS_REMOTE_PREFILL_ANNOTATION.to_string());
                        request
                    })
                } else {
                    let outcome = outcome
                        .take()
                        .ok_or_else(|| anyhow!("decode selected before the prefill handoff"))?;
                    // NVBugs 5969206: decode routing proceeds even after the
                    // client context stopped, so the KV transfer has a receiver.
                    if engine_ctx.is_stopped() || engine_ctx.is_killed() {
                        tracing::debug!(
                            "Context {} killed/stopped after prefill, allowing decode routing for KV transfer",
                            engine_ctx.id()
                        );
                    }
                    let original_max_tokens = self.original_max_tokens;
                    decode_request.map(|request| {
                        let mut decode = into_decode_request(request, outcome);
                        decode.stop_conditions.max_tokens = original_max_tokens;
                        let existing = decode.router_config_override.take();
                        decode.router_config_override =
                            Some(build_decode_router_override(existing, conditional));
                        decode
                    })
                };
                let dispatch = self.decode_host.dispatch_kv_plan(decode_request, plan);
                let response = match prefill_completion.take() {
                    Some(completion) => {
                        completion
                            .forward_decode(dispatch, Arc::clone(&engine_ctx))
                            .await?
                    }
                    None => dispatch.await?,
                };
                // Routing is over; the coordinator owns nothing after this.
                if let Err(error) = coordinator
                    .advance(
                        &mut session,
                        HostEvent::Dispatched {
                            stage: StageId::DECODE,
                            attempt: ready.attempt,
                        },
                    )
                    .await
                {
                    tracing::debug!(request_id = %self.request_id, %error, "routing session did not close cleanly");
                }
                if local_prefill {
                    let ctx = response.context();
                    let annotation = Annotated::<LLMEngineOutput>::from_annotation(
                        BYPASS_REMOTE_PREFILL_ANNOTATION,
                        &true,
                    )?;
                    let merged = stream::once(async move { annotation }).chain(response);
                    return Ok(ResponseStream::new(Box::pin(merged), ctx));
                }
                return Ok(response);
            }

            return Err(anyhow!(
                "routing coordinator executed unexpected stage {}",
                ready.stage
            ));
        }
    }

    fn routing_error(&self, error: CoordinationError) -> anyhow::Error {
        let error = anyhow::Error::from(error);
        if match_error_chain(
            host_error(&error).as_ref(),
            &[ErrorType::ResourceExhausted, ErrorType::WorkerOverloaded],
            &[],
        ) {
            tracing::warn!(
                request_id = %self.request_id,
                error = %error,
                "request rejected during stage selection (at capacity)"
            );
        } else {
            tracing::error!(
                request_id = %self.request_id,
                error = %error,
                "Stage selection failed, failing request"
            );
        }
        error
    }

    fn prefill_error(&self, error: anyhow::Error) -> anyhow::Error {
        if match_error_chain(
            error.as_ref(),
            &[ErrorType::ResourceExhausted, ErrorType::WorkerOverloaded],
            &[],
        ) {
            tracing::warn!(
                error = %error,
                "request rejected by prefill worker (at capacity)"
            );
        } else {
            tracing::error!(error = %error, "Remote prefill failed, failing request");
        }
        error
    }
}

fn handoff_outcome(
    result: PrefillResult,
    worker_link: Option<crate::protocols::common::preprocessor::TraceLink>,
    prepared: &super::PreparedPrefill,
) -> PrefillOutcome {
    if let Some(bootstrap_info) = extract_bootstrap_info(&result.disaggregated_params) {
        PrefillOutcome::Bootstrap {
            bootstrap_info,
            worker_id: prepared.worker_id,
            prefill_dp_rank: prepared.prefill_dp_rank,
        }
    } else {
        PrefillOutcome::Completed {
            result,
            worker_id: prepared.worker_id,
            worker_link,
        }
    }
}
