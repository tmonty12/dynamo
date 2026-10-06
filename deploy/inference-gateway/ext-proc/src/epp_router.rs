// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Standalone (selector) endpoint picker.
//!
//! This is the runtime-free counterpart to [`crate::epp::Router`]. It runs with
//! no Dynamo `DistributedRuntime`, no etcd/NATS, and no embedded KV router.
//! Instead it composes:
//!
//! - a [`RenderClient`] that tokenizes prompts via a render sidecar,
//! - one [`PodDiscovery`] per `InferencePool` that discovers Ready worker pods
//!   from Kubernetes (the decode pool the `HTTPRoute` targets and, when
//!   configured, a prefill pool),
//! - a [`TopologyAdapter`] per pool that registers those pods into its
//!   [`Selector`] (in-process, runtime-free selection service), and
//! - a `RoutingCoordinator` that plans every stage of a request over those
//!   selectors and books each one before the request leaves the EPP.
//!
//! On each request it tokenizes the prompt, plans and books the stages, and
//! tells Envoy where to send the request: the decode destination, plus
//! `x-prefiller-host-port` naming the prefill worker for the decode-side
//! sidecar when a prefill pool is configured.

use std::collections::HashSet;
use std::net::{IpAddr, SocketAddr};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use anyhow::Result;
use tokio::sync::Semaphore;

use dynamo_kv_router::coordination::{
    AggregatedPolicy, AttemptId, CoordinationError, CoreAdmissionMode, CoreBooking,
    CoreStageSelector, DelegatedPlan, PoolRef, PrefillDecodePolicy, PromptInput, RequestSettings,
    RoutingCoordinator, RoutingRequest, SelectionRestrictions, StageBinding, StageId,
    StageSelector, Topology, reservation_id,
};
use dynamo_kv_router::identity::RoutingPartitionId;
use dynamo_kv_router::services::selection::WorkerSelectionPolicyRegistry;
use dynamo_kv_router::{DEFAULT_ROUTING_GROUP, SessionContext, WorkerType};
use dynamo_llm::http::service::metadata::extract_metadata_from_header_pairs;
use dynamo_llm::protocols::agents::HEADER_DYNAMO_SESSION_ID;
use dynamo_llm::protocols::common::extensions::{
    AgentHints, HEADER_REQUEST_PRIORITY, HEADER_REQUEST_STRICT_PRIORITY, resolve_request_priority,
};
use serde::Deserialize;

use crate::epp_standalone_config::{EppStandaloneConfig, RendererProtocol};
use crate::picker::{
    CacheSaltForwarding, Endpoint, EndpointPicker, PickError, PickResult, RequestInfo,
    resolve_cache_namespace,
};
use crate::pod_discovery::PodDiscovery;
use crate::render_http::RenderError;
use crate::selector::Selector;
use crate::sglang_renderer_client::SglangRendererClient;
use crate::topology_adapter::{RegistrationDefaults, TopologyAdapter};
use crate::vllm_render_client::VllmRenderClient;

/// Resolve the request's scheduling policy class from the Dynamo metadata
/// headers. Goes through the frontend's metadata extractor (rather than a
/// hardcoded header name) so custom `DYN_METADATA_HEADER` prefixes, trimming,
/// and duplicate handling stay aligned with the integrated router.
pub(crate) fn requested_policy_class(
    headers: &[(String, String)],
) -> Result<Option<String>, PickError> {
    let metadata =
        extract_metadata_from_header_pairs(headers.iter().map(|(k, v)| (k.as_str(), v.as_str())))
            .map_err(PickError::MetadataHeadersTooLarge)?;
    Ok(metadata.get("policy-class").cloned())
}

/// Protocol-dispatched render client for the standalone EPP.
enum RenderClient {
    Vllm(VllmRenderClient),
    Sglang(SglangRendererClient),
}

impl RenderClient {
    async fn render_chat(&self, body: bytes::Bytes) -> Result<Vec<u32>, RenderError> {
        match self {
            Self::Vllm(c) => c.render_chat(body).await,
            Self::Sglang(c) => c.render_chat(body).await,
        }
    }
}

/// One worker pool: its selector, the reflector that resolves its pods, and
/// the reconcile loop that keeps the two in step.
struct Pool {
    selector: Arc<Selector>,
    reflector: Arc<PodDiscovery>,
    // Kept alive for the lifetime of the router; the reconcile loop runs on it.
    _adapter: TopologyAdapter,
    ready: Arc<AtomicBool>,
}

impl Pool {
    async fn spawn(
        cfg: &EppStandaloneConfig,
        pool_name: &str,
        policy_registry: WorkerSelectionPolicyRegistry,
        worker_type: WorkerType,
    ) -> Result<Self> {
        let selector = Arc::new(Selector::new_for_role(cfg, policy_registry, worker_type).await?);
        let (reflector, ready) = PodDiscovery::spawn_for_pool(cfg, pool_name).await?;
        let reflector = Arc::new(reflector);
        let adapter = TopologyAdapter::spawn(
            reflector.as_ref().clone(),
            selector.clone(),
            RegistrationDefaults::from_config(cfg),
        );
        Ok(Self {
            selector,
            reflector,
            _adapter: adapter,
            ready,
        })
    }
}

/// A coordinator stage selector over one pool's embedded selection core.
/// Admissions are booked by id so the lifecycle callbacks can address them.
fn stage_selector(selector: &Selector, pool: &str, model_name: &str) -> Arc<dyn StageSelector> {
    Arc::new(CoreStageSelector::new(
        Arc::clone(selector.core()),
        RoutingPartitionId::new(model_name, DEFAULT_ROUTING_GROUP),
        PoolRef::new(pool.to_string(), 1),
        CoreAdmissionMode::Book,
    ))
}

/// Which stages a request's plan booked, by the ids the lifecycle callbacks
/// address them with. The decode pool booking is `decode` on the
/// disaggregated path and `aggregated` on the fallback; both are tried.
fn stage_booking_id(key: &str, stage: &StageId) -> String {
    reservation_id(key, stage, AttemptId::FIRST)
}

/// Whether a disaggregated plan failed only because prefill could not be
/// selected, so the request can still be served by the decode pool alone.
fn falls_back_to_aggregated(error: &CoordinationError) -> bool {
    match error {
        CoordinationError::NoEligibleWorkers { stage, .. }
        | CoordinationError::AdmissionRejected { stage, .. } => *stage == StageId::PREFILL,
        _ => false,
    }
}

/// The two coordinators a standalone EPP plans with. `disaggregated` exists
/// only when a prefill pool is configured.
fn build_coordinators(
    model_name: &str,
    decode: &Selector,
    prefill: Option<&Selector>,
) -> Result<(RoutingCoordinator, Option<RoutingCoordinator>)> {
    let aggregated = RoutingCoordinator::new(
        Topology::aggregated(),
        [StageBinding::new(
            StageId::AGGREGATED,
            PoolRef::new("decode", 1),
            WorkerType::Aggregated,
            stage_selector(decode, "decode", model_name),
        )],
        Arc::new(|_| Box::new(AggregatedPolicy::new())),
    )?;
    let disaggregated = prefill
        .map(|prefill| {
            RoutingCoordinator::new(
                Topology::prefill_decode(),
                [
                    StageBinding::new(
                        StageId::PREFILL,
                        PoolRef::new("prefill", 1),
                        WorkerType::Prefill,
                        stage_selector(prefill, "prefill", model_name),
                    ),
                    StageBinding::new(
                        StageId::DECODE,
                        PoolRef::new("decode", 1),
                        WorkerType::Decode,
                        stage_selector(decode, "decode", model_name),
                    ),
                ],
                Arc::new(|_| Box::new(PrefillDecodePolicy::prefill_first())),
            )
        })
        .transpose()?;
    Ok((aggregated, disaggregated))
}

/// A planned request as the picker returns it: workers by stage, with every
/// booking handed over to id-addressed ownership.
struct AdoptedPlan {
    decode_worker_id: u64,
    prefill_worker_id: Option<u64>,
}

/// Plan a request: the disaggregated coordinator first, falling back to the
/// aggregated one when prefill alone cannot be selected.
async fn plan_request(
    disaggregated: Option<&RoutingCoordinator>,
    aggregated: &RoutingCoordinator,
    request: impl Fn() -> RoutingRequest,
) -> Result<DelegatedPlan, CoordinationError> {
    if let Some(disaggregated) = disaggregated {
        match disaggregated.plan_all(request()).await {
            Ok(plan) => return Ok(plan),
            Err(error) if falls_back_to_aggregated(&error) => {
                tracing::debug!(
                    %error,
                    "Prefill pool could not serve the request; routing to the decode pool alone"
                );
            }
            Err(error) => return Err(error),
        }
    }
    aggregated.plan_all(request()).await
}

/// Hand every booking in `plan` to id-addressed ownership: the lifecycle
/// callbacks free them by [`stage_booking_id`], so no per-request map is kept.
fn adopt_plan(plan: DelegatedPlan) -> Result<AdoptedPlan> {
    let mut decode_worker_id = None;
    let mut prefill_worker_id = None;
    for stage in plan.stages {
        let (target, _, lease) = stage.reservation.into_parts();
        let booking = lease.into_owner::<CoreBooking>().map_err(|_| {
            anyhow::anyhow!(
                "stage {} was not reserved through the embedded selection core",
                target.stage
            )
        })?;
        let _selection_id = booking.into_selection_id();
        if target.stage == StageId::PREFILL {
            prefill_worker_id = Some(target.worker.worker_id);
        } else {
            decode_worker_id = Some(target.worker.worker_id);
        }
    }
    Ok(AdoptedPlan {
        decode_worker_id: decode_worker_id
            .ok_or_else(|| anyhow::anyhow!("plan has no decode or aggregated stage"))?,
        prefill_worker_id,
    })
}

/// Standalone endpoint picker backed by the standalone selection service.
pub struct EppRouter {
    renderer: RenderClient,
    /// The pool the `HTTPRoute` targets; every response streams from here.
    decode: Pool,
    /// Prefill workers named to the decode-side sidecar; `None` runs aggregated.
    prefill: Option<Pool>,
    aggregated: RoutingCoordinator,
    disaggregated: Option<RoutingCoordinator>,
    /// Bounds total concurrent in-flight `pick()`s. HTTP/2 stream multiplexing
    /// means the TCP-connection cap (`MAX_CONCURRENT_CONNECTIONS`) does NOT bound
    /// requests, so without this a burst could fan out unbounded tokenizer/render
    /// calls and buffer unbounded request bodies. A permit is taken per `pick()`
    /// and released (RAII) when it returns or is dropped/cancelled; when none are
    /// available the request is shed with `PickError::Overloaded` (not queued).
    inflight: Arc<Semaphore>,
}

/// Routing inputs parsed from a standalone EPP request.
struct TokenizeResult {
    token_ids: Vec<u32>,
    priority_jump: Option<f64>,
    strict_priority: Option<u32>,
    cache_namespace: Option<String>,
    expected_output_tokens: Option<u32>,
}

impl EppRouter {
    /// Assemble the standalone runtime from the validated selector config.
    pub async fn from_selector(
        cfg: EppStandaloneConfig,
        policy_registry: WorkerSelectionPolicyRegistry,
    ) -> Result<Self> {
        let timeout = Duration::from_millis(cfg.tokenization_timeout_ms);
        let max_response_bytes = cfg.tokenizer_max_response_bytes;
        let renderer = match cfg.renderer_protocol {
            RendererProtocol::VllmRender => RenderClient::Vllm(VllmRenderClient::new(
                &cfg.tokenizer_service_url,
                timeout,
                max_response_bytes,
            )?),
            RendererProtocol::SglangRenderer => RenderClient::Sglang(SglangRendererClient::new(
                &cfg.tokenizer_service_url,
                timeout,
                max_response_bytes,
            )?),
        };
        let prefill = match &cfg.prefill_inference_pool_name {
            Some(pool_name) => {
                // Prefill admission is local to this replica: the prefill pool's
                // selector does not join the decode pool's replica sync.
                let mut prefill_cfg = cfg.clone();
                prefill_cfg.peer_replication = None;
                Some(
                    Pool::spawn(
                        &prefill_cfg,
                        pool_name,
                        policy_registry.clone(),
                        WorkerType::Prefill,
                    )
                    .await?,
                )
            }
            None => None,
        };
        let decode_role = if prefill.is_some() {
            WorkerType::Decode
        } else {
            WorkerType::Aggregated
        };
        let decode =
            Pool::spawn(&cfg, &cfg.inference_pool_name, policy_registry, decode_role).await?;
        let (aggregated, disaggregated) = build_coordinators(
            &cfg.model_name,
            &decode.selector,
            prefill.as_ref().map(|pool| pool.selector.as_ref()),
        )?;
        tracing::info!(
            decode_pool = %cfg.inference_pool_name,
            prefill_pool = ?cfg.prefill_inference_pool_name,
            "Initialized standalone EPP routing"
        );

        // Readiness is driven solely by the live pod+pool signal (see `is_ready`);
        // we do not block startup on a schedulable worker. A valid, empty pool is
        // ready immediately and returns 503 per-request until capacity appears.
        Ok(Self {
            renderer,
            decode,
            prefill,
            aggregated,
            disaggregated,
            inflight: Arc::new(Semaphore::new(cfg.max_inflight_requests)),
        })
    }

    /// Overall EPP readiness for the gRPC health signal: every pod reflector
    /// has synced workers and resolved its InferencePool. Polled by the health
    /// mirror in `main`.
    pub fn is_ready(&self) -> bool {
        self.decode.ready.load(Ordering::Acquire)
            && self
                .prefill
                .as_ref()
                .is_none_or(|prefill| prefill.ready.load(Ordering::Acquire))
    }

    /// Tokenize a chat body and resolve its routing inputs.
    async fn tokenize(
        &self,
        request_body: bytes::Bytes,
        headers: &[(String, String)],
    ) -> Result<TokenizeResult, TokenizeError> {
        // Parse only the routing hot-path fields — the worker re-parses the full
        // body anyway, so we skip allocating the large `messages`/tools fields.
        // Malformed JSON still fails here (→ 400); a well-formed body that is not
        // a valid chat request is caught by the renderer below.
        let hints: RoutingHints =
            serde_json::from_slice(&request_body).map_err(TokenizeError::InvalidBody)?;
        let priority_header = first_header(headers, HEADER_REQUEST_PRIORITY);
        let strict_priority_header = first_header(headers, HEADER_REQUEST_STRICT_PRIORITY);
        let resolved = resolve_request_priority(
            hints.nvext.as_ref().and_then(|n| n.agent_hints.as_ref()),
            priority_header,
            strict_priority_header,
        );
        let expected_output_tokens = hints
            .nvext
            .as_ref()
            .and_then(|n| n.agent_hints.as_ref())
            .and_then(|h| h.osl);
        let cache_namespace = resolve_cache_namespace(
            headers,
            hints
                .nvext
                .as_ref()
                .and_then(|nvext| nvext.cache_namespace.as_deref()),
            hints.cache_namespace.as_deref(),
        );
        // Moves the `Bytes` into reqwest (zero-copy) rather than copying.
        let token_ids = self
            .renderer
            .render_chat(request_body)
            .await
            .map_err(TokenizeError::Render)?;
        Ok(TokenizeResult {
            token_ids,
            priority_jump: resolved.priority_jump,
            strict_priority: resolved.strict_priority,
            expected_output_tokens,
            cache_namespace,
        })
    }

    /// Ready workers inside an Envoy `candidate_subset`, resolved in a single index
    /// pass (no full-ready set materialized). The reflector's endpoints are
    /// scheme-less `ip:port`, so a worker matches the subset's full `ip:port` or
    /// bare `ip`; empty means nothing matched.
    fn subset_worker_ids(&self, candidate_subset: &[String]) -> HashSet<u64> {
        let candidates: HashSet<&str> = candidate_subset.iter().map(String::as_str).collect();
        let candidate_ips: HashSet<IpAddr> = candidate_subset
            .iter()
            .filter_map(|candidate| candidate.parse().ok())
            .collect();
        // Single index pass; the predicate borrows each endpoint (no clone).
        self.decode.reflector.ready_worker_ids_matching(|endpoint| {
            endpoint_in_subset(endpoint, &candidates, &candidate_ips)
        })
    }
}

/// True if a scheme-less `ip:port` endpoint is covered by an Envoy subset,
/// matching either the full `ip:port` or the bare `ip`.
///
/// Matches the bare-IP case via `IpAddr`, never `endpoint.split(':')`: a
/// bracketed IPv6 endpoint (`[fd00::2]:8000`) splits into garbage on `:`,
/// silently never matching a bare `fd00::2` candidate. Shared with
/// [`crate::epp::Router::subset_to_worker_ids`], the other Envoy
/// candidate_subset matcher in this crate.
pub(crate) fn endpoint_in_subset(
    endpoint: &str,
    candidates: &HashSet<&str>,
    candidate_ips: &HashSet<IpAddr>,
) -> bool {
    candidates.contains(endpoint)
        || endpoint
            .parse::<SocketAddr>()
            .is_ok_and(|address| candidate_ips.contains(&address.ip()))
}

/// Minimal deserialize target for the routing hot path: only `nvext.agent_hints`
/// is needed for priority resolution and `cache_namespace`,so the large
/// `messages`/tools fields are never allocated.
/// Unknown fields are ignored (no `deny_unknown_fields`).
#[derive(Deserialize)]
struct RoutingHints {
    #[serde(default)]
    nvext: Option<RoutingNvExt>,
    /// Native vLLM top-level `cache_salt`.
    #[serde(default, rename = "cache_salt")]
    cache_namespace: Option<String>,
}

#[derive(Deserialize)]
struct RoutingNvExt {
    #[serde(default)]
    agent_hints: Option<AgentHints>,
    /// Dynamo-style `nvext.cache_salt`.
    #[serde(default, rename = "cache_salt")]
    cache_namespace: Option<String>,
}

/// Case-insensitive lookup of the first non-empty, trimmed value for `name`.
fn first_header<'a>(headers: &'a [(String, String)], name: &str) -> Option<&'a str> {
    headers
        .iter()
        .find(|(k, _)| k.eq_ignore_ascii_case(name))
        .map(|(_, v)| v.trim())
        .filter(|v| !v.is_empty())
}

#[tonic::async_trait]
impl EndpointPicker for EppRouter {
    async fn pick(
        &self,
        req: &RequestInfo,
        _endpoints: &[Endpoint],
    ) -> Result<PickResult, PickError> {
        if !self.is_ready() {
            return Err(PickError::RoutingFailed(
                "pod reflector cache not ready".to_string(),
            ));
        }

        if !self.decode.reflector.has_ready_workers() {
            return Err(PickError::NoEndpoints);
        }

        // Bound total in-flight picks. This caps the tokenizer/render fan-out,
        // `select_and_reserve`, and the buffered request bodies held for the
        // duration of the pick — the connection cap does NOT, because HTTP/2 stream
        // multiplexing lets one connection carry unbounded concurrent requests.
        // `try_acquire_owned` sheds (never blocks/awaits) so we don't grow an
        // unbounded wait queue; the permit is held until `pick()` returns or the
        // future is dropped/cancelled, releasing it (RAII).
        let _inflight_permit = self
            .inflight
            .clone()
            .try_acquire_owned()
            .map_err(|_| PickError::Overloaded)?;

        // Ordinary path: pass `None` so the SelectionService schedules over its
        // own catalog ("selector owns eligibility") — no O(worker-count) id set is
        // built per request. We accept that the catalog lags the reflector by ~ms
        // after a pod event: the system already tolerates far larger staleness
        // (pod readiness), and the post-select `resolve_endpoint` guard still
        // refuses to route to a worker the reflector can no longer resolve. The
        // freshness-preserving alternative (re-assert the ready set every request)
        // would need an `Arc`-shared set threaded through the core to stay O(1) —
        // not worth the complexity. Only a subset hint (info the selector lacks)
        // needs an explicit id set, built lazily below.
        let allowed: Option<HashSet<u64>> = if req.candidate_subset.is_empty() {
            None
        } else {
            // Honor Envoy's subset hint (`x-gateway-destination-endpoint-subset`):
            // constrain to Ready workers in the subset, refusing (not falling back
            // to the full set) when nothing matches.
            let filtered = self.subset_worker_ids(&req.candidate_subset);
            if filtered.is_empty() {
                tracing::warn!(
                    subset = ?req.candidate_subset,
                    "No Ready pod matches the subset hint; refusing to route outside the subset"
                );
                return Err(PickError::NoEndpoints);
            }
            Some(filtered)
        };

        // Body-less requests (no prompt to tokenize) route to any Ready worker,
        // staying inside the subset when one was given.
        if req.body.is_empty() {
            let endpoint = match &allowed {
                Some(ids) => {
                    let worker_id = *ids.iter().next().ok_or(PickError::NoEndpoints)?;
                    self.decode
                        .reflector
                        .resolve_endpoint(worker_id)
                        .ok_or(PickError::NoEndpoints)?
                }
                None => self
                    .decode
                    .reflector
                    .resolve_any_endpoint()
                    .ok_or(PickError::NoEndpoints)?,
            };
            return Ok(PickResult {
                endpoint,
                ..Default::default()
            });
        }

        let TokenizeResult {
            token_ids: tokens,
            priority_jump,
            strict_priority,
            cache_namespace,
            expected_output_tokens,
        } = self
            .tokenize(req.body.clone(), &req.headers)
            .await
            .map_err(|e| e.into_pick_error(&req.request_id))?;
        let policy_class = requested_policy_class(&req.headers)?;

        // EPP-minted booking key (not the reused `x-request-id`): each stage's
        // booking is `reservation_id(key, stage, attempt)`, so the lifecycle
        // callbacks address every booking from the key alone, no shared map.
        let reservation_id = uuid::Uuid::new_v4().to_string();

        let shared_tokens = Arc::new(tokens);
        let session_id = first_header(&req.headers, HEADER_DYNAMO_SESSION_ID).map(str::to_owned);
        let routing_request = || {
            let prompt = PromptInput {
                token_ids: Arc::clone(&shared_tokens),
                block_mm_infos: None,
                lora_name: None,
                cache_namespace: cache_namespace.clone(),
            };
            // `None` on the ordinary path: the selector schedules over its
            // catalog; `Some` only carries an Envoy subset constraint, and only
            // for the decode pool the subset describes.
            let decode_restrictions = SelectionRestrictions {
                allowed_worker_ids: allowed.clone(),
                ..SelectionRestrictions::default()
            };
            RoutingRequest::new(&reservation_id, prompt)
                .with_settings(RequestSettings {
                    // Effective header-over-body values; defaults only when unset everywhere.
                    priority_jump: priority_jump.unwrap_or_default(),
                    strict_priority: strict_priority.unwrap_or(0),
                    policy_class: policy_class.clone(),
                    session_context: session_id
                        .clone()
                        .map(|session_id| SessionContext::new(session_id, None, None, None)),
                    expected_output_tokens,
                    router_config_override: None,
                })
                .with_stage_restrictions(StageId::DECODE, decode_restrictions.clone())
                .with_stage_restrictions(StageId::AGGREGATED, decode_restrictions)
        };

        // Until the plan is adopted below it owns every booking: dropping this
        // future (the ext-proc stream closed after the scheduler booked) frees
        // them. Adoption is synchronous, so nothing can slip between it and the
        // server storing `booking_id`.
        let plan = match plan_request(
            self.disaggregated.as_ref(),
            &self.aggregated,
            routing_request,
        )
        .await
        {
            Ok(plan) => plan,
            Err(CoordinationError::Selector(error))
                if error
                    .downcast_ref::<dynamo_kv_router::services::selection::SelectionError>()
                    .is_some_and(|error| {
                        matches!(
                            error,
                            dynamo_kv_router::services::selection::SelectionError::BadRequest(_)
                        )
                    }) =>
            {
                return Err(PickError::InvalidRequest(error.to_string()));
            }
            Err(error) => return Err(PickError::RoutingFailed(error.to_string())),
        };

        // The reflectors own addresses and readiness. If one can no longer
        // resolve a selected worker, that pod left Ready in the race, so the
        // selection is stale: refuse rather than route to a stale address.
        let decode_stage = plan
            .stage(&StageId::DECODE)
            .or_else(|| plan.stage(&StageId::AGGREGATED))
            .ok_or_else(|| PickError::RoutingFailed("plan has no decode stage".to_string()))?;
        let Some(endpoint) = self
            .decode
            .reflector
            .resolve_endpoint(decode_stage.target.worker.worker_id)
        else {
            tracing::warn!(
                worker_id = decode_stage.target.worker.worker_id,
                "Selected worker no longer resolvable in reflector; treating selection as stale"
            );
            plan.release_all().await.ok();
            return Err(PickError::NoEndpoints);
        };
        let selected_prefill_endpoint = match (plan.stage(&StageId::PREFILL), &self.prefill) {
            (Some(prefill_stage), Some(prefill)) => {
                let Some(prefill_endpoint) = prefill
                    .reflector
                    .resolve_endpoint(prefill_stage.target.worker.worker_id)
                else {
                    tracing::warn!(
                        worker_id = prefill_stage.target.worker.worker_id,
                        "Selected prefill worker no longer resolvable in reflector; treating selection as stale"
                    );
                    plan.release_all().await.ok();
                    return Err(PickError::NoEndpoints);
                };
                Some(prefill_endpoint)
            }
            _ => None,
        };

        // Success: the caller adopts `reservation_id` synchronously (there is no
        // await between this return and the server storing `booking_id`), so the
        // lifecycle callbacks now own every booking by id.
        let adopted = adopt_plan(plan).map_err(|e| PickError::RoutingFailed(e.to_string()))?;
        tracing::debug!(
            reservation_id = %reservation_id,
            decode_worker_id = adopted.decode_worker_id,
            prefill_worker_id = ?adopted.prefill_worker_id,
            endpoint = %endpoint,
            prefill_endpoint = ?selected_prefill_endpoint,
            "Picked standalone endpoint"
        );

        // Routing comes from the destination mutation; aggregated raw workers
        // read no `x-dynamo-*` headers. A disaggregated plan names the prefill
        // worker in `x-prefiller-host-port` for the decode-side sidecar.
        Ok(PickResult {
            endpoint,
            // Worker re-tokenizes the forwarded request (llm-d parity); no inject.
            token_ids: None,
            cache_namespace,
            // Native vLLM has no Dynamo handler to tag the salt; the EPP does.
            cache_salt_forwarding: CacheSaltForwarding::NativeVllm,
            selected_prefill_endpoint,
            // Booking key for the server's lifecycle callbacks (no shared map).
            reservation_id: Some(reservation_id),
            ..Default::default()
        })
    }

    /// Response complete: release every booking from `pick`. `booking_id` is
    /// that pick's reservation key; `free_reservation` is idempotent (a
    /// body-less pick booked nothing, and the fallback plan has no prefill).
    async fn on_request_complete(&self, booking_id: &str) {
        if let Some(prefill) = &self.prefill
            && let Err(e) = prefill
                .selector
                .free_reservation(&stage_booking_id(booking_id, &StageId::PREFILL))
                .await
        {
            tracing::warn!(reservation_id = booking_id, error = %e, "Failed to free prefill reservation");
        }
        for stage in [StageId::DECODE, StageId::AGGREGATED] {
            if let Err(e) = self
                .decode
                .selector
                .free_reservation(&stage_booking_id(booking_id, &stage))
                .await
            {
                tracing::warn!(reservation_id = booking_id, error = %e, "Failed to free reservation");
            }
        }
    }

    /// First token: the prefill worker is done, so free its booking; release
    /// the decode booking's prefill load and keep its decode load until
    /// completion. `prefill_complete` is idempotent.
    async fn on_prefill_complete(&self, booking_id: &str) {
        if let Some(prefill) = &self.prefill
            && let Err(e) = prefill
                .selector
                .free_reservation(&stage_booking_id(booking_id, &StageId::PREFILL))
                .await
        {
            tracing::warn!(reservation_id = booking_id, error = %e, "Failed to free prefill reservation");
        }
        for stage in [StageId::DECODE, StageId::AGGREGATED] {
            if let Err(e) = self
                .decode
                .selector
                .prefill_complete(&stage_booking_id(booking_id, &stage))
                .await
            {
                tracing::warn!(reservation_id = booking_id, error = %e, "Failed to mark prefill complete");
            }
        }
    }
}

/// Why tokenizing a request for routing failed. Kept typed so the picker can map
/// each cause to the correct HTTP status instead of collapsing everything to 400.
enum TokenizeError {
    /// The request body could not be parsed — a genuine client (400) error.
    InvalidBody(serde_json::Error),
    /// The renderer call failed; the specific variant decides the status.
    Render(RenderError),
}

impl TokenizeError {
    /// Map to a client-safe [`PickError`], logging the detailed cause (which may
    /// include upstream URLs/bodies) server-side rather than returning it.
    fn into_pick_error(self, request_id: &str) -> PickError {
        match self {
            // The serde message describes the client's own JSON, not our
            // internals, so it is safe to surface as a 400.
            TokenizeError::InvalidBody(e) => {
                PickError::InvalidRequest(format!("invalid request body: {e}"))
            }
            TokenizeError::Render(e) => {
                tracing::warn!(request_id, error = %e, "Tokenization render failed");
                match &e {
                    RenderError::Unavailable { .. } => PickError::TokenizerUnavailable,
                    RenderError::Timeout { .. } => PickError::TokenizerTimeout,
                    RenderError::InvalidResponse { .. } | RenderError::ResponseTooLarge { .. } => {
                        PickError::TokenizerUpstreamError
                    }
                    RenderError::UpstreamStatus { status, .. } => {
                        match status.as_u16() {
                            // Only payload-validation statuses (400/422) mean the
                            // client's request was bad → surface as a client 400.
                            // Auth/misconfig (401/403/404), overload (429/503), any
                            // other 4xx, and 5xx are the renderer's or our own fault.
                            400 | 422 => PickError::InvalidRequest(
                                "request rejected by tokenization service".to_string(),
                            ),
                            // Renderer overloaded / temporarily unavailable → retryable.
                            429 | 503 => PickError::TokenizerUnavailable,
                            _ => PickError::TokenizerUpstreamError,
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    mod planning {
        use std::collections::HashMap;

        use dynamo_kv_router::config::KvRouterConfig;
        use dynamo_kv_router::services::selection::{CatalogReconciler, WorkerRequest};

        use super::super::*;
        use crate::epp_standalone_config::{EppStandaloneConfig, RendererProtocol};

        fn config() -> EppStandaloneConfig {
            EppStandaloneConfig {
                selector_threads: 1,
                peer_replication: None,
                inference_pool_name: "decode-pool".to_string(),
                prefill_inference_pool_name: Some("prefill-pool".to_string()),
                namespace: "test-ns".to_string(),
                model_name: "test-model".to_string(),
                tokenizer_service_url: "http://vllm-render:8000".to_string(),
                renderer_protocol: RendererProtocol::VllmRender,
                tokenizer_max_response_bytes: 16 * 1024 * 1024,
                tokenization_timeout_ms: 5_000,
                block_size: 16,
                data_parallel_size: 1,
                kv_event_port_stride: 1,
                kv_event_port: 5557,
                replay_port: None,
                total_kv_blocks: None,
                max_num_batched_tokens: Some(8192),
                max_inflight_requests: 1024,
                session_affinity_ttl_secs: None,
            }
        }

        fn worker(worker_id: u64) -> WorkerRequest {
            WorkerRequest {
                worker_id,
                model_name: "test-model".to_string(),
                endpoint: Some(format!("http://10.0.0.{worker_id}:8000")),
                block_size: Some(16),
                data_parallel_start_rank: Some(0),
                data_parallel_size: Some(1),
                kv_events_endpoints: HashMap::from([(
                    0u32,
                    format!("tcp://127.0.0.1:{}", 46_000 + worker_id),
                )]),
                ..Default::default()
            }
        }

        async fn selector(role: WorkerType, workers: &[u64]) -> Selector {
            let selector = Selector::new_with_kv_router_config(
                &config(),
                KvRouterConfig {
                    use_kv_events: false,
                    ..Default::default()
                },
                WorkerSelectionPolicyRegistry::default(),
                role,
            )
            .await
            .expect("selector builds");
            let registrations: Vec<WorkerRequest> = workers.iter().copied().map(worker).collect();
            CatalogReconciler::new(Arc::clone(selector.core()))
                .apply(&registrations)
                .await
                .expect("workers register");
            selector
        }

        fn active_requests(selector: &Selector) -> usize {
            selector
                .core()
                .loads(Some("test-model"), None)
                .into_iter()
                .flat_map(|model| model.loads)
                .map(|load| load.active_requests)
                .sum()
        }

        fn request(key: &str) -> impl Fn() -> RoutingRequest + '_ {
            move || RoutingRequest::new(key, PromptInput::from_tokens((1..=32).collect()))
        }

        #[tokio::test]
        async fn disaggregated_plan_books_both_pools_and_callbacks_free_by_id() {
            let decode = selector(WorkerType::Decode, &[1, 2]).await;
            let prefill = selector(WorkerType::Prefill, &[11]).await;
            let (aggregated, disaggregated) =
                build_coordinators("test-model", &decode, Some(&prefill)).unwrap();

            let plan = plan_request(disaggregated.as_ref(), &aggregated, request("req-1"))
                .await
                .unwrap();
            assert_eq!(plan.stages.len(), 2);
            assert_eq!(active_requests(&prefill), 1);
            assert_eq!(active_requests(&decode), 1);

            let adopted = adopt_plan(plan).unwrap();
            assert_eq!(adopted.prefill_worker_id, Some(11));
            assert!([1, 2].contains(&adopted.decode_worker_id));
            // Adoption hands the bookings to id-addressed ownership: nothing
            // was freed by dropping the plan.
            assert_eq!(active_requests(&prefill), 1);
            assert_eq!(active_requests(&decode), 1);

            // First token: the prefill booking is freed, decode keeps running.
            prefill
                .free_reservation(&stage_booking_id("req-1", &StageId::PREFILL))
                .await
                .unwrap();
            decode
                .prefill_complete(&stage_booking_id("req-1", &StageId::DECODE))
                .await
                .unwrap();
            // The fallback stage id is tried too and is a harmless no-op.
            decode
                .prefill_complete(&stage_booking_id("req-1", &StageId::AGGREGATED))
                .await
                .unwrap();
            assert_eq!(active_requests(&prefill), 0);
            assert_eq!(active_requests(&decode), 1);

            // Request end.
            decode
                .free_reservation(&stage_booking_id("req-1", &StageId::DECODE))
                .await
                .unwrap();
            assert_eq!(active_requests(&decode), 0);
        }

        #[tokio::test]
        async fn empty_prefill_pool_falls_back_to_the_decode_pool_alone() {
            let decode = selector(WorkerType::Decode, &[1]).await;
            let prefill = selector(WorkerType::Prefill, &[]).await;
            let (aggregated, disaggregated) =
                build_coordinators("test-model", &decode, Some(&prefill)).unwrap();

            let plan = plan_request(disaggregated.as_ref(), &aggregated, request("req-2"))
                .await
                .unwrap();
            assert_eq!(plan.stages.len(), 1);
            assert_eq!(plan.stages[0].stage, StageId::AGGREGATED);
            let adopted = adopt_plan(plan).unwrap();
            assert_eq!(adopted.prefill_worker_id, None);
            assert_eq!(adopted.decode_worker_id, 1);
            assert_eq!(active_requests(&decode), 1);
            decode
                .free_reservation(&stage_booking_id("req-2", &StageId::AGGREGATED))
                .await
                .unwrap();
            assert_eq!(active_requests(&decode), 0);
        }

        #[tokio::test]
        async fn dropping_an_unadopted_plan_frees_every_booking() {
            let decode = selector(WorkerType::Decode, &[1]).await;
            let prefill = selector(WorkerType::Prefill, &[11]).await;
            let (aggregated, disaggregated) =
                build_coordinators("test-model", &decode, Some(&prefill)).unwrap();
            let plan = plan_request(disaggregated.as_ref(), &aggregated, request("req-3"))
                .await
                .unwrap();
            assert_eq!(active_requests(&prefill) + active_requests(&decode), 2);
            drop(plan);
            tokio::time::timeout(Duration::from_secs(2), async {
                while active_requests(&prefill) + active_requests(&decode) != 0 {
                    tokio::task::yield_now().await;
                }
            })
            .await
            .expect("dropped plan frees its bookings");
        }

        #[tokio::test]
        async fn without_a_prefill_pool_only_the_aggregated_coordinator_exists() {
            let decode = selector(WorkerType::Aggregated, &[1]).await;
            let (aggregated, disaggregated) =
                build_coordinators("test-model", &decode, None).unwrap();
            assert!(disaggregated.is_none());
            let plan = plan_request(None, &aggregated, request("req-4"))
                .await
                .unwrap();
            assert_eq!(plan.stages[0].stage, StageId::AGGREGATED);
            plan.release_all().await.unwrap();
            assert_eq!(active_requests(&decode), 0);
        }
    }

    #[test]
    fn requested_policy_class_uses_frontend_metadata_extraction() {
        // The class rides a Dynamo metadata header; the extractor strips the
        // prefix, trims, and honors the first of repeated headers.
        let headers: Vec<(String, String)> = vec![
            (
                "x-dynamo-meta-policy-class".to_string(),
                " latency ".to_string(),
            ),
            (
                "x-dynamo-meta-policy-class".to_string(),
                "throughput".to_string(),
            ),
            ("x-request-id".to_string(), "irrelevant".to_string()),
        ];
        assert_eq!(
            requested_policy_class(&headers).unwrap().as_deref(),
            Some("latency")
        );

        // Mixed-case header names match as well.
        let headers: Vec<(String, String)> = vec![(
            "X-Dynamo-Meta-Policy-Class".to_string(),
            "express".to_string(),
        )];
        assert_eq!(
            requested_policy_class(&headers).unwrap().as_deref(),
            Some("express")
        );

        // No metadata header → no policy class.
        let headers: Vec<(String, String)> = vec![("x-request-id".to_string(), "r1".to_string())];
        assert_eq!(requested_policy_class(&headers).unwrap(), None);
    }

    #[test]
    fn requested_policy_class_preserves_typed_limit_error() {
        use dynamo_llm::http::service::metadata::MetadataHeaderError;

        let headers: Vec<(String, String)> = (0..65)
            .map(|i| (format!("x-dynamo-meta-key-{i:02}"), "v".to_string()))
            .collect();
        let err = requested_policy_class(&headers).expect_err("65 metadata entries must fail");
        assert!(
            matches!(
                err,
                PickError::MetadataHeadersTooLarge(MetadataHeaderError::TooManyEntries { .. })
            ),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn render_upstream_status_maps_to_correct_pick_error() {
        use reqwest::StatusCode;

        let map = |status: StatusCode| {
            TokenizeError::Render(RenderError::UpstreamStatus {
                status,
                body: String::new(),
            })
            .into_pick_error("req-1")
        };

        // Renderer validated the client's payload and rejected it → client 400.
        assert!(matches!(
            map(StatusCode::BAD_REQUEST),
            PickError::InvalidRequest(_)
        ));
        assert!(matches!(
            map(StatusCode::UNPROCESSABLE_ENTITY),
            PickError::InvalidRequest(_)
        ));

        // Auth / misconfiguration is NOT an invalid client payload → upstream 502,
        // not a misleading 400.
        for status in [
            StatusCode::UNAUTHORIZED,
            StatusCode::FORBIDDEN,
            StatusCode::NOT_FOUND,
            StatusCode::INTERNAL_SERVER_ERROR,
            StatusCode::BAD_GATEWAY,
        ] {
            assert!(
                matches!(map(status), PickError::TokenizerUpstreamError),
                "{status} should map to an upstream error, not a client 400"
            );
        }

        // Overloaded / temporarily unavailable → retryable 503.
        assert!(matches!(
            map(StatusCode::TOO_MANY_REQUESTS),
            PickError::TokenizerUnavailable
        ));
        assert!(matches!(
            map(StatusCode::SERVICE_UNAVAILABLE),
            PickError::TokenizerUnavailable
        ));
    }

    #[test]
    fn endpoint_in_subset_matches_ip_port_or_bare_ip() {
        fn matches(endpoint: &str, values: &[&str]) -> bool {
            let candidates: HashSet<&str> = values.iter().copied().collect();
            let candidate_ips: HashSet<IpAddr> = values
                .iter()
                .filter_map(|candidate| candidate.parse().ok())
                .collect();
            endpoint_in_subset(endpoint, &candidates, &candidate_ips)
        }

        // Full ip:port match.
        assert!(matches("10.0.0.1:8000", &["10.0.0.1:8000"]));
        // Bare-ip match (subset lists just the IP).
        assert!(matches("10.0.0.2:8000", &["10.0.0.2"]));
        // Subset pinned a full ip:port, so a different port on that IP does NOT match.
        assert!(!matches("10.0.0.1:9999", &["10.0.0.1:8000"]));
        // Unrelated endpoint does not match.
        assert!(!matches("10.0.0.3:8000", &["10.0.0.2"]));

        // Full bracketed IPv6 endpoint match.
        assert!(matches("[fd00::1]:8000", &["[fd00::1]:8000"]));
        // Bare IPv6 match uses the normalized address, without brackets.
        assert!(matches("[fd00::2]:8000", &["fd00::2"]));
        // A different port does not match a full-endpoint-only candidate.
        assert!(!matches("[fd00::1]:9999", &["[fd00::1]:8000"]));
    }
}
