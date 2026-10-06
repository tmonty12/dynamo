// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Coordinator behaviour over the fake selector: the request flows from the
//! DEP, placement rules, and the ownership and fencing failure matrix.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;

use crate::WorkerType;
use crate::conditional_disagg::IslBoundingPolicy;
use crate::protocols::{KvTransferEnforcement, WorkerWithDpRank};

use super::test_support::{FakeEvent, FakeStageSelector, FakeWorker};
use super::*;

const PROMPT: [u32; 16] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];

fn request(id: &str) -> RoutingRequest {
    RoutingRequest::new(id, PromptInput::from_tokens(PROMPT.to_vec()))
}

fn pool(name: &'static str) -> PoolRef {
    PoolRef::new(name, 1)
}

struct Fixture {
    prefill: Arc<FakeStageSelector>,
    decode: Arc<FakeStageSelector>,
    encode: Arc<FakeStageSelector>,
    aggregated: Arc<FakeStageSelector>,
}

impl Fixture {
    fn new() -> Self {
        Self {
            prefill: FakeStageSelector::with_workers(pool("prefill"), [11, 12]),
            decode: FakeStageSelector::with_workers(pool("decode"), [21, 22]),
            encode: FakeStageSelector::with_workers(pool("encode"), [31]),
            aggregated: FakeStageSelector::with_workers(pool("agg"), [1, 2]),
        }
    }

    fn bindings(&self) -> Vec<StageBinding> {
        vec![
            StageBinding::new(
                StageId::PREFILL,
                self.prefill.pool().clone(),
                WorkerType::Prefill,
                self.prefill.clone(),
            ),
            StageBinding::new(
                StageId::DECODE,
                self.decode.pool().clone(),
                WorkerType::Decode,
                self.decode.clone(),
            ),
            StageBinding::new(
                StageId::ENCODE,
                self.encode.pool().clone(),
                WorkerType::Encode,
                self.encode.clone(),
            ),
            StageBinding::new(
                StageId::AGGREGATED,
                self.aggregated.pool().clone(),
                WorkerType::Aggregated,
                self.aggregated.clone(),
            ),
        ]
    }

    fn coordinator(
        &self,
        topology: Topology,
        policy: CoordinationPolicyFactory,
    ) -> RoutingCoordinator {
        RoutingCoordinator::new(topology, self.bindings(), policy).unwrap()
    }

    fn outstanding(&self) -> usize {
        self.prefill.outstanding_reservations().len()
            + self.decode.outstanding_reservations().len()
            + self.encode.outstanding_reservations().len()
            + self.aggregated.outstanding_reservations().len()
    }
}

fn progressive_pd() -> CoordinationPolicyFactory {
    Arc::new(|_| Box::new(ProgressivePrefillDecodePolicy::new()))
}

fn prefill_first() -> CoordinationPolicyFactory {
    Arc::new(|_| Box::new(PrefillDecodePolicy::prefill_first()))
}

fn decode_first() -> CoordinationPolicyFactory {
    Arc::new(|_| Box::new(PrefillDecodePolicy::decode_first()))
}

fn conditional(
    enabled: bool,
    thresholds: ConditionalDisaggThresholds,
) -> CoordinationPolicyFactory {
    Arc::new(move |_| {
        Box::new(ConditionalDisaggregationPolicy::new(
            Box::new(IslBoundingPolicy::new(enabled, 2048, 0.7)),
            thresholds,
            Box::new(ProgressivePrefillDecodePolicy::new()),
        ))
    })
}

async fn expect_execute(
    coordinator: &RoutingCoordinator,
    session: &mut RouteSession,
    event: HostEvent,
    stage: &StageId,
) -> ReadyStage {
    match coordinator.advance(session, event).await.unwrap() {
        HostAction::Execute(ready) => {
            assert_eq!(ready.stage, *stage, "expected {stage} to be ready");
            *ready
        }
        other => panic!("expected Execute({stage}), got {other:?}"),
    }
}

async fn expect_wait(
    coordinator: &RoutingCoordinator,
    session: &mut RouteSession,
    event: HostEvent,
) {
    match coordinator.advance(session, event).await.unwrap() {
        HostAction::Wait => {}
        other => panic!("expected Wait, got {other:?}"),
    }
}

async fn expect_complete(
    coordinator: &RoutingCoordinator,
    session: &mut RouteSession,
    event: HostEvent,
) {
    match coordinator.advance(session, event).await.unwrap() {
        HostAction::Complete => {}
        other => panic!("expected Complete, got {other:?}"),
    }
    assert!(session.is_closed());
}

fn dispatched(ready: &ReadyStage) -> HostEvent {
    HostEvent::Dispatched {
        stage: ready.stage.clone(),
        attempt: ready.attempt,
    }
}

fn handoff(ready: &ReadyStage) -> HostEvent {
    HostEvent::HandoffReady {
        stage: ready.stage.clone(),
        attempt: ready.attempt,
    }
}

// ---------------------------------------------------------------------------
// Request flows (DEP §5)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn aggregated_serving_admits_executes_and_completes() {
    let fixture = Fixture::new();
    let coordinator = fixture.coordinator(
        Topology::aggregated(),
        Arc::new(|_| Box::new(AggregatedPolicy::new())),
    );
    let mut session = coordinator
        .start(request("agg-1"), PlanningMode::Progressive)
        .unwrap();

    let ready = expect_execute(
        &coordinator,
        &mut session,
        HostEvent::Continue,
        &StageId::AGGREGATED,
    )
    .await;
    assert!(ready.inputs.is_empty());
    assert_eq!(ready.branch, DEFAULT_BRANCH);
    assert_eq!(session.status(&StageId::AGGREGATED), StageStatus::Executing);
    // The host owns the reservation now.
    assert_eq!(fixture.aggregated.outstanding_reservations().len(), 1);

    expect_complete(&coordinator, &mut session, dispatched(&ready)).await;
    assert_eq!(
        session.status(&StageId::AGGREGATED),
        StageStatus::Dispatched
    );
    assert_eq!(fixture.aggregated.outstanding_reservations().len(), 1);
    drop(ready);
    assert_eq!(fixture.outstanding(), 0);
    assert!(matches!(
        coordinator.advance(&mut session, HostEvent::Continue).await,
        Err(CoordinationError::Finished)
    ));
}

#[tokio::test]
async fn progressive_prefill_decode_waits_for_the_handoff() {
    let fixture = Fixture::new();
    let coordinator = fixture.coordinator(Topology::prefill_decode(), progressive_pd());
    let mut session = coordinator
        .start(request("pd-1"), PlanningMode::Progressive)
        .unwrap();

    let prefill = expect_execute(
        &coordinator,
        &mut session,
        HostEvent::Continue,
        &StageId::PREFILL,
    )
    .await;
    assert_eq!(prefill.reservation.target.stage, StageId::PREFILL);
    expect_wait(&coordinator, &mut session, dispatched(&prefill)).await;
    assert_eq!(session.status(&StageId::DECODE), StageStatus::Pending);
    assert_eq!(fixture.decode.outstanding_reservations().len(), 0);

    let decode = expect_execute(
        &coordinator,
        &mut session,
        handoff(&prefill),
        &StageId::DECODE,
    )
    .await;
    assert_eq!(decode.inputs, vec![StageId::PREFILL]);
    let admit = fixture
        .decode
        .events()
        .into_iter()
        .find(|event| matches!(event, FakeEvent::Admit { .. }))
        .unwrap();
    let FakeEvent::Admit { reservation_id, .. } = admit else {
        unreachable!()
    };
    assert_eq!(reservation_id, "pd-1/decode/0");

    expect_complete(&coordinator, &mut session, dispatched(&decode)).await;
    assert_eq!(fixture.outstanding(), 2);
    drop((prefill, decode));
    assert_eq!(fixture.outstanding(), 0);
}

#[tokio::test]
async fn plan_all_selects_every_stage_before_execution() {
    let fixture = Fixture::new();
    let coordinator = fixture.coordinator(Topology::prefill_decode(), prefill_first());
    let plan = coordinator.plan_all(request("plan-1")).await.unwrap();

    assert_eq!(plan.request_id, "plan-1");
    assert_eq!(plan.branch, DEFAULT_BRANCH);
    let stages: Vec<&StageId> = plan.stages.iter().map(|planned| &planned.stage).collect();
    assert_eq!(stages, vec![&StageId::PREFILL, &StageId::DECODE]);
    assert!(plan.stage(&StageId::PREFILL).unwrap().depends_on.is_empty());
    assert_eq!(
        plan.stage(&StageId::DECODE).unwrap().depends_on,
        vec![StageId::PREFILL]
    );
    assert_eq!(fixture.outstanding(), 2);

    plan.release_all().await.unwrap();
    assert_eq!(fixture.outstanding(), 0);
    assert_eq!(fixture.prefill.release_count("plan-1/prefill/0"), 1);
    assert_eq!(fixture.decode.release_count("plan-1/decode/0"), 1);
}

#[tokio::test]
async fn plan_all_orders_a_decode_first_selection_by_execution_dependency() {
    let fixture = Fixture::new();
    let coordinator = fixture.coordinator(Topology::prefill_decode(), decode_first());
    let plan = coordinator.plan_all(request("plan-2")).await.unwrap();

    // Decode was selected first...
    let first_admit = fixture
        .decode
        .events()
        .into_iter()
        .chain(fixture.prefill.events())
        .filter(|event| matches!(event, FakeEvent::Admit { .. }))
        .map(|event| match event {
            FakeEvent::Admit { stage, .. } => stage,
            _ => unreachable!(),
        })
        .next()
        .unwrap();
    assert_eq!(first_admit, StageId::DECODE);
    // ...but the plan still runs prefill before decode.
    let stages: Vec<&StageId> = plan.stages.iter().map(|planned| &planned.stage).collect();
    assert_eq!(stages, vec![&StageId::PREFILL, &StageId::DECODE]);
    drop(plan);
    assert_eq!(fixture.outstanding(), 0);
}

#[tokio::test]
async fn progressive_decode_first_executes_prefill_first() {
    let fixture = Fixture::new();
    let coordinator = fixture.coordinator(Topology::prefill_decode(), decode_first());
    let mut session = coordinator
        .start(request("df-1"), PlanningMode::Progressive)
        .unwrap();

    let prefill = expect_execute(
        &coordinator,
        &mut session,
        HostEvent::Continue,
        &StageId::PREFILL,
    )
    .await;
    // Decode is admitted and held by the session until prefill hands off.
    assert_eq!(session.status(&StageId::DECODE), StageStatus::Admitted);
    expect_wait(&coordinator, &mut session, dispatched(&prefill)).await;
    let decode = expect_execute(
        &coordinator,
        &mut session,
        handoff(&prefill),
        &StageId::DECODE,
    )
    .await;
    expect_complete(&coordinator, &mut session, dispatched(&decode)).await;
}

#[tokio::test]
async fn conditional_disagg_bypasses_to_the_previewed_decode_worker() {
    let fixture = Fixture::new();
    // Worker 22 holds almost the whole prompt; the ISL policy bypasses.
    fixture.decode.set_signals(
        22,
        SelectionSignals {
            cached_tokens: 15,
            overlap_blocks: 3,
            potential_decode_blocks: 1,
            total_kv_blocks: Some(100),
            ..SelectionSignals::default()
        },
    );
    fixture.decode.set_signals(
        21,
        SelectionSignals {
            potential_decode_blocks: 5,
            ..SelectionSignals::default()
        },
    );
    let coordinator = fixture.coordinator(
        Topology::conditional_prefill_decode(),
        conditional(true, ConditionalDisaggThresholds::default()),
    );
    let mut session = coordinator
        .start(request("cond-1"), PlanningMode::Progressive)
        .unwrap();

    let decode = expect_execute(
        &coordinator,
        &mut session,
        HostEvent::Continue,
        &StageId::DECODE,
    )
    .await;
    assert_eq!(decode.branch, LOCAL_PREFILL_DECODE_BRANCH);
    assert_eq!(decode.target.worker.worker_id, 22);
    assert!(decode.inputs.is_empty());
    let events = fixture.decode.events();
    assert!(matches!(
        events[0],
        FakeEvent::Preview { ref stage, .. } if *stage == StageId::DECODE
    ));
    assert!(matches!(
        events[1],
        FakeEvent::Admit { from_preview: true, ref stage, .. } if *stage == StageId::DECODE
    ));
    assert!(fixture.prefill.events().is_empty());
    expect_complete(&coordinator, &mut session, dispatched(&decode)).await;
}

#[tokio::test]
async fn conditional_bypass_survives_placement_rules() {
    // Choosing the local branch happens between the decode preview and its
    // admission; with placement rules on the topology that choice must not
    // stale the preview.
    let fixture = Fixture::new();
    fixture.decode.remove_worker(21);
    fixture.decode.set_signals(
        22,
        SelectionSignals {
            cached_tokens: 16,
            ..SelectionSignals::default()
        },
    );
    let coordinator = fixture.coordinator(
        Topology::conditional_prefill_decode().with_rule(PlacementRule::TransferCompatible),
        conditional(true, ConditionalDisaggThresholds::default()),
    );
    let mut session = coordinator
        .start(request("cond-7"), PlanningMode::Progressive)
        .unwrap();
    let decode = expect_execute(
        &coordinator,
        &mut session,
        HostEvent::Continue,
        &StageId::DECODE,
    )
    .await;
    assert_eq!(decode.branch, LOCAL_PREFILL_DECODE_BRANCH);
    assert!(matches!(
        fixture.decode.events().as_slice(),
        [
            FakeEvent::Preview { .. },
            FakeEvent::Admit {
                from_preview: true,
                ..
            }
        ]
    ));
}

#[tokio::test]
async fn conditional_disagg_takes_remote_prefill_when_the_policy_declines() {
    let fixture = Fixture::new();
    let coordinator = fixture.coordinator(
        Topology::conditional_prefill_decode(),
        conditional(true, ConditionalDisaggThresholds::default()),
    );
    let mut session = coordinator
        .start(request("cond-2"), PlanningMode::Progressive)
        .unwrap();

    let prefill = expect_execute(
        &coordinator,
        &mut session,
        HostEvent::Continue,
        &StageId::PREFILL,
    )
    .await;
    assert_eq!(prefill.branch, REMOTE_PREFILL_DECODE_BRANCH);
    // The decode preview happened but nothing was admitted on decode yet.
    assert!(matches!(
        fixture.decode.events().as_slice(),
        [FakeEvent::Preview { .. }]
    ));
    expect_wait(&coordinator, &mut session, dispatched(&prefill)).await;
    let decode = expect_execute(
        &coordinator,
        &mut session,
        handoff(&prefill),
        &StageId::DECODE,
    )
    .await;
    assert_eq!(decode.inputs, vec![StageId::PREFILL]);
    expect_complete(&coordinator, &mut session, dispatched(&decode)).await;
}

#[tokio::test]
async fn conditional_disagg_decode_gate_vetoes_a_busy_decode_worker() {
    let fixture = Fixture::new();
    fixture.decode.remove_worker(21);
    fixture.decode.set_signals(
        22,
        SelectionSignals {
            cached_tokens: 15,
            potential_decode_blocks: 95,
            total_kv_blocks: Some(100),
            ..SelectionSignals::default()
        },
    );
    let coordinator = fixture.coordinator(
        Topology::conditional_prefill_decode(),
        conditional(
            true,
            ConditionalDisaggThresholds {
                prefill_busy: None,
                decode_busy: Some(0.9),
            },
        ),
    );
    let mut session = coordinator
        .start(request("cond-3"), PlanningMode::Progressive)
        .unwrap();
    let prefill = expect_execute(
        &coordinator,
        &mut session,
        HostEvent::Continue,
        &StageId::PREFILL,
    )
    .await;
    assert_eq!(prefill.branch, REMOTE_PREFILL_DECODE_BRANCH);
}

#[tokio::test]
async fn conditional_disagg_skips_the_decision_for_a_pinned_prefill_worker() {
    let fixture = Fixture::new();
    fixture.decode.set_signals(
        22,
        SelectionSignals {
            cached_tokens: 16,
            ..SelectionSignals::default()
        },
    );
    let coordinator = fixture.coordinator(
        Topology::conditional_prefill_decode(),
        conditional(true, ConditionalDisaggThresholds::default()),
    );
    let request = request("cond-4").with_stage_restrictions(
        StageId::PREFILL,
        SelectionRestrictions {
            pinned_worker: Some(WorkerWithDpRank::new(12, 0)),
            ..SelectionRestrictions::default()
        },
    );
    let mut session = coordinator
        .start(request, PlanningMode::Progressive)
        .unwrap();
    let prefill = expect_execute(
        &coordinator,
        &mut session,
        HostEvent::Continue,
        &StageId::PREFILL,
    )
    .await;
    assert_eq!(prefill.target.worker.worker_id, 12);
    assert!(
        fixture.decode.events().is_empty(),
        "no decode preview for a pinned prefill"
    );
}

#[tokio::test]
async fn conditional_disagg_disabled_goes_straight_to_remote_prefill() {
    let fixture = Fixture::new();
    let coordinator = fixture.coordinator(
        Topology::conditional_prefill_decode(),
        conditional(false, ConditionalDisaggThresholds::default()),
    );
    let plan = coordinator.plan_all(request("cond-5")).await.unwrap();
    assert_eq!(plan.branch, REMOTE_PREFILL_DECODE_BRANCH);
    assert_eq!(plan.stages.len(), 2);
    assert!(
        fixture
            .decode
            .events()
            .iter()
            .all(|event| !matches!(event, FakeEvent::Preview { .. }))
    );
}

#[tokio::test]
async fn conditional_disagg_consults_prefill_load_when_the_policy_needs_it() {
    use crate::conditional_disagg::PrefillLoadPolicy;

    let fixture = Fixture::new();
    fixture.prefill.remove_worker(12);
    fixture.prefill.set_signals(
        11,
        SelectionSignals {
            prefill_load: Some(PrefillLoadSignal {
                active_prefill_tokens: 900,
                prefill_token_capacity: 1000,
            }),
            ..SelectionSignals::default()
        },
    );
    let factory: CoordinationPolicyFactory = Arc::new(|_| {
        Box::new(ConditionalDisaggregationPolicy::new(
            Box::new(PrefillLoadPolicy::new(true)),
            ConditionalDisaggThresholds {
                prefill_busy: Some(0.5),
                decode_busy: None,
            },
            Box::new(ProgressivePrefillDecodePolicy::new()),
        ))
    });
    let coordinator = fixture.coordinator(Topology::conditional_prefill_decode(), factory);
    let mut session = coordinator
        .start(request("cond-6"), PlanningMode::Progressive)
        .unwrap();
    let decode = expect_execute(
        &coordinator,
        &mut session,
        HostEvent::Continue,
        &StageId::DECODE,
    )
    .await;
    assert_eq!(decode.branch, LOCAL_PREFILL_DECODE_BRANCH);
    assert!(matches!(
        fixture.prefill.events().as_slice(),
        [FakeEvent::Preview { .. }]
    ));
    assert_eq!(fixture.prefill.outstanding_reservations().len(), 0);
}

#[tokio::test]
async fn encode_prefill_decode_passes_each_handoff_in_order() {
    let fixture = Fixture::new();
    let coordinator = fixture.coordinator(
        Topology::encode_prefill_decode(),
        Arc::new(|_| Box::new(EncodePrefillDecodePolicy::new(true))),
    );
    let mut session = coordinator
        .start(
            request("epd-1").with_encode_input(true),
            PlanningMode::Progressive,
        )
        .unwrap();

    let encode = expect_execute(
        &coordinator,
        &mut session,
        HostEvent::Continue,
        &StageId::ENCODE,
    )
    .await;
    assert_eq!(encode.branch, ENCODE_PREFILL_DECODE_BRANCH);
    expect_wait(&coordinator, &mut session, dispatched(&encode)).await;
    let prefill = expect_execute(
        &coordinator,
        &mut session,
        handoff(&encode),
        &StageId::PREFILL,
    )
    .await;
    assert_eq!(prefill.inputs, vec![StageId::ENCODE]);
    expect_wait(&coordinator, &mut session, dispatched(&prefill)).await;
    let decode = expect_execute(
        &coordinator,
        &mut session,
        handoff(&prefill),
        &StageId::DECODE,
    )
    .await;
    assert_eq!(decode.inputs, vec![StageId::PREFILL]);
    expect_complete(&coordinator, &mut session, dispatched(&decode)).await;
}

#[tokio::test]
async fn encode_prefill_decode_skips_encode_without_multimodal_input() {
    let fixture = Fixture::new();
    let coordinator = fixture.coordinator(
        Topology::encode_prefill_decode(),
        Arc::new(|_| Box::new(EncodePrefillDecodePolicy::new(false))),
    );
    let plan = coordinator.plan_all(request("epd-2")).await.unwrap();
    assert_eq!(plan.branch, PREFILL_DECODE_BRANCH);
    assert_eq!(plan.stages.len(), 2);
    assert!(fixture.encode.events().is_empty());
}

// ---------------------------------------------------------------------------
// Cross-stage constraints (DEP §4.5)
// ---------------------------------------------------------------------------

fn zoned_worker(worker_id: u64, zone: &str) -> FakeWorker {
    FakeWorker::new(worker_id).with_facts(WorkerFacts {
        taints: HashSet::from([topology_taint("zone", zone)]),
        topology_domains: HashMap::from([("zone".to_string(), zone.to_string())]),
        ..WorkerFacts::default()
    })
}

fn transfer_worker(worker_id: u64, zone: &str, enforcement: KvTransferEnforcement) -> FakeWorker {
    FakeWorker::new(worker_id).with_facts(WorkerFacts {
        taints: HashSet::from([topology_taint("zone", zone)]),
        topology_domains: HashMap::from([("zone".to_string(), zone.to_string())]),
        kv_transfer_domain: Some("zone".to_string()),
        kv_transfer_enforcement: Some(enforcement),
        kv_transfer_preferred_weight: Some(0.5),
        ..WorkerFacts::default()
    })
}

#[tokio::test]
async fn same_domain_rule_constrains_decode_to_the_prefill_zone() {
    let fixture = Fixture::new();
    fixture.prefill.remove_worker(11);
    fixture.prefill.remove_worker(12);
    fixture.prefill.add_worker(zoned_worker(11, "b"));
    fixture.decode.remove_worker(21);
    fixture.decode.remove_worker(22);
    // Worker 21 is lighter, but in the wrong zone.
    fixture.decode.add_worker(zoned_worker(21, "a"));
    fixture
        .decode
        .add_worker(zoned_worker(22, "b").with_signals(SelectionSignals {
            potential_decode_blocks: 50,
            ..SelectionSignals::default()
        }));
    let topology = Topology::prefill_decode().with_rule(PlacementRule::SameDomain {
        domain: "zone".to_string(),
        mode: PlacementMode::Required,
    });
    let coordinator = fixture.coordinator(topology, prefill_first());
    let plan = coordinator.plan_all(request("zone-1")).await.unwrap();
    assert_eq!(
        plan.stage(&StageId::PREFILL)
            .unwrap()
            .target
            .worker
            .worker_id,
        11
    );
    assert_eq!(
        plan.stage(&StageId::DECODE)
            .unwrap()
            .target
            .worker
            .worker_id,
        22
    );
}

#[tokio::test]
async fn unsatisfiable_placement_releases_the_held_reservation() {
    let fixture = Fixture::new();
    fixture.prefill.remove_worker(12);
    fixture.prefill.remove_worker(11);
    fixture
        .prefill
        .add_worker(transfer_worker(11, "a", KvTransferEnforcement::Required));
    fixture.decode.remove_worker(21);
    fixture.decode.remove_worker(22);
    fixture.decode.add_worker(zoned_worker(21, "b"));
    let topology = Topology::prefill_decode().with_rule(PlacementRule::TransferCompatible);
    let coordinator = fixture.coordinator(topology, prefill_first());
    let error = coordinator.plan_all(request("zone-2")).await.unwrap_err();
    assert!(
        matches!(error, CoordinationError::NoEligibleWorkers { ref stage, .. } if *stage == StageId::DECODE),
        "{error}"
    );
    assert_eq!(fixture.outstanding(), 0);
    assert_eq!(fixture.prefill.release_count("zone-2/prefill/0"), 1);
}

#[tokio::test]
async fn decode_first_selection_still_satisfies_the_prefill_transfer_requirement() {
    let fixture = Fixture::new();
    fixture.prefill.remove_worker(11);
    fixture.prefill.remove_worker(12);
    fixture
        .prefill
        .add_worker(transfer_worker(11, "a", KvTransferEnforcement::Required));
    fixture.decode.remove_worker(21);
    fixture.decode.remove_worker(22);
    // Decode is selected first and lands in zone b; prefill in zone a then
    // requires zone b peers, which the pair validation catches.
    fixture.decode.add_worker(zoned_worker(21, "b"));
    let topology = Topology::prefill_decode().with_rule(PlacementRule::TransferCompatible);
    let coordinator = fixture.coordinator(topology, decode_first());
    let error = coordinator.plan_all(request("zone-3")).await.unwrap_err();
    assert!(
        matches!(error, CoordinationError::Placement { .. }),
        "{error}"
    );
    assert_eq!(fixture.outstanding(), 0);
}

#[tokio::test]
async fn caller_restrictions_apply_only_to_the_named_stage() {
    let fixture = Fixture::new();
    let coordinator = fixture.coordinator(Topology::prefill_decode(), prefill_first());
    // A gateway destination subset that names only decode worker 22 must not
    // narrow the prefill pool.
    let request = request("subset-1").with_stage_restrictions(
        StageId::DECODE,
        SelectionRestrictions {
            allowed_worker_ids: Some(HashSet::from([22])),
            ..SelectionRestrictions::default()
        },
    );
    let plan = coordinator.plan_all(request).await.unwrap();
    assert_eq!(
        plan.stage(&StageId::DECODE)
            .unwrap()
            .target
            .worker
            .worker_id,
        22
    );
    assert_eq!(
        plan.stage(&StageId::PREFILL)
            .unwrap()
            .target
            .worker
            .worker_id,
        11
    );
}

// ---------------------------------------------------------------------------
// Lifecycle and correctness (DEP §6) and the failure-injection matrix
// ---------------------------------------------------------------------------

#[tokio::test]
async fn cancel_before_admission_completes_leaves_nothing_reserved() {
    let fixture = Fixture::new();
    let coordinator = fixture.coordinator(Topology::prefill_decode(), progressive_pd());
    let mut session = coordinator
        .start(request("cancel-1"), PlanningMode::Progressive)
        .unwrap();
    fixture.prefill.set_admission_open(false);
    {
        let advance = coordinator.advance(&mut session, HostEvent::Continue);
        tokio::pin!(advance);
        assert!(futures_util::poll!(advance.as_mut()).is_pending());
        // Dropping the pinned future cancels the in-flight admission.
    }
    assert!(fixture.prefill.events().is_empty());
    assert_eq!(session.status(&StageId::PREFILL), StageStatus::Pending);
    // The session is still usable once admission opens.
    fixture.prefill.set_admission_open(true);
    let prefill = expect_execute(
        &coordinator,
        &mut session,
        HostEvent::Continue,
        &StageId::PREFILL,
    )
    .await;
    assert_eq!(prefill.attempt, AttemptId::FIRST);
}

#[tokio::test]
async fn cancel_between_admission_and_transfer_releases_the_held_stage() {
    let fixture = Fixture::new();
    let coordinator = fixture.coordinator(Topology::prefill_decode(), decode_first());
    let mut session = coordinator
        .start(request("cancel-2"), PlanningMode::Progressive)
        .unwrap();
    let prefill = expect_execute(
        &coordinator,
        &mut session,
        HostEvent::Continue,
        &StageId::PREFILL,
    )
    .await;
    assert_eq!(session.status(&StageId::DECODE), StageStatus::Admitted);
    assert_eq!(fixture.decode.outstanding_reservations().len(), 1);

    // Prefill was handed over and may be mid-dispatch, so decode is kept until
    // the host reports what became of prefill.
    expect_wait(&coordinator, &mut session, HostEvent::Cancelled).await;
    assert_eq!(session.status(&StageId::DECODE), StageStatus::Admitted);

    // The host declined to dispatch after the cancellation.
    drop(prefill);
    expect_complete(
        &coordinator,
        &mut session,
        HostEvent::Failed {
            stage: StageId::PREFILL,
            attempt: AttemptId::FIRST,
            retry: false,
        },
    )
    .await;
    assert_eq!(fixture.outstanding(), 0);
}

#[tokio::test]
async fn cancel_after_prefill_dispatch_still_routes_decode_for_kv_cleanup() {
    let fixture = Fixture::new();
    let coordinator = fixture.coordinator(Topology::prefill_decode(), progressive_pd());
    let mut session = coordinator
        .start(request("cancel-3"), PlanningMode::Progressive)
        .unwrap();
    let prefill = expect_execute(
        &coordinator,
        &mut session,
        HostEvent::Continue,
        &StageId::PREFILL,
    )
    .await;
    expect_wait(&coordinator, &mut session, dispatched(&prefill)).await;
    // The client is gone, but prefill is staging KV for one decode worker.
    expect_wait(&coordinator, &mut session, HostEvent::Cancelled).await;
    assert!(session.cancelled());
    let decode = expect_execute(
        &coordinator,
        &mut session,
        handoff(&prefill),
        &StageId::DECODE,
    )
    .await;
    assert_eq!(decode.inputs, vec![StageId::PREFILL]);
    expect_complete(&coordinator, &mut session, dispatched(&decode)).await;
}

#[tokio::test]
async fn cancel_with_a_held_dependent_of_a_dispatched_producer_keeps_it() {
    let fixture = Fixture::new();
    let coordinator = fixture.coordinator(Topology::prefill_decode(), decode_first());
    let mut session = coordinator
        .start(request("cancel-4"), PlanningMode::Progressive)
        .unwrap();
    let prefill = expect_execute(
        &coordinator,
        &mut session,
        HostEvent::Continue,
        &StageId::PREFILL,
    )
    .await;
    expect_wait(&coordinator, &mut session, dispatched(&prefill)).await;
    expect_wait(&coordinator, &mut session, HostEvent::Cancelled).await;
    // Decode stays reserved: prefill already reached a worker.
    assert_eq!(session.status(&StageId::DECODE), StageStatus::Admitted);
    let decode = expect_execute(
        &coordinator,
        &mut session,
        handoff(&prefill),
        &StageId::DECODE,
    )
    .await;
    expect_complete(&coordinator, &mut session, dispatched(&decode)).await;
}

#[tokio::test]
async fn dropping_the_host_side_after_transfer_releases_the_reservation() {
    let fixture = Fixture::new();
    let coordinator = fixture.coordinator(
        Topology::aggregated(),
        Arc::new(|_| Box::new(AggregatedPolicy::new())),
    );
    let mut session = coordinator
        .start(request("drop-1"), PlanningMode::Progressive)
        .unwrap();
    let ready = expect_execute(
        &coordinator,
        &mut session,
        HostEvent::Continue,
        &StageId::AGGREGATED,
    )
    .await;
    drop(ready);
    assert_eq!(fixture.outstanding(), 0);
    assert_eq!(fixture.aggregated.release_count("drop-1/aggregated/0"), 1);
    // The session no longer owns anything and finishes cleanly.
    expect_complete(&coordinator, &mut session, HostEvent::Continue).await;

    let plan = coordinator.plan_all(request("drop-2")).await.unwrap();
    drop(plan);
    assert_eq!(fixture.outstanding(), 0);
}

#[tokio::test]
async fn retry_fences_the_old_attempts_handoff_and_failure() {
    let fixture = Fixture::new();
    let coordinator = fixture.coordinator(Topology::prefill_decode(), progressive_pd());
    let mut session = coordinator
        .start(request("retry-1"), PlanningMode::Progressive)
        .unwrap();
    let first = expect_execute(
        &coordinator,
        &mut session,
        HostEvent::Continue,
        &StageId::PREFILL,
    )
    .await;
    assert_eq!(first.attempt, AttemptId::FIRST);
    assert_eq!(first.target.worker.worker_id, 11);
    expect_wait(&coordinator, &mut session, dispatched(&first)).await;

    // The worker failed; the host releases its reservation and asks to retry.
    let failed_worker = first.target.worker.worker_id;
    drop(first);
    let second = expect_execute(
        &coordinator,
        &mut session,
        HostEvent::Failed {
            stage: StageId::PREFILL,
            attempt: AttemptId::FIRST,
            retry: true,
        },
        &StageId::PREFILL,
    )
    .await;
    assert_eq!(second.attempt, AttemptId::new(1));
    assert_ne!(second.target.worker.worker_id, failed_worker);
    assert_eq!(second.reservation.target.attempt, AttemptId::new(1));

    // Late events for the old attempt change nothing.
    expect_wait(
        &coordinator,
        &mut session,
        HostEvent::HandoffReady {
            stage: StageId::PREFILL,
            attempt: AttemptId::FIRST,
        },
    )
    .await;
    assert_eq!(session.status(&StageId::DECODE), StageStatus::Pending);
    expect_wait(
        &coordinator,
        &mut session,
        HostEvent::Failed {
            stage: StageId::PREFILL,
            attempt: AttemptId::FIRST,
            retry: true,
        },
    )
    .await;
    assert_eq!(session.attempt(&StageId::PREFILL), AttemptId::new(1));
    assert_eq!(session.status(&StageId::PREFILL), StageStatus::Executing);

    expect_wait(&coordinator, &mut session, dispatched(&second)).await;
    let decode = expect_execute(
        &coordinator,
        &mut session,
        handoff(&second),
        &StageId::DECODE,
    )
    .await;
    expect_complete(&coordinator, &mut session, dispatched(&decode)).await;
}

#[tokio::test]
async fn duplicate_terminal_events_are_idempotent() {
    let fixture = Fixture::new();
    let coordinator = fixture.coordinator(Topology::prefill_decode(), progressive_pd());
    let mut session = coordinator
        .start(request("dup-1"), PlanningMode::Progressive)
        .unwrap();
    let prefill = expect_execute(
        &coordinator,
        &mut session,
        HostEvent::Continue,
        &StageId::PREFILL,
    )
    .await;
    expect_wait(&coordinator, &mut session, dispatched(&prefill)).await;
    let completed = HostEvent::Completed {
        stage: StageId::PREFILL,
        attempt: prefill.attempt,
    };
    // Completion implies the handoff; decode becomes ready once.
    let decode = expect_execute(
        &coordinator,
        &mut session,
        completed.clone(),
        &StageId::DECODE,
    )
    .await;
    expect_complete(&coordinator, &mut session, completed).await;
    assert_eq!(session.status(&StageId::PREFILL), StageStatus::Completed);
    assert!(matches!(
        coordinator
            .advance(
                &mut session,
                HostEvent::Completed {
                    stage: StageId::PREFILL,
                    attempt: prefill.attempt
                }
            )
            .await,
        Err(CoordinationError::Finished)
    ));
    drop((prefill, decode));
    assert_eq!(fixture.outstanding(), 0);
}

#[tokio::test]
async fn second_stage_admission_failure_releases_the_first_reservation() {
    let fixture = Fixture::new();
    fixture
        .decode
        .fail_next_admit(CoordinationError::AdmissionRejected {
            stage: StageId::DECODE,
            reason: "queue limit".to_string(),
        });
    let coordinator = fixture.coordinator(Topology::prefill_decode(), prefill_first());
    let error = coordinator.plan_all(request("fail-1")).await.unwrap_err();
    assert!(error.is_retryable(), "{error}");
    assert_eq!(fixture.outstanding(), 0);
    assert_eq!(fixture.prefill.release_count("fail-1/prefill/0"), 1);
}

#[tokio::test]
async fn admission_failure_in_a_progressive_session_aborts_it() {
    let fixture = Fixture::new();
    fixture
        .decode
        .fail_next_admit(CoordinationError::NoEligibleWorkers {
            stage: StageId::DECODE,
            reason: "none".to_string(),
        });
    let coordinator = fixture.coordinator(Topology::prefill_decode(), progressive_pd());
    let mut session = coordinator
        .start(request("fail-2"), PlanningMode::Progressive)
        .unwrap();
    let prefill = expect_execute(
        &coordinator,
        &mut session,
        HostEvent::Continue,
        &StageId::PREFILL,
    )
    .await;
    expect_wait(&coordinator, &mut session, dispatched(&prefill)).await;
    let error = coordinator
        .advance(&mut session, handoff(&prefill))
        .await
        .unwrap_err();
    assert!(matches!(error, CoordinationError::NoEligibleWorkers { .. }));
    assert!(session.is_closed());
    // Prefill is host-owned; the host releases it.
    drop(prefill);
    assert_eq!(fixture.outstanding(), 0);
}

#[tokio::test]
async fn stale_pool_generation_between_preview_and_admission_is_rejected() {
    let fixture = Fixture::new();
    fixture.decode.remove_worker(21);
    fixture.decode.set_signals(
        22,
        SelectionSignals {
            cached_tokens: 16,
            ..SelectionSignals::default()
        },
    );
    // The decode binding is rebuilt (new pool generation) with a selector that
    // rejects the old preview.
    let rebuilt = FakeStageSelector::with_workers(PoolRef::new("decode", 2), [22]);
    rebuilt.set_signals(
        22,
        SelectionSignals {
            cached_tokens: 16,
            ..SelectionSignals::default()
        },
    );
    let mut bindings = fixture.bindings();
    let previewing = fixture.decode.clone();
    struct SplitSelector {
        preview_from: Arc<FakeStageSelector>,
        admit_into: Arc<FakeStageSelector>,
    }
    #[async_trait]
    impl StageSelector for SplitSelector {
        async fn preview(&self, input: SelectionInput<'_>) -> Result<Preview, CoordinationError> {
            self.preview_from.preview(input).await
        }
        async fn admit(
            &self,
            input: SelectionInput<'_>,
            target: AdmissionTarget,
        ) -> Result<StageReservation, CoordinationError> {
            self.admit_into.admit(input, target).await
        }
    }
    for binding in &mut bindings {
        if binding.id == StageId::DECODE {
            binding.selector = Arc::new(SplitSelector {
                preview_from: previewing.clone(),
                admit_into: rebuilt.clone(),
            });
        }
    }
    let coordinator = RoutingCoordinator::new(
        Topology::conditional_prefill_decode(),
        bindings,
        conditional(true, ConditionalDisaggThresholds::default()),
    )
    .unwrap();
    let mut session = coordinator
        .start(request("stale-1"), PlanningMode::Progressive)
        .unwrap();
    let error = coordinator
        .advance(&mut session, HostEvent::Continue)
        .await
        .unwrap_err();
    assert!(
        matches!(error, CoordinationError::StalePreview { .. }),
        "{error}"
    );
    assert!(session.is_closed());
    assert_eq!(rebuilt.outstanding_reservations().len(), 0);
}

#[tokio::test(start_paused = true)]
async fn admission_timeout_releases_everything_the_session_holds() {
    let fixture = Fixture::new();
    fixture.decode.set_admission_open(false);
    let coordinator = fixture
        .coordinator(Topology::prefill_decode(), prefill_first())
        .with_limits(CoordinationLimits {
            admission_timeout: Duration::from_millis(50),
            ..CoordinationLimits::default()
        });
    let error = coordinator
        .plan_all(request("timeout-1"))
        .await
        .unwrap_err();
    assert!(
        matches!(error, CoordinationError::DeadlineExceeded { .. }),
        "{error}"
    );
    assert_eq!(fixture.outstanding(), 0);
    assert_eq!(fixture.prefill.release_count("timeout-1/prefill/0"), 1);
}

#[tokio::test(start_paused = true)]
async fn a_reservation_held_too_long_aborts_the_session() {
    let fixture = Fixture::new();
    let coordinator = fixture
        .coordinator(Topology::prefill_decode(), decode_first())
        .with_limits(CoordinationLimits {
            max_reservation_hold: Duration::from_secs(1),
            ..CoordinationLimits::default()
        });
    let mut session = coordinator
        .start(request("hold-1"), PlanningMode::Progressive)
        .unwrap();
    let prefill = expect_execute(
        &coordinator,
        &mut session,
        HostEvent::Continue,
        &StageId::PREFILL,
    )
    .await;
    assert_eq!(session.status(&StageId::DECODE), StageStatus::Admitted);
    tokio::time::advance(Duration::from_secs(2)).await;
    let error = coordinator
        .advance(&mut session, dispatched(&prefill))
        .await
        .unwrap_err();
    assert!(
        matches!(error, CoordinationError::DeadlineExceeded { .. }),
        "{error}"
    );
    assert_eq!(fixture.decode.outstanding_reservations().len(), 0);
    drop(prefill);
    assert_eq!(fixture.outstanding(), 0);
}

#[tokio::test]
async fn dropping_a_session_releases_what_it_owns() {
    let fixture = Fixture::new();
    let coordinator = fixture.coordinator(Topology::prefill_decode(), decode_first());
    let mut session = coordinator
        .start(request("drop-session"), PlanningMode::Progressive)
        .unwrap();
    let prefill = expect_execute(
        &coordinator,
        &mut session,
        HostEvent::Continue,
        &StageId::PREFILL,
    )
    .await;
    assert_eq!(fixture.decode.outstanding_reservations().len(), 1);
    drop(session);
    assert_eq!(fixture.decode.outstanding_reservations().len(), 0);
    drop(prefill);
    assert_eq!(fixture.outstanding(), 0);
}

#[tokio::test]
async fn retry_exhaustion_fails_the_stage_and_releases_dependents() {
    let fixture = Fixture::new();
    let coordinator = fixture
        .coordinator(Topology::prefill_decode(), decode_first())
        .with_limits(CoordinationLimits {
            max_attempts_per_stage: 2,
            ..CoordinationLimits::default()
        });
    let mut session = coordinator
        .start(request("exhaust-1"), PlanningMode::Progressive)
        .unwrap();
    let first = expect_execute(
        &coordinator,
        &mut session,
        HostEvent::Continue,
        &StageId::PREFILL,
    )
    .await;
    assert_eq!(session.status(&StageId::DECODE), StageStatus::Admitted);
    let first_decode_id = fixture.decode.outstanding_reservations()[0].clone();
    drop(first);
    // First failure: retry; the held decode is re-selected too.
    let second = expect_execute(
        &coordinator,
        &mut session,
        HostEvent::Failed {
            stage: StageId::PREFILL,
            attempt: AttemptId::FIRST,
            retry: true,
        },
        &StageId::PREFILL,
    )
    .await;
    assert_eq!(second.attempt, AttemptId::new(1));
    assert_eq!(fixture.decode.release_count(&first_decode_id), 1);
    assert_eq!(
        fixture.decode.outstanding_reservations(),
        vec!["exhaust-1/decode/1".to_string()]
    );
    drop(second);
    // Second failure exhausts the budget: the stage fails, decode is released.
    expect_complete(
        &coordinator,
        &mut session,
        HostEvent::Failed {
            stage: StageId::PREFILL,
            attempt: AttemptId::new(1),
            retry: true,
        },
    )
    .await;
    assert_eq!(session.status(&StageId::PREFILL), StageStatus::Failed);
    assert_eq!(fixture.outstanding(), 0);
}

#[tokio::test]
async fn non_retryable_failure_completes_routing_after_releasing_held_stages() {
    let fixture = Fixture::new();
    let coordinator = fixture.coordinator(Topology::prefill_decode(), decode_first());
    let mut session = coordinator
        .start(request("fatal-1"), PlanningMode::Progressive)
        .unwrap();
    let prefill = expect_execute(
        &coordinator,
        &mut session,
        HostEvent::Continue,
        &StageId::PREFILL,
    )
    .await;
    drop(prefill);
    expect_complete(
        &coordinator,
        &mut session,
        HostEvent::Failed {
            stage: StageId::PREFILL,
            attempt: AttemptId::FIRST,
            retry: false,
        },
    )
    .await;
    assert_eq!(fixture.outstanding(), 0);
}

#[tokio::test]
async fn upfront_planning_rejects_a_policy_that_waits() {
    struct WaitingPolicy;
    #[async_trait]
    impl CoordinationPolicy for WaitingPolicy {
        async fn next(
            &mut self,
            view: &CoordinationView<'_>,
        ) -> Result<CoordinationOp, CoordinationError> {
            if !view.is_admitted(&StageId::PREFILL) {
                return Ok(CoordinationOp::Admit(AdmissionIntent::new(
                    StageId::PREFILL,
                )));
            }
            Ok(CoordinationOp::Wait)
        }
        fn supports_upfront_planning(&self) -> bool {
            false
        }
    }
    let fixture = Fixture::new();
    let coordinator = fixture.coordinator(
        Topology::prefill_decode(),
        Arc::new(|_| Box::new(WaitingPolicy)),
    );
    let error = coordinator.plan_all(request("wait-1")).await.unwrap_err();
    assert!(
        matches!(error, CoordinationError::InvalidPolicyOperation(_)),
        "{error}"
    );
    assert_eq!(fixture.outstanding(), 0);
    assert!(
        fixture.prefill.events().is_empty(),
        "rejected before any selection"
    );

    // A policy that claims upfront support but waits anyway is caught after
    // its first admission, and that admission is released.
    struct LyingPolicy;
    #[async_trait]
    impl CoordinationPolicy for LyingPolicy {
        async fn next(
            &mut self,
            view: &CoordinationView<'_>,
        ) -> Result<CoordinationOp, CoordinationError> {
            if !view.is_admitted(&StageId::PREFILL) {
                return Ok(CoordinationOp::Admit(AdmissionIntent::new(
                    StageId::PREFILL,
                )));
            }
            Ok(CoordinationOp::Wait)
        }
    }
    let coordinator = fixture.coordinator(
        Topology::prefill_decode(),
        Arc::new(|_| Box::new(LyingPolicy)),
    );
    let error = coordinator.plan_all(request("wait-2")).await.unwrap_err();
    assert!(
        matches!(error, CoordinationError::InvalidPolicyOperation(_)),
        "{error}"
    );
    assert_eq!(fixture.outstanding(), 0);
    assert_eq!(fixture.prefill.release_count("wait-2/prefill/0"), 1);
}

#[tokio::test]
async fn finishing_with_an_unadmitted_stage_is_a_policy_error() {
    struct EagerFinish;
    #[async_trait]
    impl CoordinationPolicy for EagerFinish {
        async fn next(
            &mut self,
            view: &CoordinationView<'_>,
        ) -> Result<CoordinationOp, CoordinationError> {
            if !view.is_admitted(&StageId::PREFILL) {
                return Ok(CoordinationOp::Admit(AdmissionIntent::new(
                    StageId::PREFILL,
                )));
            }
            Ok(CoordinationOp::Finish)
        }
    }
    let fixture = Fixture::new();
    let coordinator = fixture.coordinator(
        Topology::prefill_decode(),
        Arc::new(|_| Box::new(EagerFinish)),
    );
    let error = coordinator.plan_all(request("finish-1")).await.unwrap_err();
    assert!(
        matches!(error, CoordinationError::InvalidPolicyOperation(_)),
        "{error}"
    );
    assert_eq!(fixture.outstanding(), 0);
}

#[tokio::test]
async fn unbound_topology_stages_are_rejected_at_construction() {
    let fixture = Fixture::new();
    let bindings: Vec<StageBinding> = fixture
        .bindings()
        .into_iter()
        .filter(|binding| binding.id != StageId::DECODE)
        .collect();
    let Err(error) = RoutingCoordinator::new(Topology::prefill_decode(), bindings, prefill_first())
    else {
        panic!("an unbound stage must be rejected");
    };
    assert!(matches!(error, CoordinationError::UnknownStage { stage } if stage == StageId::DECODE));
}

#[tokio::test]
async fn preview_is_refused_for_a_stage_without_preview_support() {
    struct PreviewEncode;
    #[async_trait]
    impl CoordinationPolicy for PreviewEncode {
        async fn next(
            &mut self,
            _view: &CoordinationView<'_>,
        ) -> Result<CoordinationOp, CoordinationError> {
            Ok(CoordinationOp::Preview(SelectionIntent::new(
                StageId::ENCODE,
            )))
        }
    }
    let fixture = Fixture::new();
    let coordinator = fixture.coordinator(
        Topology::single(StageId::ENCODE),
        Arc::new(|_| Box::new(PreviewEncode)),
    );
    let mut session = coordinator
        .start(request("enc-1"), PlanningMode::Progressive)
        .unwrap();
    let error = coordinator
        .advance(&mut session, HostEvent::Continue)
        .await
        .unwrap_err();
    assert!(matches!(
        error,
        CoordinationError::PreviewUnsupported { .. }
    ));
}

#[tokio::test]
async fn registry_built_policies_drive_the_coordinator() {
    let fixture = Fixture::new();
    let registry = CoordinationPolicyRegistry::with_builtins(&crate::config::KvRouterConfig {
        conditional_disagg_enabled: true,
        ..Default::default()
    });
    let coordinator = fixture.coordinator(
        Topology::conditional_prefill_decode(),
        registry.resolve(CONDITIONAL_DISAGGREGATION_POLICY).unwrap(),
    );
    // Nothing cached on decode: remote prefill.
    let mut session = coordinator
        .start(request("reg-1"), PlanningMode::Progressive)
        .unwrap();
    let prefill = expect_execute(
        &coordinator,
        &mut session,
        HostEvent::Continue,
        &StageId::PREFILL,
    )
    .await;
    assert_eq!(prefill.branch, REMOTE_PREFILL_DECODE_BRANCH);

    let coordinator = fixture.coordinator(
        Topology::aggregated(),
        registry.resolve(AGGREGATED_POLICY).unwrap(),
    );
    let plan = coordinator.plan_all(request("reg-2")).await.unwrap();
    assert_eq!(plan.stages.len(), 1);
}
