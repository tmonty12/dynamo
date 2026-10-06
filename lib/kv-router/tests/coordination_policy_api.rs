// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A custom coordination policy and a host-side stage selector written
//! against the public `dynamo_kv_router::coordination` API only. This is the
//! compile-checked version of the example in the stage-coordination docs.

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use futures_util::future::BoxFuture;

use dynamo_kv_router::WorkerType;
use dynamo_kv_router::config::KvRouterConfig;
use dynamo_kv_router::coordination::{
    AGGREGATED_POLICY, AdmissionIntent, AdmissionTarget, CONDITIONAL_DISAGGREGATION_POLICY,
    CoordinationError, CoordinationOp, CoordinationPolicy, CoordinationPolicyRegistry,
    CoordinationView, DECODE_FIRST_POLICY, ENCODE_PREFILL_DECODE_POLICY, PREFILL_FIRST_POLICY,
    PROGRESSIVE_PREFILL_DECODE_POLICY, PoolRef, Preview, ProfileName, PromptInput,
    ReservationLease, ReservationOwner, RoutingCoordinator, RoutingRequest, SelectedTarget,
    SelectionInput, SelectionSignals, StageBinding, StageId, StageReservation, StageSelector,
    Topology, WorkerFacts,
};
use dynamo_kv_router::protocols::WorkerWithDpRank;

/// Selects decode before prefill. Execution order still comes from the
/// topology: prefill runs first and decode waits for its handoff.
struct DecodeFirstThenPrefill;

#[async_trait]
impl CoordinationPolicy for DecodeFirstThenPrefill {
    async fn next(
        &mut self,
        view: &CoordinationView<'_>,
    ) -> Result<CoordinationOp, CoordinationError> {
        if !view.is_admitted(&StageId::DECODE) {
            return Ok(CoordinationOp::Admit(
                AdmissionIntent::new(StageId::DECODE).with_profile(ProfileName::DECODE_ONLY),
            ));
        }
        if !view.is_admitted(&StageId::PREFILL) {
            return Ok(CoordinationOp::Admit(AdmissionIntent::new(
                StageId::PREFILL,
            )));
        }
        Ok(CoordinationOp::Finish)
    }
}

/// A host selector over a fixed worker list. Admissions are counted so the
/// test can prove every reservation is released exactly once.
struct StaticPool {
    pool: PoolRef,
    workers: Vec<u64>,
    admitted: Arc<Mutex<Vec<StageId>>>,
    released: Arc<AtomicUsize>,
}

struct CountedLease(Arc<AtomicUsize>, bool);

impl ReservationOwner for CountedLease {
    fn release(mut self: Box<Self>) -> BoxFuture<'static, Result<(), CoordinationError>> {
        self.1 = true;
        self.0.fetch_add(1, Ordering::SeqCst);
        Box::pin(async { Ok(()) })
    }

    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
        self
    }
}

impl Drop for CountedLease {
    fn drop(&mut self) {
        if !self.1 {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }
}

impl StaticPool {
    fn target(&self, input: &SelectionInput<'_>, worker_id: u64) -> SelectedTarget {
        SelectedTarget {
            invocation: input.invocation,
            attempt: input.attempt,
            stage: input.stage.clone(),
            pool: self.pool.clone(),
            worker: WorkerWithDpRank::new(worker_id, 0),
            facts: Arc::new(WorkerFacts::default()),
        }
    }

    fn first_permitted(&self, input: &SelectionInput<'_>) -> Result<u64, CoordinationError> {
        self.workers
            .iter()
            .copied()
            .find(|worker_id| input.restrictions.permits(*worker_id))
            .ok_or_else(|| CoordinationError::NoEligibleWorkers {
                stage: input.stage.clone(),
                reason: "static pool exhausted".to_string(),
            })
    }
}

#[async_trait]
impl StageSelector for StaticPool {
    async fn preview(&self, input: SelectionInput<'_>) -> Result<Preview, CoordinationError> {
        let worker_id = self.first_permitted(&input)?;
        Ok(Preview {
            target: self.target(&input, worker_id),
            signals: SelectionSignals::default(),
        })
    }

    async fn admit(
        &self,
        input: SelectionInput<'_>,
        target: AdmissionTarget,
    ) -> Result<StageReservation, CoordinationError> {
        let worker_id = match target {
            AdmissionTarget::AnyEligible => self.first_permitted(&input)?,
            AdmissionTarget::FromPreview(preview) => preview.target.worker.worker_id,
        };
        self.admitted.lock().unwrap().push(input.stage.clone());
        Ok(StageReservation::new(
            self.target(&input, worker_id),
            SelectionSignals::default(),
            ReservationLease::new(CountedLease(Arc::clone(&self.released), false)),
        ))
    }
}

#[tokio::test]
async fn custom_policy_registers_and_plans_in_dependency_order() {
    let mut registry = CoordinationPolicyRegistry::with_builtins(&KvRouterConfig::default());
    for name in [
        AGGREGATED_POLICY,
        PREFILL_FIRST_POLICY,
        DECODE_FIRST_POLICY,
        PROGRESSIVE_PREFILL_DECODE_POLICY,
        CONDITIONAL_DISAGGREGATION_POLICY,
        ENCODE_PREFILL_DECODE_POLICY,
    ] {
        assert!(registry.resolve(name).is_ok(), "built-in {name} resolves");
    }
    registry
        .register(
            "decode_first_then_prefill",
            Arc::new(|_facts| Box::new(DecodeFirstThenPrefill)),
        )
        .unwrap();
    let factory = registry.resolve("decode_first_then_prefill").unwrap();

    let admitted = Arc::new(Mutex::new(Vec::new()));
    let released = Arc::new(AtomicUsize::new(0));
    let pool = |name: &'static str, workers: Vec<u64>| -> Arc<dyn StageSelector> {
        Arc::new(StaticPool {
            pool: PoolRef::new(name, 1),
            workers,
            admitted: Arc::clone(&admitted),
            released: Arc::clone(&released),
        })
    };
    let coordinator = RoutingCoordinator::new(
        Topology::prefill_decode(),
        [
            StageBinding::new(
                StageId::PREFILL,
                PoolRef::new("prefill", 1),
                WorkerType::Prefill,
                pool("prefill", vec![11, 12]),
            ),
            StageBinding::new(
                StageId::DECODE,
                PoolRef::new("decode", 1),
                WorkerType::Decode,
                pool("decode", vec![21]),
            ),
        ],
        factory,
    )
    .unwrap();

    let plan = coordinator
        .plan_all(RoutingRequest::new(
            "request-1",
            PromptInput::from_tokens(vec![1, 2, 3, 4]),
        ))
        .await
        .unwrap();

    // The policy admitted decode first...
    assert_eq!(
        *admitted.lock().unwrap(),
        vec![StageId::DECODE, StageId::PREFILL]
    );
    // ...but the plan runs prefill first and decode depends on it.
    let stages: Vec<&StageId> = plan.stages.iter().map(|stage| &stage.stage).collect();
    assert_eq!(stages, vec![&StageId::PREFILL, &StageId::DECODE]);
    assert_eq!(
        plan.stage(&StageId::DECODE).unwrap().depends_on,
        vec![StageId::PREFILL]
    );
    assert_eq!(
        plan.stage(&StageId::PREFILL)
            .unwrap()
            .target
            .worker
            .worker_id,
        11
    );
    assert_eq!(released.load(Ordering::SeqCst), 0);

    plan.release_all().await.unwrap();
    assert_eq!(released.load(Ordering::SeqCst), 2);
}

#[tokio::test]
async fn registry_rejects_duplicate_and_unknown_names() {
    let mut registry = CoordinationPolicyRegistry::new();
    registry
        .register("custom", Arc::new(|_| Box::new(DecodeFirstThenPrefill)))
        .unwrap();
    assert!(
        registry
            .register("custom", Arc::new(|_| Box::new(DecodeFirstThenPrefill)))
            .is_err()
    );
    assert!(registry.resolve("missing").is_err());
    let names: HashMap<&str, ()> = registry.names().map(|name| (name, ())).collect();
    assert!(names.contains_key("custom"));
}
