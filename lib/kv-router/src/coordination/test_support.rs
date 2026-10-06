// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! An in-memory [`StageSelector`] for coordinator tests in this crate and in
//! dependent crates (behind the `testing` feature).
//!
//! The fake selects deterministically (lowest projected decode load, then
//! lowest worker id), records every preview, admission, and release, and lets
//! a test inject failures or hold admissions open to exercise cancellation.

use std::any::Any;
use std::collections::VecDeque;
use std::sync::Arc;

use async_trait::async_trait;
use futures_util::future::BoxFuture;
use parking_lot::Mutex;
use tokio::sync::watch;

use crate::protocols::{WorkerId, WorkerWithDpRank};

use super::error::CoordinationError;
use super::ids::{AttemptId, StageId};
use super::selector::{
    AdmissionTarget, Preview, ReservationLease, ReservationOwner, SelectedTarget, SelectionInput,
    SelectionSignals, StageReservation, StageSelector, WorkerFacts,
};
use super::stage::PoolRef;

/// One worker the fake can select.
#[derive(Debug, Clone)]
pub struct FakeWorker {
    pub worker: WorkerWithDpRank,
    pub facts: Arc<WorkerFacts>,
    pub signals: SelectionSignals,
}

impl FakeWorker {
    pub fn new(worker_id: WorkerId) -> Self {
        Self {
            worker: WorkerWithDpRank::new(worker_id, 0),
            facts: Arc::new(WorkerFacts {
                data_parallel_size: 1,
                ..WorkerFacts::default()
            }),
            signals: SelectionSignals::default(),
        }
    }

    pub fn with_facts(mut self, facts: WorkerFacts) -> Self {
        self.facts = Arc::new(facts);
        self
    }

    pub fn with_signals(mut self, signals: SelectionSignals) -> Self {
        self.signals = signals;
        self
    }
}

/// What the fake observed, in order.
#[derive(Debug, Clone, PartialEq)]
pub enum FakeEvent {
    Preview {
        stage: StageId,
        attempt: AttemptId,
        worker: WorkerWithDpRank,
    },
    Admit {
        stage: StageId,
        attempt: AttemptId,
        worker: WorkerWithDpRank,
        reservation_id: String,
        from_preview: bool,
    },
    Release {
        reservation_id: String,
        explicit: bool,
    },
}

pub struct FakeStageSelector {
    pool: PoolRef,
    workers: Mutex<Vec<FakeWorker>>,
    events: Arc<Mutex<Vec<FakeEvent>>>,
    fail_preview: Mutex<VecDeque<CoordinationError>>,
    fail_admit: Mutex<VecDeque<CoordinationError>>,
    admission_open: watch::Sender<bool>,
}

impl FakeStageSelector {
    pub fn new(pool: PoolRef) -> Arc<Self> {
        let (admission_open, _) = watch::channel(true);
        Arc::new(Self {
            pool,
            workers: Mutex::new(Vec::new()),
            events: Arc::new(Mutex::new(Vec::new())),
            fail_preview: Mutex::new(VecDeque::new()),
            fail_admit: Mutex::new(VecDeque::new()),
            admission_open,
        })
    }

    /// A selector over `pool` with one single-rank worker per id.
    pub fn with_workers(
        pool: PoolRef,
        worker_ids: impl IntoIterator<Item = WorkerId>,
    ) -> Arc<Self> {
        let selector = Self::new(pool);
        for worker_id in worker_ids {
            selector.add_worker(FakeWorker::new(worker_id));
        }
        selector
    }

    pub fn pool(&self) -> &PoolRef {
        &self.pool
    }

    pub fn add_worker(&self, worker: FakeWorker) {
        self.workers.lock().push(worker);
    }

    pub fn remove_worker(&self, worker_id: WorkerId) {
        self.workers
            .lock()
            .retain(|worker| worker.worker.worker_id != worker_id);
    }

    pub fn set_signals(&self, worker_id: WorkerId, signals: SelectionSignals) {
        for worker in self.workers.lock().iter_mut() {
            if worker.worker.worker_id == worker_id {
                worker.signals = signals;
            }
        }
    }

    /// The next preview fails with `error` instead of selecting.
    pub fn fail_next_preview(&self, error: CoordinationError) {
        self.fail_preview.lock().push_back(error);
    }

    /// The next admission fails with `error` instead of reserving.
    pub fn fail_next_admit(&self, error: CoordinationError) {
        self.fail_admit.lock().push_back(error);
    }

    /// While closed, admissions wait; dropping a waiting admission future is
    /// a cancellation before any reservation exists.
    pub fn set_admission_open(&self, open: bool) {
        self.admission_open.send_replace(open);
    }

    pub fn events(&self) -> Vec<FakeEvent> {
        self.events.lock().clone()
    }

    pub fn clear_events(&self) {
        self.events.lock().clear();
    }

    /// Reservation ids admitted and not yet released.
    pub fn outstanding_reservations(&self) -> Vec<String> {
        let events = self.events.lock();
        let mut outstanding = Vec::new();
        for event in events.iter() {
            match event {
                FakeEvent::Admit { reservation_id, .. } => {
                    outstanding.push(reservation_id.clone());
                }
                FakeEvent::Release { reservation_id, .. } => {
                    outstanding.retain(|id| id != reservation_id);
                }
                FakeEvent::Preview { .. } => {}
            }
        }
        outstanding
    }

    pub fn release_count(&self, reservation_id: &str) -> usize {
        self.events
            .lock()
            .iter()
            .filter(|event| {
                matches!(event, FakeEvent::Release { reservation_id: id, .. } if id == reservation_id)
            })
            .count()
    }

    fn choose(
        &self,
        input: &SelectionInput<'_>,
        pinned: Option<WorkerWithDpRank>,
    ) -> Result<FakeWorker, CoordinationError> {
        let restrictions = input.restrictions;
        let workers = self.workers.lock();
        let mut candidates: Vec<&FakeWorker> = workers
            .iter()
            .filter(|candidate| restrictions.permits(candidate.worker.worker_id))
            .filter(|candidate| {
                restrictions
                    .routing_constraints
                    .is_compatible_with_worker_taints(&candidate.facts.taints)
            })
            .filter(|candidate| pinned.is_none_or(|pinned| candidate.worker == pinned))
            .collect();
        if candidates.is_empty() {
            return Err(CoordinationError::NoEligibleWorkers {
                stage: input.stage.clone(),
                reason: match pinned {
                    Some(pinned) => format!(
                        "pinned worker {} dp_rank {} is unavailable",
                        pinned.worker_id, pinned.dp_rank
                    ),
                    None => "no fake worker matches the restrictions".to_string(),
                },
            });
        }
        if let Some(affinity) = restrictions.affinity_target
            && let Some(preferred) = candidates.iter().find(|candidate| {
                candidate.worker.worker_id == affinity.worker_id
                    && affinity
                        .dp_rank
                        .is_none_or(|dp_rank| candidate.worker.dp_rank == dp_rank)
            })
        {
            return Ok((*preferred).clone());
        }
        candidates.sort_by_key(|candidate| {
            (
                candidate.signals.potential_decode_blocks,
                candidate.worker.worker_id,
                candidate.worker.dp_rank,
            )
        });
        Ok(candidates[0].clone())
    }

    fn target(&self, input: &SelectionInput<'_>, worker: &FakeWorker) -> SelectedTarget {
        SelectedTarget {
            invocation: input.invocation,
            attempt: input.attempt,
            stage: input.stage.clone(),
            pool: self.pool.clone(),
            worker: worker.worker,
            facts: Arc::clone(&worker.facts),
        }
    }
}

#[async_trait]
impl StageSelector for FakeStageSelector {
    async fn preview(&self, input: SelectionInput<'_>) -> Result<Preview, CoordinationError> {
        if let Some(error) = self.fail_preview.lock().pop_front() {
            return Err(error);
        }
        let worker = self.choose(&input, input.restrictions.pinned_worker)?;
        self.events.lock().push(FakeEvent::Preview {
            stage: input.stage.clone(),
            attempt: input.attempt,
            worker: worker.worker,
        });
        Ok(Preview {
            target: self.target(&input, &worker),
            signals: worker.signals,
        })
    }

    async fn admit(
        &self,
        input: SelectionInput<'_>,
        target: AdmissionTarget,
    ) -> Result<StageReservation, CoordinationError> {
        let mut open = self.admission_open.subscribe();
        while !*open.borrow_and_update() {
            open.changed()
                .await
                .map_err(|_| CoordinationError::Finished)?;
        }
        if let Some(error) = self.fail_admit.lock().pop_front() {
            return Err(error);
        }
        let (pinned, from_preview) = match &target {
            AdmissionTarget::AnyEligible => (input.restrictions.pinned_worker, false),
            AdmissionTarget::FromPreview(preview) => {
                if preview.target.pool != self.pool {
                    return Err(CoordinationError::StalePreview {
                        stage: input.stage.clone(),
                        reason: format!(
                            "preview selected from pool {} but the stage now draws from {}",
                            preview.target.pool, self.pool
                        ),
                    });
                }
                if preview.target.stage != *input.stage {
                    return Err(CoordinationError::StalePreview {
                        stage: input.stage.clone(),
                        reason: format!("preview belongs to stage {}", preview.target.stage),
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
                (Some(preview.target.worker), true)
            }
        };
        let worker = self.choose(&input, pinned)?;
        let reservation_id = input.reservation_id();
        self.events.lock().push(FakeEvent::Admit {
            stage: input.stage.clone(),
            attempt: input.attempt,
            worker: worker.worker,
            reservation_id: reservation_id.clone(),
            from_preview,
        });
        let lease = ReservationLease::new(FakeLease {
            reservation_id,
            events: Arc::clone(&self.events),
            released: false,
        });
        Ok(StageReservation::new(
            self.target(&input, &worker),
            worker.signals,
            lease,
        ))
    }
}

/// The fake's reservation owner; records whether release was explicit or by drop.
pub struct FakeLease {
    reservation_id: String,
    events: Arc<Mutex<Vec<FakeEvent>>>,
    released: bool,
}

impl FakeLease {
    pub fn reservation_id(&self) -> &str {
        &self.reservation_id
    }
}

impl ReservationOwner for FakeLease {
    fn release(mut self: Box<Self>) -> BoxFuture<'static, Result<(), CoordinationError>> {
        self.released = true;
        self.events.lock().push(FakeEvent::Release {
            reservation_id: self.reservation_id.clone(),
            explicit: true,
        });
        Box::pin(async { Ok(()) })
    }

    fn into_any(self: Box<Self>) -> Box<dyn Any + Send> {
        self
    }

    fn describe(&self) -> String {
        format!("fake lease {}", self.reservation_id)
    }
}

impl Drop for FakeLease {
    fn drop(&mut self) {
        if !self.released {
            self.events.lock().push(FakeEvent::Release {
                reservation_id: self.reservation_id.clone(),
                explicit: false,
            });
        }
    }
}
