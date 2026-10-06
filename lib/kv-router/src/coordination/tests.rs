// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{HashMap, HashSet};

use crate::WorkerType;
use crate::protocols::{RoutingConstraints, WorkerAffinityTarget, WorkerWithDpRank};
use crate::scheduling::config::RouterConfigOverride;

use super::test_support::{FakeEvent, FakeLease, FakeStageSelector, FakeWorker};
use super::*;

fn restrictions_with_allowlist(ids: &[u64]) -> SelectionRestrictions {
    SelectionRestrictions {
        allowed_worker_ids: Some(ids.iter().copied().collect()),
        ..SelectionRestrictions::default()
    }
}

fn prompt() -> PromptInputView<'static> {
    const TOKENS: [u32; 8] = [1, 2, 3, 4, 5, 6, 7, 8];
    PromptInputView {
        token_ids: &TOKENS,
        block_mm_infos: None,
        lora_name: None,
        cache_namespace: None,
    }
}

fn input<'a>(
    stage: &'a StageId,
    attempt: AttemptId,
    profile: &'a StageProfile,
    restrictions: &'a SelectionRestrictions,
    settings: &'a RequestSettings,
) -> SelectionInput<'a> {
    SelectionInput {
        request_id: "req-1",
        stage,
        invocation: InvocationId::new(1),
        attempt,
        prompt: prompt(),
        profile,
        restrictions,
        settings,
    }
}

#[test]
fn restrictions_merge_intersects_allowlists_and_unions_constraints() {
    let mut left = restrictions_with_allowlist(&[1, 2, 3]);
    left.routing_constraints.required_taints.insert("a".into());
    left.routing_constraints
        .preferred_taints
        .insert("zone=1".into(), 0.5);
    left.excluded_worker_ids.insert(9);

    let mut right = restrictions_with_allowlist(&[2, 3, 4]);
    right.routing_constraints.required_taints.insert("b".into());
    right
        .routing_constraints
        .preferred_taints
        .insert("zone=1".into(), 0.25);
    right
        .routing_constraints
        .preferred_taints
        .insert("rack=7".into(), 1.0);
    right.excluded_worker_ids.insert(8);

    let merged = left.merge(&right, &StageId::DECODE).unwrap();
    assert_eq!(merged.allowed_worker_ids, Some(HashSet::from([2, 3])));
    assert_eq!(merged.excluded_worker_ids, HashSet::from([8, 9]));
    assert_eq!(
        merged.routing_constraints.required_taints,
        HashSet::from(["a".to_string(), "b".to_string()])
    );
    assert_eq!(merged.routing_constraints.preferred_taints["zone=1"], 0.75);
    assert_eq!(merged.routing_constraints.preferred_taints["rack=7"], 1.0);
    assert!(merged.permits(2));
    assert!(!merged.permits(1));
    assert!(!merged.permits(9));
}

#[test]
fn restrictions_merge_keeps_an_agreed_pin_and_rejects_a_conflict() {
    let pinned = SelectionRestrictions {
        pinned_worker: Some(WorkerWithDpRank::new(3, 1)),
        ..SelectionRestrictions::default()
    };
    let unpinned = SelectionRestrictions::default();
    let merged = pinned.merge(&unpinned, &StageId::DECODE).unwrap();
    assert_eq!(merged.pinned_worker, Some(WorkerWithDpRank::new(3, 1)));

    let other_pin = SelectionRestrictions {
        pinned_worker: Some(WorkerWithDpRank::new(4, 0)),
        ..SelectionRestrictions::default()
    };
    let error = pinned.merge(&other_pin, &StageId::DECODE).unwrap_err();
    assert!(matches!(
        error,
        CoordinationError::ConflictingRestrictions { .. }
    ));
}

#[test]
fn restrictions_merge_rejects_a_pin_the_other_side_forbids() {
    let pinned = SelectionRestrictions {
        pinned_worker: Some(WorkerWithDpRank::new(3, 0)),
        ..SelectionRestrictions::default()
    };
    let excluded = SelectionRestrictions {
        excluded_worker_ids: HashSet::from([3]),
        ..SelectionRestrictions::default()
    };
    assert!(matches!(
        pinned.merge(&excluded, &StageId::PREFILL).unwrap_err(),
        CoordinationError::ConflictingRestrictions { .. }
    ));

    let allowlist = restrictions_with_allowlist(&[1, 2]);
    assert!(matches!(
        pinned.merge(&allowlist, &StageId::PREFILL).unwrap_err(),
        CoordinationError::ConflictingRestrictions { .. }
    ));
}

#[test]
fn restrictions_merge_rejects_conflicting_affinity_targets() {
    let left = SelectionRestrictions {
        affinity_target: Some(WorkerAffinityTarget::new(1, None)),
        ..SelectionRestrictions::default()
    };
    let right = SelectionRestrictions {
        affinity_target: Some(WorkerAffinityTarget::new(2, None)),
        ..SelectionRestrictions::default()
    };
    assert!(matches!(
        left.merge(&right, &StageId::DECODE).unwrap_err(),
        CoordinationError::ConflictingRestrictions { .. }
    ));
    let merged = left.merge(&left, &StageId::DECODE).unwrap();
    assert_eq!(merged.affinity_target, left.affinity_target);
}

#[test]
fn effective_allowlist_applies_exclusions_to_the_universe() {
    let restrictions = SelectionRestrictions {
        excluded_worker_ids: HashSet::from([2]),
        ..SelectionRestrictions::default()
    };
    let universe = || HashSet::from([1, 2, 3]);
    assert_eq!(
        restrictions.effective_allowed_worker_ids(universe),
        Some(HashSet::from([1, 3]))
    );

    let allowlisted = SelectionRestrictions {
        allowed_worker_ids: Some(HashSet::from([2, 3])),
        excluded_worker_ids: HashSet::from([2]),
        ..SelectionRestrictions::default()
    };
    assert_eq!(
        allowlisted.effective_allowed_worker_ids(|| unreachable!("allowlist present")),
        Some(HashSet::from([3]))
    );

    let unrestricted = SelectionRestrictions::default();
    assert_eq!(
        unrestricted.effective_allowed_worker_ids(|| unreachable!("no exclusions")),
        None
    );
}

#[test]
fn decode_only_profile_disables_prompt_accounting_and_overlap_credit() {
    let profile = StageProfile::new(
        ProfileName::DECODE_ONLY,
        WorkAccounting::DecodeOnly,
        ScoringMode::LoadOnly,
    );
    let caller = RouterConfigOverride {
        overlap_score_credit: Some(0.5),
        router_temperature: Some(0.7),
        ..Default::default()
    };
    let resolved = profile
        .resolve_router_config_override(Some(&caller))
        .unwrap();
    assert_eq!(resolved.overlap_score_credit, Some(0.0));
    assert_eq!(resolved.assume_kv_reuse, Some(false));
    assert_eq!(resolved.track_prefill_tokens, Some(false));
    assert_eq!(resolved.router_temperature, Some(0.7));
    assert_eq!(profile.resolve_expected_output_tokens(Some(64)), Some(64));
}

#[test]
fn cache_aware_full_accounting_profile_passes_the_caller_through() {
    let profile = StageProfile::new(
        ProfileName::LOCAL_PREFILL_DECODE,
        WorkAccounting::PrefillAndDecode,
        ScoringMode::CacheAware,
    );
    assert!(profile.resolve_router_config_override(None).is_none());
    let caller = RouterConfigOverride {
        overlap_score_credit: Some(0.25),
        ..Default::default()
    };
    let resolved = profile
        .resolve_router_config_override(Some(&caller))
        .unwrap();
    assert_eq!(resolved.overlap_score_credit, Some(0.25));
    assert_eq!(resolved.assume_kv_reuse, None);
    assert_eq!(resolved.track_prefill_tokens, None);
}

#[test]
fn profile_override_wins_over_the_caller_but_not_the_profile_rules() {
    let profile = StageProfile::new(
        ProfileName::DEFAULT,
        WorkAccounting::DecodeOnly,
        ScoringMode::CacheAware,
    )
    .with_router_config_override(RouterConfigOverride {
        router_temperature: Some(0.1),
        track_prefill_tokens: Some(true),
        ..Default::default()
    });
    let caller = RouterConfigOverride {
        router_temperature: Some(0.9),
        prefill_load_scale: Some(2.0),
        ..Default::default()
    };
    let resolved = profile
        .resolve_router_config_override(Some(&caller))
        .unwrap();
    assert_eq!(resolved.router_temperature, Some(0.1));
    assert_eq!(resolved.prefill_load_scale, Some(2.0));
    assert_eq!(resolved.track_prefill_tokens, Some(false));
    assert_eq!(resolved.overlap_score_credit, None);
}

#[test]
fn prefill_only_profile_projects_one_output_token() {
    let profile = StageProfile::new(
        ProfileName::DEFAULT,
        WorkAccounting::PrefillOnly,
        ScoringMode::CacheAware,
    );
    assert_eq!(profile.resolve_expected_output_tokens(Some(512)), Some(1));
    assert!(profile.resolve_router_config_override(None).is_none());

    let untracked = StageProfile::new(
        ProfileName::DEFAULT,
        WorkAccounting::None,
        ScoringMode::LoadOnly,
    );
    assert_eq!(untracked.resolve_expected_output_tokens(Some(512)), None);
    let resolved = untracked.resolve_router_config_override(None).unwrap();
    assert_eq!(resolved.track_prefill_tokens, Some(false));
    assert_eq!(resolved.overlap_score_credit, Some(0.0));
}

#[test]
fn worker_type_defaults_follow_the_stage_table() {
    let decode = StageProfiles::for_worker_type(WorkerType::Decode);
    assert_eq!(decode.default_name(), &ProfileName::DECODE_ONLY);
    assert_eq!(decode.default_profile().work, WorkAccounting::DecodeOnly);
    assert_eq!(decode.default_profile().scoring, ScoringMode::LoadOnly);
    let local = decode.get(&ProfileName::LOCAL_PREFILL_DECODE).unwrap();
    assert_eq!(local.work, WorkAccounting::PrefillAndDecode);
    assert_eq!(local.scoring, ScoringMode::CacheAware);

    let prefill = StageProfiles::for_worker_type(WorkerType::Prefill);
    assert_eq!(prefill.default_profile().work, WorkAccounting::PrefillOnly);
    assert_eq!(prefill.names().count(), 1);

    let aggregated = StageProfiles::for_worker_type(WorkerType::Aggregated);
    assert_eq!(
        aggregated.default_profile().work,
        WorkAccounting::PrefillAndDecode
    );

    let encode = StageProfiles::for_worker_type(WorkerType::Encode);
    assert_eq!(encode.default_profile().work, WorkAccounting::None);

    let capabilities = StageCapabilities::for_worker_type(WorkerType::Prefill);
    assert!(capabilities.produces_handoff && !capabilities.consumes_handoff);
    let capabilities = StageCapabilities::for_worker_type(WorkerType::Decode);
    assert!(capabilities.consumes_handoff && !capabilities.produces_handoff);
    assert!(!StageCapabilities::for_worker_type(WorkerType::Encode).supports_preview);
}

#[test]
fn binding_resolves_named_and_default_profiles() {
    let selector = FakeStageSelector::with_workers(PoolRef::new("decode", 1), [1]);
    let binding = StageBinding::new(
        StageId::DECODE,
        PoolRef::new("decode", 1),
        WorkerType::Decode,
        selector,
    );
    assert_eq!(
        binding.profile(None).unwrap().name,
        ProfileName::DECODE_ONLY
    );
    assert_eq!(
        binding
            .profile(Some(&ProfileName::LOCAL_PREFILL_DECODE))
            .unwrap()
            .work,
        WorkAccounting::PrefillAndDecode
    );
    let missing = ProfileName::new("missing");
    assert!(matches!(
        binding.profile(Some(&missing)).unwrap_err(),
        CoordinationError::UnknownProfile { stage, profile }
            if stage == StageId::DECODE && profile == missing
    ));
    assert!(format!("{binding:?}").contains("decode"));
}

#[tokio::test]
async fn dropping_a_reservation_releases_its_lease_exactly_once() {
    let pool = PoolRef::new("agg", 1);
    let selector = FakeStageSelector::with_workers(pool, [1, 2]);
    let profile = StageProfiles::for_worker_type(WorkerType::Aggregated)
        .default_profile()
        .clone();
    let restrictions = SelectionRestrictions::default();
    let settings = RequestSettings::default();
    let stage = StageId::AGGREGATED;

    let reservation = selector
        .admit(
            input(&stage, AttemptId::FIRST, &profile, &restrictions, &settings),
            AdmissionTarget::AnyEligible,
        )
        .await
        .unwrap();
    let reservation_id = "req-1/aggregated/0".to_string();
    assert_eq!(
        selector.outstanding_reservations(),
        vec![reservation_id.clone()]
    );
    assert!(reservation.lease().is_tracked());

    drop(reservation);
    assert!(selector.outstanding_reservations().is_empty());
    assert_eq!(selector.release_count(&reservation_id), 1);
    assert!(matches!(
        selector.events().last(),
        Some(FakeEvent::Release {
            explicit: false,
            ..
        })
    ));
}

#[tokio::test]
async fn explicit_release_is_awaited_and_not_repeated_on_drop() {
    let pool = PoolRef::new("agg", 1);
    let selector = FakeStageSelector::with_workers(pool, [1]);
    let profile = StageProfiles::for_worker_type(WorkerType::Aggregated)
        .default_profile()
        .clone();
    let restrictions = SelectionRestrictions::default();
    let settings = RequestSettings::default();
    let stage = StageId::AGGREGATED;

    let reservation = selector
        .admit(
            input(
                &stage,
                AttemptId::new(2),
                &profile,
                &restrictions,
                &settings,
            ),
            AdmissionTarget::AnyEligible,
        )
        .await
        .unwrap();
    reservation.release().await.unwrap();
    assert_eq!(selector.release_count("req-1/aggregated/2"), 1);
    assert!(matches!(
        selector.events().last(),
        Some(FakeEvent::Release { explicit: true, .. })
    ));
}

#[tokio::test]
async fn lease_downcasts_to_its_owner_and_survives_a_miss() {
    let pool = PoolRef::new("agg", 1);
    let selector = FakeStageSelector::with_workers(pool, [1]);
    let profile = StageProfiles::for_worker_type(WorkerType::Aggregated)
        .default_profile()
        .clone();
    let restrictions = SelectionRestrictions::default();
    let settings = RequestSettings::default();
    let stage = StageId::AGGREGATED;

    struct Other;
    impl ReservationOwner for Other {
        fn release(
            self: Box<Self>,
        ) -> futures_util::future::BoxFuture<'static, Result<(), CoordinationError>> {
            Box::pin(async { Ok(()) })
        }
        fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
            self
        }
    }

    let reservation = selector
        .admit(
            input(&stage, AttemptId::FIRST, &profile, &restrictions, &settings),
            AdmissionTarget::AnyEligible,
        )
        .await
        .unwrap();
    let (_, _, lease) = reservation.into_parts();
    // A miss hands the lease back intact and still releases on drop.
    let Err(lease) = lease.into_owner::<Other>() else {
        panic!("a fake lease must not downcast to another owner type");
    };
    assert!(lease.is_tracked());
    assert!(selector.outstanding_reservations().len() == 1);
    // After a miss the owner is opaque; dropping it still releases.
    drop(lease);
    assert!(selector.outstanding_reservations().is_empty());

    let reservation = selector
        .admit(
            input(
                &stage,
                AttemptId::new(1),
                &profile,
                &restrictions,
                &settings,
            ),
            AdmissionTarget::AnyEligible,
        )
        .await
        .unwrap();
    let (_, _, lease) = reservation.into_parts();
    let owner = lease.into_owner::<FakeLease>().ok().unwrap();
    assert_eq!(owner.reservation_id(), "req-1/aggregated/1");
    drop(owner);
    assert!(selector.outstanding_reservations().is_empty());

    assert!(!ReservationLease::untracked().is_tracked());
    ReservationLease::untracked().release().await.unwrap();
}

#[tokio::test]
async fn admission_from_preview_pins_the_previewed_worker() {
    let pool = PoolRef::new("decode", 3);
    let selector = FakeStageSelector::with_workers(pool, [1, 2]);
    // Worker 2 is the lighter one, so a fresh selection would pick it.
    selector.set_signals(
        1,
        SelectionSignals {
            potential_decode_blocks: 10,
            ..SelectionSignals::default()
        },
    );
    let profiles = StageProfiles::for_worker_type(WorkerType::Decode);
    let profile = profiles.get(&ProfileName::LOCAL_PREFILL_DECODE).unwrap();
    let settings = RequestSettings::default();
    let stage = StageId::DECODE;

    let pinned_to_one = SelectionRestrictions {
        pinned_worker: Some(WorkerWithDpRank::new(1, 0)),
        ..SelectionRestrictions::default()
    };
    let preview = selector
        .preview(input(
            &stage,
            AttemptId::FIRST,
            profile,
            &pinned_to_one,
            &settings,
        ))
        .await
        .unwrap();
    assert_eq!(preview.target.worker, WorkerWithDpRank::new(1, 0));
    assert_eq!(preview.signals.potential_decode_blocks, 10);

    let unrestricted = SelectionRestrictions::default();
    let reservation = selector
        .admit(
            input(&stage, AttemptId::FIRST, profile, &unrestricted, &settings),
            AdmissionTarget::FromPreview(preview),
        )
        .await
        .unwrap();
    assert_eq!(reservation.target.worker, WorkerWithDpRank::new(1, 0));
    assert!(matches!(
        selector.events().last(),
        Some(FakeEvent::Admit {
            from_preview: true,
            ..
        })
    ));
    reservation.release().await.unwrap();
}

#[tokio::test]
async fn admission_from_preview_rejects_stale_pool_or_forbidden_worker() {
    let old_pool = PoolRef::new("decode", 1);
    let selector = FakeStageSelector::with_workers(old_pool, [1]);
    let profiles = StageProfiles::for_worker_type(WorkerType::Decode);
    let profile = profiles.default_profile();
    let settings = RequestSettings::default();
    let restrictions = SelectionRestrictions::default();
    let stage = StageId::DECODE;

    let preview = selector
        .preview(input(
            &stage,
            AttemptId::FIRST,
            profile,
            &restrictions,
            &settings,
        ))
        .await
        .unwrap();

    // Discovery rebuilt the binding: a new pool generation serves the stage.
    let rebuilt = FakeStageSelector::with_workers(PoolRef::new("decode", 2), [1]);
    let error = rebuilt
        .admit(
            input(&stage, AttemptId::FIRST, profile, &restrictions, &settings),
            AdmissionTarget::FromPreview(preview.clone()),
        )
        .await
        .unwrap_err();
    assert!(
        matches!(error, CoordinationError::StalePreview { .. }),
        "{error}"
    );
    assert!(rebuilt.outstanding_reservations().is_empty());

    // The previewed worker failed an earlier attempt.
    let excluded = SelectionRestrictions {
        excluded_worker_ids: HashSet::from([1]),
        ..SelectionRestrictions::default()
    };
    let error = selector
        .admit(
            input(&stage, AttemptId::new(1), profile, &excluded, &settings),
            AdmissionTarget::FromPreview(preview.clone()),
        )
        .await
        .unwrap_err();
    assert!(
        matches!(error, CoordinationError::StalePreview { .. }),
        "{error}"
    );

    // The preview belongs to another stage.
    let error = selector
        .admit(
            input(
                &StageId::PREFILL,
                AttemptId::FIRST,
                profile,
                &restrictions,
                &settings,
            ),
            AdmissionTarget::FromPreview(preview),
        )
        .await
        .unwrap_err();
    assert!(
        matches!(error, CoordinationError::StalePreview { .. }),
        "{error}"
    );
    assert!(selector.outstanding_reservations().is_empty());
}

#[tokio::test]
async fn fake_selector_honours_taints_exclusions_and_affinity() {
    let pool = PoolRef::new("prefill", 1);
    let selector = FakeStageSelector::new(pool);
    selector.add_worker(FakeWorker::new(1).with_facts(WorkerFacts {
        taints: HashSet::from(["zone=a".to_string()]),
        topology_domains: HashMap::from([("zone".to_string(), "a".to_string())]),
        ..WorkerFacts::default()
    }));
    selector.add_worker(FakeWorker::new(2).with_facts(WorkerFacts {
        taints: HashSet::from(["zone=b".to_string()]),
        ..WorkerFacts::default()
    }));
    let profiles = StageProfiles::for_worker_type(WorkerType::Prefill);
    let profile = profiles.default_profile();
    let settings = RequestSettings::default();
    let stage = StageId::PREFILL;

    let zone_b = SelectionRestrictions {
        routing_constraints: RoutingConstraints {
            required_taints: HashSet::from(["zone=b".to_string()]),
            preferred_taints: HashMap::new(),
        },
        ..SelectionRestrictions::default()
    };
    let preview = selector
        .preview(input(&stage, AttemptId::FIRST, profile, &zone_b, &settings))
        .await
        .unwrap();
    assert_eq!(preview.target.worker.worker_id, 2);
    assert_eq!(preview.target.facts.topology_value("zone"), None);

    let affinity = SelectionRestrictions {
        affinity_target: Some(WorkerAffinityTarget::new(2, None)),
        ..SelectionRestrictions::default()
    };
    let preview = selector
        .preview(input(
            &stage,
            AttemptId::FIRST,
            profile,
            &affinity,
            &settings,
        ))
        .await
        .unwrap();
    assert_eq!(preview.target.worker.worker_id, 2);

    let nothing_left = SelectionRestrictions {
        excluded_worker_ids: HashSet::from([1, 2]),
        ..SelectionRestrictions::default()
    };
    let error = selector
        .preview(input(
            &stage,
            AttemptId::FIRST,
            profile,
            &nothing_left,
            &settings,
        ))
        .await
        .unwrap_err();
    assert!(matches!(error, CoordinationError::NoEligibleWorkers { .. }));
    assert!(!error.is_retryable());
}

#[tokio::test]
async fn cancelled_admission_leaves_no_reservation_behind() {
    let pool = PoolRef::new("agg", 1);
    let selector = FakeStageSelector::with_workers(pool, [1]);
    let profile = StageProfiles::for_worker_type(WorkerType::Aggregated)
        .default_profile()
        .clone();
    let restrictions = SelectionRestrictions::default();
    let settings = RequestSettings::default();
    let stage = StageId::AGGREGATED;

    selector.set_admission_open(false);
    {
        let admission = selector.admit(
            input(&stage, AttemptId::FIRST, &profile, &restrictions, &settings),
            AdmissionTarget::AnyEligible,
        );
        tokio::pin!(admission);
        assert!(
            futures_util::poll!(admission.as_mut()).is_pending(),
            "admission must wait while the gate is closed"
        );
        // The pinned future drops at the end of this block: a cancellation
        // before any reservation exists.
    }
    assert!(selector.events().is_empty());

    selector.set_admission_open(true);
    let reservation = selector
        .admit(
            input(&stage, AttemptId::FIRST, &profile, &restrictions, &settings),
            AdmissionTarget::AnyEligible,
        )
        .await
        .unwrap();
    assert_eq!(selector.outstanding_reservations().len(), 1);
    drop(reservation);
}

#[test]
fn worker_facts_capture_config_metadata() {
    use crate::test_utils::SimpleWorkerConfig;

    let config = SimpleWorkerConfig {
        data_parallel_start_rank: 2,
        data_parallel_size: 4,
        total_kv_blocks: Some(1000),
        taints: HashSet::from(["gpu=h100".to_string()]),
        ..Default::default()
    };
    let facts = WorkerFacts::from_config(&config);
    assert_eq!(facts.data_parallel_start_rank, 2);
    assert_eq!(facts.data_parallel_size, 4);
    assert_eq!(facts.total_kv_blocks, Some(1000));
    assert!(facts.taints.contains("gpu=h100"));
    assert!(facts.topology_domains.is_empty());
    assert_eq!(facts.kv_transfer_domain, None);
}

#[test]
fn signals_report_load_thresholds_only_when_known() {
    let signals = SelectionSignals {
        potential_decode_blocks: 90,
        total_kv_blocks: Some(100),
        prefill_load: Some(PrefillLoadSignal {
            active_prefill_tokens: 900,
            prefill_token_capacity: 1000,
        }),
        ..SelectionSignals::default()
    };
    assert_eq!(signals.decode_load_exceeds(0.8), Some(true));
    assert_eq!(signals.decode_load_exceeds(0.95), Some(false));
    assert_eq!(signals.prefill_load_exceeds(0.5), Some(true));
    assert_eq!(signals.prefill_load_exceeds(0.9), Some(false));

    let unknown = SelectionSignals::default();
    assert_eq!(unknown.decode_load_exceeds(0.5), None);
    assert_eq!(unknown.prefill_load_exceeds(0.5), None);
}

#[test]
fn selection_input_derives_a_unique_reservation_id_per_attempt() {
    let profile = StageProfiles::for_worker_type(WorkerType::Decode)
        .default_profile()
        .clone();
    let restrictions = SelectionRestrictions::default();
    let settings = RequestSettings {
        expected_output_tokens: Some(32),
        ..RequestSettings::default()
    };
    let stage = StageId::DECODE;
    let first = input(&stage, AttemptId::FIRST, &profile, &restrictions, &settings);
    let retry = input(
        &stage,
        AttemptId::new(1),
        &profile,
        &restrictions,
        &settings,
    );
    assert_eq!(first.reservation_id(), "req-1/decode/0");
    assert_eq!(retry.reservation_id(), "req-1/decode/1");
    assert_eq!(first.expected_output_tokens(), Some(32));
    let config_override = first.router_config_override().unwrap();
    assert_eq!(config_override.track_prefill_tokens, Some(false));
}

#[cfg(feature = "standalone-selection")]
mod core_selector {
    use std::sync::Arc;
    use std::time::Duration;

    use tokio_util::sync::CancellationToken;

    use crate::identity::RoutingPartitionId;
    use crate::services::selection::{SelectionCacheConfig, SelectionCore, WorkerRequest};

    use super::*;

    fn core() -> Arc<SelectionCore> {
        let config = crate::config::KvRouterConfig {
            use_kv_events: false,
            router_queue_threshold: None,
            ..Default::default()
        };
        Arc::new(
            SelectionCore::try_new_local(
                config,
                1,
                CancellationToken::new(),
                SelectionCacheConfig::default(),
                Arc::new(|config, role, _| {
                    crate::WorkerSelectionPolicy::reference(
                        config.clone(),
                        role.default_selector_label(),
                    )
                }),
            )
            .expect("valid test config"),
        )
    }

    fn worker(worker_id: u64) -> WorkerRequest {
        WorkerRequest {
            worker_id,
            model_name: "model".to_string(),
            routing_group: "default".to_string(),
            endpoint: Some(format!("http://worker-{worker_id}:8000")),
            block_size: Some(4),
            max_num_batched_tokens: Some(1024),
            total_kv_blocks: Some(500),
            taints: HashSet::from([format!("worker={worker_id}")]),
            topology_domains: HashMap::from([("zone".to_string(), format!("z{worker_id}"))]),
            ..WorkerRequest::default()
        }
    }

    async fn core_with_workers(ids: &[u64]) -> Arc<SelectionCore> {
        let core = core();
        for id in ids {
            core.upsert_worker(worker(*id)).await.unwrap();
        }
        core
    }

    fn partition() -> RoutingPartitionId {
        RoutingPartitionId::new("model", "default")
    }

    fn prefill_tokens(core: &SelectionCore) -> usize {
        core.loads(Some("model"), Some("default"))
            .into_iter()
            .flat_map(|model| model.loads)
            .map(|load| load.potential_prefill_tokens)
            .sum()
    }

    fn active_requests(core: &SelectionCore) -> usize {
        core.loads(Some("model"), Some("default"))
            .into_iter()
            .flat_map(|model| model.loads)
            .map(|load| load.active_requests)
            .sum()
    }

    async fn wait_until(what: &str, mut condition: impl FnMut() -> bool) {
        tokio::time::timeout(Duration::from_secs(2), async {
            while !condition() {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap_or_else(|_| panic!("timed out waiting for {what}"));
    }

    #[tokio::test]
    async fn preview_is_advisory_and_books_nothing() {
        let core = core_with_workers(&[1, 2]).await;
        let selector = CoreStageSelector::new(
            Arc::clone(&core),
            partition(),
            PoolRef::new("prefill", 1),
            CoreAdmissionMode::Lease,
        );
        let profiles = StageProfiles::for_worker_type(WorkerType::Prefill);
        let profile = profiles.default_profile();
        let restrictions = SelectionRestrictions::default();
        let settings = RequestSettings::default();
        let stage = StageId::PREFILL;

        let preview = selector
            .preview(input(
                &stage,
                AttemptId::FIRST,
                profile,
                &restrictions,
                &settings,
            ))
            .await
            .unwrap();
        assert!(preview.signals.prefill_load.is_some());
        assert_eq!(preview.signals.total_kv_blocks, Some(500));
        assert!(
            preview
                .target
                .facts
                .taints
                .contains(&format!("worker={}", preview.target.worker.worker_id))
        );
        assert_eq!(
            preview.target.facts.topology_value("zone"),
            Some(format!("z{}", preview.target.worker.worker_id).as_str())
        );
        assert_eq!(active_requests(&core), 0);
        assert_eq!(prefill_tokens(&core), 0);
    }

    #[tokio::test]
    async fn lease_admission_books_and_release_frees() {
        let core = core_with_workers(&[1, 2]).await;
        let selector = CoreStageSelector::new(
            Arc::clone(&core),
            partition(),
            PoolRef::new("prefill", 1),
            CoreAdmissionMode::Lease,
        );
        let profiles = StageProfiles::for_worker_type(WorkerType::Prefill);
        let profile = profiles.default_profile();
        let restrictions = SelectionRestrictions::default();
        let settings = RequestSettings::default();
        let stage = StageId::PREFILL;

        let reservation = selector
            .admit(
                input(&stage, AttemptId::FIRST, profile, &restrictions, &settings),
                AdmissionTarget::AnyEligible,
            )
            .await
            .unwrap();
        assert!(reservation.lease().is_tracked());
        assert_eq!(active_requests(&core), 1);
        assert!(prefill_tokens(&core) > 0, "prefill-only work is accounted");

        reservation.release().await.unwrap();
        wait_until("lease release", || active_requests(&core) == 0).await;
        assert_eq!(prefill_tokens(&core), 0);
    }

    #[tokio::test]
    async fn dropped_lease_frees_its_booking() {
        let core = core_with_workers(&[1]).await;
        let selector = CoreStageSelector::new(
            Arc::clone(&core),
            partition(),
            PoolRef::new("agg", 1),
            CoreAdmissionMode::Lease,
        );
        let profiles = StageProfiles::for_worker_type(WorkerType::Aggregated);
        let profile = profiles.default_profile();
        let restrictions = SelectionRestrictions::default();
        let settings = RequestSettings::default();
        let stage = StageId::AGGREGATED;

        let reservation = selector
            .admit(
                input(&stage, AttemptId::FIRST, profile, &restrictions, &settings),
                AdmissionTarget::AnyEligible,
            )
            .await
            .unwrap();
        assert_eq!(active_requests(&core), 1);
        drop(reservation);
        wait_until("dropped lease release", || active_requests(&core) == 0).await;
    }

    #[tokio::test]
    async fn book_admission_is_addressable_by_id_until_released() {
        let core = core_with_workers(&[1, 2]).await;
        let selector = CoreStageSelector::new(
            Arc::clone(&core),
            partition(),
            PoolRef::new("decode", 1),
            CoreAdmissionMode::Book,
        );
        let profiles = StageProfiles::for_worker_type(WorkerType::Decode);
        let profile = profiles.default_profile();
        let restrictions = SelectionRestrictions::default();
        let settings = RequestSettings {
            expected_output_tokens: Some(16),
            ..RequestSettings::default()
        };
        let stage = StageId::DECODE;

        let reservation = selector
            .admit(
                input(&stage, AttemptId::FIRST, profile, &restrictions, &settings),
                AdmissionTarget::AnyEligible,
            )
            .await
            .unwrap();
        assert_eq!(active_requests(&core), 1);
        assert_eq!(
            prefill_tokens(&core),
            0,
            "decode-only work accounts no prompt load"
        );

        let (_, _, lease) = reservation.into_parts();
        let booking = lease.into_owner::<CoreBooking>().ok().unwrap();
        assert_eq!(booking.selection_id(), "req-1/decode/0");
        // Lifecycle calls resolve the same booking by id.
        core.prefill_complete("req-1/decode/0").await.unwrap();
        booking.prefill_complete().await.unwrap();
        booking.add_output_block(None).unwrap();

        // Dropping the owner frees it through the runtime.
        drop(booking);
        wait_until("book release", || active_requests(&core) == 0).await;
        assert!(core.free_reservation("req-1/decode/0").await.is_err());
    }

    #[tokio::test]
    async fn book_admission_explicit_release_is_idempotent_with_the_index() {
        let core = core_with_workers(&[1]).await;
        let selector = CoreStageSelector::new(
            Arc::clone(&core),
            partition(),
            PoolRef::new("agg", 1),
            CoreAdmissionMode::Book,
        );
        let profiles = StageProfiles::for_worker_type(WorkerType::Aggregated);
        let profile = profiles.default_profile();
        let restrictions = SelectionRestrictions::default();
        let settings = RequestSettings::default();
        let stage = StageId::AGGREGATED;

        let reservation = selector
            .admit(
                input(&stage, AttemptId::FIRST, profile, &restrictions, &settings),
                AdmissionTarget::AnyEligible,
            )
            .await
            .unwrap();
        // The host freed it by id first (for example from a response observer).
        core.free_reservation("req-1/aggregated/0").await.unwrap();
        reservation.release().await.unwrap();
        assert_eq!(active_requests(&core), 0);
    }

    #[tokio::test]
    async fn admission_from_preview_reserves_the_previewed_worker() {
        let core = core_with_workers(&[1, 2, 3]).await;
        let selector = CoreStageSelector::new(
            Arc::clone(&core),
            partition(),
            PoolRef::new("decode", 1),
            CoreAdmissionMode::Lease,
        );
        let profiles = StageProfiles::for_worker_type(WorkerType::Decode);
        let profile = profiles.get(&ProfileName::LOCAL_PREFILL_DECODE).unwrap();
        let settings = RequestSettings::default();
        let stage = StageId::DECODE;

        // Steer the preview onto worker 2 so it differs from a fresh choice.
        let only_two = restrictions_with_allowlist(&[2]);
        let preview = selector
            .preview(input(
                &stage,
                AttemptId::FIRST,
                profile,
                &only_two,
                &settings,
            ))
            .await
            .unwrap();
        assert_eq!(preview.target.worker.worker_id, 2);

        let unrestricted = SelectionRestrictions::default();
        let reservation = selector
            .admit(
                input(&stage, AttemptId::FIRST, profile, &unrestricted, &settings),
                AdmissionTarget::FromPreview(preview.clone()),
            )
            .await
            .unwrap();
        assert_eq!(reservation.target.worker, preview.target.worker);
        assert_eq!(active_requests(&core), 1);
        reservation.release().await.unwrap();
        wait_until("release", || active_requests(&core) == 0).await;

        // A stale generation is rejected before any scheduler call.
        let rebuilt = CoreStageSelector::new(
            Arc::clone(&core),
            partition(),
            PoolRef::new("decode", 2),
            CoreAdmissionMode::Lease,
        );
        let error = rebuilt
            .admit(
                input(&stage, AttemptId::FIRST, profile, &unrestricted, &settings),
                AdmissionTarget::FromPreview(preview),
            )
            .await
            .unwrap_err();
        assert!(matches!(error, CoordinationError::StalePreview { .. }));
        assert_eq!(active_requests(&core), 0);
    }

    #[tokio::test]
    async fn exclusions_narrow_the_candidate_set() {
        let core = core_with_workers(&[1, 2]).await;
        let selector = CoreStageSelector::new(
            Arc::clone(&core),
            partition(),
            PoolRef::new("prefill", 1),
            CoreAdmissionMode::Lease,
        );
        let profiles = StageProfiles::for_worker_type(WorkerType::Prefill);
        let profile = profiles.default_profile();
        let settings = RequestSettings::default();
        let stage = StageId::PREFILL;

        let excluded = SelectionRestrictions {
            excluded_worker_ids: HashSet::from([1]),
            ..SelectionRestrictions::default()
        };
        let preview = selector
            .preview(input(
                &stage,
                AttemptId::new(1),
                profile,
                &excluded,
                &settings,
            ))
            .await
            .unwrap();
        assert_eq!(preview.target.worker.worker_id, 2);

        let everything = SelectionRestrictions {
            excluded_worker_ids: HashSet::from([1, 2]),
            ..SelectionRestrictions::default()
        };
        let error = selector
            .admit(
                input(&stage, AttemptId::new(1), profile, &everything, &settings),
                AdmissionTarget::AnyEligible,
            )
            .await
            .unwrap_err();
        assert!(
            matches!(error, CoordinationError::NoEligibleWorkers { .. }),
            "{error}"
        );
        assert_eq!(active_requests(&core), 0);
    }

    #[tokio::test]
    async fn no_schedulable_workers_is_not_an_internal_error() {
        let core = core();
        let selector = CoreStageSelector::new(
            Arc::clone(&core),
            partition(),
            PoolRef::new("prefill", 1),
            CoreAdmissionMode::Lease,
        );
        let profiles = StageProfiles::for_worker_type(WorkerType::Prefill);
        let profile = profiles.default_profile();
        let settings = RequestSettings::default();
        let restrictions = SelectionRestrictions::default();
        let stage = StageId::PREFILL;

        let error = selector
            .preview(input(
                &stage,
                AttemptId::FIRST,
                profile,
                &restrictions,
                &settings,
            ))
            .await
            .unwrap_err();
        assert!(
            matches!(error, CoordinationError::NoEligibleWorkers { .. }),
            "{error}"
        );
    }
}
