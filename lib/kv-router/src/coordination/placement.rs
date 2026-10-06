// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Cross-stage placement: derive the restrictions a stage inherits from the
//! workers already selected, and validate selected pairs after the fact, in
//! either selection order.

use crate::protocols::{KvTransferEnforcement, RoutingConstraints};

use super::error::CoordinationError;
use super::ids::StageId;
use super::selector::{SelectedTarget, SelectionRestrictions, WorkerFacts};
use super::topology::{PlacementMode, PlacementRule};

/// Prefix of the canonical taint a worker publishes for each topology domain.
///
/// Matches the frontend's `dynamo.topology/<domain>=<value>` taints so that
/// constraints derived here select against the same worker metadata.
pub const TOPOLOGY_TAINT_PREFIX: &str = "dynamo.topology/";

/// The canonical taint for one topology domain value.
pub fn topology_taint(domain: &str, value: &str) -> String {
    format!("{TOPOLOGY_TAINT_PREFIX}{domain}={value}")
}

/// The constraint one already-selected worker imposes on another stage.
enum DerivedConstraint {
    Required(String),
    Preferred(String, f32),
    None,
}

fn same_domain_constraint(
    selected: &SelectedTarget,
    stage: &StageId,
    domain: &str,
    mode: PlacementMode,
) -> Result<DerivedConstraint, CoordinationError> {
    let Some(value) = selected.facts.topology_value(domain) else {
        return match mode {
            PlacementMode::Required => Err(CoordinationError::Placement {
                first: selected.stage.clone(),
                second: stage.clone(),
                reason: format!(
                    "worker {} publishes no topology domain {domain:?}",
                    selected.worker.worker_id
                ),
            }),
            PlacementMode::Preferred { .. } => Ok(DerivedConstraint::None),
        };
    };
    let taint = topology_taint(domain, value);
    Ok(match mode {
        PlacementMode::Required => DerivedConstraint::Required(taint),
        PlacementMode::Preferred { weight } => DerivedConstraint::Preferred(taint, weight),
    })
}

/// The transfer constraint a worker publishes for its peers: the same
/// semantics as the frontend's KV-transfer topology constraints.
fn transfer_constraint(
    selected: &SelectedTarget,
    stage: &StageId,
) -> Result<DerivedConstraint, CoordinationError> {
    let facts: &WorkerFacts = &selected.facts;
    let Some(domain) = facts.kv_transfer_domain.as_deref() else {
        return Ok(DerivedConstraint::None);
    };
    let placement = |reason: String| CoordinationError::Placement {
        first: selected.stage.clone(),
        second: stage.clone(),
        reason,
    };
    let value = facts.topology_value(domain).ok_or_else(|| {
        placement(format!(
            "worker {} configured kv_transfer_domain={domain:?}, but topology_domains does not contain that domain",
            selected.worker.worker_id
        ))
    })?;
    let taint = topology_taint(domain, value);
    match facts.kv_transfer_enforcement {
        Some(KvTransferEnforcement::Required) => Ok(DerivedConstraint::Required(taint)),
        Some(KvTransferEnforcement::Preferred) => {
            let weight = facts.kv_transfer_preferred_weight.ok_or_else(|| {
                placement(format!(
                    "worker {} configured preferred KV transfer enforcement for domain {domain:?}, but kv_transfer_preferred_weight is missing",
                    selected.worker.worker_id
                ))
            })?;
            Ok(DerivedConstraint::Preferred(taint, weight))
        }
        None => Err(placement(format!(
            "worker {} configured kv_transfer_domain={domain:?}, but kv_transfer_enforcement is missing",
            selected.worker.worker_id
        ))),
    }
}

/// The restrictions `stage` inherits from `selected` under `rules`.
///
/// Required constraints union; preferred weights add. Returns empty
/// restrictions when no rule applies.
pub fn derived_restrictions<'a>(
    rules: &[PlacementRule],
    selected: impl IntoIterator<Item = &'a SelectedTarget>,
    stage: &StageId,
) -> Result<SelectionRestrictions, CoordinationError> {
    let mut constraints = RoutingConstraints::default();
    for target in selected {
        if target.stage == *stage {
            continue;
        }
        for rule in rules {
            let derived = match rule {
                PlacementRule::SameDomain { domain, mode } => {
                    same_domain_constraint(target, stage, domain, *mode)?
                }
                PlacementRule::TransferCompatible => transfer_constraint(target, stage)?,
            };
            match derived {
                DerivedConstraint::Required(taint) => {
                    constraints.required_taints.insert(taint);
                }
                DerivedConstraint::Preferred(taint, weight) => {
                    *constraints.preferred_taints.entry(taint).or_insert(0.0) += weight;
                }
                DerivedConstraint::None => {}
            }
        }
    }
    Ok(SelectionRestrictions {
        routing_constraints: constraints,
        ..SelectionRestrictions::default()
    })
}

/// Check that two selected workers satisfy every required rule in both
/// directions, so a pair chosen in either order is validated the same way.
pub fn validate_pair(
    rules: &[PlacementRule],
    first: &SelectedTarget,
    second: &SelectedTarget,
) -> Result<(), CoordinationError> {
    for (source, peer) in [(first, second), (second, first)] {
        for rule in rules {
            let derived = match rule {
                PlacementRule::SameDomain { domain, mode } => {
                    same_domain_constraint(source, &peer.stage, domain, *mode)?
                }
                PlacementRule::TransferCompatible => transfer_constraint(source, &peer.stage)?,
            };
            if let DerivedConstraint::Required(taint) = derived
                && !peer.facts.taints.contains(&taint)
            {
                return Err(CoordinationError::Placement {
                    first: source.stage.clone(),
                    second: peer.stage.clone(),
                    reason: format!(
                        "worker {} requires peers tainted {taint:?}, but worker {} is not",
                        source.worker.worker_id, peer.worker.worker_id
                    ),
                });
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::collections::{HashMap, HashSet};
    use std::sync::Arc;

    use crate::protocols::WorkerWithDpRank;

    use super::super::ids::{AttemptId, InvocationId};
    use super::super::stage::PoolRef;
    use super::*;

    fn target(stage: StageId, worker_id: u64, facts: WorkerFacts) -> SelectedTarget {
        SelectedTarget {
            invocation: InvocationId::new(1),
            attempt: AttemptId::FIRST,
            stage,
            pool: PoolRef::new("pool", 1),
            worker: WorkerWithDpRank::new(worker_id, 0),
            facts: Arc::new(facts),
        }
    }

    fn zoned(zone: &str) -> WorkerFacts {
        WorkerFacts {
            taints: HashSet::from([topology_taint("zone", zone)]),
            topology_domains: HashMap::from([("zone".to_string(), zone.to_string())]),
            ..WorkerFacts::default()
        }
    }

    fn transfer(
        zone: &str,
        enforcement: Option<KvTransferEnforcement>,
        weight: Option<f32>,
    ) -> WorkerFacts {
        WorkerFacts {
            kv_transfer_domain: Some("zone".to_string()),
            kv_transfer_enforcement: enforcement,
            kv_transfer_preferred_weight: weight,
            ..zoned(zone)
        }
    }

    #[test]
    fn same_domain_rules_derive_required_and_preferred_taints() {
        let prefill = target(StageId::PREFILL, 1, zoned("a"));
        let required = [PlacementRule::SameDomain {
            domain: "zone".to_string(),
            mode: PlacementMode::Required,
        }];
        let derived = derived_restrictions(&required, [&prefill], &StageId::DECODE).unwrap();
        assert_eq!(
            derived.routing_constraints.required_taints,
            HashSet::from([topology_taint("zone", "a")])
        );

        let preferred = [PlacementRule::SameDomain {
            domain: "zone".to_string(),
            mode: PlacementMode::Preferred { weight: 0.5 },
        }];
        let derived = derived_restrictions(&preferred, [&prefill], &StageId::DECODE).unwrap();
        assert_eq!(
            derived.routing_constraints.preferred_taints[&topology_taint("zone", "a")],
            0.5
        );

        // The stage's own selection never constrains itself.
        let derived = derived_restrictions(&required, [&prefill], &StageId::PREFILL).unwrap();
        assert!(derived.routing_constraints.is_empty());
    }

    #[test]
    fn required_same_domain_needs_the_domain_but_preferred_tolerates_its_absence() {
        let unzoned = target(StageId::PREFILL, 1, WorkerFacts::default());
        let required = [PlacementRule::SameDomain {
            domain: "zone".to_string(),
            mode: PlacementMode::Required,
        }];
        assert!(matches!(
            derived_restrictions(&required, [&unzoned], &StageId::DECODE).unwrap_err(),
            CoordinationError::Placement { .. }
        ));
        let preferred = [PlacementRule::SameDomain {
            domain: "zone".to_string(),
            mode: PlacementMode::Preferred { weight: 1.0 },
        }];
        let derived = derived_restrictions(&preferred, [&unzoned], &StageId::DECODE).unwrap();
        assert!(derived.routing_constraints.is_empty());
    }

    #[test]
    fn transfer_compatible_follows_the_selected_workers_metadata() {
        let rules = [PlacementRule::TransferCompatible];
        let required = target(
            StageId::PREFILL,
            1,
            transfer("a", Some(KvTransferEnforcement::Required), None),
        );
        let derived = derived_restrictions(&rules, [&required], &StageId::DECODE).unwrap();
        assert_eq!(
            derived.routing_constraints.required_taints,
            HashSet::from([topology_taint("zone", "a")])
        );

        let preferred = target(
            StageId::PREFILL,
            2,
            transfer("b", Some(KvTransferEnforcement::Preferred), Some(0.85)),
        );
        let derived = derived_restrictions(&rules, [&preferred], &StageId::DECODE).unwrap();
        assert_eq!(
            derived.routing_constraints.preferred_taints[&topology_taint("zone", "b")],
            0.85
        );

        let no_domain = target(StageId::PREFILL, 3, zoned("c"));
        let derived = derived_restrictions(&rules, [&no_domain], &StageId::DECODE).unwrap();
        assert!(derived.routing_constraints.is_empty());

        let missing_enforcement = target(StageId::PREFILL, 4, transfer("d", None, None));
        assert!(matches!(
            derived_restrictions(&rules, [&missing_enforcement], &StageId::DECODE).unwrap_err(),
            CoordinationError::Placement { .. }
        ));
        let missing_weight = target(
            StageId::PREFILL,
            5,
            transfer("e", Some(KvTransferEnforcement::Preferred), None),
        );
        assert!(matches!(
            derived_restrictions(&rules, [&missing_weight], &StageId::DECODE).unwrap_err(),
            CoordinationError::Placement { .. }
        ));
        let missing_value = target(
            StageId::PREFILL,
            6,
            WorkerFacts {
                kv_transfer_domain: Some("zone".to_string()),
                kv_transfer_enforcement: Some(KvTransferEnforcement::Required),
                ..WorkerFacts::default()
            },
        );
        assert!(matches!(
            derived_restrictions(&rules, [&missing_value], &StageId::DECODE).unwrap_err(),
            CoordinationError::Placement { .. }
        ));
    }

    #[test]
    fn constraints_from_several_selections_combine() {
        let encode = target(StageId::ENCODE, 1, zoned("a"));
        let prefill = target(
            StageId::PREFILL,
            2,
            transfer("a", Some(KvTransferEnforcement::Preferred), Some(0.25)),
        );
        let rules = [
            PlacementRule::SameDomain {
                domain: "zone".to_string(),
                mode: PlacementMode::Preferred { weight: 0.5 },
            },
            PlacementRule::TransferCompatible,
        ];
        let derived = derived_restrictions(&rules, [&encode, &prefill], &StageId::DECODE).unwrap();
        // 0.5 from encode's zone, 0.5 from prefill's zone, 0.25 from transfer.
        assert_eq!(
            derived.routing_constraints.preferred_taints[&topology_taint("zone", "a")],
            1.25
        );
        assert!(derived.routing_constraints.required_taints.is_empty());
    }

    #[test]
    fn pair_validation_checks_both_directions() {
        let rules = [PlacementRule::TransferCompatible];
        let prefill = target(
            StageId::PREFILL,
            1,
            transfer("a", Some(KvTransferEnforcement::Required), None),
        );
        let decode_same = target(StageId::DECODE, 2, zoned("a"));
        let decode_other = target(StageId::DECODE, 3, zoned("b"));
        validate_pair(&rules, &prefill, &decode_same).unwrap();
        validate_pair(&rules, &decode_same, &prefill).unwrap();
        assert!(matches!(
            validate_pair(&rules, &prefill, &decode_other).unwrap_err(),
            CoordinationError::Placement { first, second, .. }
                if first == StageId::PREFILL && second == StageId::DECODE
        ));
        assert!(matches!(
            validate_pair(&rules, &decode_other, &prefill).unwrap_err(),
            CoordinationError::Placement { first, second, .. }
                if first == StageId::PREFILL && second == StageId::DECODE
        ));

        // A decode-side requirement constrains prefill too (decode-first order).
        let strict_decode = target(
            StageId::DECODE,
            4,
            transfer("b", Some(KvTransferEnforcement::Required), None),
        );
        let loose_prefill = target(StageId::PREFILL, 5, zoned("a"));
        assert!(matches!(
            validate_pair(&rules, &loose_prefill, &strict_decode).unwrap_err(),
            CoordinationError::Placement { first, second, .. }
                if first == StageId::DECODE && second == StageId::PREFILL
        ));

        let same_zone = [PlacementRule::SameDomain {
            domain: "zone".to_string(),
            mode: PlacementMode::Required,
        }];
        validate_pair(&same_zone, &loose_prefill, &decode_same).unwrap();
        assert!(validate_pair(&same_zone, &loose_prefill, &decode_other).is_err());
    }
}
