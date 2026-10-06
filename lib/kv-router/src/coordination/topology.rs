// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Execution topology: the branches a request may follow, the handoff
//! dependencies between stages on each branch, and the placement rules that
//! connect stages.

use super::error::CoordinationError;
use super::ids::{BranchId, StageId};

/// The branch a single-path topology declares.
pub const DEFAULT_BRANCH: BranchId = BranchId::from_static("default");
/// Conditional disaggregation: decode runs prefill locally.
pub const LOCAL_PREFILL_DECODE_BRANCH: BranchId = BranchId::from_static("local_prefill_decode");
/// Conditional disaggregation: a prefill worker hands off to decode.
pub const REMOTE_PREFILL_DECODE_BRANCH: BranchId = BranchId::from_static("remote_prefill_decode");
/// Encode, prefill, and decode for a request with multimodal input.
pub const ENCODE_PREFILL_DECODE_BRANCH: BranchId = BranchId::from_static("encode_prefill_decode");
/// Prefill and decode for a request without multimodal input.
pub const PREFILL_DECODE_BRANCH: BranchId = BranchId::from_static("prefill_decode");

/// Data or transfer information one stage produces and a later stage needs
/// before it can run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Handoff {
    pub from: StageId,
    pub to: StageId,
}

/// One execution path through the topology.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Branch {
    pub id: BranchId,
    /// Stages in execution order.
    pub path: Vec<StageId>,
    pub handoffs: Vec<Handoff>,
}

impl Branch {
    /// A branch where every stage hands off to the next one on the path.
    pub fn linear(id: impl Into<BranchId>, path: Vec<StageId>) -> Self {
        let handoffs = path
            .windows(2)
            .map(|pair| Handoff {
                from: pair[0].clone(),
                to: pair[1].clone(),
            })
            .collect();
        Self {
            id: id.into(),
            path,
            handoffs,
        }
    }

    /// A branch with explicit handoffs; stages without one may run as soon as
    /// they are admitted.
    pub fn new(id: impl Into<BranchId>, path: Vec<StageId>, handoffs: Vec<Handoff>) -> Self {
        Self {
            id: id.into(),
            path,
            handoffs,
        }
    }

    pub fn contains(&self, stage: &StageId) -> bool {
        self.path.contains(stage)
    }

    /// Stages whose handoff `stage` waits for.
    pub fn inputs_of<'a>(&'a self, stage: &'a StageId) -> impl Iterator<Item = &'a StageId> + 'a {
        self.handoffs
            .iter()
            .filter(move |handoff| handoff.to == *stage)
            .map(|handoff| &handoff.from)
    }

    /// Stages that wait for `stage`'s handoff.
    pub fn dependents_of<'a>(
        &'a self,
        stage: &'a StageId,
    ) -> impl Iterator<Item = &'a StageId> + 'a {
        self.handoffs
            .iter()
            .filter(move |handoff| handoff.from == *stage)
            .map(|handoff| &handoff.to)
    }

    fn validate(&self) -> Result<(), CoordinationError> {
        if self.path.is_empty() {
            return Err(CoordinationError::InvalidTopology(format!(
                "branch {} has no stages",
                self.id
            )));
        }
        for (index, stage) in self.path.iter().enumerate() {
            if self.path[..index].contains(stage) {
                return Err(CoordinationError::InvalidTopology(format!(
                    "branch {} lists stage {stage} more than once",
                    self.id
                )));
            }
        }
        for handoff in &self.handoffs {
            let from = self.path.iter().position(|stage| *stage == handoff.from);
            let to = self.path.iter().position(|stage| *stage == handoff.to);
            match (from, to) {
                (Some(from), Some(to)) if from < to => {}
                (Some(_), Some(_)) => {
                    return Err(CoordinationError::InvalidTopology(format!(
                        "branch {} hands off from {} to {} against path order",
                        self.id, handoff.from, handoff.to
                    )));
                }
                _ => {
                    return Err(CoordinationError::InvalidTopology(format!(
                        "branch {} hands off between stages not on its path ({} -> {})",
                        self.id, handoff.from, handoff.to
                    )));
                }
            }
        }
        Ok(())
    }
}

/// How strictly a placement rule applies.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PlacementMode {
    Required,
    Preferred { weight: f32 },
}

/// A rule connecting the workers selected for different stages.
#[derive(Debug, Clone, PartialEq)]
pub enum PlacementRule {
    /// Stages share the selected worker's value for one topology domain, such
    /// as a rack or zone.
    SameDomain { domain: String, mode: PlacementMode },
    /// Stages satisfy each selected worker's own KV-transfer domain,
    /// enforcement, and weight, as published in its metadata.
    TransferCompatible,
}

/// Execution paths, their handoff dependencies, and placement rules.
#[derive(Debug, Clone)]
pub struct Topology {
    stages: Vec<StageId>,
    branches: Vec<Branch>,
    default_branch: Option<BranchId>,
    rules: Vec<PlacementRule>,
}

impl Topology {
    /// Validate and build a topology. A single branch is its own default.
    pub fn new(branches: Vec<Branch>) -> Result<Self, CoordinationError> {
        if branches.is_empty() {
            return Err(CoordinationError::InvalidTopology(
                "a topology needs at least one branch".to_string(),
            ));
        }
        let mut stages: Vec<StageId> = Vec::new();
        for (index, branch) in branches.iter().enumerate() {
            branch.validate()?;
            if branches[..index].iter().any(|other| other.id == branch.id) {
                return Err(CoordinationError::InvalidTopology(format!(
                    "branch {} is declared more than once",
                    branch.id
                )));
            }
            for stage in &branch.path {
                if !stages.contains(stage) {
                    stages.push(stage.clone());
                }
            }
        }
        let default_branch = (branches.len() == 1).then(|| branches[0].id.clone());
        Ok(Self {
            stages,
            branches,
            default_branch,
            rules: Vec::new(),
        })
    }

    /// A topology with one stage and one branch.
    pub fn single(stage: StageId) -> Self {
        Self::new(vec![Branch::linear(DEFAULT_BRANCH, vec![stage])])
            .expect("a single-stage branch is valid")
    }

    /// Aggregated serving: one stage.
    pub fn aggregated() -> Self {
        Self::single(StageId::AGGREGATED)
    }

    /// Prefill hands off to decode.
    pub fn prefill_decode() -> Self {
        Self::new(vec![Branch::linear(
            DEFAULT_BRANCH,
            vec![StageId::PREFILL, StageId::DECODE],
        )])
        .expect("prefill/decode is valid")
    }

    /// Conditional disaggregation: either decode runs prefill locally, or
    /// prefill hands off to decode. The policy chooses the branch.
    pub fn conditional_prefill_decode() -> Self {
        Self::new(vec![
            Branch::linear(LOCAL_PREFILL_DECODE_BRANCH, vec![StageId::DECODE]),
            Branch::linear(
                REMOTE_PREFILL_DECODE_BRANCH,
                vec![StageId::PREFILL, StageId::DECODE],
            ),
        ])
        .expect("conditional prefill/decode is valid")
    }

    /// Encode hands off to prefill, which hands off to decode; requests
    /// without multimodal input take the prefill/decode branch.
    pub fn encode_prefill_decode() -> Self {
        Self::new(vec![
            Branch::linear(
                ENCODE_PREFILL_DECODE_BRANCH,
                vec![StageId::ENCODE, StageId::PREFILL, StageId::DECODE],
            ),
            Branch::linear(
                PREFILL_DECODE_BRANCH,
                vec![StageId::PREFILL, StageId::DECODE],
            ),
        ])
        .expect("encode/prefill/decode is valid")
    }

    /// Declare the branch used when the policy never selects one.
    pub fn with_default_branch(mut self, branch: BranchId) -> Result<Self, CoordinationError> {
        if self.branch(&branch).is_none() {
            return Err(CoordinationError::UnknownBranch { branch });
        }
        self.default_branch = Some(branch);
        Ok(self)
    }

    pub fn with_rule(mut self, rule: PlacementRule) -> Self {
        self.rules.push(rule);
        self
    }

    pub fn with_rules(mut self, rules: impl IntoIterator<Item = PlacementRule>) -> Self {
        self.rules.extend(rules);
        self
    }

    /// Every stage any branch names, in first-seen order.
    pub fn stages(&self) -> &[StageId] {
        &self.stages
    }

    pub fn branches(&self) -> &[Branch] {
        &self.branches
    }

    pub fn branch(&self, id: &BranchId) -> Option<&Branch> {
        self.branches.iter().find(|branch| branch.id == *id)
    }

    pub fn default_branch(&self) -> Option<&Branch> {
        self.default_branch.as_ref().and_then(|id| self.branch(id))
    }

    pub fn rules(&self) -> &[PlacementRule] {
        &self.rules
    }

    pub fn has_stage(&self, stage: &StageId) -> bool {
        self.stages.contains(stage)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_branches_hand_off_along_the_path() {
        let branch = Branch::linear(
            "epd",
            vec![StageId::ENCODE, StageId::PREFILL, StageId::DECODE],
        );
        assert_eq!(branch.handoffs.len(), 2);
        assert_eq!(
            branch.inputs_of(&StageId::DECODE).collect::<Vec<_>>(),
            vec![&StageId::PREFILL]
        );
        assert_eq!(
            branch.dependents_of(&StageId::ENCODE).collect::<Vec<_>>(),
            vec![&StageId::PREFILL]
        );
        assert_eq!(branch.inputs_of(&StageId::ENCODE).count(), 0);
    }

    #[test]
    fn single_branch_is_the_default_and_multi_branch_needs_a_choice() {
        let single = Topology::prefill_decode();
        assert_eq!(single.default_branch().unwrap().id, DEFAULT_BRANCH);
        assert_eq!(single.stages(), &[StageId::PREFILL, StageId::DECODE]);

        let conditional = Topology::conditional_prefill_decode();
        assert!(conditional.default_branch().is_none());
        assert_eq!(conditional.stages(), &[StageId::DECODE, StageId::PREFILL]);
        let with_default = conditional
            .with_default_branch(REMOTE_PREFILL_DECODE_BRANCH)
            .unwrap();
        assert_eq!(
            with_default.default_branch().unwrap().id,
            REMOTE_PREFILL_DECODE_BRANCH
        );
    }

    #[test]
    fn invalid_branches_are_rejected() {
        assert!(matches!(
            Topology::new(vec![]).unwrap_err(),
            CoordinationError::InvalidTopology(_)
        ));
        assert!(matches!(
            Topology::new(vec![Branch::linear("empty", vec![])]).unwrap_err(),
            CoordinationError::InvalidTopology(_)
        ));
        let backwards = Branch::new(
            "backwards",
            vec![StageId::PREFILL, StageId::DECODE],
            vec![Handoff {
                from: StageId::DECODE,
                to: StageId::PREFILL,
            }],
        );
        assert!(matches!(
            Topology::new(vec![backwards]).unwrap_err(),
            CoordinationError::InvalidTopology(_)
        ));
        let off_path = Branch::new(
            "off-path",
            vec![StageId::DECODE],
            vec![Handoff {
                from: StageId::PREFILL,
                to: StageId::DECODE,
            }],
        );
        assert!(matches!(
            Topology::new(vec![off_path]).unwrap_err(),
            CoordinationError::InvalidTopology(_)
        ));
        let duplicate = vec![
            Branch::linear("a", vec![StageId::DECODE]),
            Branch::linear("a", vec![StageId::PREFILL]),
        ];
        assert!(matches!(
            Topology::new(duplicate).unwrap_err(),
            CoordinationError::InvalidTopology(_)
        ));
        assert!(matches!(
            Topology::aggregated()
                .with_default_branch(BranchId::new("missing"))
                .unwrap_err(),
            CoordinationError::UnknownBranch { .. }
        ));
    }
}
