// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Stage configuration: the pool a stage draws from, the selection profiles it
//! offers, and the capabilities a coordination policy may rely on.

use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::WorkerType;
use crate::scheduling::config::RouterConfigOverride;

use super::error::CoordinationError;
use super::ids::{PoolId, ProfileName, StageId};
use super::selector::StageSelector;

/// A stage's worker pool together with the generation of the binding that
/// selected it. A selection retains its `PoolRef`, so a binding rebuilt by
/// discovery cannot be mistaken for the one a preview came from.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PoolRef {
    pub id: PoolId,
    pub generation: u64,
}

impl PoolRef {
    pub fn new(id: impl Into<PoolId>, generation: u64) -> Self {
        Self {
            id: id.into(),
            generation,
        }
    }
}

impl fmt::Display for PoolRef {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}@{}", self.id, self.generation)
    }
}

/// Which work a selection accounts against the chosen worker.
///
/// Accounting is independent of scoring: a decode stage that scores on load
/// alone still accounts the decode blocks it will hold, and a stage that scores
/// on cache overlap still accounts the full prompt when it runs prefill.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkAccounting {
    /// The worker runs prefill and decode for this request.
    #[default]
    PrefillAndDecode,
    /// The worker runs prefill only; its output is one handoff token.
    PrefillOnly,
    /// The worker decodes from a remote prefill; prompt work is not its load.
    DecodeOnly,
    /// The stage keeps no scheduler accounting (for example, an encoder pool
    /// selected round-robin).
    None,
}

/// How a stage scores candidate workers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScoringMode {
    /// Cache overlap earns credit against prefill load.
    #[default]
    CacheAware,
    /// Overlap earns no credit; selection follows load alone.
    LoadOnly,
}

/// The settings used to choose a worker for one stage of one request.
///
/// Profiles are fixed when a stage is bound and chosen per request by the
/// coordination policy.
#[derive(Debug, Clone)]
pub struct StageProfile {
    pub name: ProfileName,
    pub work: WorkAccounting,
    pub scoring: ScoringMode,
    /// Operator-configured router settings for this profile. Fields set here
    /// take precedence over the caller's per-request override; the `work` and
    /// `scoring` rules take precedence over both.
    pub router_config_override: Option<RouterConfigOverride>,
}

impl StageProfile {
    pub fn new(name: impl Into<ProfileName>, work: WorkAccounting, scoring: ScoringMode) -> Self {
        Self {
            name: name.into(),
            work,
            scoring,
            router_config_override: None,
        }
    }

    pub fn with_router_config_override(mut self, config_override: RouterConfigOverride) -> Self {
        self.router_config_override = Some(config_override);
        self
    }

    /// The router override a selector should apply for this profile, layered
    /// over the caller's per-request override.
    ///
    /// Returns `None` when nothing needs overriding so selectors can pass the
    /// request through untouched.
    pub fn resolve_router_config_override(
        &self,
        caller: Option<&RouterConfigOverride>,
    ) -> Option<RouterConfigOverride> {
        let mut resolved = caller.cloned().unwrap_or_default();
        if let Some(profile) = &self.router_config_override {
            apply_override(&mut resolved, profile);
        }
        match self.work {
            WorkAccounting::DecodeOnly => {
                resolved.assume_kv_reuse = Some(false);
                resolved.track_prefill_tokens = Some(false);
            }
            WorkAccounting::None => {
                resolved.track_prefill_tokens = Some(false);
            }
            WorkAccounting::PrefillAndDecode | WorkAccounting::PrefillOnly => {}
        }
        if self.scoring == ScoringMode::LoadOnly {
            resolved.overlap_score_credit = Some(0.0);
        }
        if is_noop_override(&resolved) {
            None
        } else {
            Some(resolved)
        }
    }

    /// The output length a selector should project for this profile.
    pub fn resolve_expected_output_tokens(&self, caller: Option<u32>) -> Option<u32> {
        match self.work {
            WorkAccounting::PrefillOnly => Some(1),
            WorkAccounting::None => None,
            WorkAccounting::PrefillAndDecode | WorkAccounting::DecodeOnly => caller,
        }
    }
}

fn apply_override(target: &mut RouterConfigOverride, source: &RouterConfigOverride) {
    let RouterConfigOverride {
        overlap_score_credit,
        prefill_load_scale,
        router_temperature,
        assume_kv_reuse,
        track_prefill_tokens,
        shared_cache_multiplier,
    } = source;
    if overlap_score_credit.is_some() {
        target.overlap_score_credit = *overlap_score_credit;
    }
    if prefill_load_scale.is_some() {
        target.prefill_load_scale = *prefill_load_scale;
    }
    if router_temperature.is_some() {
        target.router_temperature = *router_temperature;
    }
    if assume_kv_reuse.is_some() {
        target.assume_kv_reuse = *assume_kv_reuse;
    }
    if track_prefill_tokens.is_some() {
        target.track_prefill_tokens = *track_prefill_tokens;
    }
    if shared_cache_multiplier.is_some() {
        target.shared_cache_multiplier = *shared_cache_multiplier;
    }
}

fn is_noop_override(config_override: &RouterConfigOverride) -> bool {
    let RouterConfigOverride {
        overlap_score_credit,
        prefill_load_scale,
        router_temperature,
        assume_kv_reuse,
        track_prefill_tokens,
        shared_cache_multiplier,
    } = config_override;
    overlap_score_credit.is_none()
        && prefill_load_scale.is_none()
        && router_temperature.is_none()
        && assume_kv_reuse.is_none()
        && track_prefill_tokens.is_none()
        && shared_cache_multiplier.is_none()
}

/// The named profiles one stage offers, with the one used when a policy names
/// none.
#[derive(Debug, Clone)]
pub struct StageProfiles {
    default: ProfileName,
    profiles: HashMap<ProfileName, StageProfile>,
}

impl StageProfiles {
    /// A profile set whose only (and default) profile is `default`.
    pub fn single(default: StageProfile) -> Self {
        let name = default.name.clone();
        Self {
            default: name.clone(),
            profiles: HashMap::from([(name, default)]),
        }
    }

    /// Add or replace a profile by name. The default is unchanged.
    pub fn with_profile(mut self, profile: StageProfile) -> Self {
        self.profiles.insert(profile.name.clone(), profile);
        self
    }

    /// The defaults for a worker role, following the DEP's stage table.
    pub fn for_worker_type(worker_type: WorkerType) -> Self {
        match worker_type {
            WorkerType::Aggregated => Self::single(StageProfile::new(
                ProfileName::DEFAULT,
                WorkAccounting::PrefillAndDecode,
                ScoringMode::CacheAware,
            )),
            WorkerType::Prefill => Self::single(StageProfile::new(
                ProfileName::DEFAULT,
                WorkAccounting::PrefillOnly,
                ScoringMode::CacheAware,
            )),
            WorkerType::Decode => Self::single(StageProfile::new(
                ProfileName::DECODE_ONLY,
                WorkAccounting::DecodeOnly,
                ScoringMode::LoadOnly,
            ))
            .with_profile(StageProfile::new(
                ProfileName::LOCAL_PREFILL_DECODE,
                WorkAccounting::PrefillAndDecode,
                ScoringMode::CacheAware,
            )),
            WorkerType::Encode => Self::single(StageProfile::new(
                ProfileName::DEFAULT,
                WorkAccounting::None,
                ScoringMode::LoadOnly,
            )),
        }
    }

    pub fn get(&self, name: &ProfileName) -> Option<&StageProfile> {
        self.profiles.get(name)
    }

    pub fn default_profile(&self) -> &StageProfile {
        self.profiles
            .get(&self.default)
            .expect("the default profile is always present")
    }

    pub fn default_name(&self) -> &ProfileName {
        &self.default
    }

    pub fn names(&self) -> impl Iterator<Item = &ProfileName> {
        self.profiles.keys()
    }
}

/// What a coordination policy may rely on a stage to do.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StageCapabilities {
    pub worker_type: WorkerType,
    /// The selector can report a candidate and its signals without admitting.
    pub supports_preview: bool,
    /// The stage produces a handoff a later stage waits for.
    pub produces_handoff: bool,
    /// The stage needs an earlier stage's handoff before it can run.
    pub consumes_handoff: bool,
}

impl StageCapabilities {
    pub fn for_worker_type(worker_type: WorkerType) -> Self {
        match worker_type {
            WorkerType::Aggregated => Self {
                worker_type,
                supports_preview: true,
                produces_handoff: false,
                consumes_handoff: false,
            },
            WorkerType::Prefill => Self {
                worker_type,
                supports_preview: true,
                produces_handoff: true,
                consumes_handoff: false,
            },
            WorkerType::Decode => Self {
                worker_type,
                supports_preview: true,
                produces_handoff: false,
                consumes_handoff: true,
            },
            WorkerType::Encode => Self {
                worker_type,
                supports_preview: false,
                produces_handoff: true,
                consumes_handoff: false,
            },
        }
    }

    pub fn with_preview(mut self, supports_preview: bool) -> Self {
        self.supports_preview = supports_preview;
        self
    }
}

/// Connects a stage to its worker pool, selector, profiles, and capabilities.
#[derive(Clone)]
pub struct StageBinding {
    pub id: StageId,
    pub pool: PoolRef,
    pub selector: Arc<dyn StageSelector>,
    pub profiles: StageProfiles,
    pub capabilities: StageCapabilities,
}

impl StageBinding {
    /// Bind `id` to `selector` with the defaults for `worker_type`.
    pub fn new(
        id: impl Into<StageId>,
        pool: PoolRef,
        worker_type: WorkerType,
        selector: Arc<dyn StageSelector>,
    ) -> Self {
        Self {
            id: id.into(),
            pool,
            selector,
            profiles: StageProfiles::for_worker_type(worker_type),
            capabilities: StageCapabilities::for_worker_type(worker_type),
        }
    }

    pub fn with_profiles(mut self, profiles: StageProfiles) -> Self {
        self.profiles = profiles;
        self
    }

    pub fn with_capabilities(mut self, capabilities: StageCapabilities) -> Self {
        self.capabilities = capabilities;
        self
    }

    /// Resolve a policy's profile choice, falling back to the stage default.
    pub fn profile(&self, name: Option<&ProfileName>) -> Result<&StageProfile, CoordinationError> {
        match name {
            None => Ok(self.profiles.default_profile()),
            Some(name) => {
                self.profiles
                    .get(name)
                    .ok_or_else(|| CoordinationError::UnknownProfile {
                        stage: self.id.clone(),
                        profile: name.clone(),
                    })
            }
        }
    }
}

impl fmt::Debug for StageBinding {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("StageBinding")
            .field("id", &self.id)
            .field("pool", &self.pool)
            .field("profiles", &self.profiles)
            .field("capabilities", &self.capabilities)
            .finish_non_exhaustive()
    }
}
