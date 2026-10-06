// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Registration of coordination policies by name.
//!
//! Mirrors the worker-selection plugin registry: hosts register compiled Rust
//! policies at startup and resolve them by the name a topology or deployment
//! configuration refers to. The built-in names cover the flows the DEP
//! describes.

use std::collections::HashMap;
use std::sync::Arc;

use crate::conditional_disagg::make_conditional_disagg_policy;
use crate::config::KvRouterConfig;

use super::builtin_policies::{
    AggregatedPolicy, ConditionalDisaggThresholds, ConditionalDisaggregationPolicy,
    EncodePrefillDecodePolicy, PrefillDecodePolicy, ProgressivePrefillDecodePolicy,
};
use super::policy::CoordinationPolicyFactory;

pub const AGGREGATED_POLICY: &str = "aggregated";
pub const PREFILL_FIRST_POLICY: &str = "prefill_first";
pub const DECODE_FIRST_POLICY: &str = "decode_first";
pub const PROGRESSIVE_PREFILL_DECODE_POLICY: &str = "progressive_prefill_decode";
pub const CONDITIONAL_DISAGGREGATION_POLICY: &str = "conditional_disaggregation";
pub const ENCODE_PREFILL_DECODE_POLICY: &str = "encode_prefill_decode";

/// Why a policy could not be registered or resolved.
#[derive(Debug, thiserror::Error)]
pub enum CoordinationPolicyRegistryError {
    #[error("coordination policy name must not be empty")]
    EmptyName,
    #[error("coordination policy {name:?} is already registered")]
    Duplicate { name: String },
    #[error("unknown coordination policy {name:?}; registered policies: {available}")]
    Unknown { name: String, available: String },
}

/// Coordination policy factories keyed by name.
#[derive(Clone, Default)]
pub struct CoordinationPolicyRegistry {
    factories: HashMap<String, CoordinationPolicyFactory>,
}

impl CoordinationPolicyRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// A registry holding every built-in policy, configured from `config`.
    pub fn with_builtins(config: &KvRouterConfig) -> Self {
        let mut registry = Self::new();
        registry
            .register(
                AGGREGATED_POLICY,
                Arc::new(|_| Box::new(AggregatedPolicy::new())),
            )
            .expect("fresh registry");
        registry
            .register(
                PREFILL_FIRST_POLICY,
                Arc::new(|_| Box::new(PrefillDecodePolicy::prefill_first())),
            )
            .expect("fresh registry");
        registry
            .register(
                DECODE_FIRST_POLICY,
                Arc::new(|_| Box::new(PrefillDecodePolicy::decode_first())),
            )
            .expect("fresh registry");
        registry
            .register(
                PROGRESSIVE_PREFILL_DECODE_POLICY,
                Arc::new(|_| Box::new(ProgressivePrefillDecodePolicy::new())),
            )
            .expect("fresh registry");
        let conditional_config = config.clone();
        registry
            .register(
                CONDITIONAL_DISAGGREGATION_POLICY,
                Arc::new(move |_| {
                    Box::new(ConditionalDisaggregationPolicy::new(
                        make_conditional_disagg_policy(Some(&conditional_config)),
                        ConditionalDisaggThresholds::from_config(&conditional_config),
                        Box::new(ProgressivePrefillDecodePolicy::new()),
                    ))
                }),
            )
            .expect("fresh registry");
        registry
            .register(
                ENCODE_PREFILL_DECODE_POLICY,
                Arc::new(|_| Box::new(EncodePrefillDecodePolicy::new(true))),
            )
            .expect("fresh registry");
        registry
    }

    pub fn register(
        &mut self,
        name: impl Into<String>,
        factory: CoordinationPolicyFactory,
    ) -> Result<(), CoordinationPolicyRegistryError> {
        let name = name.into();
        if name.is_empty() {
            return Err(CoordinationPolicyRegistryError::EmptyName);
        }
        if self.factories.contains_key(&name) {
            return Err(CoordinationPolicyRegistryError::Duplicate { name });
        }
        self.factories.insert(name, factory);
        Ok(())
    }

    pub fn resolve(
        &self,
        name: &str,
    ) -> Result<CoordinationPolicyFactory, CoordinationPolicyRegistryError> {
        self.factories.get(name).cloned().ok_or_else(|| {
            let mut available: Vec<&str> = self.factories.keys().map(String::as_str).collect();
            available.sort_unstable();
            CoordinationPolicyRegistryError::Unknown {
                name: name.to_string(),
                available: if available.is_empty() {
                    "<none>".to_string()
                } else {
                    available.join(", ")
                },
            }
        })
    }

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.factories.keys().map(String::as_str)
    }

    pub fn is_empty(&self) -> bool {
        self.factories.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtins_are_registered_and_resolve() {
        let registry = CoordinationPolicyRegistry::with_builtins(&KvRouterConfig::default());
        for name in [
            AGGREGATED_POLICY,
            PREFILL_FIRST_POLICY,
            DECODE_FIRST_POLICY,
            PROGRESSIVE_PREFILL_DECODE_POLICY,
            CONDITIONAL_DISAGGREGATION_POLICY,
            ENCODE_PREFILL_DECODE_POLICY,
        ] {
            registry
                .resolve(name)
                .unwrap_or_else(|error| panic!("{error}"));
        }
        let Err(error) = registry.resolve("custom_order") else {
            panic!("unknown policy must not resolve");
        };
        assert!(matches!(
            error,
            CoordinationPolicyRegistryError::Unknown { .. }
        ));
        assert!(error.to_string().contains(AGGREGATED_POLICY));
    }

    #[test]
    fn registration_rejects_empty_and_duplicate_names() {
        let mut registry = CoordinationPolicyRegistry::new();
        assert!(registry.is_empty());
        assert!(matches!(
            registry.register("", Arc::new(|_| Box::new(AggregatedPolicy::new()))),
            Err(CoordinationPolicyRegistryError::EmptyName)
        ));
        registry
            .register("custom", Arc::new(|_| Box::new(AggregatedPolicy::new())))
            .unwrap();
        assert!(matches!(
            registry.register("custom", Arc::new(|_| Box::new(AggregatedPolicy::new()))),
            Err(CoordinationPolicyRegistryError::Duplicate { .. })
        ));
        assert_eq!(registry.names().collect::<Vec<_>>(), vec!["custom"]);
        let Err(error) = CoordinationPolicyRegistry::new().resolve("x") else {
            panic!("empty registry must not resolve");
        };
        assert!(error.to_string().contains("<none>"));
    }
}
