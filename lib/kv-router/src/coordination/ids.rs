// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Identifiers shared by every coordination contract.
//!
//! Text identifiers (`StageId`, `BranchId`, `PoolId`, `ProfileName`) are
//! configuration names: cheap to clone when built from a static string, and
//! owned otherwise. Numeric identifiers (`InvocationId`, `AttemptId`) are
//! allocated per request by the coordinator.

use std::borrow::Cow;
use std::fmt;

use serde::{Deserialize, Serialize};

macro_rules! text_id {
    ($(#[$meta:meta])* $name:ident) => {
        $(#[$meta])*
        #[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
        #[serde(transparent)]
        pub struct $name(Cow<'static, str>);

        impl $name {
            /// Build from a static name without allocating.
            pub const fn from_static(name: &'static str) -> Self {
                Self(Cow::Borrowed(name))
            }

            /// Build from an owned or borrowed runtime name.
            pub fn new(name: impl Into<String>) -> Self {
                Self(Cow::Owned(name.into()))
            }

            pub fn as_str(&self) -> &str {
                &self.0
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str(&self.0)
            }
        }

        impl From<&'static str> for $name {
            fn from(name: &'static str) -> Self {
                Self::from_static(name)
            }
        }

        impl From<String> for $name {
            fn from(name: String) -> Self {
                Self::new(name)
            }
        }

        impl AsRef<str> for $name {
            fn as_ref(&self) -> &str {
                &self.0
            }
        }
    };
}

text_id! {
    /// A named stage in a routing topology, such as `prefill` or `decode`.
    ///
    /// Stage names are operator configuration. The built-in names below match
    /// the canonical [`crate::WorkerType`] roles, but a topology may bind any
    /// name to any pool.
    StageId
}

impl StageId {
    pub const AGGREGATED: Self = Self::from_static("aggregated");
    pub const PREFILL: Self = Self::from_static("prefill");
    pub const DECODE: Self = Self::from_static("decode");
    pub const ENCODE: Self = Self::from_static("encode");
}

text_id! {
    /// A named execution path declared by a topology.
    BranchId
}

text_id! {
    /// A worker pool a stage draws from. Stages that share a pool share its
    /// capacity accounting.
    PoolId
}

text_id! {
    /// A named selection profile on a stage: the scoring and work-accounting
    /// settings used to choose a worker for one request.
    ProfileName
}

impl ProfileName {
    /// The profile a stage uses when the policy names none.
    pub const DEFAULT: Self = Self::from_static("default");
    /// Decode after remote prefill: load-only scoring, decode-only accounting.
    pub const DECODE_ONLY: Self = Self::from_static("decode_only");
    /// Decode that also runs prefill locally: cache-aware scoring, full accounting.
    pub const LOCAL_PREFILL_DECODE: Self = Self::from_static("local_prefill_decode");
}

/// One stage's work for one request. Stable across retries of that work.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct InvocationId(u64);

impl InvocationId {
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    pub const fn get(self) -> u64 {
        self.0
    }
}

impl fmt::Display for InvocationId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(formatter)
    }
}

/// One attempt at an invocation. A retry keeps the [`InvocationId`] and takes
/// the next attempt, so events from a superseded attempt can be fenced.
///
/// This is a coordination-level identity, distinct from the scheduler's
/// internal booking attempt id.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct AttemptId(u32);

impl AttemptId {
    pub const FIRST: Self = Self(0);

    pub const fn new(value: u32) -> Self {
        Self(value)
    }

    pub const fn get(self) -> u32 {
        self.0
    }

    /// The attempt that replaces this one.
    pub const fn next(self) -> Self {
        Self(self.0.saturating_add(1))
    }
}

impl Default for AttemptId {
    fn default() -> Self {
        Self::FIRST
    }
}

impl fmt::Display for AttemptId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(formatter)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn static_and_owned_stage_ids_compare_by_name() {
        let owned = StageId::new("prefill".to_string());
        assert_eq!(owned, StageId::PREFILL);
        assert_eq!(owned.to_string(), "prefill");
        assert_eq!(StageId::from("decode"), StageId::DECODE);
    }

    #[test]
    fn attempt_ids_advance_and_saturate() {
        assert_eq!(AttemptId::FIRST.next(), AttemptId::new(1));
        assert_eq!(AttemptId::new(u32::MAX).next(), AttemptId::new(u32::MAX));
        assert_eq!(AttemptId::default(), AttemptId::FIRST);
    }

    #[test]
    fn ids_serialize_transparently() {
        assert_eq!(
            serde_json::to_string(&StageId::PREFILL).unwrap(),
            "\"prefill\""
        );
        assert_eq!(serde_json::to_string(&InvocationId::new(7)).unwrap(), "7");
        assert_eq!(
            serde_json::from_str::<ProfileName>("\"decode_only\"").unwrap(),
            ProfileName::DECODE_ONLY
        );
    }
}
