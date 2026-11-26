//! Smartcore model payload shared by `train.smartcore` and the smartcore predictor.

use serde::{Deserialize, Serialize};

/// File name of the JSON model payload inside the output artifact.
pub(crate) const MODEL_FILE: &str = "model.json";
/// Artifact flavor for binary logistic regression models.
pub(crate) const FLAVOR_LOGISTIC: &str = "smartcore-logistic";
/// Artifact flavor for ordinary least-squares linear regression models.
pub(crate) const FLAVOR_LINEAR: &str = "smartcore-linear";
/// Schema version of `model.json`.
pub(crate) const MODEL_SCHEMA_VERSION: u32 = 1;

/// Algorithm supported by `std/train.smartcore` and the smartcore predictor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum Algorithm {
    LogisticRegression,
    LinearRegression,
}

/// Typed binary classes serialized into `model.json` for logistic regression.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub(crate) enum ModelClasses {
    Int64(Vec<i64>),
    Bool(Vec<bool>),
    String(Vec<String>),
}

/// Serializable payload written to `model.json` (fixed field order).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ModelPayload {
    pub(crate) v: u32,
    pub(crate) algorithm: Algorithm,
    pub(crate) features: Vec<String>,
    pub(crate) target: String,
    pub(crate) intercept: f64,
    pub(crate) coefficients: Vec<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) classes: Option<ModelClasses>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) threshold: Option<f64>,
}
