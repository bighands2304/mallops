//! Fitted transform artifact of `std/preprocess.tabular` (`kind: transform`, `format: json`),
//! reloaded by `package` and `serve` to apply the exact same preprocessing (FR-PRE-3).
//!
//! Payload: a single `transform.json` file, `{"v":1,"target":…,"input":[…],"steps":[…]}`,
//! serialized deterministically (struct field order, `BTreeMap` keys).

use std::collections::BTreeMap;
use std::path::Path;

use arrow_array::RecordBatch;
use mallops_core::Error;
use serde::{Deserialize, Serialize};

use crate::ingest::Column;

/// File name of the payload of a `transform` artifact.
pub const TRANSFORM_FILE: &str = "transform.json";

/// Format version of [`TRANSFORM_FILE`].
const VERSION: u32 = 1;

/// A fitted preprocessing pipeline: replays `select`, `impute`, `scale` and `encode` in order
/// (`filter` and `split` only shape the training data and are not replayed).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FittedTransform {
    v: u32,
    /// Target column (absent from inference inputs; passed through when present).
    pub(crate) target: String,
    /// Raw input columns expected by the first step, in order (target included).
    pub(crate) input: Vec<Column>,
    /// Fitted steps in application order.
    pub(crate) steps: Vec<FittedStep>,
}

/// Parameters of a fitted step, keyed by column name.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub(crate) enum FittedStep {
    /// Keep these columns, in this order (target kept when present).
    Select { columns: Vec<String> },
    /// Fill nulls of each column with its value (number or string).
    Impute {
        values: BTreeMap<String, serde_json::Value>,
    },
    /// `x' = (x - offset) / scale` per column, output `float64` (`scale == 0` maps to `0.0`).
    Scale { params: BTreeMap<String, Affine> },
    /// Category index per column, output `int64`; unseen or null = `-1`.
    Ordinal {
        categories: BTreeMap<String, Vec<String>>,
    },
    /// One `float64` column `<col>=<category>` per category replacing `<col>`; unseen = all `0`.
    OneHot {
        categories: BTreeMap<String, Vec<String>>,
    },
}

/// Affine scaling parameters of one column.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Affine {
    pub(crate) offset: f64,
    pub(crate) scale: f64,
}

impl FittedTransform {
    /// Creates a transform from its parts.
    pub(crate) fn new(target: String, input: Vec<Column>, steps: Vec<FittedStep>) -> Self {
        Self {
            v: VERSION,
            target,
            input,
            steps,
        }
    }

    /// Canonical JSON bytes of this transform (the [`TRANSFORM_FILE`] payload).
    pub(crate) fn to_json(&self) -> Result<Vec<u8>, Error> {
        serde_json::to_vec(self).map_err(|e| Error::Runtime(format!("transform: {e}")))
    }

    /// Target column name.
    #[must_use]
    pub fn target(&self) -> &str {
        &self.target
    }

    /// Loads the transform from a materialized `transform` artifact directory.
    ///
    /// # Errors
    ///
    /// [`Error::Runtime`] if [`TRANSFORM_FILE`] is missing, malformed or has another version.
    pub fn from_dir(dir: &Path) -> Result<Self, Error> {
        let _ = dir;
        Err(Error::Runtime(
            "FittedTransform::from_dir: not implemented yet".into(),
        )) // 12.1
    }

    /// Applies every fitted step to `batch` (raw columns as in the training input).
    ///
    /// # Errors
    ///
    /// [`Error::Runtime`] if a required column is missing or has an unexpected type.
    pub fn apply_batch(&self, batch: &RecordBatch) -> Result<RecordBatch, Error> {
        let _ = batch;
        Err(Error::Runtime(
            "FittedTransform::apply_batch: not implemented yet".into(),
        )) // 12.1
    }

    /// Builds a batch from JSON rows (an array of objects keyed by input column; target optional)
    /// and applies [`Self::apply_batch`].
    ///
    /// # Errors
    ///
    /// [`Error::Runtime`] if `rows` is not an array of objects or a value does not match the
    /// input column type.
    pub fn apply_json(&self, rows: &serde_json::Value) -> Result<RecordBatch, Error> {
        let _ = rows;
        Err(Error::Runtime(
            "FittedTransform::apply_json: not implemented yet".into(),
        )) // 12.1
    }
}
