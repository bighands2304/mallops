//! Arrow kernels of the preprocessing steps: stateless `filter`, and `fit` of `select`, `impute`,
//! `scale`, `encode` into a [`FittedStep`] (applied with [`FittedStep::apply`]).

use arrow_array::RecordBatch;
use mallops_core::Error;

use super::transform::FittedStep;
use super::{FilterStep, Step};

/// Keeps the rows of `batch` matching `step` (nulls never match a comparison).
pub(crate) fn filter(step: &FilterStep, batch: &RecordBatch) -> Result<RecordBatch, Error> {
    let _ = (step, batch);
    Err(Error::Runtime("filter: not implemented yet".into())) // 12.1
}

/// Fits `step` (`select`, `impute`, `scale` or `encode`) on `train`; `target` is never
/// transformed and is excluded from default column sets.
///
/// # Errors
///
/// [`Error::Runtime`] if a named column is missing or has an unsupported type;
/// [`Error::Config`] for `filter` / `split` (not fitted steps).
pub(crate) fn fit(step: &Step, target: &str, train: &RecordBatch) -> Result<FittedStep, Error> {
    let _ = (step, target, train);
    Err(Error::Runtime("fit: not implemented yet".into())) // 12.1
}

impl FittedStep {
    /// Applies this fitted step to `batch`; a missing target column is allowed (serving).
    pub(crate) fn apply(&self, batch: &RecordBatch) -> Result<RecordBatch, Error> {
        let _ = batch;
        Err(Error::Runtime("apply: not implemented yet".into())) // 12.1
    }
}
