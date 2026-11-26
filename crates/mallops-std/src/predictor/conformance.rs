//! Reusable conformance suite for [`Predictor`] implementations.
//!
//! Returns `Err(message)` on the first violation instead of panicking, so callers decide how to
//! report it (typically `check(..).await.unwrap()` in a test).

use std::path::Path;

use mallops_core::Artifact;

use super::{PredictInput, Predictor};

/// Checks `predictor` against `artifact` (payload in `payload_dir`).
///
/// `cases` are `(input, expected output)` pairs; `malformed` must make `predict` fail.
///
/// # Errors
///
/// A description of the first violated rule: manifest flavor not in `flavors()`, failing `load`,
/// a wrong `predict` / `predict_batch` result (order matters), a non-empty result for an empty
/// batch, or `malformed` being accepted.
pub async fn check(
    predictor: &dyn Predictor,
    artifact: &Artifact,
    payload_dir: &Path,
    cases: &[(PredictInput, PredictInput)],
    malformed: PredictInput,
) -> Result<(), String> {
    if let Some(flavor) = &artifact.manifest().flavor
        && !predictor.flavors().contains(&flavor.as_str())
    {
        return Err(format!("manifest flavor `{flavor}` not in flavors()"));
    }
    let loaded = predictor
        .load(artifact, payload_dir)
        .await
        .map_err(|e| format!("load failed: {e}"))?;
    for (i, (input, expected)) in cases.iter().enumerate() {
        let got = loaded
            .predict(input.clone())
            .await
            .map_err(|e| format!("predict case {i} failed: {e}"))?;
        if &got != expected {
            return Err(format!(
                "predict case {i}: got {got:?}, expected {expected:?}"
            ));
        }
    }
    let inputs: Vec<_> = cases.iter().map(|(i, _)| i.clone()).collect();
    let expected: Vec<_> = cases.iter().map(|(_, e)| e.clone()).collect();
    let got = loaded
        .predict_batch(inputs)
        .await
        .map_err(|e| format!("predict_batch failed: {e}"))?;
    if got != expected {
        return Err(format!("predict_batch: got {got:?}, expected {expected:?}"));
    }
    let empty = loaded
        .predict_batch(Vec::new())
        .await
        .map_err(|e| format!("empty predict_batch failed: {e}"))?;
    if !empty.is_empty() {
        return Err(format!("empty predict_batch returned {empty:?}"));
    }
    if loaded.predict(malformed).await.is_ok() {
        return Err("malformed input was accepted".into());
    }
    Ok(())
}
