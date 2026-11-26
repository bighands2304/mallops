//! Predictor for the `sklearn-pickle` flavor, via an external Python worker.

use std::path::Path;

use mallops_core::{Artifact, BoxFuture, Error};

use super::{LoadedPredictor, Predictor};

/// Serves `sklearn-pickle` models through a Python subprocess (executes code: needs explicit trust).
#[derive(Debug, Default)]
pub struct ExternalPythonPredictor;

impl ExternalPythonPredictor {
    /// Creates the predictor.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

impl Predictor for ExternalPythonPredictor {
    fn flavors(&self) -> &[&str] {
        &["sklearn-pickle"]
    }

    fn executes_code(&self) -> bool {
        true
    }

    fn load<'a>(
        &'a self,
        _artifact: &'a Artifact,
        _payload_dir: &'a Path,
    ) -> BoxFuture<'a, Result<Box<dyn LoadedPredictor>, Error>> {
        Box::pin(async { Err(Error::Runtime("not implemented".into())) })
    }
}
