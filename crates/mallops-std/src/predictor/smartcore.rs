//! Predictor for the native smartcore flavors.

use std::path::Path;

use mallops_core::{Artifact, BoxFuture, Error};

use super::{LoadedPredictor, Predictor};

/// Serves `smartcore-logistic` and `smartcore-linear` models in-process.
#[derive(Debug, Default)]
pub struct SmartcorePredictor;

impl SmartcorePredictor {
    /// Creates the predictor.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

impl Predictor for SmartcorePredictor {
    fn flavors(&self) -> &[&str] {
        &["smartcore-logistic", "smartcore-linear"]
    }

    fn executes_code(&self) -> bool {
        false
    }

    fn load<'a>(
        &'a self,
        _artifact: &'a Artifact,
        _payload_dir: &'a Path,
    ) -> BoxFuture<'a, Result<Box<dyn LoadedPredictor>, Error>> {
        Box::pin(async { Err(Error::Runtime("not implemented".into())) })
    }
}
