//! Predictor contract (ADR-0007 D6/D7): turns a model artifact into something that answers
//! prediction requests. The core stays model-agnostic; consumers pick a [`Predictor`] by the
//! artifact's `flavor` through a [`PredictorRegistry`].

pub mod conformance;
#[cfg(feature = "predict-external")]
pub mod external;
#[cfg(feature = "predict-smartcore")]
pub mod smartcore;

use std::{collections::HashMap, path::Path, sync::Arc};

use mallops_core::{Artifact, BoxFuture, Error};

#[cfg(feature = "predict-external")]
pub use external::ExternalPythonPredictor;
#[cfg(feature = "predict-smartcore")]
pub use smartcore::SmartcorePredictor;

/// A prediction request or response body.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum PredictInput {
    /// A JSON document.
    Json(serde_json::Value),
    /// Opaque bytes with their media type.
    Bytes {
        /// Media type of `data`.
        content_type: String,
        /// The raw payload.
        data: Vec<u8>,
    },
}

/// Loads model artifacts of one or more flavors.
pub trait Predictor: Send + Sync {
    /// The artifact `flavor` values this predictor can load (non-empty).
    fn flavors(&self) -> &[&str];

    /// Whether loading a model runs code shipped in the artifact (pickle, joblib, …).
    /// Such predictors are refused by [`PredictorRegistry::get`] unless trust is given.
    fn executes_code(&self) -> bool;

    /// Loads the model whose payload files are in `payload_dir`.
    ///
    /// # Errors
    ///
    /// Any [`Error`] if the payload is missing or malformed.
    fn load<'a>(
        &'a self,
        artifact: &'a Artifact,
        payload_dir: &'a Path,
    ) -> BoxFuture<'a, Result<Box<dyn LoadedPredictor>, Error>>;
}

/// A model loaded in memory, ready to predict.
pub trait LoadedPredictor: Send + Sync {
    /// Predicts one input.
    ///
    /// # Errors
    ///
    /// Any [`Error`] if the input is malformed or the model fails.
    fn predict(&self, input: PredictInput) -> BoxFuture<'_, Result<PredictInput, Error>>;

    /// Predicts a batch; outputs are in input order.
    ///
    /// # Errors
    ///
    /// Any [`Error`] if any input is malformed or the model fails.
    fn predict_batch(
        &self,
        inputs: Vec<PredictInput>,
    ) -> BoxFuture<'_, Result<Vec<PredictInput>, Error>>;
}

/// Resolves a [`Predictor`] by artifact flavor. Local instance, no global state.
#[derive(Default)]
pub struct PredictorRegistry {
    by_flavor: HashMap<String, Arc<dyn Predictor>>,
}

impl PredictorRegistry {
    /// Creates an empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers `predictor` for all its flavors, atomically.
    ///
    /// # Errors
    ///
    /// [`Error::Validation`] if it has no flavors, repeats a flavor, or a flavor is already
    /// registered; the registry is left unchanged.
    pub fn register(&mut self, predictor: Arc<dyn Predictor>) -> Result<(), Error> {
        let flavors = predictor.flavors();
        if flavors.is_empty() {
            return Err(Error::validation("", "predictor declares no flavors"));
        }
        for (i, f) in flavors.iter().enumerate() {
            if flavors[..i].contains(f) || self.by_flavor.contains_key(*f) {
                return Err(Error::validation(
                    "",
                    format!("duplicate predictor flavor `{f}`"),
                ));
            }
        }
        for f in flavors {
            self.by_flavor
                .insert((*f).to_owned(), Arc::clone(&predictor));
        }
        Ok(())
    }

    /// Returns the predictor for `flavor`, enforcing the code-execution trust policy.
    ///
    /// # Errors
    ///
    /// [`Error::Config`] if no predictor handles `flavor`, or it executes code and
    /// `trust_unsafe_flavors` is false.
    pub fn get(
        &self,
        flavor: &str,
        trust_unsafe_flavors: bool,
    ) -> Result<Arc<dyn Predictor>, Error> {
        let predictor = self
            .by_flavor
            .get(flavor)
            .ok_or_else(|| Error::Config(format!("no predictor for flavor `{flavor}`")))?;
        if predictor.executes_code() && !trust_unsafe_flavors {
            return Err(Error::Config(format!(
                "flavor `{flavor}` executes code from the artifact; pass --trust-unsafe-flavors to serve it"
            )));
        }
        Ok(Arc::clone(predictor))
    }
}

/// Registers every feature-enabled standard predictor into `registry`.
///
/// # Errors
///
/// [`Error::Validation`] if a standard flavor is already registered.
#[allow(unused_variables)] // with only `predictor` enabled nothing is registered
pub fn register_std_predictors(registry: &mut PredictorRegistry) -> Result<(), Error> {
    #[cfg(feature = "predict-smartcore")]
    registry.register(Arc::new(SmartcorePredictor::new()))?;
    #[cfg(feature = "predict-external")]
    registry.register(Arc::new(ExternalPythonPredictor::new()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::predictor::conformance;
    use mallops_core::{ArtifactId, ArtifactKind, Manifest};

    struct Echo(&'static [&'static str], bool);

    impl Predictor for Echo {
        fn flavors(&self) -> &[&str] {
            self.0
        }
        fn executes_code(&self) -> bool {
            self.1
        }
        fn load<'a>(
            &'a self,
            _: &'a Artifact,
            _: &'a Path,
        ) -> BoxFuture<'a, Result<Box<dyn LoadedPredictor>, Error>> {
            Box::pin(async { Ok(Box::new(EchoLoaded) as Box<dyn LoadedPredictor>) })
        }
    }

    struct EchoLoaded;

    impl LoadedPredictor for EchoLoaded {
        fn predict(&self, input: PredictInput) -> BoxFuture<'_, Result<PredictInput, Error>> {
            Box::pin(async move {
                match input {
                    PredictInput::Json(serde_json::Value::Null) => {
                        Err(Error::Runtime("malformed".into()))
                    }
                    other => Ok(other),
                }
            })
        }
        fn predict_batch(
            &self,
            inputs: Vec<PredictInput>,
        ) -> BoxFuture<'_, Result<Vec<PredictInput>, Error>> {
            Box::pin(async move {
                let mut out = Vec::new();
                for i in inputs {
                    out.push(self.predict(i).await?);
                }
                Ok(out)
            })
        }
    }

    fn reg(p: Echo) -> Result<PredictorRegistry, Error> {
        let mut r = PredictorRegistry::new();
        r.register(Arc::new(p))?;
        Ok(r)
    }

    #[test]
    fn predictor_registry_resolves_by_flavor() {
        let r = reg(Echo(&["a", "b"], false)).unwrap();
        assert!(r.get("a", false).is_ok());
        assert!(r.get("b", false).is_ok());
    }

    #[test]
    fn predictor_registry_unknown_flavor_is_config_error() {
        let r = reg(Echo(&["a"], false)).unwrap();
        let Err(Error::Config(m)) = r.get("zzz", true) else {
            panic!("expected Config")
        };
        assert_eq!(m, "no predictor for flavor `zzz`");
    }

    #[test]
    fn predictor_registry_blocks_code_execution_without_trust() {
        let r = reg(Echo(&["pk"], true)).unwrap();
        let Err(Error::Config(m)) = r.get("pk", false) else {
            panic!("expected Config")
        };
        assert!(m.contains("--trust-unsafe-flavors"), "{m}");
        assert!(r.get("pk", true).is_ok());
    }

    #[test]
    fn predictor_registry_rejects_duplicates_atomically() {
        let mut r = reg(Echo(&["a"], false)).unwrap();
        // "b" is new but "a" collides: nothing may be registered.
        let e = r.register(Arc::new(Echo(&["b", "a"], false)));
        assert!(matches!(e, Err(Error::Validation(_))));
        assert!(r.get("b", false).is_err());
        let e = r.register(Arc::new(Echo(&["c", "c"], false)));
        assert!(matches!(e, Err(Error::Validation(_))));
        assert!(r.get("c", false).is_err());
    }

    #[test]
    fn predictor_registry_rejects_empty_flavors() {
        assert!(matches!(reg(Echo(&[], false)), Err(Error::Validation(_))));
    }

    #[test]
    fn predictor_registry_register_std_predictors_twice_fails() {
        let mut r = PredictorRegistry::new();
        register_std_predictors(&mut r).unwrap();
        let second = register_std_predictors(&mut r);
        if cfg!(any(
            feature = "predict-smartcore",
            feature = "predict-external"
        )) {
            assert!(matches!(second, Err(Error::Validation(_))), "{second:?}");
        } else {
            assert!(second.is_ok());
        }
    }

    #[tokio::test]
    async fn predictor_registry_conformance_suite_accepts_echo() {
        let artifact = Artifact::new(
            ArtifactId::from_bytes([0; 32]),
            Manifest::new(ArtifactKind::Model, "x").with_flavor("a"),
        );
        let j = |n: i64| PredictInput::Json(serde_json::json!(n));
        let cases = [(j(1), j(1)), (j(2), j(2))];
        let r = conformance::check(
            &Echo(&["a"], false),
            &artifact,
            Path::new("."),
            &cases,
            PredictInput::Json(serde_json::Value::Null),
        )
        .await;
        assert_eq!(r, Ok(()));
        // A flavor the predictor does not declare is reported.
        let r = conformance::check(
            &Echo(&["other"], false),
            &artifact,
            Path::new("."),
            &cases,
            PredictInput::Json(serde_json::Value::Null),
        )
        .await;
        assert!(r.is_err());
    }
}
