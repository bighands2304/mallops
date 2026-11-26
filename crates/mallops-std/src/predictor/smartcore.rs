//! Predictor for the native smartcore flavors (`smartcore-logistic`, `smartcore-linear`).
//!
//! Reads the JSON model payload (`model.json`) produced by `std/train.smartcore` and evaluates
//! its documented linear/logistic inference rule in-process without depending on `smartcore`.
//!
//! # Input / output shape
//!
//! - **Input:** [`PredictInput::Json`] containing a JSON object whose keys match the model's
//!   `features` list **exactly** (missing or extra keys are rejected). Each value must be a
//!   finite JSON number or a boolean (`true` = `1.0`, `false` = `0.0`). [`PredictInput::Bytes`]
//!   and non-object JSON values are rejected with [`Error::Runtime`].
//! - **Output (`smartcore-logistic`):** [`PredictInput::Json`] object
//!   `{"prediction": <class>, "probability": <f64>}`, where
//!   `p = 1 / (1 + exp(-(intercept + Σ coefficients[i] * x[i])))`, and `prediction` is
//!   `classes[1]` (with its original JSON type: integer, boolean, or string) when
//!   `p >= threshold`, otherwise `classes[0]`.
//! - **Output (`smartcore-linear`):** [`PredictInput::Json`] object `{"prediction": <f64>}`,
//!   where `y = intercept + Σ coefficients[i] * x[i]`.

use std::path::Path;

use mallops_core::{Artifact, ArtifactKind, BoxFuture, Error};
use serde_json::{Number, Value, json};

use crate::smartcore_model::{
    Algorithm, FLAVOR_LINEAR, FLAVOR_LOGISTIC, MODEL_FILE, MODEL_SCHEMA_VERSION, ModelClasses,
    ModelPayload,
};

use super::{LoadedPredictor, PredictInput, Predictor};

const FLAVORS: &[&str] = &[FLAVOR_LOGISTIC, FLAVOR_LINEAR];

/// Serves `smartcore-logistic` and `smartcore-linear` models in-process.
///
/// # Input / output shape
///
/// - **Input:** [`PredictInput::Json`] object with exactly the keys listed in `model.json`'s
///   `features` array; values must be finite numbers or booleans (`true` = `1.0`, `false` = `0.0`).
/// - **Output (`smartcore-logistic`):** `{"prediction": <class>, "probability": <f64>}` preserving
///   the original JSON type of `classes`.
/// - **Output (`smartcore-linear`):** `{"prediction": <f64>}`.
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
        FLAVORS
    }

    fn executes_code(&self) -> bool {
        false
    }

    fn load<'a>(
        &'a self,
        artifact: &'a Artifact,
        payload_dir: &'a Path,
    ) -> BoxFuture<'a, Result<Box<dyn LoadedPredictor>, Error>> {
        Box::pin(async move {
            let manifest = artifact.manifest();
            if manifest.kind != ArtifactKind::Model {
                return Err(Error::Config(format!(
                    "expected a model artifact, got `{}`",
                    manifest.kind
                )));
            }
            if manifest.format != "json" {
                return Err(Error::Config(format!(
                    "expected model format `json`, got `{}`",
                    manifest.format
                )));
            }
            let flavor = manifest.flavor.as_deref().ok_or_else(|| {
                Error::Config("model artifact manifest is missing `flavor`".into())
            })?;
            if !self.flavors().contains(&flavor) {
                return Err(Error::Config(format!(
                    "unsupported smartcore flavor `{flavor}`"
                )));
            }

            let model_path = payload_dir.join(MODEL_FILE);
            let bytes = std::fs::read(&model_path)
                .map_err(|e| Error::Runtime(format!("{}: {e}", model_path.display())))?;
            let payload: ModelPayload = serde_json::from_slice(&bytes)
                .map_err(|e| Error::Runtime(format!("invalid {MODEL_FILE}: {e}")))?;

            let loaded = LoadedSmartcore::from_payload(flavor, payload)?;
            Ok(Box::new(loaded) as Box<dyn LoadedPredictor>)
        })
    }
}

enum LoadedKind {
    Logistic { classes: [Value; 2], threshold: f64 },
    Linear,
}

struct LoadedSmartcore {
    features: Vec<String>,
    intercept: f64,
    coefficients: Vec<f64>,
    kind: LoadedKind,
}

fn bad(msg: impl Into<String>) -> Error {
    Error::Runtime(msg.into())
}

fn extract_binary_classes(classes: ModelClasses) -> Result<[Value; 2], Error> {
    let values: Vec<Value> = match classes {
        ModelClasses::Int64(v) => v.into_iter().map(Value::from).collect(),
        ModelClasses::Bool(v) => v.into_iter().map(Value::from).collect(),
        ModelClasses::String(v) => v.into_iter().map(Value::from).collect(),
    };
    let len = values.len();
    match <[Value; 2]>::try_from(values) {
        Ok([a, b]) if a != b => Ok([a, b]),
        Ok(_) => Err(bad("logistic_regression model requires 2 distinct classes")),
        Err(_) => Err(bad(format!(
            "logistic_regression model requires exactly 2 classes, got {len}"
        ))),
    }
}

impl LoadedSmartcore {
    fn from_payload(flavor: &str, payload: ModelPayload) -> Result<Self, Error> {
        if payload.v != MODEL_SCHEMA_VERSION {
            return Err(bad(format!(
                "unsupported model schema version {}, expected {MODEL_SCHEMA_VERSION}",
                payload.v
            )));
        }
        match (flavor, payload.algorithm) {
            (FLAVOR_LOGISTIC, Algorithm::LogisticRegression)
            | (FLAVOR_LINEAR, Algorithm::LinearRegression) => {}
            (_, alg) => {
                return Err(bad(format!(
                    "manifest flavor `{flavor}` does not match model algorithm `{alg:?}`"
                )));
            }
        }
        if payload.features.is_empty() {
            return Err(bad("model `features` list must not be empty"));
        }
        let fs = &payload.features;
        if let Some((_, f)) = fs.iter().enumerate().find(|(i, f)| fs[..*i].contains(f)) {
            return Err(bad(format!("duplicate feature `{f}` in model payload")));
        }
        if payload.target.is_empty() {
            return Err(bad("model `target` must not be empty"));
        }
        if payload.coefficients.len() != payload.features.len() {
            return Err(bad(format!(
                "model has {} coefficients for {} features",
                payload.coefficients.len(),
                payload.features.len()
            )));
        }
        if !payload.intercept.is_finite() {
            return Err(bad("model `intercept` must be finite"));
        }
        if payload.coefficients.iter().any(|c| !c.is_finite()) {
            return Err(bad("model `coefficients` must all be finite"));
        }

        let kind = match payload.algorithm {
            Algorithm::LogisticRegression => {
                let classes = extract_binary_classes(
                    payload
                        .classes
                        .ok_or_else(|| bad("logistic_regression model is missing `classes`"))?,
                )?;
                let threshold = payload
                    .threshold
                    .ok_or_else(|| bad("logistic_regression model is missing `threshold`"))?;
                if !threshold.is_finite() {
                    return Err(bad("logistic_regression `threshold` must be finite"));
                }
                LoadedKind::Logistic { classes, threshold }
            }
            Algorithm::LinearRegression => {
                if payload.classes.is_some() {
                    return Err(bad("linear_regression model must not specify `classes`"));
                }
                if payload.threshold.is_some() {
                    return Err(bad("linear_regression model must not specify `threshold`"));
                }
                LoadedKind::Linear
            }
        };

        Ok(Self {
            features: payload.features,
            intercept: payload.intercept,
            coefficients: payload.coefficients,
            kind,
        })
    }

    fn predict_sync(&self, input: PredictInput) -> Result<PredictInput, Error> {
        let map = match input {
            PredictInput::Json(Value::Object(map)) => map,
            PredictInput::Json(_) => {
                return Err(Error::Runtime(
                    "smartcore predictor expects a JSON object of feature values".into(),
                ));
            }
            _ => {
                return Err(Error::Runtime(
                    "smartcore predictor expects JSON input".into(),
                ));
            }
        };

        let mut z = self.intercept;
        for (name, &coef) in self.features.iter().zip(&self.coefficients) {
            let raw = map
                .get(name)
                .ok_or_else(|| Error::Runtime(format!("missing feature `{name}`")))?;
            let x = raw
                .as_f64()
                .or_else(|| raw.as_bool().map(f64::from))
                .filter(|v| v.is_finite())
                .ok_or_else(|| {
                    bad(format!(
                        "feature `{name}` must be a finite number or boolean"
                    ))
                })?;
            z += coef * x;
        }

        if map.len() != self.features.len() {
            let extra = map
                .keys()
                .find(|k| !self.features.iter().any(|f| f == *k))
                .map_or("<unknown>", String::as_str);
            return Err(Error::Runtime(format!("unexpected feature `{extra}`")));
        }

        match &self.kind {
            LoadedKind::Logistic { classes, threshold } => {
                let p = 1.0 / (1.0 + (-z).exp());
                let prob = Number::from_f64(p).ok_or_else(|| {
                    Error::Runtime("prediction produced non-finite probability".into())
                })?;
                let pred = if p >= *threshold {
                    classes[1].clone()
                } else {
                    classes[0].clone()
                };
                Ok(PredictInput::Json(json!({
                    "prediction": pred,
                    "probability": prob,
                })))
            }
            LoadedKind::Linear => {
                let pred = Number::from_f64(z)
                    .ok_or_else(|| Error::Runtime("prediction produced non-finite value".into()))?;
                Ok(PredictInput::Json(json!({
                    "prediction": pred,
                })))
            }
        }
    }
}

impl LoadedPredictor for LoadedSmartcore {
    fn predict(&self, input: PredictInput) -> BoxFuture<'_, Result<PredictInput, Error>> {
        Box::pin(async move { self.predict_sync(input) })
    }

    fn predict_batch(
        &self,
        inputs: Vec<PredictInput>,
    ) -> BoxFuture<'_, Result<Vec<PredictInput>, Error>> {
        Box::pin(async move { inputs.into_iter().map(|i| self.predict_sync(i)).collect() })
    }
}

#[cfg(test)]
mod tests {
    use mallops_core::{ArtifactId, Manifest};

    use super::*;

    fn model_artifact(flavor: Option<&str>) -> Artifact {
        let mut manifest = Manifest::new(ArtifactKind::Model, "json");
        if let Some(f) = flavor {
            manifest = manifest.with_flavor(f);
        }
        Artifact::new(ArtifactId::from_bytes([0; 32]), manifest)
    }

    fn write_model(json_value: &Value) -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join(MODEL_FILE),
            serde_json::to_vec(json_value).unwrap(),
        )
        .unwrap();
        dir
    }

    fn valid_logistic_json() -> Value {
        json!({
            "v": 1,
            "algorithm": "logistic_regression",
            "features": ["x1", "flag"],
            "target": "y",
            "intercept": -0.5,
            "coefficients": [1.0, 0.5],
            "classes": [0, 1],
            "threshold": 0.5
        })
    }

    fn valid_linear_json() -> Value {
        json!({
            "v": 1,
            "algorithm": "linear_regression",
            "features": ["x1", "flag"],
            "target": "y",
            "intercept": 1.0,
            "coefficients": [2.0, -3.0]
        })
    }

    /// Mutates a valid payload (`true` = logistic, else linear); the load must fail with Runtime.
    type Mutate = fn(&mut Value);

    #[tokio::test]
    async fn smartcore_predictor_load_validates_manifest_and_payload() {
        let predictor = SmartcorePredictor::new();
        assert!(!predictor.executes_code());
        assert_eq!(predictor.flavors(), &[FLAVOR_LOGISTIC, FLAVOR_LINEAR]);
        let log_art = model_artifact(Some(FLAVOR_LOGISTIC));
        let lin_art = model_artifact(Some(FLAVOR_LINEAR));
        let dir = write_model(&valid_logistic_json());

        // Manifest problems: Error::Config.
        let manifest_cases = [
            Manifest::new(ArtifactKind::Features, "json").with_flavor(FLAVOR_LOGISTIC),
            Manifest::new(ArtifactKind::Model, "parquet").with_flavor(FLAVOR_LOGISTIC),
            Manifest::new(ArtifactKind::Model, "json"),
            Manifest::new(ArtifactKind::Model, "json").with_flavor("sklearn-pickle"),
        ];
        for manifest in manifest_cases {
            let art = Artifact::new(ArtifactId::from_bytes([0; 32]), manifest.clone());
            let r = predictor.load(&art, dir.path()).await.map(drop);
            assert!(matches!(r, Err(Error::Config(_))), "{manifest:?}: {r:?}");
        }

        // Missing model.json
        let empty_dir = tempfile::tempdir().unwrap();
        let r = predictor.load(&log_art, empty_dir.path()).await.map(drop);
        assert!(matches!(r, Err(Error::Runtime(_))), "{r:?}");

        // Payload problems: (logistic?, mutation).
        let payload_cases: [(bool, Mutate); 17] = [
            (true, |m| m["extra_field"] = json!(123)),
            (true, |m| m["v"] = json!(2)),
            (false, |m| {
                m["features"] = json!([]);
                m["coefficients"] = json!([]);
            }),
            (false, |m| m["features"] = json!(["x1", "x1"])),
            (false, |m| m["features"] = json!(["a", "b", "a"])),
            (false, |m| m["target"] = json!("")),
            (false, |m| m["coefficients"] = json!([2.0])),
            (true, |m| _ = m.as_object_mut().unwrap().remove("classes")),
            (true, |m| m["classes"] = json!([])),
            (true, |m| m["classes"] = json!([0])),
            (true, |m| m["classes"] = json!([0, 1, 2])),
            (true, |m| m["classes"] = json!([1, 1])),
            (true, |m| m["classes"] = json!(["a", "a"])),
            (true, |m| _ = m.as_object_mut().unwrap().remove("threshold")),
            (false, |m| m["classes"] = json!([0, 1])),
            (false, |m| m["threshold"] = json!(0.5)),
            (true, |m| m["algorithm"] = json!("linear_regression")),
        ];
        for (i, (logistic, mutate)) in payload_cases.into_iter().enumerate() {
            let mut model = if logistic {
                valid_logistic_json()
            } else {
                valid_linear_json()
            };
            mutate(&mut model);
            let dir = write_model(&model);
            let art = if logistic { &log_art } else { &lin_art };
            let r = predictor.load(art, dir.path()).await.map(drop);
            assert!(matches!(r, Err(Error::Runtime(_))), "case {i}: {r:?}");
        }

        // Flavor / algorithm mismatch (linear model under the logistic flavor).
        let dir_lin = write_model(&valid_linear_json());
        let r = predictor.load(&log_art, dir_lin.path()).await.map(drop);
        assert!(matches!(r, Err(Error::Runtime(_))), "{r:?}");

        // Non-finite intercept, coefficient, and threshold (via from_payload).
        let base = ModelPayload {
            v: MODEL_SCHEMA_VERSION,
            algorithm: Algorithm::LogisticRegression,
            features: vec!["x".into()],
            target: "y".into(),
            intercept: 0.0,
            coefficients: vec![1.0],
            classes: Some(ModelClasses::Int64(vec![0, 1])),
            threshold: Some(0.5),
        };
        let non_finite: [fn(&mut ModelPayload); 3] = [
            |p| p.intercept = f64::NAN,
            |p| p.coefficients = vec![f64::INFINITY],
            |p| p.threshold = Some(f64::NEG_INFINITY),
        ];
        for mutate in non_finite {
            let mut p = base.clone();
            mutate(&mut p);
            let r = LoadedSmartcore::from_payload(FLAVOR_LOGISTIC, p);
            assert!(matches!(r, Err(Error::Runtime(_))));
        }
    }

    #[tokio::test]
    async fn smartcore_predictor_predict_logistic_and_linear_hand_calculated() {
        let predictor = SmartcorePredictor::new();

        // Logistic: intercept = -0.5, coefs = [1.0, 0.5], threshold = 0.5
        // Input {x1: 0.0, flag: true} -> z = -0.5 + 0 + 0.5 = 0.0 -> p = 0.5 >= 0.5 -> classes[1]
        // Input {x1: -1.0, flag: false} -> z = -1.5 -> p = 1/(1+exp(1.5)) < 0.5 -> classes[0]
        for (classes, c0, c1) in [
            (json!([0, 1]), json!(0), json!(1)),
            (json!([false, true]), json!(false), json!(true)),
            (json!(["no", "yes"]), json!("no"), json!("yes")),
        ] {
            let mut model = valid_logistic_json();
            model["classes"] = classes;
            let dir = write_model(&model);
            let loaded = predictor
                .load(&model_artifact(Some(FLAVOR_LOGISTIC)), dir.path())
                .await
                .unwrap();

            let p_pos = 1.0 / (1.0 + (-0.0_f64).exp());
            let p_neg = 1.0 / (1.0 + 1.5_f64.exp());

            let in_pos = PredictInput::Json(json!({ "x1": 0.0, "flag": true }));
            let in_neg = PredictInput::Json(json!({ "x1": -1.0, "flag": false }));
            let exp_pos = PredictInput::Json(json!({ "prediction": c1, "probability": p_pos }));
            let exp_neg = PredictInput::Json(json!({ "prediction": c0, "probability": p_neg }));

            assert_eq!(loaded.predict(in_pos.clone()).await.unwrap(), exp_pos);
            assert_eq!(loaded.predict(in_neg.clone()).await.unwrap(), exp_neg);
            assert_eq!(
                loaded.predict_batch(vec![in_neg, in_pos]).await.unwrap(),
                vec![exp_neg, exp_pos]
            );
            assert_eq!(loaded.predict_batch(Vec::new()).await.unwrap(), Vec::new());
        }

        // Linear: y = 1.0 + 2.0 * x1 - 3.0 * flag
        let dir = write_model(&valid_linear_json());
        let loaded = predictor
            .load(&model_artifact(Some(FLAVOR_LINEAR)), dir.path())
            .await
            .unwrap();
        let out = loaded
            .predict(PredictInput::Json(json!({ "x1": 2.5, "flag": true })))
            .await
            .unwrap();
        assert_eq!(out, PredictInput::Json(json!({ "prediction": 3.0 })));

        // Malformed inputs -> Error::Runtime
        let bad_inputs = [
            PredictInput::Bytes {
                content_type: "application/octet-stream".into(),
                data: vec![1, 2, 3],
            },
            PredictInput::Json(json!([1.0, 2.0])),
            PredictInput::Json(json!(null)),
            PredictInput::Json(json!({ "x1": 1.0 })), // missing flag
            PredictInput::Json(json!({ "x1": 1.0, "flag": true, "extra": 0 })), // extra key
            PredictInput::Json(json!({ "x1": "not-a-number", "flag": true })), // string value
            PredictInput::Json(json!({ "x1": null, "flag": true })), // null value
            PredictInput::Json(json!({ "x1": 1e308, "flag": false })), // overflow -> non-finite
        ];
        for bad in bad_inputs {
            let err = loaded.predict(bad.clone()).await;
            assert!(
                matches!(err, Err(Error::Runtime(_))),
                "expected Error::Runtime for {bad:?}, got {err:?}"
            );
        }
    }

    #[cfg(feature = "train-smartcore")]
    mod conformance_tests {
        use std::sync::Arc;

        use arrow_array::{ArrayRef, Float64Array, Int64Array, RecordBatch};
        use mallops_core::{Artifact, ArtifactId, DraftPayload, MemoryStore};
        use serde_json::json;

        use crate::predictor::{PredictInput, Predictor, SmartcorePredictor, conformance};
        use crate::train::smartcore::MODEL;
        use crate::train::smartcore::tests::{
            read_model_json, run_block, separable_logistic_batch_int64, store_features,
        };

        #[tokio::test]
        async fn smartcore_predictor_conformance_logistic() {
            let store = Arc::new(MemoryStore::new());
            let batch = separable_logistic_batch_int64();
            let inputs = store_features(&store, &batch, Some("y")).await;
            let (_workdir, drafts) = run_block(
                &store,
                &inputs,
                json!({ "algorithm": "logistic_regression" }),
            )
            .await
            .unwrap();
            let draft = drafts.get(MODEL).unwrap();
            let DraftPayload::File(model_path) = &draft.payload else {
                panic!("expected DraftPayload::File");
            };
            let payload_dir = model_path.parent().unwrap();
            let artifact = Artifact::new(ArtifactId::from_bytes([1; 32]), draft.manifest.clone());

            let model = read_model_json(draft);
            let intercept = model["intercept"].as_f64().unwrap();
            let c0 = model["coefficients"][0].as_f64().unwrap();
            let c1 = model["coefficients"][1].as_f64().unwrap();

            let prob = |x1: f64, x2: f64| 1.0 / (1.0 + (-(intercept + c0 * x1 + c1 * x2)).exp());
            let p_low = prob(-4.0, 0.0);
            let p_high = prob(4.0, 1.0);
            assert!(p_low < 0.5, "expected class 0 for x1=-4.0, got p={p_low}");
            assert!(p_high >= 0.5, "expected class 1 for x1=4.0, got p={p_high}");

            let cases = [
                (
                    PredictInput::Json(json!({ "x1": -4.0, "x2": 0 })),
                    PredictInput::Json(json!({ "prediction": 0, "probability": p_low })),
                ),
                (
                    PredictInput::Json(json!({ "x1": 4.0, "x2": 1 })),
                    PredictInput::Json(json!({ "prediction": 1, "probability": p_high })),
                ),
            ];
            let malformed = PredictInput::Json(json!({ "x1": 1.0 }));

            let predictor = SmartcorePredictor::new();
            assert!(!predictor.executes_code());
            conformance::check(&predictor, &artifact, payload_dir, &cases, malformed)
                .await
                .unwrap();
        }

        #[tokio::test]
        async fn smartcore_predictor_conformance_linear() {
            let store = Arc::new(MemoryStore::new());
            // Exact linear relation: y = 2*x1 - 3*x2 + 1
            let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 2.0];
            let x2 = vec![0.0, 1.0, 0.0, 2.0, 1.0, 3.0];
            let y: Vec<i64> = x1
                .iter()
                .zip(&x2)
                .map(|(&a, &b)| (2.0 * a - 3.0 * b + 1.0) as i64)
                .collect();
            let batch = RecordBatch::try_from_iter([
                ("x1", Arc::new(Float64Array::from(x1)) as ArrayRef),
                ("x2", Arc::new(Float64Array::from(x2)) as _),
                ("y", Arc::new(Int64Array::from(y)) as _),
            ])
            .unwrap();
            let inputs = store_features(&store, &batch, Some("y")).await;
            let (_workdir, drafts) =
                run_block(&store, &inputs, json!({ "algorithm": "linear_regression" }))
                    .await
                    .unwrap();
            let draft = drafts.get(MODEL).unwrap();
            let DraftPayload::File(model_path) = &draft.payload else {
                panic!("expected DraftPayload::File");
            };
            let payload_dir = model_path.parent().unwrap();
            let artifact = Artifact::new(ArtifactId::from_bytes([2; 32]), draft.manifest.clone());

            let model = read_model_json(draft);
            let intercept = model["intercept"].as_f64().unwrap();
            let c0 = model["coefficients"][0].as_f64().unwrap();
            let c1 = model["coefficients"][1].as_f64().unwrap();

            let y_a = intercept + c0 * 3.0 + c1 * 1.0; // true: 2*3 - 3*1 + 1 = 4.0
            let y_b = intercept + c0 * 0.0 + c1 * 2.0; // true: 2*0 - 3*2 + 1 = -5.0
            assert!((y_a - 4.0).abs() < 1e-6, "y_a={y_a}");
            assert!((y_b - (-5.0)).abs() < 1e-6, "y_b={y_b}");

            let cases = [
                (
                    PredictInput::Json(json!({ "x1": 3.0, "x2": 1.0 })),
                    PredictInput::Json(json!({ "prediction": y_a })),
                ),
                (
                    PredictInput::Json(json!({ "x1": 0.0, "x2": 2.0 })),
                    PredictInput::Json(json!({ "prediction": y_b })),
                ),
            ];
            let malformed = PredictInput::Json(json!({ "x1": "bad", "x2": 1.0 }));

            let predictor = SmartcorePredictor::new();
            assert!(!predictor.executes_code());
            conformance::check(&predictor, &artifact, payload_dir, &cases, malformed)
                .await
                .unwrap();
        }
    }
}
