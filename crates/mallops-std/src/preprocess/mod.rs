//! `std/preprocess.tabular`: tabular transforms on a Parquet `dataset` into feature splits plus a
//! fitted, serializable `transform` artifact (FR-PRE-1, FR-PRE-3).
//!
//! Steps run in declared order. `split` appears exactly once; only stateless steps (`select`,
//! `filter`) may precede it, so stateful steps (`impute`, `scale`, `encode`) are always fitted on
//! the training partition only (no leakage) and applied to every partition.

pub(crate) mod steps;
pub mod transform;

use mallops_core::{
    ArtifactKind, Block, BlockOutcome, BlockSpec, BoxFuture, Context, Error, Inputs, Port,
    ResolvedPorts, Stage,
};
use serde::Deserialize;

use crate::ingest::parse_config;

/// Input port: the dataset to transform.
pub(crate) const INPUT: &str = "data";
/// Output port: training features.
pub(crate) const TRAIN: &str = "train";
/// Output port: validation features (only when `split.validation` is set).
pub(crate) const VALIDATION: &str = "validation";
/// Output port: test features.
pub(crate) const TEST: &str = "test";
/// Output port: fitted transform.
pub(crate) const TRANSFORM: &str = "transform";

/// `with:` config of `std/preprocess.tabular`.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct PreprocessConfig {
    /// Target column: kept by `select`, never transformed, recorded in `metadata["target"]`.
    pub(crate) target: String,
    /// Steps in execution order.
    pub(crate) steps: Vec<Step>,
}

/// One preprocessing step (`- <name>: { … }` in YAML).
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum Step {
    /// Keep only these feature columns (the target is always kept).
    Select(SelectStep),
    /// Keep rows matching a predicate (not replayed at serving time).
    Filter(FilterStep),
    /// Replace nulls with a value fitted on the training partition.
    Impute(ImputeStep),
    /// Scale numeric columns with parameters fitted on the training partition.
    Scale(ScaleStep),
    /// Encode string columns with categories fitted on the training partition.
    Encode(EncodeStep),
    /// Seeded row split into train / validation / test.
    Split(SplitStep),
}

impl Step {
    fn name(&self) -> &'static str {
        match self {
            Self::Select(_) => "select",
            Self::Filter(_) => "filter",
            Self::Impute(_) => "impute",
            Self::Scale(_) => "scale",
            Self::Encode(_) => "encode",
            Self::Split(_) => "split",
        }
    }

    fn is_fitted(&self) -> bool {
        matches!(self, Self::Impute(_) | Self::Scale(_) | Self::Encode(_))
    }
}

/// `select: { columns: [..] }`.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct SelectStep {
    pub(crate) columns: Vec<String>,
}

/// `filter: { column, op, value? }`.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct FilterStep {
    pub(crate) column: String,
    pub(crate) op: FilterOp,
    /// Required for comparisons, forbidden for `is_null` / `not_null`.
    pub(crate) value: Option<serde_json::Value>,
}

/// Comparison of a [`FilterStep`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum FilterOp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    IsNull,
    NotNull,
}

/// `impute: { strategy, value?, columns? }`; default columns: numeric non-target.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ImputeStep {
    pub(crate) strategy: ImputeStrategy,
    /// Required for `constant`, forbidden otherwise.
    pub(crate) value: Option<serde_json::Value>,
    pub(crate) columns: Option<Vec<String>>,
}

/// Strategy of an [`ImputeStep`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ImputeStrategy {
    Mean,
    Median,
    Constant,
}

/// `scale: { method, columns? }`; default columns: numeric non-target.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ScaleStep {
    pub(crate) method: ScaleMethod,
    pub(crate) columns: Option<Vec<String>>,
}

/// Method of a [`ScaleStep`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ScaleMethod {
    Standard,
    Minmax,
}

/// `encode: { method, columns? }`; default columns: string non-target.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct EncodeStep {
    pub(crate) method: EncodeMethod,
    pub(crate) columns: Option<Vec<String>>,
}

/// Method of an [`EncodeStep`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum EncodeMethod {
    Ordinal,
    OneHot,
}

/// `split: { test, validation?, seed }`.
#[derive(Debug, Clone, Copy, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct SplitStep {
    /// Fraction of rows in `test`, in `(0, 1)`.
    pub(crate) test: f64,
    /// Fraction of rows in `validation`, in `(0, 1)`; absent = no `validation` output.
    pub(crate) validation: Option<f64>,
    pub(crate) seed: u64,
}

impl PreprocessConfig {
    /// Parses and checks the config; every error is an [`Error::Config`].
    pub(crate) fn parse(config: &serde_json::Value) -> Result<Self, Error> {
        let cfg: Self = parse_config(config)?;
        cfg.check()?;
        Ok(cfg)
    }

    /// The single `split` step (present after [`Self::parse`]).
    pub(crate) fn split(&self) -> Option<SplitStep> {
        self.steps.iter().find_map(|s| match s {
            Step::Split(s) => Some(*s),
            _ => None,
        })
    }

    fn check(&self) -> Result<(), Error> {
        let err = |i: usize, msg: String| Err(Error::Config(format!("steps[{i}]: {msg}")));
        if self.target.is_empty() {
            return Err(Error::Config("target must not be empty".into()));
        }
        let mut split_at = None;
        for (i, step) in self.steps.iter().enumerate() {
            match step {
                Step::Split(s) => {
                    if split_at.is_some() {
                        return err(i, "`split` must appear exactly once".into());
                    }
                    split_at = Some(i);
                    let ratio_ok = |r: f64| r > 0.0 && r < 1.0;
                    if !ratio_ok(s.test) {
                        return err(i, format!("split.test {} must be in (0, 1)", s.test));
                    }
                    if let Some(v) = s.validation {
                        if !ratio_ok(v) || s.test + v >= 1.0 {
                            return err(
                                i,
                                format!(
                                    "split.validation {v} must be in (0, 1) with test + validation < 1"
                                ),
                            );
                        }
                    }
                }
                s if s.is_fitted() && split_at.is_none() => {
                    return err(
                        i,
                        format!(
                            "`{}` must come after `split`: fitted steps are fitted on the training split only",
                            s.name()
                        ),
                    );
                }
                Step::Select(s) if s.columns.is_empty() => {
                    return err(i, "select.columns must not be empty".into());
                }
                Step::Filter(f) => {
                    let unary = matches!(f.op, FilterOp::IsNull | FilterOp::NotNull);
                    match (&f.value, unary) {
                        (Some(_), true) => {
                            return err(
                                i,
                                "filter.value is not allowed with is_null/not_null".into(),
                            );
                        }
                        (None, false) => return err(i, "filter.value is required".into()),
                        (Some(v), false) if !(v.is_number() || v.is_string() || v.is_boolean()) => {
                            return err(i, "filter.value must be a number, string or bool".into());
                        }
                        _ => {}
                    }
                }
                Step::Impute(s) => match (s.strategy, &s.value) {
                    (ImputeStrategy::Constant, None) => {
                        return err(i, "impute.value is required with strategy constant".into());
                    }
                    (ImputeStrategy::Constant, Some(v)) if !(v.is_number() || v.is_string()) => {
                        return err(i, "impute.value must be a number or string".into());
                    }
                    (ImputeStrategy::Mean | ImputeStrategy::Median, Some(_)) => {
                        return err(
                            i,
                            "impute.value is only allowed with strategy constant".into(),
                        );
                    }
                    _ => {}
                },
                _ => {}
            }
        }
        if split_at.is_none() {
            return Err(Error::Config("steps: a `split` step is required".into()));
        }
        Ok(())
    }
}

/// Features output port (`kind: features`, `format: parquet`).
fn features_port(name: &'static str) -> Port {
    Port::new(name, ArtifactKind::Features).with_format("parquet")
}

/// `std/preprocess.tabular` block.
pub(crate) struct PreprocessTabular(BlockSpec);

impl PreprocessTabular {
    pub(crate) fn new() -> Self {
        Self(
            BlockSpec::new("std/preprocess.tabular", "0.1.0", Stage::Preprocess)
                .with_input(Port::new(INPUT, ArtifactKind::Dataset).with_format("parquet"))
                .with_output(features_port(TRAIN))
                .with_output(features_port(TEST))
                .with_output(Port::new(TRANSFORM, ArtifactKind::Transform).with_format("json")),
        )
    }
}

impl Block for PreprocessTabular {
    fn spec(&self) -> &BlockSpec {
        &self.0
    }

    fn validate_config(&self, config: &serde_json::Value) -> Result<(), Error> {
        PreprocessConfig::parse(config).map(|_| ())
    }

    fn resolve_ports(&self, config: &serde_json::Value) -> Result<ResolvedPorts, Error> {
        let cfg = PreprocessConfig::parse(config)?;
        let mut ports = ResolvedPorts::from(self.spec());
        if cfg.split().and_then(|s| s.validation).is_some() {
            ports.outputs.push(features_port(VALIDATION));
        }
        Ok(ports)
    }

    fn run<'a>(
        &'a self,
        ctx: &'a Context,
        _inputs: Inputs,
    ) -> BoxFuture<'a, Result<BlockOutcome, Error>> {
        // ponytail: stub until 12.2 (read, split, fit on train, write outputs).
        Box::pin(async move {
            let _cfg = PreprocessConfig::parse(ctx.config())?;
            Err(Error::Runtime(
                "std/preprocess.tabular: not implemented yet".into(),
            ))
        })
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn cfg(steps: serde_json::Value) -> Result<PreprocessConfig, Error> {
        PreprocessConfig::parse(&json!({ "target": "y", "steps": steps }))
    }

    fn config_err(steps: serde_json::Value) -> String {
        match cfg(steps) {
            Err(Error::Config(m)) => m,
            other => panic!("expected Error::Config, got {other:?}"),
        }
    }

    #[test]
    fn preprocess_steps_config_parses_every_step() {
        let c = cfg(json!([
            { "select": { "columns": ["a", "b"] } },
            { "filter": { "column": "a", "op": "gt", "value": 0 } },
            { "filter": { "column": "b", "op": "not_null" } },
            { "split": { "test": 0.2, "validation": 0.1, "seed": 42 } },
            { "impute": { "strategy": "median" } },
            { "impute": { "strategy": "constant", "value": "n/a", "columns": ["b"] } },
            { "scale": { "method": "minmax" } },
            { "encode": { "method": "one_hot", "columns": ["b"] } },
        ]))
        .unwrap();
        assert_eq!(c.steps.len(), 8);
        assert_eq!(
            c.split(),
            Some(SplitStep {
                test: 0.2,
                validation: Some(0.1),
                seed: 42
            })
        );
    }

    #[test]
    fn preprocess_steps_config_rejects_invalid() {
        let split = json!({ "split": { "test": 0.2, "seed": 1 } });
        for (steps, needle) in [
            (json!([]), "`split` step is required"),
            (json!([split, split]), "exactly once"),
            (
                json!([{ "scale": { "method": "standard" } }, split]),
                "after `split`",
            ),
            (
                json!([{ "split": { "test": 1.0, "seed": 1 } }]),
                "split.test",
            ),
            (
                json!([{ "split": { "test": 0.5, "validation": 0.5, "seed": 1 } }]),
                "split.validation",
            ),
            (
                json!([split, { "impute": { "strategy": "constant" } }]),
                "required with strategy constant",
            ),
            (
                json!([split, { "impute": { "strategy": "mean", "value": 1 } }]),
                "only allowed",
            ),
            (
                json!([{ "filter": { "column": "a", "op": "eq" } }, split]),
                "filter.value is required",
            ),
            (
                json!([{ "filter": { "column": "a", "op": "is_null", "value": 1 } }, split]),
                "not allowed",
            ),
            (
                json!([{ "select": { "columns": [] } }, split]),
                "must not be empty",
            ),
            (
                json!([split, { "scale": { "method": "robust" } }]),
                "unknown variant",
            ),
            (
                json!([split, { "encode": { "method": "ordinal", "extra": 1 } }]),
                "unknown field",
            ),
        ] {
            let msg = config_err(steps.clone());
            assert!(msg.contains(needle), "{steps}: {msg}");
        }
        assert!(matches!(
            PreprocessConfig::parse(&json!({ "steps": [split] })),
            Err(Error::Config(_))
        ));
    }

    #[test]
    fn preprocess_validation_split_port_only_when_configured() {
        let block = PreprocessTabular::new();
        let outputs = |split: serde_json::Value| -> Vec<String> {
            let c = json!({ "target": "y", "steps": [{ "split": split }] });
            block
                .resolve_ports(&c)
                .unwrap()
                .outputs
                .iter()
                .map(|p| p.name.to_string())
                .collect()
        };
        assert_eq!(
            outputs(json!({ "test": 0.2, "seed": 1 })),
            [TRAIN, TEST, TRANSFORM]
        );
        assert_eq!(
            outputs(json!({ "test": 0.2, "validation": 0.1, "seed": 1 })),
            [TRAIN, TEST, TRANSFORM, VALIDATION]
        );
    }
}
