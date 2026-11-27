//! `std/validate.tabular`: check a Parquet dataset against declared expectations (schema, row
//! count, per-column null fraction / range / allowed values / uniqueness) and gate the pipeline.
//!
//! The `data` input passes through unchanged (same payload and manifest; the runtime gives it a
//! new `ArtifactId` because `parents` = the input, ADR-0007 D3); the `report` output is a
//! `report.json` listing, for every check, `passed`, `observed` and `expected`.
//!
//! Every problem *in the data* is a failed check, never an error; only a bad `with:` or an
//! unreadable input is an error. The gate (`on_fail`) is `fail` (default, [`Error::Validation`]
//! with one entry per failed check) or `skip` (downstream nodes are skipped, report kept).
//!
//! Limits (v0.1): the whole dataset is read into memory; the report has no row-level detail
//! beyond up to 5 sorted example values for `allowed` / `unique`.

use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};

use arrow_array::cast::AsArray;
use arrow_array::types::{Date32Type, Float64Type, Int64Type};
use arrow_array::{Array, ArrayRef, RecordBatch};
use arrow_schema::DataType;
use mallops_core::{
    ArtifactKind, Block, BlockOutcome, BlockSpec, BoxFuture, Context, DraftPayload, Error, Inputs,
    Manifest, OutputDraft, OutputDrafts, Port, Stage, ValidationError,
};
use serde::{Deserialize, Serialize};
use serde_json::{Number, Value, json};

use crate::has_secret_ref;
use crate::ingest::{Column, DATA_FILE, blocking, parse_config, read_parquet_batch};

const DATA: &str = "data";
const REPORT: &str = "report";
/// File name of the `report` payload.
const REPORT_FILE: &str = "report.json";
/// Max example values listed per `allowed` / `unique` check.
const EXAMPLES: usize = 5;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
enum OnFail {
    #[default]
    Fail,
    Skip,
}

/// Checks of one column; at least one must be active.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ColumnChecks {
    max_null_fraction: Option<f64>,
    min: Option<Number>,
    max: Option<Number>,
    allowed: Option<Vec<Value>>,
    #[serde(default)]
    unique: bool,
}

const fn yes() -> bool {
    true
}

/// Validated `with:` configuration.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Config {
    schema: Option<Vec<Column>>,
    #[serde(default = "yes")]
    allow_extra_columns: bool,
    min_rows: Option<u64>,
    #[serde(default)]
    checks: BTreeMap<String, ColumnChecks>,
    #[serde(default)]
    on_fail: OnFail,
}

impl Config {
    fn parse(config: &Value) -> Result<Self, Error> {
        if has_secret_ref(config) {
            return Err(Error::Config(
                "`${secret:...}` is not supported in `std/validate.tabular` config".into(),
            ));
        }
        let cfg: Self = parse_config(config)?;
        let err = |m: String| Err(Error::Config(m));
        if cfg.schema.is_none() && !cfg.allow_extra_columns {
            return err("`allow_extra_columns: false` needs a `schema`".into());
        }
        for (col, c) in &cfg.checks {
            if c.max_null_fraction.is_none()
                && c.min.is_none()
                && c.max.is_none()
                && c.allowed.is_none()
                && !c.unique
            {
                return err(format!("`checks.{col}` has no active check"));
            }
            if c.max_null_fraction
                .is_some_and(|f| !(0.0..=1.0).contains(&f))
            {
                return err(format!(
                    "`checks.{col}.max_null_fraction` must be in [0, 1]"
                ));
            }
            if let (Some(lo), Some(hi)) = (&c.min, &c.max)
                && lo
                    .as_i64()
                    .zip(hi.as_i64())
                    .map_or_else(|| lo.as_f64() > hi.as_f64(), |(a, b)| a > b)
            {
                return err(format!("`checks.{col}.min` must not exceed `max`"));
            }
            if let Some(al) = &c.allowed
                && (al.is_empty()
                    || !al
                        .iter()
                        .all(|v| v.is_string() || v.is_i64() || v.is_boolean()))
            {
                return err(format!(
                    "`checks.{col}.allowed` must be a non-empty list of strings, integers or booleans"
                ));
            }
        }
        Ok(cfg)
    }
}

/// Result of one check, as serialized in `report.json`.
#[derive(Debug, Serialize)]
struct Check {
    check: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    column: Option<String>,
    passed: bool,
    observed: Value,
    expected: Value,
    /// Gate errors of a failed check; paths and counts only, never data values.
    #[serde(skip)]
    errors: Vec<ValidationError>,
}

impl Check {
    fn new(
        check: &'static str,
        column: Option<&str>,
        passed: bool,
        observed: Value,
        expected: Value,
    ) -> Self {
        let mut errors = Vec::new();
        if !passed {
            let path = column.map_or_else(|| check.to_owned(), |c| format!("checks.{c}.{check}"));
            let message = observed.get("count").and_then(Value::as_u64).map_or_else(
                || "check failed".to_owned(),
                |n| format!("{n} violating row(s)"),
            );
            errors.push(ValidationError::new(path, message));
        }
        Self {
            check,
            column: column.map(str::to_owned),
            passed,
            observed,
            expected,
            errors,
        }
    }
}

#[derive(Serialize)]
struct Report<'a> {
    v: u32,
    passed: bool,
    checks: &'a [Check],
}

/// Finite floats as JSON numbers; NaN / inf / -inf as strings (JSON has no such numbers).
fn fnum(x: f64) -> Value {
    if x.is_finite() {
        json!(x)
    } else {
        json!(x.to_string()) // "NaN" / "inf" / "-inf"
    }
}

/// `(passed, {count, examples})` of the values flagged by `bad`; examples are the first
/// [`EXAMPLES`] distinct flagged values in sorted order.
fn tally<T: Ord>(
    vals: impl Iterator<Item = Option<T>>,
    mut bad: impl FnMut(&T) -> bool,
    js: impl Fn(&T) -> Value,
) -> (bool, Value) {
    let (mut count, mut distinct) = (0u64, BTreeSet::new());
    for v in vals.flatten() {
        if bad(&v) {
            count += 1;
            distinct.insert(v);
        }
    }
    let examples: Vec<Value> = distinct.iter().take(EXAMPLES).map(js).collect();
    (count == 0, json!({ "count": count, "examples": examples }))
}

/// Total-order, involutive i64 key of a float (`-0.0` = `0.0`, one NaN), so floats sort and dedup
/// in a `BTreeSet`.
fn float_key(x: f64) -> i64 {
    let x = if x == 0.0 {
        0.0
    } else if x.is_nan() {
        f64::NAN
    } else {
        x
    };
    let b = x.to_bits() as i64;
    b ^ (((b >> 63) as u64) >> 1) as i64
}

fn float_from_key(k: i64) -> f64 {
    f64::from_bits((k ^ (((k >> 63) as u64) >> 1) as i64) as u64)
}

/// `min` / `max` check of one column: `(passed, observed extremum)`. NaN always fails; nulls are
/// ignored and an all-null column passes.
fn bound(a: &ArrayRef, b: &Number, is_min: bool) -> (bool, Value) {
    let want = b.as_f64().unwrap_or(f64::NAN);
    let ok_f = |x: f64| if is_min { x >= want } else { x <= want };
    match a.data_type() {
        DataType::Int64 => {
            let vals = a.as_primitive::<Int64Type>().iter().flatten();
            let Some(x) = (if is_min { vals.min() } else { vals.max() }) else {
                return (true, Value::Null);
            };
            let ord = match b.as_i64() {
                Some(bi) => x.cmp(&bi),
                None if b.is_u64() => Ordering::Less, // bound above i64::MAX
                None => (x as f64).total_cmp(&want),
            };
            let passed = if is_min { ord.is_ge() } else { ord.is_le() };
            (passed, json!(x))
        }
        DataType::Float64 => {
            let vals: Vec<f64> = a.as_primitive::<Float64Type>().iter().flatten().collect();
            if vals.iter().any(|v| v.is_nan()) {
                return (false, json!("NaN"));
            }
            let ext = vals
                .into_iter()
                .reduce(|p, q| if is_min == (q < p) { q } else { p });
            ext.map_or((true, Value::Null), |x| (ok_f(x), fnum(x)))
        }
        dt => (
            false,
            json!(format!("non-numeric column of type {}", type_name(dt))),
        ),
    }
}

/// `allowed` check: only string / int64 / bool columns.
fn allowed(a: &ArrayRef, list: &[Value]) -> (bool, Value) {
    match a.data_type() {
        DataType::Utf8 => {
            let ok: BTreeSet<&str> = list.iter().filter_map(Value::as_str).collect();
            tally(
                a.as_string::<i32>().iter(),
                |v| !ok.contains(v),
                |v| json!(v),
            )
        }
        DataType::Int64 => {
            let ok: BTreeSet<i64> = list.iter().filter_map(Value::as_i64).collect();
            tally(
                a.as_primitive::<Int64Type>().iter(),
                |v| !ok.contains(v),
                |v| json!(v),
            )
        }
        DataType::Boolean => {
            let ok: BTreeSet<bool> = list.iter().filter_map(Value::as_bool).collect();
            tally(a.as_boolean().iter(), |v| !ok.contains(v), |v| json!(v))
        }
        dt => (
            false,
            json!(format!("`allowed` is not supported on {}", type_name(dt))),
        ),
    }
}

/// `unique` check: `count` is the number of rows repeating an earlier value.
fn unique(a: &ArrayRef) -> (bool, Value) {
    fn dups<T: Ord + Copy>(
        it: impl Iterator<Item = Option<T>>,
        js: impl Fn(&T) -> Value,
    ) -> (bool, Value) {
        let mut seen = BTreeSet::new();
        tally(it, |v| !seen.insert(*v), js)
    }
    match a.data_type() {
        DataType::Utf8 => dups(a.as_string::<i32>().iter(), |v| json!(v)),
        DataType::Int64 => dups(a.as_primitive::<Int64Type>().iter(), |v| json!(v)),
        DataType::Date32 => dups(a.as_primitive::<Date32Type>().iter(), |v| json!(v)),
        DataType::Boolean => dups(a.as_boolean().iter(), |v| json!(v)),
        DataType::Float64 => dups(
            a.as_primitive::<Float64Type>()
                .iter()
                .map(|o| o.map(float_key)),
            |v| fnum(float_from_key(*v)),
        ),
        dt => (
            false,
            json!(format!("`unique` is not supported on {}", type_name(dt))),
        ),
    }
}

/// Column type name as in `with.schema` (`int64`, `string`, ...); Arrow's own name otherwise.
fn type_name(dt: &DataType) -> String {
    match dt {
        DataType::Boolean => "bool".into(),
        DataType::Int64 => "int64".into(),
        DataType::Float64 => "float64".into(),
        DataType::Utf8 => "string".into(),
        DataType::Date32 => "date32".into(),
        other => other.to_string(),
    }
}

fn schema_check(batch: &RecordBatch, schema: &[Column], allow_extra: bool) -> Check {
    let bs = batch.schema();
    let (mut missing, mut mismatched) = (Vec::new(), Vec::new());
    for c in schema {
        match bs.field_with_name(&c.name) {
            Err(_) => missing.push(c.name.clone()),
            Ok(f) if *f.data_type() != c.ty.data_type() => {
                mismatched.push((c.name.clone(), type_name(f.data_type())));
            }
            Ok(_) => {}
        }
    }
    let extra: Vec<String> = if allow_extra {
        Vec::new()
    } else {
        bs.fields()
            .iter()
            .map(|f| f.name())
            .filter(|n| !schema.iter().any(|c| &c.name == *n))
            .cloned()
            .collect()
    };
    let passed = missing.is_empty() && mismatched.is_empty() && extra.is_empty();
    let observed = json!({
        "missing": missing,
        "mismatched": mismatched.iter().map(|(c, t)| json!({ "column": c, "found": t })).collect::<Vec<_>>(),
        "extra": extra,
    });
    let mut check = Check::new("schema", None, passed, observed, json!(schema));
    // one gate entry per offending column (names only, never values)
    check.errors = missing
        .iter()
        .map(|c| (c, "missing column".to_owned()))
        .chain(
            mismatched
                .iter()
                .map(|(c, t)| (c, format!("unexpected type {t}"))),
        )
        .chain(extra.iter().map(|c| (c, "unexpected column".to_owned())))
        .map(|(c, m)| ValidationError::new(format!("schema.{c}"), m))
        .collect();
    check
}

/// Runs every configured check, in fixed order: schema, min_rows, then columns alphabetically
/// with `max_null_fraction`, `min`, `max`, `allowed`, `unique`.
fn evaluate(batch: &RecordBatch, cfg: &Config) -> Vec<Check> {
    let mut out = Vec::new();
    if let Some(schema) = &cfg.schema {
        out.push(schema_check(batch, schema, cfg.allow_extra_columns));
    }
    if let Some(n) = cfg.min_rows {
        let rows = batch.num_rows() as u64;
        out.push(Check::new(
            "min_rows",
            None,
            rows >= n,
            json!(rows),
            json!(n),
        ));
    }
    for (name, c) in &cfg.checks {
        let col = batch.column_by_name(name);
        let mut push = |check, expected: Value, f: &dyn Fn(&ArrayRef) -> (bool, Value)| {
            let (passed, observed) = col.map_or((false, json!("missing column")), f);
            out.push(Check::new(check, Some(name), passed, observed, expected));
        };
        if let Some(max) = c.max_null_fraction {
            push("max_null_fraction", json!(max), &|a| {
                let f = if a.is_empty() {
                    0.0
                } else {
                    a.null_count() as f64 / a.len() as f64
                };
                (f <= max, json!(f))
            });
        }
        if let Some(b) = &c.min {
            push("min", Value::Number(b.clone()), &|a| bound(a, b, true));
        }
        if let Some(b) = &c.max {
            push("max", Value::Number(b.clone()), &|a| bound(a, b, false));
        }
        if let Some(list) = &c.allowed {
            push("allowed", json!(list), &|a| allowed(a, list));
        }
        if c.unique {
            push("unique", json!({ "count": 0 }), &unique);
        }
    }
    out
}

/// The `std/validate.tabular` block.
pub(crate) struct ValidateTabular {
    spec: BlockSpec,
}

impl ValidateTabular {
    /// Creates `std/validate.tabular`.
    ///
    /// Note: bump `BlockSpec.version` when the semantics of a check change.
    pub(crate) fn new() -> Self {
        let dataset = |name| Port::new(name, ArtifactKind::Dataset).with_format("parquet");
        let spec = BlockSpec::new("std/validate.tabular", "0.1.0", Stage::Preprocess)
            .with_input(dataset(DATA))
            .with_output(dataset(DATA))
            .with_output(Port::new(REPORT, ArtifactKind::Metrics).with_format("json"));
        Self { spec }
    }
}

impl Block for ValidateTabular {
    fn spec(&self) -> &BlockSpec {
        &self.spec
    }

    fn validate_config(&self, config: &Value) -> Result<(), Error> {
        Config::parse(config).map(|_| ())
    }

    fn run<'a>(
        &'a self,
        ctx: &'a Context,
        inputs: Inputs,
    ) -> BoxFuture<'a, Result<BlockOutcome, Error>> {
        Box::pin(async move {
            if ctx.cancel().is_cancelled() {
                return Err(Error::Cancelled);
            }
            let cfg = Config::parse(ctx.config())?;
            let on_fail = cfg.on_fail;
            let input = inputs.get(DATA)?;
            let dir = ctx
                .store()
                .fetch(&input.id(), &ctx.workdir().join(DATA))
                .await?;
            if ctx.cancel().is_cancelled() {
                return Err(Error::Cancelled);
            }
            let (file, cancel) = (dir.join(DATA_FILE), ctx.cancel().clone());
            let checks = blocking(move || {
                let batch = read_parquet_batch(&file, &cancel)?;
                Ok(evaluate(&batch, &cfg))
            })
            .await?;
            let passed = checks.iter().all(|c| c.passed);

            let report = serde_json::to_vec(&Report {
                v: 1,
                passed,
                checks: &checks,
            })
            .map_err(|e| Error::Runtime(format!("failed to serialize report: {e}")))?;
            let report_path = ctx.workdir().join(REPORT_FILE);
            std::fs::write(&report_path, report)
                .map_err(|e| Error::Runtime(format!("{}: {e}", report_path.display())))?;
            let manifest = Manifest::new(ArtifactKind::Metrics, "json")
                .with_metadata_entry("passed", json!(passed));
            let drafts = OutputDrafts::new().with(
                REPORT,
                OutputDraft::new(manifest, DraftPayload::File(report_path)),
            );

            if passed {
                let data = OutputDraft::new(input.manifest().clone(), DraftPayload::Dir(dir));
                return Ok(BlockOutcome::Produced(drafts.with(DATA, data)));
            }
            let errors: Vec<ValidationError> = checks.into_iter().flat_map(|c| c.errors).collect();
            match on_fail {
                OnFail::Fail => Err(Error::Validation(errors)),
                OnFail::Skip => Ok(BlockOutcome::Skipped {
                    reason: format!(
                        "validation failed: {}",
                        errors
                            .iter()
                            .map(|e| e.path.as_str())
                            .collect::<Vec<_>>()
                            .join(", ")
                    ),
                    outputs: drafts,
                }),
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;
    use std::sync::Arc;

    use arrow_array::{
        BooleanArray, Date32Array, Float64Array, Int64Array, RecordBatchIterator, StringArray,
    };
    use mallops_core::{
        ArtifactStore, BlockRegistry, CancelSource, MemorySecretProvider, MemoryStore,
        PipelineBuilder, RunId, RunSummary, Runtime, RuntimeConfig, Status,
    };

    use super::*;
    use crate::ingest::{hash_file, write_parquet};

    fn batch() -> RecordBatch {
        RecordBatch::try_from_iter([
            (
                "x",
                Arc::new(Int64Array::from(vec![1, 2, 3, 4])) as ArrayRef,
            ),
            (
                "f",
                Arc::new(Float64Array::from(vec![
                    Some(0.5),
                    None,
                    Some(2.5),
                    Some(1.0),
                ])) as _,
            ),
            (
                "s",
                Arc::new(StringArray::from(vec![
                    Some("a"),
                    Some("b"),
                    Some("a"),
                    None,
                ])) as _,
            ),
            (
                "b",
                Arc::new(BooleanArray::from(vec![true, false, true, true])) as _,
            ),
            ("d", Arc::new(Date32Array::from(vec![1, 2, 3, 3])) as _),
        ])
        .unwrap()
    }

    fn floats(v: Vec<Option<f64>>) -> RecordBatch {
        RecordBatch::try_from_iter([("f", Arc::new(Float64Array::from(v)) as ArrayRef)]).unwrap()
    }

    fn run_checks(batch: &RecordBatch, cfg: Value) -> Vec<Check> {
        evaluate(batch, &Config::parse(&cfg).unwrap())
    }

    /// The single check produced by `cfg`: `(passed, observed, expected)`.
    fn one(batch: &RecordBatch, cfg: Value) -> (bool, Value, Value) {
        let mut c = run_checks(batch, cfg);
        assert_eq!(c.len(), 1, "{c:?}");
        let c = c.remove(0);
        (c.passed, c.observed, c.expected)
    }

    fn col(name: &str, checks: Value) -> Value {
        json!({ "checks": { name: checks } })
    }

    #[test]
    fn validate_tabular_checks_schema_and_min_rows() {
        let b = batch();
        let (ok, ..) = one(
            &b,
            json!({ "schema": [{"name":"x","type":"int64"},{"name":"f","type":"float64"}] }),
        );
        assert!(ok);

        let (ok, obs, exp) = one(
            &b,
            json!({ "schema": [{"name":"x","type":"float64"},{"name":"zz","type":"string"}] }),
        );
        assert!(!ok);
        assert_eq!(obs["missing"], json!(["zz"]));
        assert_eq!(obs["mismatched"], json!([{"column":"x","found":"int64"}]));
        assert_eq!(obs["extra"], json!([]));
        assert_eq!(exp[0], json!({"name":"x","type":"float64"}));

        let only_x = json!([{"name":"x","type":"int64"}]);
        let (ok, obs, _) = one(
            &b,
            json!({ "schema": only_x, "allow_extra_columns": false }),
        );
        assert!(!ok);
        assert_eq!(obs["extra"], json!(["f", "s", "b", "d"]));

        assert!(one(&b, json!({ "min_rows": 4 })).0);
        let (ok, obs, exp) = one(&b, json!({ "min_rows": 5 }));
        assert!(!ok);
        assert_eq!((obs, exp), (json!(4), json!(5)));
    }

    #[test]
    fn validate_tabular_checks_null_fraction_and_range() {
        let b = batch();
        assert!(one(&b, col("f", json!({"max_null_fraction": 0.25}))).0);
        let (ok, obs, exp) = one(&b, col("f", json!({"max_null_fraction": 0.1})));
        assert!(!ok);
        assert_eq!((obs, exp), (json!(0.25), json!(0.1)));

        assert!(one(&b, col("x", json!({"min": 1}))).0);
        let (ok, obs, exp) = one(&b, col("x", json!({"min": 2})));
        assert!(!ok);
        assert_eq!((obs, exp), (json!(1), json!(2)));
        // bound above i64::MAX: every i64 is below it
        assert!(!one(&b, col("x", json!({"min": 9223372036854775808_u64}))).0);
        assert!(one(&b, col("x", json!({"max": 9223372036854775808_u64}))).0);
        // non-integer bound on an int column is compared as float
        assert!(one(&b, col("x", json!({"min": 0.5}))).0);
        assert!(!one(&b, col("x", json!({"min": 1.5}))).0);

        assert!(one(&b, col("x", json!({"max": 4}))).0);
        let (ok, obs, exp) = one(&b, col("x", json!({"max": 3})));
        assert!(!ok);
        assert_eq!((obs, exp), (json!(4), json!(3)));
        // float column, nulls ignored
        let c = run_checks(&b, col("f", json!({"min": 0.5, "max": 2.4})));
        assert!(c[0].passed && !c[1].passed);
        assert_eq!(c[1].observed, json!(2.5));

        // non-numeric column is a failed check, not an error
        let (ok, obs, _) = one(&b, col("s", json!({"min": 1})));
        assert!(!ok);
        assert!(obs.as_str().unwrap().contains("non-numeric"));
    }

    #[test]
    fn validate_tabular_checks_allowed_and_unique() {
        let b = batch();
        assert!(one(&b, col("s", json!({"allowed": ["a", "b"]}))).0);
        let (ok, obs, exp) = one(&b, col("s", json!({"allowed": ["a"]})));
        assert!(!ok);
        assert_eq!(obs, json!({"count": 1, "examples": ["b"]}));
        assert_eq!(exp, json!(["a"]));
        assert!(one(&b, col("x", json!({"allowed": [1, 2, 3, 4]}))).0);
        assert_eq!(
            one(&b, col("x", json!({"allowed": [1]}))).1,
            json!({"count": 3, "examples": [2, 3, 4]})
        );
        assert!(!one(&b, col("b", json!({"allowed": [true]}))).0);
        assert!(!one(&b, col("f", json!({"allowed": [1]}))).0);

        assert!(one(&b, col("x", json!({"unique": true}))).0);
        let (ok, obs, _) = one(&b, col("s", json!({"unique": true})));
        assert!(!ok); // null ignored, "a" repeats
        assert_eq!(obs, json!({"count": 1, "examples": ["a"]}));
        assert_eq!(
            one(&b, col("d", json!({"unique": true}))).1["examples"],
            json!([3])
        );
        assert!(!one(&b, col("b", json!({"unique": true}))).0);
        // -0.0 and 0.0 are the same value; NaNs too
        let z = floats(vec![
            Some(0.0),
            Some(-0.0),
            Some(f64::NAN),
            Some(f64::NAN),
            Some(1.0),
        ]);
        let (ok, obs, _) = one(&z, col("f", json!({"unique": true})));
        assert!(!ok);
        assert_eq!(obs["count"], 2);
        assert_eq!(obs["examples"], json!([0.0, "NaN"]));
    }

    #[test]
    fn validate_tabular_checks_examples_are_capped_and_sorted() {
        let vals = ["h", "c", "g", "a", "f", "b", "e", "d", "z"];
        let b = RecordBatch::try_from_iter([(
            "s",
            Arc::new(StringArray::from(vals.to_vec())) as ArrayRef,
        )])
        .unwrap();
        let (ok, obs, _) = one(&b, col("s", json!({"allowed": ["z"]})));
        assert!(!ok);
        assert_eq!(
            obs,
            json!({"count": 8, "examples": ["a", "b", "c", "d", "e"]})
        );

        let dup: Vec<i64> = (0..8).rev().chain(0..8).collect();
        let b = RecordBatch::try_from_iter([("x", Arc::new(Int64Array::from(dup)) as ArrayRef)])
            .unwrap();
        let (ok, obs, _) = one(&b, col("x", json!({"unique": true})));
        assert!(!ok);
        assert_eq!(obs, json!({"count": 8, "examples": [0, 1, 2, 3, 4]}));
    }

    #[test]
    fn validate_tabular_checks_missing_column_zero_rows_and_all_null() {
        let b = batch();
        let c = run_checks(
            &b,
            col(
                "nope",
                json!({"max_null_fraction": 0.5, "min": 0, "unique": true}),
            ),
        );
        assert_eq!(c.len(), 3);
        assert!(
            c.iter()
                .all(|c| !c.passed && c.observed == json!("missing column"))
        );

        let empty = floats(vec![]);
        assert!(one(&empty, col("f", json!({"max_null_fraction": 0.0}))).0);
        let nulls = floats(vec![None, None]);
        let c = run_checks(
            &nulls,
            col("f", json!({"min": 0, "max": 1, "unique": true})),
        );
        assert!(c.iter().all(|c| c.passed));
    }

    #[test]
    fn validate_tabular_checks_nan_fails_min_and_max() {
        let b = floats(vec![Some(1.0), Some(f64::NAN)]);
        for (cfg, name) in [(json!({"min": -10}), "min"), (json!({"max": 10}), "max")] {
            let (ok, obs, _) = one(&b, col("f", cfg));
            assert!(!ok, "NaN must fail {name}");
            assert_eq!(obs, json!("NaN"));
        }
        // NaN anywhere (not only last) fails both bounds
        for v in [
            vec![Some(f64::NAN), Some(1.0)],
            vec![Some(1.0), Some(f64::NAN), Some(2.0)],
            vec![None, Some(f64::NAN)],
        ] {
            let b = floats(v);
            for cfg in [json!({"min": -10}), json!({"max": 10})] {
                let (ok, obs, _) = one(&b, col("f", cfg));
                assert!(!ok);
                assert_eq!(obs, json!("NaN"));
            }
        }
        // infinities are ordinary out-of-range values, rendered as strings
        let b = floats(vec![Some(f64::INFINITY), Some(f64::NEG_INFINITY)]);
        assert_eq!(one(&b, col("f", json!({"max": 1}))).1, json!("inf"));
        assert_eq!(one(&b, col("f", json!({"min": 1}))).1, json!("-inf"));
    }

    #[test]
    fn validate_tabular_checks_report_layout_and_order() {
        let cfg = json!({
            "min_rows": 1,
            "schema": [{"name":"x","type":"int64"}],
            "checks": {
                "x": {"unique": true, "max": 9, "min": 0, "max_null_fraction": 0.0},
                "f": {"min": 0}
            }
        });
        let c = run_checks(&batch(), cfg);
        let order: Vec<_> = c.iter().map(|c| (c.check, c.column.as_deref())).collect();
        assert_eq!(
            order,
            [
                ("schema", None),
                ("min_rows", None),
                ("min", Some("f")),
                ("max_null_fraction", Some("x")),
                ("min", Some("x")),
                ("max", Some("x")),
                ("unique", Some("x")),
            ]
        );
        let raw = serde_json::to_string(&Report {
            v: 1,
            passed: true,
            checks: &c[1..2],
        })
        .unwrap();
        assert_eq!(
            raw,
            r#"{"v":1,"passed":true,"checks":[{"check":"min_rows","passed":true,"observed":4,"expected":1}]}"#
        );
    }

    #[test]
    fn validate_tabular_config_validation() {
        let ok = |v: Value| Config::parse(&v).is_ok();
        assert!(ok(json!({})));
        assert!(ok(json!({
            "schema": [{"name":"x","type":"date32"}], "allow_extra_columns": false,
            "min_rows": 3, "on_fail": "skip",
            "checks": {"x": {"min": 1, "max": 1.5, "allowed": ["a", 1, true], "unique": false}}
        })));
        assert!(!ok(json!({"bogus": 1})));
        assert!(!ok(json!({"on_fail": "nope"})));
        assert!(!ok(json!({"allow_extra_columns": false})));
        assert!(!ok(json!({"checks": {"x": {}}})));
        assert!(!ok(json!({"checks": {"x": {"unique": false}}})));
        assert!(!ok(json!({"checks": {"x": {"bogus": 1}}})));
        assert!(!ok(json!({"checks": {"x": {"min": 2, "max": 1}}})));
        assert!(!ok(
            json!({"checks": {"x": {"min": 9007199254740993_i64, "max": 9007199254740992_i64}}})
        ));
        assert!(ok(
            json!({"checks": {"x": {"min": 9007199254740992_i64, "max": 9007199254740993_i64}}})
        ));
        assert!(!ok(json!({"checks": {"x": {"max_null_fraction": 1.5}}})));
        assert!(!ok(json!({"checks": {"x": {"max_null_fraction": -0.1}}})));
        assert!(!ok(json!({"checks": {"x": {"allowed": []}}})));
        assert!(!ok(json!({"checks": {"x": {"allowed": [1.5]}}})));
        assert!(!ok(json!({"checks": {"x": {"min": "1"}}})));
        assert!(!ok(json!({"schema": [{"name":"x","type":"int"}]})));
        assert!(!ok(json!({"checks": {"x": {"allowed": ["${secret:K}"]}}})));
        let e = Config::parse(&json!({"checks": {"x": {}}})).unwrap_err();
        assert!(matches!(e, Error::Config(_)));
    }

    // ---- block + gate inside the runtime ----

    fn write_test_parquet(batch: &RecordBatch, path: &Path) {
        let cancel = CancelSource::new().token();
        let iter = RecordBatchIterator::new([Ok(batch.clone())], batch.schema());
        write_parquet(iter, path, path, &cancel).unwrap();
    }

    fn data_manifest() -> Manifest {
        Manifest::new(ArtifactKind::Dataset, "parquet").with_metadata_entry("target", json!("y"))
    }

    struct Src(BlockSpec);

    impl Block for Src {
        fn spec(&self) -> &BlockSpec {
            &self.0
        }
        fn run<'a>(
            &'a self,
            ctx: &'a Context,
            _: Inputs,
        ) -> BoxFuture<'a, Result<BlockOutcome, Error>> {
            Box::pin(async move {
                let path = ctx.workdir().join(DATA_FILE);
                write_test_parquet(&batch(), &path);
                let d = OutputDraft::new(data_manifest(), DraftPayload::File(path));
                Ok(BlockOutcome::Produced(OutputDrafts::new().with("out", d)))
            })
        }
    }

    struct Down(BlockSpec);

    impl Block for Down {
        fn spec(&self) -> &BlockSpec {
            &self.0
        }
        fn run<'a>(
            &'a self,
            _: &'a Context,
            _: Inputs,
        ) -> BoxFuture<'a, Result<BlockOutcome, Error>> {
            Box::pin(async {
                let d = OutputDraft::new(
                    Manifest::new(ArtifactKind::Blob, "text"),
                    DraftPayload::Bytes(b"x".to_vec()),
                );
                Ok(BlockOutcome::Produced(OutputDrafts::new().with("out", d)))
            })
        }
    }

    async fn gate_run(cfg: Value) -> (RunSummary, Arc<MemoryStore>) {
        let dataset = |n| Port::new(n, ArtifactKind::Dataset).with_format("parquet");
        let mut reg = BlockRegistry::new();
        reg.register(Arc::new(Src(
            BlockSpec::new("t/src", "1", Stage::Ingest).with_output(dataset("out"))
        )))
        .unwrap();
        reg.register(Arc::new(ValidateTabular::new())).unwrap();
        reg.register(Arc::new(Down(
            BlockSpec::new("t/down", "1", Stage::Train)
                .with_input(dataset("in"))
                .with_output(Port::new("out", ArtifactKind::Blob)),
        )))
        .unwrap();
        let mut b = PipelineBuilder::new("validate");
        b.block("src", "t/src".parse().unwrap());
        b.block("val", "std/validate.tabular".parse().unwrap())
            .input("data", "src.out")
            .with(cfg);
        b.block("down", "t/down".parse().unwrap())
            .input("in", "val.data");
        let p = b.build(&reg).unwrap();
        let store = Arc::new(MemoryStore::new());
        let rt = Runtime::new(RuntimeConfig::new(
            Arc::clone(&store) as Arc<dyn ArtifactStore>
        ));
        (rt.run(&p).await.unwrap(), store)
    }

    async fn payload_digest(store: &MemoryStore, s: &RunSummary, node: &str, port: &str) -> String {
        let dir = tempfile::tempdir().unwrap();
        let out = store
            .fetch(&s.nodes[node].outputs[port], &dir.path().join("p"))
            .await
            .unwrap();
        hash_file(&out.join(DATA_FILE)).unwrap().to_string()
    }

    #[tokio::test]
    async fn validate_tabular_gate_pass_passes_data_through() {
        let cfg = json!({ "min_rows": 4, "checks": { "x": { "unique": true } } });
        let (s, store) = gate_run(cfg).await;
        assert!(s.is_success());
        assert_eq!(s.nodes["val"].status, Status::Succeeded);
        assert_eq!(s.nodes["down"].status, Status::Succeeded);

        let (src_id, out_id) = (
            s.nodes["src"].outputs["out"],
            s.nodes["val"].outputs["data"],
        );
        assert_ne!(src_id, out_id); // parents enter the id (ADR-0007 D3)
        let input = store.get(&src_id).await.unwrap().unwrap();
        let output = store.get(&out_id).await.unwrap().unwrap();
        assert_eq!(output.manifest().kind, input.manifest().kind);
        assert_eq!(output.manifest().format, input.manifest().format);
        assert_eq!(output.manifest().metadata, input.manifest().metadata);
        assert_eq!(output.manifest().parents, vec![src_id]);
        assert_eq!(
            payload_digest(&store, &s, "src", "out").await,
            payload_digest(&store, &s, "val", "data").await
        );

        let report = store
            .get(&s.nodes["val"].outputs["report"])
            .await
            .unwrap()
            .unwrap();
        assert_eq!(report.manifest().kind, ArtifactKind::Metrics);
        assert_eq!(report.manifest().metadata["passed"], true);
    }

    #[tokio::test]
    async fn validate_tabular_gate_fail_names_failed_checks() {
        let cfg = json!({
            "schema": [{"name":"x","type":"string"}, {"name":"zz","type":"int64"}],
            "min_rows": 99,
            "checks": { "s": { "allowed": ["a"] }, "x": { "min": 0 } }
        });
        let (s, _) = gate_run(cfg).await;
        assert!(!s.is_success());
        assert_eq!(s.nodes["val"].status, Status::Failed);
        assert!(s.nodes["val"].outputs.is_empty());
        assert_eq!(s.nodes["down"].status, Status::Cancelled);
        let Some(Error::Validation(errs)) = &s.nodes["val"].error else {
            panic!("expected a validation error: {:?}", s.nodes["val"].error);
        };
        let got: Vec<(&str, &str)> = errs
            .iter()
            .map(|e| (e.path.as_str(), e.message.as_str()))
            .collect();
        assert_eq!(
            got,
            [
                ("schema.zz", "missing column"),
                ("schema.x", "unexpected type int64"),
                ("min_rows", "check failed"),
                ("checks.s.allowed", "1 violating row(s)"),
            ]
        );
    }

    #[tokio::test]
    async fn validate_tabular_gate_skip_skips_downstream_keeps_report() {
        let cfg = json!({ "on_fail": "skip", "min_rows": 99 });
        let (s, store) = gate_run(cfg).await;
        assert!(s.is_success());
        assert_eq!(s.nodes["val"].status, Status::Skipped);
        assert_eq!(
            s.nodes["val"].reason.as_deref(),
            Some("validation failed: min_rows")
        );
        assert_eq!(s.nodes["down"].status, Status::Skipped);
        assert!(!s.nodes["val"].outputs.contains_key("data"));
        let report = store
            .get(&s.nodes["val"].outputs["report"])
            .await
            .unwrap()
            .unwrap();
        assert_eq!(report.manifest().metadata["passed"], false);

        let dir = tempfile::tempdir().unwrap();
        let p = store
            .fetch(&s.nodes["val"].outputs["report"], &dir.path().join("r"))
            .await
            .unwrap();
        let doc: Value =
            serde_json::from_slice(&std::fs::read(p.join(REPORT_FILE)).unwrap()).unwrap();
        assert_eq!(doc["v"], 1);
        assert_eq!(doc["passed"], false);
        assert_eq!(doc["checks"][0]["observed"], 4);
        assert_eq!(doc["checks"][0]["expected"], 99);
    }

    #[tokio::test]
    async fn validate_tabular_gate_missing_payload_is_runtime_error() {
        let store = Arc::new(MemoryStore::new());
        let m = Manifest::new(ArtifactKind::Dataset, "parquet");
        let id = store
            .put(m.clone(), mallops_core::PayloadSource::Bytes(b"x"))
            .await
            .unwrap();
        let wd = tempfile::tempdir().unwrap();
        let ctx = Context::new(
            RunId::new(),
            wd.path().to_path_buf(),
            Arc::clone(&store) as Arc<dyn ArtifactStore>,
            json!({}),
            CancelSource::new().token(),
            Arc::new(MemorySecretProvider::default()),
        );
        let inputs = Inputs::new().with(DATA, mallops_core::Artifact::new(id, m));
        let e = ValidateTabular::new().run(&ctx, inputs).await;
        assert!(matches!(e, Err(Error::Runtime(_))), "{:?}", e.err());
    }

    #[tokio::test]
    async fn validate_tabular_cancelled_before_run() {
        let store = Arc::new(MemoryStore::new());
        let m = Manifest::new(ArtifactKind::Dataset, "parquet");
        let id = store
            .put(m.clone(), mallops_core::PayloadSource::Bytes(b"x"))
            .await
            .unwrap();
        let wd = tempfile::tempdir().unwrap();
        let cancel = CancelSource::new();
        cancel.cancel();
        let ctx = Context::new(
            RunId::new(),
            wd.path().to_path_buf(),
            Arc::clone(&store) as Arc<dyn ArtifactStore>,
            json!({}),
            cancel.token(),
            Arc::new(MemorySecretProvider::default()),
        );
        let inputs = Inputs::new().with(DATA, mallops_core::Artifact::new(id, m));
        let e = ValidateTabular::new().run(&ctx, inputs).await;
        assert!(matches!(e, Err(Error::Cancelled)), "{:?}", e.err());
    }
}
