//! Fitted transform artifact of `std/preprocess.tabular` (`kind: transform`, `format: json`),
//! reloaded by `package` and `serve` to apply the exact same preprocessing (FR-PRE-3).
//!
//! Payload: a single `transform.json` file, `{"v":1,"target":…,"input":[…],"steps":[…]}`,
//! serialized deterministically (struct field order, `BTreeMap` keys).

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;

use arrow_array::{
    ArrayRef, BooleanArray, Date32Array, Float64Array, Int64Array, RecordBatch, RecordBatchOptions,
    StringArray,
};
use arrow_schema::{Field, Schema};
use mallops_core::Error;
use serde::{Deserialize, Serialize};

use crate::ingest::{Column, ColumnType};

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
        let path = dir.join(TRANSFORM_FILE);
        let bytes = std::fs::read(&path)
            .map_err(|e| Error::Runtime(format!("cannot read {}: {e}", path.display())))?;
        let transform: Self = serde_json::from_slice(&bytes)
            .map_err(|e| Error::Runtime(format!("{}: {e}", path.display())))?;
        if transform.v != VERSION {
            return Err(Error::Runtime(format!(
                "{}: unsupported transform version {}, expected {VERSION}",
                path.display(),
                transform.v
            )));
        }
        Ok(transform)
    }

    /// Applies every fitted step to `batch` (raw columns as in the training input).
    ///
    /// # Errors
    ///
    /// [`Error::Runtime`] if a required column is missing or has an unexpected type.
    pub fn apply_batch(&self, batch: &RecordBatch) -> Result<RecordBatch, Error> {
        let mut current = batch.clone();
        for step in &self.steps {
            current = step.apply_with_target(&current, Some(&self.target))?;
        }
        Ok(current)
    }

    /// Builds a batch from JSON rows (an array of objects keyed by input column; target optional)
    /// and applies [`Self::apply_batch`].
    ///
    /// # Errors
    ///
    /// [`Error::Runtime`] if `rows` is not an array of objects or a value does not match the
    /// input column type.
    pub fn apply_json(&self, rows: &serde_json::Value) -> Result<RecordBatch, Error> {
        let items = rows
            .as_array()
            .ok_or_else(|| Error::Runtime("apply_json: expected a JSON array of objects".into()))?;
        let mut objs = Vec::with_capacity(items.len());
        for (row_idx, item) in items.iter().enumerate() {
            let obj = item.as_object().ok_or_else(|| {
                Error::Runtime(format!("apply_json: row {row_idx}: expected a JSON object"))
            })?;
            objs.push(obj);
        }

        let has_target = objs.iter().any(|o| o.contains_key(&self.target));
        let mut fields = Vec::with_capacity(self.input.len());
        let mut arrays = Vec::with_capacity(self.input.len());

        for col in &self.input {
            if col.name == self.target && !has_target {
                continue;
            }
            let array = build_column(col, &objs)?;
            fields.push(Field::new(&col.name, col.ty.data_type(), true));
            arrays.push(array);
        }

        let options = RecordBatchOptions::new().with_row_count(Some(objs.len()));
        let batch =
            RecordBatch::try_new_with_options(Arc::new(Schema::new(fields)), arrays, &options)
                .map_err(|e| Error::Runtime(format!("apply_json: {e}")))?;
        self.apply_batch(&batch)
    }
}

fn build_column(
    col: &Column,
    rows: &[&serde_json::Map<String, serde_json::Value>],
) -> Result<ArrayRef, Error> {
    let type_err = |row_idx: usize, expected: &str| {
        Error::Runtime(format!(
            "apply_json: row {row_idx}, column '{}': expected {expected}",
            col.name
        ))
    };

    match col.ty {
        ColumnType::Bool => {
            let mut vals = Vec::with_capacity(rows.len());
            for (i, obj) in rows.iter().enumerate() {
                match obj.get(&col.name) {
                    None | Some(serde_json::Value::Null) => vals.push(None),
                    Some(v) => vals.push(Some(v.as_bool().ok_or_else(|| type_err(i, "bool"))?)),
                }
            }
            Ok(Arc::new(BooleanArray::from(vals)))
        }
        ColumnType::Int64 => {
            let mut vals = Vec::with_capacity(rows.len());
            for (i, obj) in rows.iter().enumerate() {
                match obj.get(&col.name) {
                    None | Some(serde_json::Value::Null) => vals.push(None),
                    Some(v) => vals.push(Some(v.as_i64().ok_or_else(|| type_err(i, "int64"))?)),
                }
            }
            Ok(Arc::new(Int64Array::from(vals)))
        }
        ColumnType::Float64 => {
            let mut vals = Vec::with_capacity(rows.len());
            for (i, obj) in rows.iter().enumerate() {
                match obj.get(&col.name) {
                    None | Some(serde_json::Value::Null) => vals.push(None),
                    Some(v) => {
                        let f = v
                            .as_f64()
                            .filter(|f| f.is_finite())
                            .ok_or_else(|| type_err(i, "float64"))?;
                        vals.push(Some(f));
                    }
                }
            }
            Ok(Arc::new(Float64Array::from(vals)))
        }
        ColumnType::String => {
            let mut vals: Vec<Option<&str>> = Vec::with_capacity(rows.len());
            for (i, obj) in rows.iter().enumerate() {
                match obj.get(&col.name) {
                    None | Some(serde_json::Value::Null) => vals.push(None),
                    Some(v) => vals.push(Some(v.as_str().ok_or_else(|| type_err(i, "string"))?)),
                }
            }
            Ok(Arc::new(StringArray::from(vals)))
        }
        ColumnType::Date32 => {
            let mut vals = Vec::with_capacity(rows.len());
            for (i, obj) in rows.iter().enumerate() {
                match obj.get(&col.name) {
                    None | Some(serde_json::Value::Null) => vals.push(None),
                    Some(v) => {
                        let s = v
                            .as_str()
                            .ok_or_else(|| type_err(i, "date32 (YYYY-MM-DD)"))?;
                        let days = parse_iso_date32(s)
                            .ok_or_else(|| type_err(i, "date32 (YYYY-MM-DD)"))?;
                        vals.push(Some(days));
                    }
                }
            }
            Ok(Arc::new(Date32Array::from(vals)))
        }
    }
}

fn parse_iso_date32(s: &str) -> Option<i32> {
    let bytes = s.as_bytes();
    if bytes.len() != 10 || bytes[4] != b'-' || bytes[7] != b'-' {
        return None;
    }
    if !bytes[0..4].iter().all(u8::is_ascii_digit)
        || !bytes[5..7].iter().all(u8::is_ascii_digit)
        || !bytes[8..10].iter().all(u8::is_ascii_digit)
    {
        return None;
    }
    let year: i32 = s[0..4].parse().ok()?;
    let month: u32 = s[5..7].parse().ok()?;
    let day: u32 = s[8..10].parse().ok()?;
    if !(1..=12).contains(&month) {
        return None;
    }
    let is_leap = (year % 4 == 0 && year % 100 != 0) || year % 400 == 0;
    let max_day = match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if is_leap => 29,
        2 => 28,
        _ => return None,
    };
    if !(1..=max_day).contains(&day) {
        return None;
    }
    // Days since Unix epoch (1970-01-01), Howard Hinnant's civil calendar algorithm.
    let y = i64::from(year) - i64::from(month <= 2);
    let era = y.div_euclid(400);
    let yoe = y.rem_euclid(400) as u64;
    let m = u64::from(month);
    let d = u64::from(day);
    let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    let days = era * 146_097 + doe as i64 - 719_468;
    i32::try_from(days).ok()
}

#[cfg(test)]
mod tests {
    use arrow_array::Array;
    use serde_json::json;

    use super::*;

    fn sample_transform() -> FittedTransform {
        FittedTransform::new(
            "y".into(),
            vec![
                Column {
                    name: "flag".into(),
                    ty: ColumnType::Bool,
                },
                Column {
                    name: "age".into(),
                    ty: ColumnType::Int64,
                },
                Column {
                    name: "score".into(),
                    ty: ColumnType::Float64,
                },
                Column {
                    name: "city".into(),
                    ty: ColumnType::String,
                },
                Column {
                    name: "day".into(),
                    ty: ColumnType::Date32,
                },
                Column {
                    name: "y".into(),
                    ty: ColumnType::Int64,
                },
            ],
            vec![
                FittedStep::Select {
                    columns: vec![
                        "flag".into(),
                        "age".into(),
                        "score".into(),
                        "city".into(),
                        "day".into(),
                        "y".into(),
                    ],
                },
                FittedStep::Impute {
                    values: BTreeMap::from([
                        ("age".into(), json!(30)),
                        ("score".into(), json!(5.0)),
                    ]),
                },
                FittedStep::Scale {
                    params: BTreeMap::from([(
                        "score".into(),
                        Affine {
                            offset: 5.0,
                            scale: 2.0,
                        },
                    )]),
                },
                FittedStep::OneHot {
                    categories: BTreeMap::from([(
                        "city".into(),
                        vec!["paris".into(), "rome".into()],
                    )]),
                },
            ],
        )
    }

    #[test]
    fn preprocess_steps_transform_from_dir_and_roundtrip_bytes() {
        let dir = tempfile::tempdir().unwrap();
        let t = sample_transform();
        let bytes = t.to_json().unwrap();
        std::fs::write(dir.path().join(TRANSFORM_FILE), &bytes).unwrap();

        let loaded = FittedTransform::from_dir(dir.path()).unwrap();
        assert_eq!(loaded, t);
        assert_eq!(loaded.target(), "y");
        assert_eq!(loaded.to_json().unwrap(), bytes);

        // Wrong version fails.
        let bad_dir = tempfile::tempdir().unwrap();
        let mut val: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        val["v"] = json!(2);
        std::fs::write(
            bad_dir.path().join(TRANSFORM_FILE),
            serde_json::to_vec(&val).unwrap(),
        )
        .unwrap();
        let err = FittedTransform::from_dir(bad_dir.path()).unwrap_err();
        assert!(matches!(err, Error::Runtime(_)));
        assert!(err.to_string().contains("version 2"), "{err}");

        // Missing file fails.
        let empty_dir = tempfile::tempdir().unwrap();
        assert!(matches!(
            FittedTransform::from_dir(empty_dir.path()),
            Err(Error::Runtime(_))
        ));
    }

    fn col_names(batch: &RecordBatch) -> Vec<String> {
        batch
            .schema()
            .fields()
            .iter()
            .map(|f| f.name().clone())
            .collect()
    }

    #[test]
    fn preprocess_steps_transform_apply_json_with_and_without_target() {
        let t = sample_transform();

        // Serving rows: no target, unknown key ignored, null imputed, unseen city -> all 0.
        let rows = json!([
            { "flag": true, "age": null, "score": 9.0, "city": "rome", "day": "2026-01-15", "extra": 123 },
            { "flag": false, "age": 40, "score": null, "city": "berlin", "day": "1970-01-01" }
        ]);
        let out = t.apply_json(&rows).unwrap();
        assert_eq!(
            col_names(&out),
            vec!["flag", "age", "score", "city=paris", "city=rome", "day"]
        );

        let flags = out
            .column(0)
            .as_any()
            .downcast_ref::<BooleanArray>()
            .unwrap();
        assert!(flags.value(0));
        assert!(!flags.value(1));

        let ages = out.column(1).as_any().downcast_ref::<Int64Array>().unwrap();
        assert_eq!(ages.value(0), 30);
        assert_eq!(ages.value(1), 40);

        let scores = out
            .column(2)
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        assert!((scores.value(0) - 2.0).abs() < 1e-12);
        assert!((scores.value(1) - 0.0).abs() < 1e-12);

        let paris = out
            .column(3)
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        let rome = out
            .column(4)
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        assert_eq!((paris.value(0), rome.value(0)), (0.0, 1.0));
        assert_eq!((paris.value(1), rome.value(1)), (0.0, 0.0));

        let days = out
            .column(5)
            .as_any()
            .downcast_ref::<Date32Array>()
            .unwrap();
        assert_eq!(days.value(0), 20468);
        assert_eq!(days.value(1), 0);

        // Rows with target keep target at the end.
        let with_target = json!([
            { "flag": true, "age": 20, "score": 5, "city": "paris", "day": "1969-12-31", "y": 1 }
        ]);
        let out_t = t.apply_json(&with_target).unwrap();
        assert_eq!(
            col_names(&out_t),
            vec![
                "flag",
                "age",
                "score",
                "city=paris",
                "city=rome",
                "day",
                "y"
            ]
        );
        let y = out_t
            .column(6)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        assert_eq!(y.value(0), 1);
        let d = out_t
            .column(5)
            .as_any()
            .downcast_ref::<Date32Array>()
            .unwrap();
        assert_eq!(d.value(0), -1);

        // Empty array produces 0-row batch.
        let empty = t.apply_json(&json!([])).unwrap();
        assert_eq!(empty.num_rows(), 0);
        assert_eq!(empty.num_columns(), 6);
    }

    #[test]
    fn preprocess_steps_transform_apply_json_errors_with_row_and_column() {
        let t = sample_transform();

        // Not an array.
        assert!(matches!(t.apply_json(&json!({})), Err(Error::Runtime(_))));
        // Element not an object.
        let err = t.apply_json(&json!([{}, 42])).unwrap_err();
        assert!(err.to_string().contains("row 1"), "{err}");

        for (bad_rows, want_row, want_col) in [
            (json!([{}, { "flag": "yes" }]), "row 1", "flag"),
            (json!([{ "age": 3.5 }]), "row 0", "age"),
            (json!([{ "score": "high" }]), "row 0", "score"),
            (json!([{ "city": 7 }]), "row 0", "city"),
            (json!([{ "day": "2025-02-29" }]), "row 0", "day"),
            (json!([{ "day": 12345 }]), "row 0", "day"),
        ] {
            let err = t.apply_json(&bad_rows).unwrap_err();
            assert!(matches!(err, Error::Runtime(_)), "{bad_rows}: {err:?}");
            let msg = err.to_string();
            assert!(
                msg.contains(want_row) && msg.contains(want_col),
                "{bad_rows}: expected {want_row} and {want_col}, got: {msg}"
            );
        }
    }

    #[test]
    fn preprocess_steps_parse_iso_date32_leap_and_bounds() {
        assert_eq!(parse_iso_date32("1970-01-01"), Some(0));
        assert_eq!(parse_iso_date32("1969-12-31"), Some(-1));
        assert_eq!(parse_iso_date32("2024-02-29"), Some(19782));
        assert_eq!(parse_iso_date32("2026-01-15"), Some(20468));
        assert_eq!(parse_iso_date32("2025-02-29"), None);
        assert_eq!(parse_iso_date32("2025-13-01"), None);
        assert_eq!(parse_iso_date32("2025-00-01"), None);
        assert_eq!(parse_iso_date32("2025-01-00"), None);
        assert_eq!(parse_iso_date32("2025-01-32"), None);
        assert_eq!(parse_iso_date32("+025-01-01"), None);
    }

    #[test]
    fn preprocess_steps_transform_apply_batch_serving_and_missing_column() {
        let t = sample_transform();

        // Batch without target column succeeds and omits target from output.
        let batch = RecordBatch::try_from_iter([
            ("flag", Arc::new(BooleanArray::from(vec![true])) as ArrayRef),
            ("age", Arc::new(Int64Array::from(vec![None])) as _),
            ("score", Arc::new(Float64Array::from(vec![7.0])) as _),
            ("city", Arc::new(StringArray::from(vec!["paris"])) as _),
            ("day", Arc::new(Date32Array::from(vec![20468])) as _),
        ])
        .unwrap();
        let out = t.apply_batch(&batch).unwrap();
        assert_eq!(
            col_names(&out),
            vec!["flag", "age", "score", "city=paris", "city=rome", "day"]
        );
        let age = out.column(1).as_any().downcast_ref::<Int64Array>().unwrap();
        assert_eq!(age.value(0), 30);
        let score = out
            .column(2)
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        assert!((score.value(0) - 1.0).abs() < 1e-12);

        // Missing required feature column returns Error::Runtime naming the column.
        let missing = RecordBatch::try_from_iter([
            ("flag", Arc::new(BooleanArray::from(vec![true])) as ArrayRef),
            ("age", Arc::new(Int64Array::from(vec![20])) as _),
        ])
        .unwrap();
        let err = t.apply_batch(&missing).unwrap_err();
        assert!(matches!(err, Error::Runtime(_)));
        assert!(err.to_string().contains("score"), "{err}");
    }
}
