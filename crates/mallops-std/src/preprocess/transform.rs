//! Fitted transform artifact of `std/preprocess.tabular` (`kind: transform`, `format: json`),
//! reloaded by `package` and `serve` to apply the exact same preprocessing (FR-PRE-3).
//!
//! Payload: a single `transform.json` file, `{"v":1,"target":…,"input":[…],"steps":[…]}`,
//! serialized deterministically (struct field order, `BTreeMap` keys).

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;

use arrow_array::types::Date32Type;
use arrow_array::{
    ArrayRef, BooleanArray, Date32Array, Float64Array, Int64Array, RecordBatch, RecordBatchOptions,
    StringArray,
};
use arrow_cast::parse::Parser;
use arrow_schema::{Field, Schema};
use mallops_core::Error;
use serde::{Deserialize, Serialize};

use super::steps::json_as_exact_i64;
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

    /// Projects `batch` onto the expected `input` columns in order (target optional, extra
    /// columns ignored), checks their types, and applies every fitted step.
    ///
    /// # Errors
    ///
    /// [`Error::Runtime`] if a required column is missing or has an unexpected type.
    pub fn apply_batch(&self, batch: &RecordBatch) -> Result<RecordBatch, Error> {
        let projected = self.project_input(batch)?;
        self.apply_steps(&projected)
    }

    /// Builds a batch from JSON rows (an array of objects keyed by input column; target optional,
    /// explicit `null` allowed, missing non-target keys rejected) and applies [`Self::apply_batch`].
    ///
    /// # Errors
    ///
    /// [`Error::Runtime`] if `rows` is not an array of objects, a non-target column key is
    /// missing, or a value does not match the input column type.
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
            let is_target = col.name == self.target;
            if is_target && !has_target {
                continue;
            }
            let array = build_column(col, is_target, &objs)?;
            fields.push(Field::new(&col.name, col.ty.data_type(), true));
            arrays.push(array);
        }

        let options = RecordBatchOptions::new().with_row_count(Some(objs.len()));
        let batch =
            RecordBatch::try_new_with_options(Arc::new(Schema::new(fields)), arrays, &options)
                .map_err(|e| Error::Runtime(format!("apply_json: {e}")))?;
        self.apply_steps(&batch)
    }

    fn project_input(&self, batch: &RecordBatch) -> Result<RecordBatch, Error> {
        let schema = batch.schema();
        let mut fields = Vec::with_capacity(self.input.len());
        let mut arrays = Vec::with_capacity(self.input.len());
        for col in &self.input {
            let idx = match schema.index_of(&col.name) {
                Ok(i) => i,
                Err(_) if col.name == self.target => continue,
                Err(_) => {
                    return Err(Error::Runtime(format!(
                        "apply_batch: missing column '{}'",
                        col.name
                    )));
                }
            };
            let field = schema.field(idx);
            let expected = col.ty.data_type();
            if field.data_type() != &expected {
                return Err(Error::Runtime(format!(
                    "apply_batch: column '{}' type mismatch: expected {expected:?}, found {:?}",
                    col.name,
                    field.data_type()
                )));
            }
            fields.push(field.as_ref().clone());
            arrays.push(Arc::clone(batch.column(idx)));
        }
        let options = RecordBatchOptions::new().with_row_count(Some(batch.num_rows()));
        RecordBatch::try_new_with_options(Arc::new(Schema::new(fields)), arrays, &options)
            .map_err(|e| Error::Runtime(format!("apply_batch: {e}")))
    }

    fn apply_steps(&self, batch: &RecordBatch) -> Result<RecordBatch, Error> {
        let mut current = batch.clone();
        for step in &self.steps {
            current = step.apply_with_target(&current, Some(&self.target))?;
        }
        Ok(current)
    }
}

fn map_column<'a, T, F>(
    name: &str,
    optional: bool,
    rows: &[&'a serde_json::Map<String, serde_json::Value>],
    expected: &str,
    mut parse: F,
) -> Result<Vec<Option<T>>, Error>
where
    F: FnMut(&'a serde_json::Value) -> Option<T>,
{
    rows.iter()
        .enumerate()
        .map(|(i, obj)| match obj.get(name) {
            None if optional => Ok(None),
            None => Err(Error::Runtime(format!(
                "apply_json: row {i}: missing column '{name}'"
            ))),
            Some(serde_json::Value::Null) => Ok(None),
            Some(v) => parse(v).map(Some).ok_or_else(|| {
                Error::Runtime(format!(
                    "apply_json: row {i}, column '{name}': expected {expected}"
                ))
            }),
        })
        .collect()
}

fn build_column(
    col: &Column,
    optional: bool,
    rows: &[&serde_json::Map<String, serde_json::Value>],
) -> Result<ArrayRef, Error> {
    let n = &col.name;
    Ok(match col.ty {
        ColumnType::Bool => Arc::new(BooleanArray::from(map_column(
            n,
            optional,
            rows,
            "bool",
            serde_json::Value::as_bool,
        )?)),
        ColumnType::Int64 => Arc::new(Int64Array::from(map_column(
            n,
            optional,
            rows,
            "int64",
            json_as_exact_i64,
        )?)),
        ColumnType::Float64 => Arc::new(Float64Array::from(map_column(
            n,
            optional,
            rows,
            "float64",
            |v| v.as_f64().filter(|f| f.is_finite()),
        )?)),
        ColumnType::String => Arc::new(StringArray::from(map_column(
            n,
            optional,
            rows,
            "string",
            serde_json::Value::as_str,
        )?)),
        ColumnType::Date32 => Arc::new(Date32Array::from(map_column(
            n,
            optional,
            rows,
            "date32 (YYYY-MM-DD)",
            |v| v.as_str().and_then(parse_iso_date32),
        )?)),
    })
}

fn parse_iso_date32(s: &str) -> Option<i32> {
    let bytes = s.as_bytes();
    if bytes.len() != 10
        || bytes[4] != b'-'
        || bytes[7] != b'-'
        || !bytes[0..4].iter().all(u8::is_ascii_digit)
        || !bytes[5..7].iter().all(u8::is_ascii_digit)
        || !bytes[8..10].iter().all(u8::is_ascii_digit)
    {
        return None;
    }
    Date32Type::parse(s)
}

#[cfg(test)]
mod tests {
    use arrow_array::cast::AsArray;
    use arrow_array::types::{Date32Type, Float64Type, Int64Type};
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

    fn col_names(batch: &RecordBatch) -> Vec<String> {
        batch
            .schema()
            .fields()
            .iter()
            .map(|f| f.name().clone())
            .collect()
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

    #[test]
    fn preprocess_steps_transform_apply_json_with_and_without_target() {
        let t = sample_transform();

        // Serving rows: no target, unknown key ignored, explicit null imputed, whole-number float
        // accepted for int64 (`40.0`), unseen city -> all 0.
        let rows = json!([
            { "flag": true, "age": null, "score": 9.0, "city": "rome", "day": "2026-01-15", "extra": 123 },
            { "flag": false, "age": 40.0, "score": null, "city": "berlin", "day": "1970-01-01" }
        ]);
        let out = t.apply_json(&rows).unwrap();
        assert_eq!(
            col_names(&out),
            vec!["flag", "age", "score", "city=paris", "city=rome", "day"]
        );

        let flags = out.column(0).as_boolean();
        assert!(flags.value(0));
        assert!(!flags.value(1));

        let ages = out.column(1).as_primitive::<Int64Type>();
        assert_eq!(ages.value(0), 30);
        assert_eq!(ages.value(1), 40);

        let scores = out.column(2).as_primitive::<Float64Type>();
        assert!((scores.value(0) - 2.0).abs() < 1e-12);
        assert!((scores.value(1) - 0.0).abs() < 1e-12);

        let paris = out.column(3).as_primitive::<Float64Type>();
        let rome = out.column(4).as_primitive::<Float64Type>();
        assert_eq!((paris.value(0), rome.value(0)), (0.0, 1.0));
        assert_eq!((paris.value(1), rome.value(1)), (0.0, 0.0));

        let days = out.column(5).as_primitive::<Date32Type>();
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
        let y = out_t.column(6).as_primitive::<Int64Type>();
        assert_eq!(y.value(0), 1);
        let d = out_t.column(5).as_primitive::<Date32Type>();
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

        // Missing non-target key fails naming row and column.
        let missing_key = json!([
            { "flag": true, "age": 20, "score": 5.0, "city": "rome", "day": "2026-01-15" },
            { "flag": true, "score": 5.0, "city": "rome", "day": "2026-01-15" }
        ]);
        let err = t.apply_json(&missing_key).unwrap_err();
        assert!(matches!(err, Error::Runtime(_)));
        assert!(
            err.to_string().contains("row 1: missing column 'age'"),
            "{err}"
        );

        let base = json!({ "flag": null, "age": null, "score": null, "city": null, "day": null });
        let row_with = |key: &str, val: serde_json::Value| {
            let mut r = base.clone();
            r[key] = val;
            r
        };

        for (bad_rows, want_row, want_col) in [
            (
                json!([base.clone(), row_with("flag", json!("yes"))]),
                "row 1",
                "flag",
            ),
            (json!([row_with("age", json!(3.5))]), "row 0", "age"),
            (json!([row_with("score", json!("high"))]), "row 0", "score"),
            (json!([row_with("city", json!(7))]), "row 0", "city"),
            (
                json!([row_with("day", json!("2025-02-29"))]),
                "row 0",
                "day",
            ),
            (json!([row_with("day", json!(12345))]), "row 0", "day"),
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
    fn preprocess_steps_transform_apply_batch_projects_and_validates_types() {
        let t = sample_transform();

        // Batch without target, with reordered columns and an extra unknown column: projected onto
        // `input` order and produces the exact same output as `apply_json`.
        let batch = RecordBatch::try_from_iter([
            ("extra", Arc::new(Int64Array::from(vec![999])) as ArrayRef),
            ("day", Arc::new(Date32Array::from(vec![20468])) as _),
            ("city", Arc::new(StringArray::from(vec!["paris"])) as _),
            ("score", Arc::new(Float64Array::from(vec![7.0])) as _),
            ("age", Arc::new(Int64Array::from(vec![None])) as _),
            ("flag", Arc::new(BooleanArray::from(vec![true])) as _),
        ])
        .unwrap();
        let out = t.apply_batch(&batch).unwrap();
        assert_eq!(
            col_names(&out),
            vec!["flag", "age", "score", "city=paris", "city=rome", "day"]
        );
        assert_eq!(out.column(1).as_primitive::<Int64Type>().value(0), 30);
        assert!((out.column(2).as_primitive::<Float64Type>().value(0) - 1.0).abs() < 1e-12);

        // Missing required feature column returns Error::Runtime naming the column.
        let missing = RecordBatch::try_from_iter([
            ("flag", Arc::new(BooleanArray::from(vec![true])) as ArrayRef),
            ("age", Arc::new(Int64Array::from(vec![20])) as _),
        ])
        .unwrap();
        let err = t.apply_batch(&missing).unwrap_err();
        assert!(matches!(err, Error::Runtime(_)));
        assert!(err.to_string().contains("score"), "{err}");

        // Wrong column type returns Error::Runtime naming column, expected, and found types.
        let wrong_type = RecordBatch::try_from_iter([
            ("flag", Arc::new(BooleanArray::from(vec![true])) as ArrayRef),
            ("age", Arc::new(StringArray::from(vec!["20"])) as _),
            ("score", Arc::new(Float64Array::from(vec![7.0])) as _),
            ("city", Arc::new(StringArray::from(vec!["paris"])) as _),
            ("day", Arc::new(Date32Array::from(vec![20468])) as _),
        ])
        .unwrap();
        let err = t.apply_batch(&wrong_type).unwrap_err();
        assert!(matches!(err, Error::Runtime(_)));
        let msg = err.to_string();
        assert!(
            msg.contains("age") && msg.contains("Int64") && msg.contains("Utf8"),
            "{msg}"
        );
    }
}
