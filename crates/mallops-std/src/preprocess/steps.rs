//! Arrow kernels of the preprocessing steps: stateless `filter`, and `fit` of `select`, `impute`,
//! `scale`, `encode` into a [`FittedStep`] (applied with [`FittedStep::apply`]).

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::{Float64Type, Int64Type};
use arrow_array::{
    Array, ArrayRef, BooleanArray, Datum, Float64Array, Int64Array, RecordBatch,
    RecordBatchOptions, StringArray,
};
use arrow_cast::cast;
use arrow_ord::cmp;
use arrow_schema::{ArrowError, DataType, Field, Schema};
use arrow_select::filter::filter_record_batch;
use mallops_core::Error;

use super::transform::{Affine, FittedStep};
use super::{
    EncodeMethod, EncodeStep, FilterOp, FilterStep, ImputeStep, ImputeStrategy, ScaleMethod,
    ScaleStep, SelectStep, Step,
};

type CmpFn = fn(&dyn Datum, &dyn Datum) -> Result<BooleanArray, ArrowError>;

/// Keeps the rows of `batch` matching `step` (nulls never match a comparison).
pub(crate) fn filter(step: &FilterStep, batch: &RecordBatch) -> Result<RecordBatch, Error> {
    let idx = batch
        .schema()
        .index_of(&step.column)
        .map_err(|_| Error::Runtime(format!("filter: unknown column '{}'", step.column)))?;
    let col = batch.column(idx);

    let predicate = match step.op {
        FilterOp::IsNull => BooleanArray::from_iter((0..col.len()).map(|i| Some(col.is_null(i)))),
        FilterOp::NotNull => BooleanArray::from_iter((0..col.len()).map(|i| Some(col.is_valid(i)))),
        FilterOp::Eq => compare_column(&step.column, col, cmp::eq, step.value.as_ref())?,
        FilterOp::Ne => compare_column(&step.column, col, cmp::neq, step.value.as_ref())?,
        FilterOp::Lt => compare_column(&step.column, col, cmp::lt, step.value.as_ref())?,
        FilterOp::Le => compare_column(&step.column, col, cmp::lt_eq, step.value.as_ref())?,
        FilterOp::Gt => compare_column(&step.column, col, cmp::gt, step.value.as_ref())?,
        FilterOp::Ge => compare_column(&step.column, col, cmp::gt_eq, step.value.as_ref())?,
    };

    filter_record_batch(batch, &predicate).map_err(|e| Error::Runtime(format!("filter: {e}")))
}

fn compare_column(
    name: &str,
    col: &ArrayRef,
    cmp_fn: CmpFn,
    value: Option<&serde_json::Value>,
) -> Result<BooleanArray, Error> {
    let value = value.ok_or_else(|| Error::Config("filter.value is required".into()))?;
    let map_err = |e: ArrowError| Error::Runtime(format!("filter: {e}"));
    match col.data_type() {
        DataType::Int64 => {
            if let Some(i) = value.as_i64() {
                cmp_fn(col, &Int64Array::new_scalar(i)).map_err(map_err)
            } else if let Some(f) = value.as_f64() {
                let fcol = cast(col.as_ref(), &DataType::Float64)
                    .map_err(|e| Error::Runtime(format!("filter: column '{name}': {e}")))?;
                cmp_fn(&fcol, &Float64Array::new_scalar(f)).map_err(map_err)
            } else {
                Err(Error::Runtime(format!(
                    "filter: column '{name}' is int64, incompatible value {value}"
                )))
            }
        }
        DataType::Float64 => {
            let f = value.as_f64().ok_or_else(|| {
                Error::Runtime(format!(
                    "filter: column '{name}' is float64, incompatible value {value}"
                ))
            })?;
            cmp_fn(col, &Float64Array::new_scalar(f)).map_err(map_err)
        }
        DataType::Utf8 => {
            let s = value.as_str().ok_or_else(|| {
                Error::Runtime(format!(
                    "filter: column '{name}' is string, incompatible value {value}"
                ))
            })?;
            cmp_fn(col, &StringArray::new_scalar(s)).map_err(map_err)
        }
        DataType::Boolean => {
            let b = value.as_bool().ok_or_else(|| {
                Error::Runtime(format!(
                    "filter: column '{name}' is bool, incompatible value {value}"
                ))
            })?;
            cmp_fn(col, &BooleanArray::new_scalar(b)).map_err(map_err)
        }
        dt => Err(Error::Runtime(format!(
            "filter: column '{name}' has unsupported type {dt:?}"
        ))),
    }
}

/// Fits `step` (`select`, `impute`, `scale` or `encode`) on `train`; `target` is never
/// transformed and is excluded from default column sets.
///
/// # Errors
///
/// [`Error::Runtime`] if a named column is missing or has an unsupported type;
/// [`Error::Config`] for `filter` / `split` (not fitted steps).
pub(crate) fn fit(step: &Step, target: &str, train: &RecordBatch) -> Result<FittedStep, Error> {
    match step {
        Step::Select(s) => fit_select(s, target, train),
        Step::Impute(s) => fit_impute(s, target, train),
        Step::Scale(s) => fit_scale(s, target, train),
        Step::Encode(s) => fit_encode(s, target, train),
        Step::Filter(_) => Err(Error::Config("`filter` is not a fitted step".into())),
        Step::Split(_) => Err(Error::Config("`split` is not a fitted step".into())),
    }
}

fn fit_select(step: &SelectStep, target: &str, train: &RecordBatch) -> Result<FittedStep, Error> {
    let schema = train.schema();
    let mut columns = Vec::with_capacity(step.columns.len() + 1);
    for name in &step.columns {
        if schema.index_of(name).is_err() {
            return Err(Error::Runtime(format!("select: unknown column '{name}'")));
        }
        columns.push(name.clone());
    }
    if !columns.iter().any(|c| c == target) && schema.index_of(target).is_ok() {
        columns.push(target.to_owned());
    }
    Ok(FittedStep::Select { columns })
}

fn is_numeric(dt: &DataType) -> bool {
    matches!(dt, DataType::Int64 | DataType::Float64)
}

fn is_utf8(dt: &DataType) -> bool {
    matches!(dt, DataType::Utf8)
}

fn resolve_columns(
    step_name: &str,
    explicit: Option<&[String]>,
    target: &str,
    train: &RecordBatch,
    default_pred: fn(&DataType) -> bool,
) -> Result<Vec<(String, usize)>, Error> {
    let schema = train.schema();
    match explicit {
        Some(cols) => {
            let mut resolved = Vec::with_capacity(cols.len());
            for name in cols {
                if name == target {
                    return Err(Error::Runtime(format!(
                        "{step_name}: target column '{target}' cannot be transformed"
                    )));
                }
                let idx = schema
                    .index_of(name)
                    .map_err(|_| Error::Runtime(format!("{step_name}: unknown column '{name}'")))?;
                resolved.push((name.clone(), idx));
            }
            Ok(resolved)
        }
        None => Ok(schema
            .fields()
            .iter()
            .enumerate()
            .filter(|(_, f)| f.name() != target && default_pred(f.data_type()))
            .map(|(idx, f)| (f.name().clone(), idx))
            .collect()),
    }
}

fn to_float64(step_name: &str, name: &str, array: &ArrayRef) -> Result<ArrayRef, Error> {
    match array.data_type() {
        DataType::Float64 => Ok(Arc::clone(array)),
        DataType::Int64 => cast(array.as_ref(), &DataType::Float64)
            .map_err(|e| Error::Runtime(format!("{step_name}: column '{name}': {e}"))),
        dt => Err(Error::Runtime(format!(
            "{step_name}: column '{name}' has unsupported type {dt:?}"
        ))),
    }
}

fn check_finite(step_name: &str, name: &str, v: f64) -> Result<f64, Error> {
    if !v.is_finite() {
        return Err(Error::Runtime(format!(
            "{step_name}: column '{name}' contains non-finite value"
        )));
    }
    Ok(v)
}

fn numeric_values(step_name: &str, name: &str, array: &ArrayRef) -> Result<Vec<f64>, Error> {
    let fcol = to_float64(step_name, name, array)?;
    let arr = fcol.as_primitive::<Float64Type>();
    let mut vals = Vec::with_capacity(arr.len() - arr.null_count());
    for v in arr.iter().flatten() {
        vals.push(check_finite(step_name, name, v)?);
    }
    Ok(vals)
}

fn normalize_zero(v: f64) -> f64 {
    if v == 0.0 { 0.0 } else { v }
}

fn json_number(step_name: &str, name: &str, v: f64) -> Result<serde_json::Value, Error> {
    let n = serde_json::Number::from_f64(normalize_zero(v)).ok_or_else(|| {
        Error::Runtime(format!(
            "{step_name}: column '{name}' produced non-finite value"
        ))
    })?;
    Ok(serde_json::Value::Number(n))
}

fn fit_impute(step: &ImputeStep, target: &str, train: &RecordBatch) -> Result<FittedStep, Error> {
    let mut values = BTreeMap::new();

    match step.strategy {
        ImputeStrategy::Mean | ImputeStrategy::Median => {
            let cols =
                resolve_columns("impute", step.columns.as_deref(), target, train, is_numeric)?;
            for (name, idx) in cols {
                let mut vals = numeric_values("impute", &name, train.column(idx))?;
                if vals.is_empty() {
                    return Err(Error::Runtime(format!(
                        "impute: column '{name}' has no non-null values"
                    )));
                }
                let stat = if matches!(step.strategy, ImputeStrategy::Mean) {
                    vals.iter().sum::<f64>() / (vals.len() as f64)
                } else {
                    vals.sort_by(f64::total_cmp);
                    let mid = vals.len() / 2;
                    if vals.len() % 2 == 1 {
                        vals[mid]
                    } else {
                        vals[mid - 1] + (vals[mid] - vals[mid - 1]) / 2.0
                    }
                };
                values.insert(name.clone(), json_number("impute", &name, stat)?);
            }
        }
        ImputeStrategy::Constant => {
            let val = step.value.as_ref().ok_or_else(|| {
                Error::Config("impute.value is required with strategy constant".into())
            })?;
            let default_pred = if val.is_string() { is_utf8 } else { is_numeric };
            let cols = resolve_columns(
                "impute",
                step.columns.as_deref(),
                target,
                train,
                default_pred,
            )?;
            for (name, idx) in cols {
                let dt = train.column(idx).data_type();
                match dt {
                    DataType::Int64 | DataType::Float64 if val.is_number() => {}
                    DataType::Utf8 if val.is_string() => {}
                    DataType::Int64 | DataType::Float64 => {
                        return Err(Error::Runtime(format!(
                            "impute: column '{name}' is numeric, incompatible constant {val}"
                        )));
                    }
                    DataType::Utf8 => {
                        return Err(Error::Runtime(format!(
                            "impute: column '{name}' is string, incompatible constant {val}"
                        )));
                    }
                    _ => {
                        return Err(Error::Runtime(format!(
                            "impute: column '{name}' has unsupported type {dt:?}"
                        )));
                    }
                }
                values.insert(name, val.clone());
            }
        }
    }

    Ok(FittedStep::Impute { values })
}

fn fit_scale(step: &ScaleStep, target: &str, train: &RecordBatch) -> Result<FittedStep, Error> {
    let cols = resolve_columns("scale", step.columns.as_deref(), target, train, is_numeric)?;
    let mut params = BTreeMap::new();

    for (name, idx) in cols {
        let vals = numeric_values("scale", &name, train.column(idx))?;
        if vals.is_empty() {
            return Err(Error::Runtime(format!(
                "scale: column '{name}' has no non-null values"
            )));
        }
        let affine = match step.method {
            ScaleMethod::Standard => {
                let n = vals.len() as f64;
                let mean = vals.iter().sum::<f64>() / n;
                let var = vals.iter().map(|&x| (x - mean) * (x - mean)).sum::<f64>() / n;
                Affine {
                    offset: normalize_zero(mean),
                    scale: normalize_zero(var.sqrt()),
                }
            }
            ScaleMethod::Minmax => {
                let (min, max) = vals
                    .iter()
                    .copied()
                    .fold((vals[0], vals[0]), |(lo, hi), v| (lo.min(v), hi.max(v)));
                let range = max - min;
                if !range.is_finite() {
                    return Err(Error::Runtime(format!(
                        "scale: column '{name}' range overflows f64"
                    )));
                }
                Affine {
                    offset: normalize_zero(min),
                    scale: normalize_zero(range),
                }
            }
        };
        if !affine.offset.is_finite() || !affine.scale.is_finite() {
            return Err(Error::Runtime(format!(
                "scale: column '{name}' produced non-finite parameters"
            )));
        }
        params.insert(name, affine);
    }

    Ok(FittedStep::Scale { params })
}

fn fit_encode(step: &EncodeStep, target: &str, train: &RecordBatch) -> Result<FittedStep, Error> {
    let cols = resolve_columns("encode", step.columns.as_deref(), target, train, is_utf8)?;
    let mut categories = BTreeMap::new();

    for (name, idx) in cols {
        let array = train.column(idx);
        if array.data_type() != &DataType::Utf8 {
            return Err(Error::Runtime(format!(
                "encode: column '{name}' has unsupported type {:?}",
                array.data_type()
            )));
        }
        let seen: BTreeSet<String> = array
            .as_string::<i32>()
            .iter()
            .flatten()
            .map(str::to_owned)
            .collect();
        categories.insert(name, seen.into_iter().collect());
    }

    Ok(match step.method {
        EncodeMethod::Ordinal => FittedStep::Ordinal { categories },
        EncodeMethod::OneHot => FittedStep::OneHot { categories },
    })
}

/// Extracts an exact `i64` from a JSON number, accepting whole-number floats in `i64` range.
pub(super) fn json_as_exact_i64(v: &serde_json::Value) -> Option<i64> {
    if let Some(i) = v.as_i64() {
        return Some(i);
    }
    let f = v.as_f64()?;
    if f.is_finite() && f.fract() == 0.0 && f >= (i64::MIN as f64) && f < -(i64::MIN as f64) {
        let i = f as i64;
        if (i as f64) == f {
            return Some(i);
        }
    }
    None
}

fn build_batch(
    fields: Vec<Field>,
    arrays: Vec<ArrayRef>,
    num_rows: usize,
) -> Result<RecordBatch, Error> {
    let options = RecordBatchOptions::new().with_row_count(Some(num_rows));
    RecordBatch::try_new_with_options(Arc::new(Schema::new(fields)), arrays, &options)
        .map_err(|e| Error::Runtime(format!("preprocess: {e}")))
}

fn map_columns<P, F>(
    step_name: &str,
    params: &BTreeMap<String, P>,
    batch: &RecordBatch,
    mut transform: F,
) -> Result<RecordBatch, Error>
where
    F: FnMut(&Field, &ArrayRef, &P, &mut Vec<Field>, &mut Vec<ArrayRef>) -> Result<(), Error>,
{
    let schema = batch.schema();
    for name in params.keys() {
        if schema.index_of(name).is_err() {
            return Err(Error::Runtime(format!(
                "{step_name}: missing column '{name}'"
            )));
        }
    }

    let mut fields = Vec::with_capacity(schema.fields().len());
    let mut arrays = Vec::with_capacity(schema.fields().len());
    for (idx, field) in schema.fields().iter().enumerate() {
        let array = batch.column(idx);
        match params.get(field.name()) {
            Some(param) => transform(field, array, param, &mut fields, &mut arrays)?,
            None => {
                fields.push(field.as_ref().clone());
                arrays.push(Arc::clone(array));
            }
        }
    }

    build_batch(fields, arrays, batch.num_rows())
}

fn category_lookup(cats: &[String]) -> HashMap<&str, usize> {
    cats.iter()
        .enumerate()
        .map(|(i, c)| (c.as_str(), i))
        .collect()
}

impl FittedStep {
    /// Applies this fitted step to `batch`; a missing target column is allowed (serving).
    pub(crate) fn apply(&self, batch: &RecordBatch) -> Result<RecordBatch, Error> {
        self.apply_with_target(batch, None)
    }

    /// Applies this fitted step to `batch`; when `target` is `Some`, a missing `target` column in
    /// `Select` is skipped (serving), while any other missing required column returns an error.
    pub(super) fn apply_with_target(
        &self,
        batch: &RecordBatch,
        target: Option<&str>,
    ) -> Result<RecordBatch, Error> {
        match self {
            Self::Select { columns } => apply_select(columns, target, batch),
            Self::Impute { values } => apply_impute(values, batch),
            Self::Scale { params } => apply_scale(params, batch),
            Self::Ordinal { categories } => apply_ordinal(categories, batch),
            Self::OneHot { categories } => apply_one_hot(categories, batch),
        }
    }
}

fn apply_select(
    columns: &[String],
    target: Option<&str>,
    batch: &RecordBatch,
) -> Result<RecordBatch, Error> {
    let schema = batch.schema();
    let mut fields = Vec::with_capacity(columns.len());
    let mut arrays = Vec::with_capacity(columns.len());
    for name in columns {
        match schema.index_of(name) {
            Ok(idx) => {
                fields.push(schema.field(idx).as_ref().clone());
                arrays.push(Arc::clone(batch.column(idx)));
            }
            Err(_) if target == Some(name.as_str()) => {}
            Err(_) => {
                return Err(Error::Runtime(format!("select: missing column '{name}'")));
            }
        }
    }
    build_batch(fields, arrays, batch.num_rows())
}

fn fill_float64(name: &str, array: &ArrayRef, fill: f64) -> Result<Float64Array, Error> {
    let arr = array.as_primitive::<Float64Type>();
    let mut vals = Vec::with_capacity(arr.len());
    for opt in arr.iter() {
        vals.push(match opt {
            Some(v) => check_finite("impute", name, v)?,
            None => fill,
        });
    }
    Ok(Float64Array::from(vals))
}

fn apply_impute(
    values: &BTreeMap<String, serde_json::Value>,
    batch: &RecordBatch,
) -> Result<RecordBatch, Error> {
    map_columns(
        "impute",
        values,
        batch,
        |field, array, val, fields, arrays| {
            let name = field.name();
            match array.data_type() {
                DataType::Int64 => {
                    if !val.is_number() {
                        return Err(Error::Runtime(format!(
                            "impute: column '{name}' is int64, incompatible value {val}"
                        )));
                    }
                    if let Some(fill) = val.as_i64() {
                        let arr = array.as_primitive::<Int64Type>();
                        let out: Int64Array = arr.iter().map(|opt| opt.unwrap_or(fill)).collect();
                        fields.push(Field::new(name, DataType::Int64, false));
                        arrays.push(Arc::new(out));
                    } else {
                        let fill = val.as_f64().filter(|f| f.is_finite()).ok_or_else(|| {
                            Error::Runtime(format!(
                                "impute: column '{name}' has non-finite value {val}"
                            ))
                        })?;
                        let casted = to_float64("impute", name, array)?;
                        fields.push(Field::new(name, DataType::Float64, false));
                        arrays.push(Arc::new(fill_float64(name, &casted, fill)?));
                    }
                }
                DataType::Float64 => {
                    let fill = val.as_f64().filter(|f| f.is_finite()).ok_or_else(|| {
                        Error::Runtime(format!(
                            "impute: column '{name}' is float64, incompatible value {val}"
                        ))
                    })?;
                    fields.push(Field::new(name, DataType::Float64, false));
                    arrays.push(Arc::new(fill_float64(name, array, fill)?));
                }
                DataType::Utf8 => {
                    let fill = val.as_str().ok_or_else(|| {
                        Error::Runtime(format!(
                            "impute: column '{name}' is string, incompatible value {val}"
                        ))
                    })?;
                    let out: StringArray = array
                        .as_string::<i32>()
                        .iter()
                        .map(|opt| Some(opt.unwrap_or(fill)))
                        .collect();
                    fields.push(Field::new(name, DataType::Utf8, false));
                    arrays.push(Arc::new(out));
                }
                dt => {
                    return Err(Error::Runtime(format!(
                        "impute: column '{name}' has unsupported type {dt:?}"
                    )));
                }
            }
            Ok(())
        },
    )
}

fn apply_scale(
    params: &BTreeMap<String, Affine>,
    batch: &RecordBatch,
) -> Result<RecordBatch, Error> {
    map_columns(
        "scale",
        params,
        batch,
        |field, array, affine, fields, arrays| {
            let name = field.name();
            let fcol = to_float64("scale", name, array)?;
            let arr = fcol.as_primitive::<Float64Type>();
            let mut vals = Vec::with_capacity(arr.len());
            for opt in arr.iter() {
                match opt {
                    Some(x) => {
                        check_finite("scale", name, x)?;
                        let scaled = if affine.scale == 0.0 {
                            0.0
                        } else {
                            (x - affine.offset) / affine.scale
                        };
                        check_finite("scale", name, scaled)?;
                        vals.push(Some(scaled));
                    }
                    None => vals.push(None),
                }
            }
            fields.push(Field::new(name, DataType::Float64, field.is_nullable()));
            arrays.push(Arc::new(Float64Array::from(vals)));
            Ok(())
        },
    )
}

fn apply_ordinal(
    categories: &BTreeMap<String, Vec<String>>,
    batch: &RecordBatch,
) -> Result<RecordBatch, Error> {
    map_columns(
        "encode",
        categories,
        batch,
        |field, array, cats, fields, arrays| {
            let name = field.name();
            if array.data_type() != &DataType::Utf8 {
                return Err(Error::Runtime(format!(
                    "encode: column '{name}' has unsupported type {:?}",
                    array.data_type()
                )));
            }
            let lookup = category_lookup(cats);
            let out: Int64Array = array
                .as_string::<i32>()
                .iter()
                .map(|opt| {
                    opt.and_then(|v| lookup.get(v).map(|&i| i as i64))
                        .unwrap_or(-1)
                })
                .collect();
            fields.push(Field::new(name, DataType::Int64, false));
            arrays.push(Arc::new(out));
            Ok(())
        },
    )
}

fn apply_one_hot(
    categories: &BTreeMap<String, Vec<String>>,
    batch: &RecordBatch,
) -> Result<RecordBatch, Error> {
    let schema = batch.schema();
    let other_names: HashSet<&str> = schema
        .fields()
        .iter()
        .map(|f| f.name().as_str())
        .filter(|name| !categories.contains_key(*name))
        .collect();
    let mut produced = HashSet::new();
    let n_rows = batch.num_rows();

    map_columns(
        "encode",
        categories,
        batch,
        |field, array, cats, fields, arrays| {
            let name = field.name();
            if array.data_type() != &DataType::Utf8 {
                return Err(Error::Runtime(format!(
                    "encode: column '{name}' has unsupported type {:?}",
                    array.data_type()
                )));
            }
            let lookup = category_lookup(cats);
            let mut buffers = vec![vec![0.0_f64; n_rows]; cats.len()];
            for (row, opt) in array.as_string::<i32>().iter().enumerate() {
                if let Some(val) = opt
                    && let Some(&cat_idx) = lookup.get(val)
                {
                    buffers[cat_idx][row] = 1.0;
                }
            }
            for (cat, buf) in cats.iter().zip(buffers) {
                let col_name = format!("{name}={cat}");
                if other_names.contains(col_name.as_str()) || !produced.insert(col_name.clone()) {
                    return Err(Error::Runtime(format!(
                        "encode: generated column '{col_name}' collides with an existing column"
                    )));
                }
                fields.push(Field::new(col_name, DataType::Float64, false));
                arrays.push(Arc::new(Float64Array::from(buf)));
            }
            Ok(())
        },
    )
}

#[cfg(test)]
mod tests {
    use arrow_array::Date32Array;
    use serde_json::json;

    use super::*;
    use crate::preprocess::SplitStep;

    fn batch(cols: &[(&str, ArrayRef)]) -> RecordBatch {
        let fields: Vec<Field> = cols
            .iter()
            .map(|(name, arr)| Field::new(*name, arr.data_type().clone(), true))
            .collect();
        let arrays: Vec<ArrayRef> = cols.iter().map(|(_, arr)| Arc::clone(arr)).collect();
        RecordBatch::try_new(Arc::new(Schema::new(fields)), arrays).unwrap()
    }

    fn i64_col(batch: &RecordBatch, name: &str) -> Vec<Option<i64>> {
        let idx = batch.schema().index_of(name).unwrap();
        batch
            .column(idx)
            .as_primitive::<Int64Type>()
            .iter()
            .collect()
    }

    fn f64_col(batch: &RecordBatch, name: &str) -> Vec<Option<f64>> {
        let idx = batch.schema().index_of(name).unwrap();
        batch
            .column(idx)
            .as_primitive::<Float64Type>()
            .iter()
            .collect()
    }

    fn str_col(batch: &RecordBatch, name: &str) -> Vec<Option<String>> {
        let idx = batch.schema().index_of(name).unwrap();
        batch
            .column(idx)
            .as_string::<i32>()
            .iter()
            .map(|opt| opt.map(str::to_owned))
            .collect()
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
    fn preprocess_steps_filter_comparisons_and_nulls() {
        let b = batch(&[
            (
                "i",
                Arc::new(Int64Array::from(vec![Some(1), Some(2), None, Some(3)])),
            ),
            (
                "f",
                Arc::new(Float64Array::from(vec![
                    Some(1.5),
                    None,
                    Some(2.5),
                    Some(3.5),
                ])),
            ),
            (
                "s",
                Arc::new(StringArray::from(vec![
                    Some("a"),
                    Some("b"),
                    Some("c"),
                    None,
                ])),
            ),
            (
                "flag",
                Arc::new(BooleanArray::from(vec![
                    Some(true),
                    Some(false),
                    None,
                    Some(true),
                ])),
            ),
            (
                "d",
                Arc::new(Date32Array::from(vec![Some(10), None, Some(20), Some(30)])),
            ),
        ]);

        // Comparisons on int64: null never matches (including ne).
        let eq = filter(
            &FilterStep {
                column: "i".into(),
                op: FilterOp::Eq,
                value: Some(json!(2)),
            },
            &b,
        )
        .unwrap();
        assert_eq!(i64_col(&eq, "i"), vec![Some(2)]);

        let ne = filter(
            &FilterStep {
                column: "i".into(),
                op: FilterOp::Ne,
                value: Some(json!(2)),
            },
            &b,
        )
        .unwrap();
        assert_eq!(i64_col(&ne, "i"), vec![Some(1), Some(3)]);

        let gt_float = filter(
            &FilterStep {
                column: "i".into(),
                op: FilterOp::Gt,
                value: Some(json!(1.5)),
            },
            &b,
        )
        .unwrap();
        assert_eq!(i64_col(&gt_float, "i"), vec![Some(2), Some(3)]);

        // Float64 comparisons with integer and float JSON numbers.
        let ge = filter(
            &FilterStep {
                column: "f".into(),
                op: FilterOp::Ge,
                value: Some(json!(2)),
            },
            &b,
        )
        .unwrap();
        assert_eq!(f64_col(&ge, "f"), vec![Some(2.5), Some(3.5)]);

        let lt = filter(
            &FilterStep {
                column: "f".into(),
                op: FilterOp::Lt,
                value: Some(json!(2.5)),
            },
            &b,
        )
        .unwrap();
        assert_eq!(f64_col(&lt, "f"), vec![Some(1.5)]);

        // String comparisons.
        let le = filter(
            &FilterStep {
                column: "s".into(),
                op: FilterOp::Le,
                value: Some(json!("b")),
            },
            &b,
        )
        .unwrap();
        assert_eq!(str_col(&le, "s"), vec![Some("a".into()), Some("b".into())]);

        // Bool comparison: null never matches.
        let b_eq = filter(
            &FilterStep {
                column: "flag".into(),
                op: FilterOp::Eq,
                value: Some(json!(true)),
            },
            &b,
        )
        .unwrap();
        assert_eq!(i64_col(&b_eq, "i"), vec![Some(1), Some(3)]);

        // is_null and not_null (including on date32).
        let is_null = filter(
            &FilterStep {
                column: "d".into(),
                op: FilterOp::IsNull,
                value: None,
            },
            &b,
        )
        .unwrap();
        assert_eq!(i64_col(&is_null, "i"), vec![Some(2)]);

        let not_null = filter(
            &FilterStep {
                column: "i".into(),
                op: FilterOp::NotNull,
                value: None,
            },
            &b,
        )
        .unwrap();
        assert_eq!(i64_col(&not_null, "i"), vec![Some(1), Some(2), Some(3)]);
    }

    #[test]
    fn preprocess_steps_filter_errors_on_bad_types_and_missing_column() {
        let b = batch(&[
            ("i", Arc::new(Int64Array::from(vec![1, 2]))),
            ("f", Arc::new(Float64Array::from(vec![1.0, 2.0]))),
            ("s", Arc::new(StringArray::from(vec!["a", "b"]))),
            ("flag", Arc::new(BooleanArray::from(vec![true, false]))),
            ("d", Arc::new(Date32Array::from(vec![10, 20]))),
        ]);

        for (col, val) in [
            ("missing", json!(1)),
            ("i", json!("1")),
            ("i", json!(true)),
            ("f", json!("1.0")),
            ("s", json!(1)),
            ("flag", json!(1)),
            ("d", json!("2026-01-01")),
        ] {
            let err = filter(
                &FilterStep {
                    column: col.into(),
                    op: FilterOp::Eq,
                    value: Some(val),
                },
                &b,
            )
            .unwrap_err();
            assert!(matches!(err, Error::Runtime(_)), "{col}: {err:?}");
            assert!(err.to_string().contains(col), "{col}: {err}");
        }
    }

    #[test]
    fn preprocess_steps_fit_rejects_non_fitted_steps() {
        let b = batch(&[("x", Arc::new(Int64Array::from(vec![1])))]);
        let filter_step = Step::Filter(FilterStep {
            column: "x".into(),
            op: FilterOp::NotNull,
            value: None,
        });
        let split_step = Step::Split(SplitStep {
            test: 0.2,
            validation: None,
            seed: 1,
        });
        assert!(matches!(fit(&filter_step, "y", &b), Err(Error::Config(_))));
        assert!(matches!(fit(&split_step, "y", &b), Err(Error::Config(_))));
    }

    #[test]
    fn preprocess_steps_select_keeps_order_and_optional_target() {
        let train = batch(&[
            ("a", Arc::new(Int64Array::from(vec![1, 2]))),
            ("b", Arc::new(StringArray::from(vec!["x", "y"]))),
            ("c", Arc::new(Float64Array::from(vec![3.0, 4.0]))),
            ("y", Arc::new(Int64Array::from(vec![0, 1]))),
        ]);
        let step = Step::Select(SelectStep {
            columns: vec!["c".into(), "a".into()],
        });
        let fitted = fit(&step, "y", &train).unwrap();
        assert_eq!(
            fitted,
            FittedStep::Select {
                columns: vec!["c".into(), "a".into(), "y".into()]
            }
        );

        let applied = fitted.apply(&train).unwrap();
        assert_eq!(col_names(&applied), vec!["c", "a", "y"]);

        // Serving batch without target.
        let serving = batch(&[
            ("a", Arc::new(Int64Array::from(vec![10]))),
            ("b", Arc::new(StringArray::from(vec!["z"]))),
            ("c", Arc::new(Float64Array::from(vec![9.0]))),
        ]);
        let applied_serving = fitted.apply_with_target(&serving, Some("y")).unwrap();
        assert_eq!(col_names(&applied_serving), vec!["c", "a"]);

        // Missing required feature column fails.
        let missing_feat = batch(&[("a", Arc::new(Int64Array::from(vec![10])))]);
        let err = fitted
            .apply_with_target(&missing_feat, Some("y"))
            .unwrap_err();
        assert!(matches!(err, Error::Runtime(_)));
        assert!(err.to_string().contains('c'));

        // Unknown column at fit time fails with column name.
        let bad = Step::Select(SelectStep {
            columns: vec!["nope".into()],
        });
        let err = fit(&bad, "y", &train).unwrap_err();
        assert!(matches!(err, Error::Runtime(_)));
        assert!(err.to_string().contains("nope"));

        // Target already listed is not duplicated.
        let with_target = Step::Select(SelectStep {
            columns: vec!["a".into(), "y".into()],
        });
        assert_eq!(
            fit(&with_target, "y", &train).unwrap(),
            FittedStep::Select {
                columns: vec!["a".into(), "y".into()]
            }
        );
    }

    #[test]
    fn preprocess_steps_impute_mean_median_constant_and_int_vs_float() {
        let train = batch(&[
            (
                "i_exact",
                Arc::new(Int64Array::from(vec![Some(2), Some(4), None, Some(3)])),
            ),
            (
                "i_frac",
                Arc::new(Int64Array::from(vec![Some(1), Some(2), None, Some(1)])),
            ),
            (
                "f",
                Arc::new(Float64Array::from(vec![
                    Some(1.0),
                    Some(10.0),
                    Some(2.0),
                    None,
                ])),
            ),
            (
                "s",
                Arc::new(StringArray::from(vec![Some("a"), None, Some("b"), None])),
            ),
            (
                "y",
                Arc::new(Int64Array::from(vec![Some(1), None, Some(0), Some(1)])),
            ),
        ]);

        // Mean: default columns are numeric non-target; int64 ALWAYS becomes float64 (data-independent schema).
        let mean_step = Step::Impute(ImputeStep {
            strategy: ImputeStrategy::Mean,
            value: None,
            columns: None,
        });
        let fitted_mean = fit(&mean_step, "y", &train).unwrap();
        let out = fitted_mean.apply(&train).unwrap();
        assert_eq!(col_names(&out), col_names(&train));
        assert_eq!(out.schema().field(0).data_type(), &DataType::Float64);
        assert_eq!(
            f64_col(&out, "i_exact"),
            vec![Some(2.0), Some(4.0), Some(3.0), Some(3.0)]
        );
        assert_eq!(out.schema().field(1).data_type(), &DataType::Float64);
        let frac_vals = f64_col(&out, "i_frac");
        assert!((frac_vals[2].unwrap() - 4.0 / 3.0).abs() < 1e-12);
        let f_vals = f64_col(&out, "f");
        assert!((f_vals[3].unwrap() - 13.0 / 3.0).abs() < 1e-12);
        // s and y untouched
        assert_eq!(
            str_col(&out, "s"),
            vec![Some("a".into()), None, Some("b".into()), None]
        );
        assert_eq!(i64_col(&out, "y"), vec![Some(1), None, Some(0), Some(1)]);

        // Median: int64 also becomes float64 even when median is whole (1.0).
        let med_step = Step::Impute(ImputeStep {
            strategy: ImputeStrategy::Median,
            value: None,
            columns: Some(vec!["f".into(), "i_frac".into()]),
        });
        let fitted_med = fit(&med_step, "y", &train).unwrap();
        let out_med = fitted_med.apply(&train).unwrap();
        assert_eq!(out_med.schema().field(1).data_type(), &DataType::Float64);
        assert_eq!(
            f64_col(&out_med, "i_frac"),
            vec![Some(1.0), Some(2.0), Some(1.0), Some(1.0)]
        );
        assert_eq!(
            f64_col(&out_med, "f"),
            vec![Some(1.0), Some(10.0), Some(2.0), Some(2.0)]
        );

        // Constant string without explicit columns defaults to utf8 non-target columns.
        let const_str_default = Step::Impute(ImputeStep {
            strategy: ImputeStrategy::Constant,
            value: Some(json!("missing")),
            columns: None,
        });
        let out_str = fit(&const_str_default, "y", &train)
            .unwrap()
            .apply(&train)
            .unwrap();
        assert_eq!(
            str_col(&out_str, "s"),
            vec![
                Some("a".into()),
                Some("missing".into()),
                Some("b".into()),
                Some("missing".into()),
            ]
        );
        // Numeric columns left untouched by string constant default.
        assert_eq!(
            i64_col(&out_str, "i_exact"),
            vec![Some(2), Some(4), None, Some(3)]
        );

        // Constant integer on int64 stays int64; constant float casts to float64.
        let const_int = Step::Impute(ImputeStep {
            strategy: ImputeStrategy::Constant,
            value: Some(json!(99)),
            columns: Some(vec!["i_exact".into()]),
        });
        let out_ci = fit(&const_int, "y", &train).unwrap().apply(&train).unwrap();
        assert_eq!(out_ci.schema().field(0).data_type(), &DataType::Int64);
        assert_eq!(
            i64_col(&out_ci, "i_exact"),
            vec![Some(2), Some(4), Some(99), Some(3)]
        );

        let const_float = Step::Impute(ImputeStep {
            strategy: ImputeStrategy::Constant,
            value: Some(json!(0.5)),
            columns: Some(vec!["i_exact".into()]),
        });
        let out_cf = fit(&const_float, "y", &train)
            .unwrap()
            .apply(&train)
            .unwrap();
        assert_eq!(out_cf.schema().field(0).data_type(), &DataType::Float64);
        assert_eq!(
            f64_col(&out_cf, "i_exact"),
            vec![Some(2.0), Some(4.0), Some(0.5), Some(3.0)]
        );

        // Even-length median on int64 producing non-integer (2.5) casts to float64.
        let even_batch = batch(&[(
            "x",
            Arc::new(Int64Array::from(vec![
                Some(1),
                Some(2),
                Some(3),
                Some(10),
                None,
            ])),
        )]);
        let even_med = fit(
            &Step::Impute(ImputeStep {
                strategy: ImputeStrategy::Median,
                value: None,
                columns: None,
            }),
            "y",
            &even_batch,
        )
        .unwrap()
        .apply(&even_batch)
        .unwrap();
        assert_eq!(even_med.schema().field(0).data_type(), &DataType::Float64);
        assert_eq!(
            f64_col(&even_med, "x"),
            vec![Some(1.0), Some(2.0), Some(3.0), Some(10.0), Some(2.5)]
        );

        // All-null column errors on mean/median.
        let all_null = batch(&[("x", Arc::new(Float64Array::from(vec![None, None])))]);
        let err = fit(&mean_step, "y", &all_null).unwrap_err();
        assert!(matches!(err, Error::Runtime(_)));
        assert!(err.to_string().contains('x'));

        // Explicit target or unsupported type errors with column name.
        let err_target = fit(
            &Step::Impute(ImputeStep {
                strategy: ImputeStrategy::Mean,
                value: None,
                columns: Some(vec!["y".into()]),
            }),
            "y",
            &train,
        )
        .unwrap_err();
        assert!(err_target.to_string().contains('y'));

        let err_type = fit(
            &Step::Impute(ImputeStep {
                strategy: ImputeStrategy::Mean,
                value: None,
                columns: Some(vec!["s".into()]),
            }),
            "y",
            &train,
        )
        .unwrap_err();
        assert!(err_type.to_string().contains('s'));

        // Non-finite values error at apply time too.
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let bad_batch = batch(&[("f", Arc::new(Float64Array::from(vec![Some(bad), None])))]);
            let fitted = FittedStep::Impute {
                values: BTreeMap::from([("f".into(), json!(0.0))]),
            };
            let err = fitted.apply(&bad_batch).unwrap_err();
            assert!(matches!(err, Error::Runtime(_)));
            assert!(
                err.to_string().contains("'f'") && err.to_string().contains("non-finite"),
                "{err}"
            );
        }
    }

    #[test]
    fn preprocess_steps_scale_standard_minmax_zero_variance_and_nulls() {
        let train = batch(&[
            (
                "x",
                Arc::new(Int64Array::from(vec![Some(10), Some(20), Some(30), None])),
            ),
            (
                "const_col",
                Arc::new(Float64Array::from(vec![
                    Some(5.0),
                    Some(5.0),
                    Some(5.0),
                    None,
                ])),
            ),
            ("y", Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0, 4.0]))),
        ]);

        let std_step = Step::Scale(ScaleStep {
            method: ScaleMethod::Standard,
            columns: None,
        });
        let fitted_std = fit(&std_step, "y", &train).unwrap();
        let out_std = fitted_std.apply(&train).unwrap();
        assert_eq!(out_std.schema().field(0).data_type(), &DataType::Float64);
        let x_std = f64_col(&out_std, "x");
        let expected_std = (200.0_f64 / 3.0).sqrt();
        assert!((x_std[0].unwrap() - (-10.0 / expected_std)).abs() < 1e-12);
        assert!(x_std[1].unwrap().abs() < 1e-12);
        assert!((x_std[2].unwrap() - (10.0 / expected_std)).abs() < 1e-12);
        assert_eq!(x_std[3], None);
        assert_eq!(
            f64_col(&out_std, "const_col"),
            vec![Some(0.0), Some(0.0), Some(0.0), None]
        );
        assert_eq!(
            f64_col(&out_std, "y"),
            vec![Some(1.0), Some(2.0), Some(3.0), Some(4.0)]
        );

        let mm_step = Step::Scale(ScaleStep {
            method: ScaleMethod::Minmax,
            columns: None,
        });
        let fitted_mm = fit(&mm_step, "y", &train).unwrap();
        let out_mm = fitted_mm.apply(&train).unwrap();
        assert_eq!(
            f64_col(&out_mm, "x"),
            vec![Some(0.0), Some(0.5), Some(1.0), None]
        );
        assert_eq!(
            f64_col(&out_mm, "const_col"),
            vec![Some(0.0), Some(0.0), Some(0.0), None]
        );

        // All-null, missing column, target column error with column name.
        let all_null = batch(&[("z", Arc::new(Float64Array::from(vec![None, None])))]);
        assert!(
            fit(&std_step, "y", &all_null)
                .unwrap_err()
                .to_string()
                .contains('z')
        );

        for (cols, needle) in [(vec!["missing".into()], "missing"), (vec!["y".into()], "y")] {
            let err = fit(
                &Step::Scale(ScaleStep {
                    method: ScaleMethod::Standard,
                    columns: Some(cols),
                }),
                "y",
                &train,
            )
            .unwrap_err();
            assert!(matches!(err, Error::Runtime(_)));
            assert!(err.to_string().contains(needle), "{err}");
        }

        // Non-finite values error at apply time too.
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let bad_batch = batch(&[("x", Arc::new(Float64Array::from(vec![Some(bad)])))]);
            let fitted = FittedStep::Scale {
                params: BTreeMap::from([(
                    "x".into(),
                    Affine {
                        offset: 0.0,
                        scale: 1.0,
                    },
                )]),
            };
            let err = fitted.apply(&bad_batch).unwrap_err();
            assert!(matches!(err, Error::Runtime(_)));
            assert!(
                err.to_string().contains("'x'") && err.to_string().contains("non-finite"),
                "{err}"
            );
        }
    }

    #[test]
    fn preprocess_steps_encode_ordinal_and_one_hot_with_unseen_and_null() {
        let train = batch(&[
            ("id", Arc::new(Int64Array::from(vec![1, 2, 3, 4]))),
            (
                "color",
                Arc::new(StringArray::from(vec![
                    Some("red"),
                    Some("blue"),
                    None,
                    Some("red"),
                ])),
            ),
            (
                "y",
                Arc::new(StringArray::from(vec!["yes", "no", "yes", "no"])),
            ),
        ]);
        let test = batch(&[
            ("id", Arc::new(Int64Array::from(vec![10, 11, 12, 13]))),
            (
                "color",
                Arc::new(StringArray::from(vec![
                    Some("blue"),
                    Some("green"), // unseen
                    None,
                    Some("red"),
                ])),
            ),
            (
                "y",
                Arc::new(StringArray::from(vec!["no", "yes", "no", "yes"])),
            ),
        ]);

        let ord_step = Step::Encode(EncodeStep {
            method: EncodeMethod::Ordinal,
            columns: None,
        });
        let fitted_ord = fit(&ord_step, "y", &train).unwrap();
        assert_eq!(
            fitted_ord,
            FittedStep::Ordinal {
                categories: BTreeMap::from([("color".into(), vec!["blue".into(), "red".into()])]),
            }
        );
        let out_ord = fitted_ord.apply(&test).unwrap();
        assert_eq!(col_names(&out_ord), vec!["id", "color", "y"]);
        assert_eq!(
            i64_col(&out_ord, "color"),
            vec![Some(0), Some(-1), Some(-1), Some(1)]
        );
        assert_eq!(
            str_col(&out_ord, "y"),
            vec![
                Some("no".into()),
                Some("yes".into()),
                Some("no".into()),
                Some("yes".into()),
            ]
        );

        let oh_step = Step::Encode(EncodeStep {
            method: EncodeMethod::OneHot,
            columns: None,
        });
        let fitted_oh = fit(&oh_step, "y", &train).unwrap();
        let out_oh = fitted_oh.apply(&test).unwrap();
        assert_eq!(
            col_names(&out_oh),
            vec!["id", "color=blue", "color=red", "y"]
        );
        assert_eq!(
            f64_col(&out_oh, "color=blue"),
            vec![Some(1.0), Some(0.0), Some(0.0), Some(0.0)]
        );
        assert_eq!(
            f64_col(&out_oh, "color=red"),
            vec![Some(0.0), Some(0.0), Some(0.0), Some(1.0)]
        );

        // OneHot rejects collision with an existing column or duplicate generated name.
        let colliding = batch(&[
            ("color", Arc::new(StringArray::from(vec!["red"]))),
            ("color=red", Arc::new(Float64Array::from(vec![1.0]))),
        ]);
        let err = fitted_oh.apply(&colliding).unwrap_err();
        assert!(matches!(err, Error::Runtime(_)));
        assert!(err.to_string().contains("color=red"), "{err}");

        let dup_oh = FittedStep::OneHot {
            categories: BTreeMap::from([
                ("a".into(), vec!["b=c".into()]),
                ("a=b".into(), vec!["c".into()]),
            ]),
        };
        let dup_batch = batch(&[
            ("a", Arc::new(StringArray::from(vec!["b=c"]))),
            ("a=b", Arc::new(StringArray::from(vec!["c"]))),
        ]);
        let err = dup_oh.apply(&dup_batch).unwrap_err();
        assert!(matches!(err, Error::Runtime(_)));
        assert!(err.to_string().contains("a=b=c"), "{err}");

        // Encode errors on non-string, missing, or target column.
        for (cols, needle) in [
            (vec!["id".into()], "id"),
            (vec!["missing".into()], "missing"),
            (vec!["y".into()], "y"),
        ] {
            let err = fit(
                &Step::Encode(EncodeStep {
                    method: EncodeMethod::Ordinal,
                    columns: Some(cols),
                }),
                "y",
                &train,
            )
            .unwrap_err();
            assert!(matches!(err, Error::Runtime(_)));
            assert!(err.to_string().contains(needle), "{err}");
        }
    }

    #[test]
    fn preprocess_steps_pipeline_sequence_select_impute_scale_encode() {
        let train = batch(&[
            ("drop_me", Arc::new(Int64Array::from(vec![9, 9, 9]))),
            (
                "num",
                Arc::new(Int64Array::from(vec![Some(10), Some(20), None])),
            ),
            (
                "cat",
                Arc::new(StringArray::from(vec![Some("b"), Some("a"), None])),
            ),
            ("y", Arc::new(Int64Array::from(vec![0, 1, 0]))),
        ]);

        let steps = [
            Step::Select(SelectStep {
                columns: vec!["num".into(), "cat".into()],
            }),
            Step::Impute(ImputeStep {
                strategy: ImputeStrategy::Mean,
                value: None,
                columns: None,
            }),
            Step::Impute(ImputeStep {
                strategy: ImputeStrategy::Constant,
                value: Some(json!("a")),
                columns: None,
            }),
            Step::Scale(ScaleStep {
                method: ScaleMethod::Minmax,
                columns: None,
            }),
            Step::Encode(EncodeStep {
                method: EncodeMethod::OneHot,
                columns: None,
            }),
        ];

        let mut cur = train.clone();
        for s in &steps {
            let f = fit(s, "y", &cur).unwrap();
            cur = f.apply(&cur).unwrap();
        }

        assert_eq!(col_names(&cur), vec!["num", "cat=a", "cat=b", "y"]);
        assert_eq!(f64_col(&cur, "num"), vec![Some(0.0), Some(1.0), Some(0.5)]);
        assert_eq!(
            f64_col(&cur, "cat=a"),
            vec![Some(0.0), Some(1.0), Some(1.0)]
        );
        assert_eq!(
            f64_col(&cur, "cat=b"),
            vec![Some(1.0), Some(0.0), Some(0.0)]
        );
        assert_eq!(i64_col(&cur, "y"), vec![Some(0), Some(1), Some(0)]);
    }
}
