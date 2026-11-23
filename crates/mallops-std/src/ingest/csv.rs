//! `std/ingest.csv`: reads a local CSV file into a Parquet `dataset`.

use std::fs::File;
use std::io::Seek;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow_csv::ReaderBuilder;
use arrow_csv::reader::Format;
use arrow_schema::{Field, Schema};
use mallops_core::{
    Block, BlockOutcome, BlockSpec, BoxFuture, CachePolicy, CancelToken, Context, Error, Inputs,
    Stage,
};
use serde::Deserialize;

use super::{
    Column, DATA_FILE, blocking, dataset_outcome, dataset_port, parse_config, source_policy,
    write_parquet,
};

fn default_delimiter() -> char {
    ','
}

fn default_has_header() -> bool {
    true
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CsvConfig {
    path: PathBuf,
    #[serde(default = "default_delimiter")]
    delimiter: char,
    #[serde(default = "default_has_header")]
    has_header: bool,
    schema: Option<Vec<Column>>,
}

impl CsvConfig {
    fn check(self) -> Result<Self, Error> {
        if !self.delimiter.is_ascii() || matches!(self.delimiter, '\n' | '\r' | '"') {
            return Err(Error::Config(format!(
                "delimiter {:?} must be an ASCII character other than '\\n', '\\r', or '\"'",
                self.delimiter
            )));
        }
        Ok(self)
    }
}

/// Arrow schema (all columns nullable) of an expected column list.
fn arrow_schema(columns: &[Column]) -> Schema {
    Schema::new(
        columns
            .iter()
            .map(|c| Field::new(&c.name, c.ty.data_type(), true))
            .collect::<Vec<_>>(),
    )
}

fn check_header_names(expected: &[Column], header: &Schema) -> Result<(), Error> {
    let found: Vec<&str> = header.fields().iter().map(|f| f.name().as_str()).collect();
    let want: Vec<&str> = expected.iter().map(|c| c.name.as_str()).collect();
    if found == want {
        return Ok(());
    }
    Err(Error::Runtime(format!(
        "schema mismatch: expected [{}], found [{}]",
        want.join(", "),
        found.join(", "),
    )))
}

fn read_and_write_csv(cfg: &CsvConfig, out: &Path, cancel: &CancelToken) -> Result<(), Error> {
    let read_err = |e: &dyn std::fmt::Display| {
        Error::Runtime(format!("cannot read {}: {e}", cfg.path.display()))
    };
    let mut file = File::open(&cfg.path).map_err(|e| read_err(&e))?;
    let format = Format::default()
        .with_header(cfg.has_header)
        .with_delimiter(cfg.delimiter as u8);

    let schema = match &cfg.schema {
        Some(expected) => {
            let (probe, _) = format
                .infer_schema(&mut file, Some(0))
                .map_err(|e| read_err(&e))?;
            if probe.fields().is_empty() {
                return Err(Error::Runtime(format!(
                    "{}: empty CSV: no columns",
                    cfg.path.display()
                )));
            }
            if cfg.has_header {
                check_header_names(expected, &probe)?;
            } else if probe.fields().len() != expected.len() {
                return Err(Error::Runtime(format!(
                    "schema mismatch: expected {} columns, found {}",
                    expected.len(),
                    probe.fields().len()
                )));
            }
            file.rewind().map_err(|e| read_err(&e))?;
            arrow_schema(expected)
        }
        None => {
            let (inferred, _) = format
                .infer_schema(&mut file, None)
                .map_err(|e| read_err(&e))?;
            if inferred.fields().is_empty() {
                return Err(Error::Runtime(format!(
                    "{}: empty CSV: no columns",
                    cfg.path.display()
                )));
            }
            file.rewind().map_err(|e| read_err(&e))?;
            inferred
        }
    };

    let reader = ReaderBuilder::new(Arc::new(schema))
        .with_format(format)
        .build(file)
        .map_err(|e| read_err(&e))?;
    write_parquet(reader, out, &cfg.path, cancel)?;
    Ok(())
}

/// `std/ingest.csv` block.
pub(crate) struct IngestCsv(BlockSpec);

impl IngestCsv {
    pub(crate) fn new() -> Self {
        Self(BlockSpec::new("std/ingest.csv", "0.1.0", Stage::Ingest).with_output(dataset_port()))
    }
}

impl Block for IngestCsv {
    fn spec(&self) -> &BlockSpec {
        &self.0
    }

    fn validate_config(&self, config: &serde_json::Value) -> Result<(), Error> {
        parse_config::<CsvConfig>(config)?.check().map(|_| ())
    }

    fn cache_policy<'a>(&'a self, ctx: &'a Context) -> BoxFuture<'a, Result<CachePolicy, Error>> {
        Box::pin(async move {
            let cfg = ctx.config_as::<CsvConfig>()?.check()?;
            source_policy(cfg.path).await
        })
    }

    fn run<'a>(
        &'a self,
        ctx: &'a Context,
        _inputs: Inputs,
    ) -> BoxFuture<'a, Result<BlockOutcome, Error>> {
        Box::pin(async move {
            if ctx.cancel().is_cancelled() {
                return Err(Error::Cancelled);
            }
            let cfg = ctx.config_as::<CsvConfig>()?.check()?;
            let out = ctx.workdir().join(DATA_FILE);
            let cancel = ctx.cancel().clone();
            blocking(move || read_and_write_csv(&cfg, &out, &cancel)).await?;
            Ok(dataset_outcome(ctx.workdir().join(DATA_FILE)))
        })
    }
}

#[cfg(test)]
mod tests {
    use std::fmt::Write as _;

    use ::parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
    use arrow_array::{
        ArrayRef, BooleanArray, Date32Array, Float64Array, Int64Array, RecordBatch,
        RecordBatchReader, StringArray,
    };
    use arrow_schema::DataType;
    use mallops_core::{CancelSource, DraftPayload, MemorySecretProvider, MemoryStore, RunId};
    use serde_json::json;

    use super::*;
    use crate::ingest::{ColumnType, OUTPUT};

    fn make_ctx(workdir: &Path, config: serde_json::Value) -> (Context, CancelSource) {
        let cancel = CancelSource::new();
        let ctx = Context::new(
            RunId::new(),
            workdir.to_path_buf(),
            Arc::new(MemoryStore::new()),
            config,
            cancel.token(),
            Arc::new(MemorySecretProvider::default()),
        );
        (ctx, cancel)
    }

    fn nullable_batch(cols: &[(&str, ArrayRef)]) -> RecordBatch {
        let fields: Vec<Field> = cols
            .iter()
            .map(|(name, arr)| Field::new(*name, arr.data_type().clone(), true))
            .collect();
        let arrays: Vec<ArrayRef> = cols.iter().map(|(_, arr)| Arc::clone(arr)).collect();
        RecordBatch::try_new(Arc::new(Schema::new(fields)), arrays).unwrap()
    }

    fn read_parquet_batches(path: &Path) -> (Arc<Schema>, Vec<RecordBatch>) {
        let file = File::open(path).unwrap();
        let reader = ParquetRecordBatchReaderBuilder::try_new(file)
            .unwrap()
            .build()
            .unwrap();
        let schema = reader.schema();
        let batches: Vec<RecordBatch> = reader.map(|b| b.unwrap()).collect();
        (schema, batches)
    }

    fn write_csv(dir: &Path, name: &str, content: &str) -> PathBuf {
        let path = dir.join(name);
        std::fs::write(&path, content).unwrap();
        path
    }

    #[test]
    fn ingest_csv_arrow_schema_builds_nullable_fields() {
        let cols = [
            Column {
                name: "id".to_owned(),
                ty: ColumnType::Int64,
            },
            Column {
                name: "name".to_owned(),
                ty: ColumnType::String,
            },
        ];
        let s = arrow_schema(&cols);
        assert_eq!(s.fields().len(), 2);
        assert_eq!(s.field(0).name(), "id");
        assert_eq!(s.field(0).data_type(), &DataType::Int64);
        assert!(s.field(0).is_nullable());
        assert_eq!(s.field(1).data_type(), &DataType::Utf8);
    }

    #[test]
    fn ingest_csv_rejects_unknown_config_and_invalid_delimiter() {
        let block = IngestCsv::new();
        assert!(block.validate_config(&json!({ "path": "a.csv" })).is_ok());
        assert!(
            block
                .validate_config(&json!({
                    "path": "a.csv",
                    "delimiter": ";",
                    "has_header": false,
                    "schema": [{ "name": "x", "type": "int64" }]
                }))
                .is_ok()
        );

        for bad in [
            json!({ "path": "a.csv", "unknown_field": true }),
            json!({}),
            json!({ "path": "a.csv", "delimiter": "€" }),
            json!({ "path": "a.csv", "delimiter": "é" }),
            json!({ "path": "a.csv", "delimiter": "\n" }),
            json!({ "path": "a.csv", "delimiter": "\r" }),
            json!({ "path": "a.csv", "delimiter": "\"" }),
            json!({ "path": "a.csv", "delimiter": ";;" }),
            json!({ "path": "a.csv", "schema": [{ "name": "x", "type": "int64", "extra": 1 }] }),
        ] {
            let err = block.validate_config(&bad).unwrap_err();
            assert!(
                matches!(err, Error::Config(_)),
                "expected Config for {bad}: {err}"
            );
        }
    }

    #[tokio::test]
    async fn ingest_csv_custom_semicolon_delimiter() {
        let dir = tempfile::tempdir().unwrap();
        let csv = write_csv(
            dir.path(),
            "semi.csv",
            "id;name;score\n1;alice;9.5\n2;bob;8.0\n",
        );
        let workdir = tempfile::tempdir().unwrap();
        let (ctx, _cancel) = make_ctx(workdir.path(), json!({ "path": csv, "delimiter": ";" }));

        let outcome = IngestCsv::new().run(&ctx, Inputs::new()).await.unwrap();
        let BlockOutcome::Produced(drafts) = outcome else {
            panic!("expected Produced");
        };
        let DraftPayload::File(out_path) = &drafts.get(OUTPUT).unwrap().payload else {
            panic!("expected File payload");
        };

        let (_, batches) = read_parquet_batches(out_path);
        let expected = nullable_batch(&[
            ("id", Arc::new(Int64Array::from(vec![1, 2]))),
            ("name", Arc::new(StringArray::from(vec!["alice", "bob"]))),
            ("score", Arc::new(Float64Array::from(vec![9.5, 8.0]))),
        ]);
        assert_eq!(batches, vec![expected]);
    }

    #[tokio::test]
    async fn ingest_csv_without_header() {
        let dir = tempfile::tempdir().unwrap();
        let csv = write_csv(dir.path(), "nohdr.csv", "1,alice,2.5\n2,bob,3.5\n");

        // Inferred schema without header.
        let workdir = tempfile::tempdir().unwrap();
        let (ctx, _cancel) = make_ctx(workdir.path(), json!({ "path": csv, "has_header": false }));
        IngestCsv::new().run(&ctx, Inputs::new()).await.unwrap();
        let (_, batches) = read_parquet_batches(&workdir.path().join(DATA_FILE));
        let expected = nullable_batch(&[
            ("column_1", Arc::new(Int64Array::from(vec![1, 2]))),
            (
                "column_2",
                Arc::new(StringArray::from(vec!["alice", "bob"])),
            ),
            ("column_3", Arc::new(Float64Array::from(vec![2.5, 3.5]))),
        ]);
        assert_eq!(batches, vec![expected]);

        // Explicit schema without header assigns column names and keeps row 1 as data.
        let workdir2 = tempfile::tempdir().unwrap();
        let (ctx2, _cancel2) = make_ctx(
            workdir2.path(),
            json!({
                "path": csv,
                "has_header": false,
                "schema": [
                    { "name": "id", "type": "int64" },
                    { "name": "name", "type": "string" },
                    { "name": "val", "type": "float64" }
                ]
            }),
        );
        IngestCsv::new().run(&ctx2, Inputs::new()).await.unwrap();
        let (_, batches2) = read_parquet_batches(&workdir2.path().join(DATA_FILE));
        let expected2 = nullable_batch(&[
            ("id", Arc::new(Int64Array::from(vec![1, 2]))),
            ("name", Arc::new(StringArray::from(vec!["alice", "bob"]))),
            ("val", Arc::new(Float64Array::from(vec![2.5, 3.5]))),
        ]);
        assert_eq!(batches2, vec![expected2]);

        // Explicit schema with wrong column count -> schema mismatch.
        let workdir3 = tempfile::tempdir().unwrap();
        let (ctx3, _cancel3) = make_ctx(
            workdir3.path(),
            json!({
                "path": csv,
                "has_header": false,
                "schema": [{ "name": "id", "type": "int64" }]
            }),
        );
        let err = IngestCsv::new()
            .run(&ctx3, Inputs::new())
            .await
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("schema mismatch: expected 1 columns, found 3"),
            "unexpected error: {err}"
        );
    }

    #[tokio::test]
    async fn ingest_csv_infers_types_across_entire_file() {
        let dir = tempfile::tempdir().unwrap();
        // >100 rows so inference cannot rely on a small default prefix sample.
        let mut body = String::from("id,metric,label\n");
        for i in 1..=150 {
            let _ = writeln!(body, "{i},{i},{i}");
        }
        // Row 151 widens `metric` to Float64 and `label` to Utf8.
        body.push_str("151,151.5,tail\n");
        let csv = write_csv(dir.path(), "infer.csv", &body);

        let workdir = tempfile::tempdir().unwrap();
        let (ctx, _cancel) = make_ctx(workdir.path(), json!({ "path": csv }));
        IngestCsv::new().run(&ctx, Inputs::new()).await.unwrap();

        let (schema, batches) = read_parquet_batches(&workdir.path().join(DATA_FILE));
        assert_eq!(schema.field(0).data_type(), &DataType::Int64);
        assert_eq!(schema.field(1).data_type(), &DataType::Float64);
        assert_eq!(schema.field(2).data_type(), &DataType::Utf8);

        let total_rows: usize = batches.iter().map(RecordBatch::num_rows).sum();
        assert_eq!(total_rows, 151);
    }

    #[tokio::test]
    async fn ingest_csv_schema_ok() {
        let dir = tempfile::tempdir().unwrap();
        // `"42"` parsed as string and `"2"` parsed as float64 confirm explicit schema rules.
        let csv = write_csv(
            dir.path(),
            "typed.csv",
            "flag,id,val,code,day\ntrue,1,2,42,2026-01-15\nfalse,2,3.5,99,2026-01-16\n",
        );
        let workdir = tempfile::tempdir().unwrap();
        let (ctx, _cancel) = make_ctx(
            workdir.path(),
            json!({
                "path": csv,
                "schema": [
                    { "name": "flag", "type": "bool" },
                    { "name": "id", "type": "int64" },
                    { "name": "val", "type": "float64" },
                    { "name": "code", "type": "string" },
                    { "name": "day", "type": "date32" }
                ]
            }),
        );

        IngestCsv::new().run(&ctx, Inputs::new()).await.unwrap();
        let (_, batches) = read_parquet_batches(&workdir.path().join(DATA_FILE));
        // 2026-01-15 is 20468 days since 1970-01-01.
        let expected = nullable_batch(&[
            ("flag", Arc::new(BooleanArray::from(vec![true, false]))),
            ("id", Arc::new(Int64Array::from(vec![1, 2]))),
            ("val", Arc::new(Float64Array::from(vec![2.0, 3.5]))),
            ("code", Arc::new(StringArray::from(vec!["42", "99"]))),
            ("day", Arc::new(Date32Array::from(vec![20468, 20469]))),
        ]);
        assert_eq!(batches, vec![expected]);
    }

    #[tokio::test]
    async fn ingest_csv_schema_ko_names_and_types() {
        let dir = tempfile::tempdir().unwrap();
        let csv = write_csv(dir.path(), "data.csv", "id,score\n1,3.5\n");
        let block = IngestCsv::new();

        // Wrong column name, wrong order, extra/missing column -> "schema mismatch".
        for bad_schema in [
            json!([{ "name": "id", "type": "int64" }, { "name": "other", "type": "float64" }]),
            json!([{ "name": "score", "type": "float64" }, { "name": "id", "type": "int64" }]),
            json!([{ "name": "id", "type": "int64" }]),
            json!([
                { "name": "id", "type": "int64" },
                { "name": "score", "type": "float64" },
                { "name": "extra", "type": "string" }
            ]),
        ] {
            let workdir = tempfile::tempdir().unwrap();
            let (ctx, _cancel) =
                make_ctx(workdir.path(), json!({ "path": csv, "schema": bad_schema }));
            let err = block
                .run(&ctx, Inputs::new())
                .await
                .unwrap_err()
                .to_string();
            assert!(
                err.contains("schema mismatch"),
                "expected schema mismatch, got: {err}"
            );
        }

        // Incompatible column type (float value "3.5" declared as int64).
        let workdir = tempfile::tempdir().unwrap();
        let (ctx, _cancel) = make_ctx(
            workdir.path(),
            json!({
                "path": csv,
                "schema": [
                    { "name": "id", "type": "int64" },
                    { "name": "score", "type": "int64" }
                ]
            }),
        );
        let err = block.run(&ctx, Inputs::new()).await.unwrap_err();
        assert!(
            matches!(err, Error::Runtime(_)),
            "expected Runtime error, got: {err}"
        );
    }

    #[tokio::test]
    async fn ingest_csv_rejects_unconvertible_value() {
        let dir = tempfile::tempdir().unwrap();
        let csv = write_csv(
            dir.path(),
            "badval.csv",
            "id,day\n1,2026-01-01\nnope,2026-01-02\n",
        );
        let workdir = tempfile::tempdir().unwrap();
        let (ctx, _cancel) = make_ctx(
            workdir.path(),
            json!({
                "path": csv,
                "schema": [
                    { "name": "id", "type": "int64" },
                    { "name": "day", "type": "date32" }
                ]
            }),
        );

        let err = IngestCsv::new().run(&ctx, Inputs::new()).await.unwrap_err();
        assert!(
            matches!(err, Error::Runtime(_)),
            "expected Runtime error, got: {err}"
        );
        let msg = err.to_string();
        assert!(
            msg.contains(&csv.display().to_string()),
            "error should name source file, got: {msg}"
        );
        assert!(
            !msg.contains(DATA_FILE),
            "error should not name output file, got: {msg}"
        );
    }

    #[tokio::test]
    async fn ingest_csv_empty_errors_and_header_only_succeeds() {
        let dir = tempfile::tempdir().unwrap();
        let empty = write_csv(dir.path(), "empty.csv", "");
        let block = IngestCsv::new();

        for cfg in [
            json!({ "path": empty }),
            json!({ "path": empty, "has_header": false }),
            json!({ "path": empty, "schema": [{ "name": "id", "type": "int64" }] }),
            json!({ "path": empty, "has_header": false, "schema": [{ "name": "id", "type": "int64" }] }),
        ] {
            let workdir = tempfile::tempdir().unwrap();
            let (ctx, _cancel) = make_ctx(workdir.path(), cfg);
            let err = block
                .run(&ctx, Inputs::new())
                .await
                .unwrap_err()
                .to_string();
            assert!(
                err.contains("empty CSV: no columns"),
                "expected empty CSV error, got: {err}"
            );
        }

        // Header-only (0 data rows) is valid both with and without explicit schema.
        let hdr_only = write_csv(dir.path(), "hdr.csv", "id,name\n");
        for cfg in [
            json!({ "path": hdr_only }),
            json!({
                "path": hdr_only,
                "schema": [
                    { "name": "id", "type": "int64" },
                    { "name": "name", "type": "string" }
                ]
            }),
        ] {
            let workdir = tempfile::tempdir().unwrap();
            let (ctx, _cancel) = make_ctx(workdir.path(), cfg);
            block.run(&ctx, Inputs::new()).await.unwrap();
            let (schema, batches) = read_parquet_batches(&workdir.path().join(DATA_FILE));
            assert_eq!(schema.fields().len(), 2);
            let total_rows: usize = batches.iter().map(RecordBatch::num_rows).sum();
            assert_eq!(total_rows, 0);
        }
    }

    #[tokio::test]
    async fn ingest_csv_respects_cancellation_and_missing_file() {
        let dir = tempfile::tempdir().unwrap();
        let csv = write_csv(dir.path(), "ok.csv", "x\n1\n");
        let workdir = tempfile::tempdir().unwrap();
        let (ctx, cancel) = make_ctx(workdir.path(), json!({ "path": csv }));
        cancel.cancel();

        let err = IngestCsv::new().run(&ctx, Inputs::new()).await.unwrap_err();
        assert!(matches!(err, Error::Cancelled));

        let (missing_ctx, _cancel2) = make_ctx(
            workdir.path(),
            json!({ "path": dir.path().join("nope.csv") }),
        );
        let err = IngestCsv::new()
            .run(&missing_ctx, Inputs::new())
            .await
            .unwrap_err();
        assert!(matches!(err, Error::Runtime(_)));
        assert!(IngestCsv::new().cache_policy(&missing_ctx).await.is_err());
    }
}
