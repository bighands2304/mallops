//! `std/ingest.parquet`: reads a local Parquet file into a Parquet `dataset`.

use std::collections::HashMap;
use std::fs::File;
use std::path::PathBuf;
use std::sync::Arc;

use ::parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use arrow_array::{RecordBatch, RecordBatchReader};
use arrow_schema::{ArrowError, Schema, SchemaRef};
use mallops_core::{
    Block, BlockOutcome, BlockSpec, BoxFuture, CachePolicy, Context, Error, Inputs, Stage,
};
use serde::Deserialize;

use super::{
    Column, DATA_FILE, blocking, dataset_outcome, dataset_port, parse_config, source_policy,
    write_parquet,
};

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Config {
    path: PathBuf,
    schema: Option<Vec<Column>>,
}

/// Checks that `actual` has exactly the `expected` column names and types, in order
/// (nullability is ignored).
fn check_schema(expected: &[Column], actual: &Schema) -> Result<(), Error> {
    let want: Vec<String> = expected
        .iter()
        .map(|c| format!("{}: {}", c.name, c.ty.data_type()))
        .collect();
    let found: Vec<String> = actual
        .fields()
        .iter()
        .map(|f| format!("{}: {}", f.name(), f.data_type()))
        .collect();
    if want == found {
        return Ok(());
    }
    Err(Error::Runtime(format!(
        "schema mismatch: expected [{}], found [{}]",
        want.join(", "),
        found.join(", "),
    )))
}

/// Reader wrapper that drops source schema and field key-value metadata (e.g. `pandas`)
/// so identical data from different writers produces identical Parquet bytes.
struct StripSchemaMetadata<R> {
    inner: R,
    schema: SchemaRef,
}

impl<R: RecordBatchReader> StripSchemaMetadata<R> {
    fn new(inner: R) -> Self {
        let fields: Vec<_> = inner
            .schema()
            .fields()
            .iter()
            .map(|f| f.as_ref().clone().with_metadata(HashMap::new()))
            .collect();
        Self {
            inner,
            schema: Arc::new(Schema::new(fields)),
        }
    }
}

impl<R: RecordBatchReader> Iterator for StripSchemaMetadata<R> {
    type Item = Result<RecordBatch, ArrowError>;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(|res| {
            res.and_then(|batch| {
                RecordBatch::try_new(Arc::clone(&self.schema), batch.columns().to_vec())
            })
        })
    }
}

impl<R: RecordBatchReader> RecordBatchReader for StripSchemaMetadata<R> {
    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }
}

/// `std/ingest.parquet` block.
pub(crate) struct IngestParquet(BlockSpec);

impl IngestParquet {
    pub(crate) fn new() -> Self {
        Self(
            BlockSpec::new("std/ingest.parquet", "0.1.0", Stage::Ingest)
                .with_output(dataset_port()),
        )
    }
}

impl Block for IngestParquet {
    fn spec(&self) -> &BlockSpec {
        &self.0
    }

    fn validate_config(&self, config: &serde_json::Value) -> Result<(), Error> {
        parse_config::<Config>(config).map(|_| ())
    }

    fn cache_policy<'a>(&'a self, ctx: &'a Context) -> BoxFuture<'a, Result<CachePolicy, Error>> {
        Box::pin(async move {
            let cfg: Config = ctx.config_as()?;
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
            let cfg: Config = ctx.config_as()?;
            let out = ctx.workdir().join(DATA_FILE);
            let cancel = ctx.cancel().clone();
            blocking(move || {
                let file = File::open(&cfg.path).map_err(|e| {
                    Error::Runtime(format!("cannot read {}: {e}", cfg.path.display()))
                })?;
                let builder = ParquetRecordBatchReaderBuilder::try_new(file)
                    .map_err(|e| Error::Runtime(format!("{}: {e}", cfg.path.display())))?;
                if let Some(expected) = &cfg.schema {
                    check_schema(expected, builder.schema())?;
                }
                let reader = builder
                    .build()
                    .map_err(|e| Error::Runtime(format!("{}: {e}", cfg.path.display())))?;
                write_parquet(StripSchemaMetadata::new(reader), &out, &cfg.path, &cancel)?;
                Ok(())
            })
            .await?;
            Ok(dataset_outcome(ctx.workdir().join(DATA_FILE)))
        })
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use ::parquet::arrow::ArrowWriter;
    use ::parquet::basic::Compression;
    use ::parquet::file::properties::WriterProperties;
    use arrow_array::{
        ArrayRef, BooleanArray, Float64Array, Int64Array, RecordBatchIterator, StringArray,
    };
    use arrow_schema::Field;
    use mallops_core::{CancelSource, DraftPayload, MemorySecretProvider, MemoryStore, RunId};
    use serde_json::json;

    use super::*;
    use crate::ingest::{CREATED_BY, ColumnType, OUTPUT};

    fn sample_batch() -> RecordBatch {
        RecordBatch::try_from_iter([
            ("id", Arc::new(Int64Array::from(vec![1, 2, 3])) as ArrayRef),
            (
                "name",
                Arc::new(StringArray::from(vec!["alice", "bob", "carol"])) as _,
            ),
            (
                "score",
                Arc::new(Float64Array::from(vec![1.5, 2.5, 3.5])) as _,
            ),
            (
                "active",
                Arc::new(BooleanArray::from(vec![true, false, true])) as _,
            ),
        ])
        .unwrap()
    }

    fn write_sample_parquet(path: &Path, batch: &RecordBatch) {
        let cancel = CancelSource::new().token();
        let iter = RecordBatchIterator::new([Ok(batch.clone())], batch.schema());
        write_parquet(iter, path, path, &cancel).unwrap();
    }

    fn test_context(workdir: PathBuf, config: serde_json::Value) -> Context {
        Context::new(
            RunId::new(),
            workdir,
            Arc::new(MemoryStore::new()),
            config,
            CancelSource::new().token(),
            Arc::new(MemorySecretProvider::default()),
        )
    }

    fn cols(spec: &[(&str, ColumnType)]) -> Vec<Column> {
        spec.iter()
            .map(|(n, t)| Column {
                name: (*n).to_owned(),
                ty: *t,
            })
            .collect()
    }

    #[test]
    fn ingest_parquet_check_schema_matches_names_types_and_order() {
        let actual = sample_batch().schema();
        let ok = cols(&[
            ("id", ColumnType::Int64),
            ("name", ColumnType::String),
            ("score", ColumnType::Float64),
            ("active", ColumnType::Bool),
        ]);
        check_schema(&ok, &actual).unwrap();

        for bad in [
            cols(&[
                ("id", ColumnType::Float64),
                ("name", ColumnType::String),
                ("score", ColumnType::Float64),
                ("active", ColumnType::Bool),
            ]),
            cols(&[
                ("name", ColumnType::String),
                ("id", ColumnType::Int64),
                ("score", ColumnType::Float64),
                ("active", ColumnType::Bool),
            ]),
            cols(&[("id", ColumnType::Int64)]),
        ] {
            let err = check_schema(&bad, &actual).unwrap_err().to_string();
            assert!(err.contains("schema mismatch"), "{err}");
        }
    }

    #[tokio::test]
    async fn ingest_parquet_reads_and_preserves_rows_and_values() {
        let dir = tempfile::tempdir().unwrap();
        let src = dir.path().join("input.parquet");
        let workdir = dir.path().join("work");
        std::fs::create_dir(&workdir).unwrap();

        let expected = sample_batch();
        write_sample_parquet(&src, &expected);

        let block = IngestParquet::new();
        let ctx = test_context(workdir.clone(), json!({ "path": src }));
        let outcome = block.run(&ctx, Inputs::new()).await.unwrap();

        let BlockOutcome::Produced(drafts) = outcome else {
            panic!("expected Produced");
        };
        let draft = drafts.get(OUTPUT).expect("dataset output");
        let DraftPayload::File(out_path) = &draft.payload else {
            panic!("expected DraftPayload::File");
        };
        assert_eq!(out_path, &workdir.join(DATA_FILE));

        let reader = ParquetRecordBatchReaderBuilder::try_new(File::open(out_path).unwrap())
            .unwrap()
            .build()
            .unwrap();
        let batches: Vec<RecordBatch> = reader.collect::<Result<_, _>>().unwrap();
        assert_eq!(batches, vec![expected]);
    }

    #[tokio::test]
    async fn ingest_parquet_reads_compressed_sources() {
        let dir = tempfile::tempdir().unwrap();
        let expected = sample_batch();
        let block = IngestParquet::new();

        for (label, codec) in [
            ("snappy", Compression::SNAPPY),
            ("lz4_raw", Compression::LZ4_RAW),
            ("gzip", Compression::GZIP(Default::default())),
            ("zstd", Compression::ZSTD(Default::default())),
        ] {
            let src = dir.path().join(format!("{label}.parquet"));
            let props = WriterProperties::builder().set_compression(codec).build();
            let mut writer =
                ArrowWriter::try_new(File::create(&src).unwrap(), expected.schema(), Some(props))
                    .unwrap();
            writer.write(&expected).unwrap();
            writer.close().unwrap();

            let workdir = dir.path().join(format!("work_{label}"));
            std::fs::create_dir(&workdir).unwrap();
            let ctx = test_context(workdir.clone(), json!({ "path": src }));
            block.run(&ctx, Inputs::new()).await.unwrap();

            let out_path = workdir.join(DATA_FILE);
            let builder =
                ParquetRecordBatchReaderBuilder::try_new(File::open(&out_path).unwrap()).unwrap();
            assert_eq!(
                builder.metadata().row_group(0).column(0).compression(),
                Compression::UNCOMPRESSED,
                "output for {label} must be uncompressed"
            );
            assert_eq!(
                builder.metadata().file_metadata().created_by(),
                Some(CREATED_BY)
            );
            let batches: Vec<RecordBatch> =
                builder.build().unwrap().collect::<Result<_, _>>().unwrap();
            assert_eq!(batches, vec![expected.clone()], "rows mismatch for {label}");
        }
    }

    #[tokio::test]
    async fn ingest_parquet_strips_source_metadata_for_identical_bytes() {
        let dir = tempfile::tempdir().unwrap();
        let base = sample_batch();
        let block = IngestParquet::new();

        let write_with_meta = |name: &str, schema_meta: (&str, &str), field_meta: (&str, &str)| {
            let fields: Vec<Field> = base
                .schema()
                .fields()
                .iter()
                .map(|f| {
                    f.as_ref().clone().with_metadata(HashMap::from([(
                        field_meta.0.to_owned(),
                        field_meta.1.to_owned(),
                    )]))
                })
                .collect();
            let schema = Arc::new(Schema::new_with_metadata(
                fields,
                HashMap::from([(schema_meta.0.to_owned(), schema_meta.1.to_owned())]),
            ));
            let batch = RecordBatch::try_new(schema, base.columns().to_vec()).unwrap();
            let src = dir.path().join(name);
            let mut writer =
                ArrowWriter::try_new(File::create(&src).unwrap(), batch.schema(), None).unwrap();
            writer.write(&batch).unwrap();
            writer.close().unwrap();
            src
        };

        let src_a = write_with_meta("a.parquet", ("pandas", "{\"v\":1}"), ("note", "alpha"));
        let src_b = write_with_meta("b.parquet", ("polars", "{\"v\":2}"), ("note", "beta"));

        let mut outputs = Vec::new();
        for (src, label) in [(src_a, "out_a"), (src_b, "out_b")] {
            let workdir = dir.path().join(label);
            std::fs::create_dir(&workdir).unwrap();
            let ctx = test_context(workdir.clone(), json!({ "path": src }));
            block.run(&ctx, Inputs::new()).await.unwrap();
            outputs.push(std::fs::read(workdir.join(DATA_FILE)).unwrap());
        }
        assert_eq!(outputs[0], outputs[1]);
    }

    #[tokio::test]
    async fn ingest_parquet_schema_ok() {
        let dir = tempfile::tempdir().unwrap();
        let src = dir.path().join("input.parquet");
        let workdir = dir.path().join("work");
        std::fs::create_dir(&workdir).unwrap();
        write_sample_parquet(&src, &sample_batch());

        let config = json!({
            "path": src,
            "schema": [
                { "name": "id", "type": "int64" },
                { "name": "name", "type": "string" },
                { "name": "score", "type": "float64" },
                { "name": "active", "type": "bool" }
            ]
        });
        let block = IngestParquet::new();
        block.validate_config(&config).unwrap();

        let ctx = test_context(workdir, config);
        let outcome = block.run(&ctx, Inputs::new()).await.unwrap();
        assert!(matches!(outcome, BlockOutcome::Produced(_)));
    }

    #[tokio::test]
    async fn ingest_parquet_schema_mismatch_errors() {
        let dir = tempfile::tempdir().unwrap();
        let src = dir.path().join("input.parquet");
        let workdir = dir.path().join("work");
        std::fs::create_dir(&workdir).unwrap();
        write_sample_parquet(&src, &sample_batch());

        let config = json!({
            "path": src,
            "schema": [
                { "name": "id", "type": "float64" },
                { "name": "name", "type": "string" },
                { "name": "score", "type": "float64" },
                { "name": "active", "type": "bool" }
            ]
        });
        let block = IngestParquet::new();
        let ctx = test_context(workdir.clone(), config);
        let err = block.run(&ctx, Inputs::new()).await.unwrap_err();
        assert!(
            matches!(&err, Error::Runtime(msg) if msg.contains("schema mismatch")),
            "unexpected error: {err}"
        );
        assert!(!workdir.join(DATA_FILE).exists());
    }

    #[tokio::test]
    async fn ingest_parquet_non_parquet_file_errors() {
        let dir = tempfile::tempdir().unwrap();
        let src = dir.path().join("not_a_parquet.txt");
        let workdir = dir.path().join("work");
        std::fs::create_dir(&workdir).unwrap();
        std::fs::write(&src, b"this is definitely not a parquet file").unwrap();

        let block = IngestParquet::new();
        let ctx = test_context(workdir, json!({ "path": src }));
        let err = block.run(&ctx, Inputs::new()).await.unwrap_err();
        assert!(matches!(err, Error::Runtime(_)), "unexpected error: {err}");
    }

    #[test]
    fn ingest_parquet_rejects_unknown_config_field() {
        let block = IngestParquet::new();
        let err = block
            .validate_config(&json!({ "path": "data.parquet", "unknown": 42 }))
            .unwrap_err();
        assert!(matches!(err, Error::Config(_)), "unexpected error: {err}");

        let missing = block.validate_config(&json!({})).unwrap_err();
        assert!(
            matches!(missing, Error::Config(_)),
            "unexpected error: {missing}"
        );
    }
}
