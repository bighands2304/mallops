//! `std/ingest.parquet`: reads a local Parquet file into a Parquet `dataset`.

use std::fs::File;
use std::path::PathBuf;

use ::parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use mallops_core::{
    Block, BlockOutcome, BlockSpec, BoxFuture, CachePolicy, Context, Error, Inputs, Stage,
};
use serde::Deserialize;

use super::{
    Column, DATA_FILE, blocking, check_schema, dataset_outcome, dataset_port, parse_config,
    source_policy, write_parquet,
};

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Config {
    path: PathBuf,
    schema: Option<Vec<Column>>,
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
            let cfg: Config = parse_config(ctx.config())?;
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
            let cfg: Config = parse_config(ctx.config())?;
            let out = ctx.workdir().join(DATA_FILE);
            let out_path = out.clone();
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
                write_parquet(reader, &out_path)?;
                Ok(())
            })
            .await?;
            Ok(dataset_outcome(out))
        })
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;
    use std::sync::Arc;

    use arrow_array::{
        ArrayRef, BooleanArray, Float64Array, Int64Array, RecordBatch, RecordBatchIterator,
        StringArray,
    };
    use mallops_core::{
        ArtifactKind, CancelSource, DraftPayload, MemorySecretProvider, MemoryStore, RunId,
    };
    use serde_json::json;

    use super::*;
    use crate::ingest::OUTPUT;

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
        let iter = RecordBatchIterator::new([Ok(batch.clone())], batch.schema());
        write_parquet(iter, path).unwrap();
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
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0], expected);
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

    #[tokio::test]
    async fn ingest_parquet_manifest_dataset_parquet() {
        let dir = tempfile::tempdir().unwrap();
        let src = dir.path().join("input.parquet");
        let workdir = dir.path().join("work");
        std::fs::create_dir(&workdir).unwrap();
        write_sample_parquet(&src, &sample_batch());

        let block = IngestParquet::new();
        let ctx = test_context(workdir, json!({ "path": src }));
        let BlockOutcome::Produced(drafts) = block.run(&ctx, Inputs::new()).await.unwrap() else {
            panic!("expected Produced");
        };
        let draft = drafts.get(OUTPUT).expect("dataset output");
        assert_eq!(draft.manifest.kind, ArtifactKind::Dataset);
        assert_eq!(draft.manifest.format, "parquet");
    }

    #[tokio::test]
    async fn ingest_parquet_cache_policy_extra_hash_changes_on_modification() {
        let dir = tempfile::tempdir().unwrap();
        let src = dir.path().join("input.parquet");
        let workdir = dir.path().join("work");
        std::fs::create_dir(&workdir).unwrap();
        write_sample_parquet(&src, &sample_batch());

        let block = IngestParquet::new();
        let ctx = test_context(workdir, json!({ "path": &src }));

        let CachePolicy::Deterministic {
            extra_hash: Some(h1),
        } = block.cache_policy(&ctx).await.unwrap()
        else {
            panic!("expected Deterministic policy with extra_hash");
        };

        let modified = RecordBatch::try_from_iter([
            ("id", Arc::new(Int64Array::from(vec![10, 20])) as ArrayRef),
            ("name", Arc::new(StringArray::from(vec!["x", "y"])) as _),
        ])
        .unwrap();
        write_sample_parquet(&src, &modified);

        let CachePolicy::Deterministic {
            extra_hash: Some(h2),
        } = block.cache_policy(&ctx).await.unwrap()
        else {
            panic!("expected Deterministic policy with extra_hash");
        };

        assert_ne!(h1, h2);
    }
}
