//! Shared plumbing of the `std/ingest.*` blocks: expected schema, Parquet output, source hashing.
//!
//! Every ingest block emits one `dataset` output with `format: parquet` (ADR-0002): a single
//! uncompressed `data.parquet` file written into the block workdir, committed by the runtime.

#[cfg(feature = "ingest-csv")]
pub(crate) mod csv;
#[cfg(feature = "ingest-parquet")]
pub(crate) mod parquet;

use std::fs::File;
use std::path::{Path, PathBuf};

use ::parquet::arrow::ArrowWriter;
#[cfg(any(feature = "preprocess-tabular", feature = "train-smartcore"))]
use ::parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use ::parquet::basic::Compression;
use ::parquet::file::properties::WriterProperties;
#[cfg(any(feature = "preprocess-tabular", feature = "train-smartcore"))]
use arrow_array::RecordBatch;
use arrow_array::RecordBatchReader;
use arrow_schema::DataType;
#[cfg(any(feature = "preprocess-tabular", feature = "train-smartcore"))]
use arrow_select::concat::concat_batches;
use mallops_core::{
    ArtifactKind, BlockOutcome, CachePolicy, CancelToken, Digest, DraftPayload, Error, Manifest,
    OutputDraft, OutputDrafts, Port,
};
use serde::{Deserialize, Serialize};

/// Name of the output port of every ingest block.
pub(crate) const OUTPUT: &str = "dataset";

/// File name of the Parquet payload inside the output artifact.
pub(crate) const DATA_FILE: &str = "data.parquet";

/// `created_by` string written into every Parquet footer so output bytes do not drift on
/// patch upgrades of the `parquet` crate.
pub(crate) const CREATED_BY: &str = "mallops-std";

/// Column type accepted in an expected `schema:` (v0.1 subset).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub(crate) enum ColumnType {
    Bool,
    Int64,
    Float64,
    String,
    Date32,
}

impl ColumnType {
    pub(crate) fn data_type(self) -> DataType {
        match self {
            Self::Bool => DataType::Boolean,
            Self::Int64 => DataType::Int64,
            Self::Float64 => DataType::Float64,
            Self::String => DataType::Utf8,
            Self::Date32 => DataType::Date32,
        }
    }

    /// Inverse of [`Self::data_type`]; `None` for types outside the v0.1 subset.
    #[cfg(any(feature = "preprocess-tabular", feature = "train-smartcore"))]
    pub(crate) fn from_data_type(dt: &DataType) -> Option<Self> {
        Some(match dt {
            DataType::Boolean => Self::Bool,
            DataType::Int64 => Self::Int64,
            DataType::Float64 => Self::Float64,
            DataType::Utf8 => Self::String,
            DataType::Date32 => Self::Date32,
            _ => return None,
        })
    }
}

/// One expected column: `{name: <column>, type: <ColumnType>}`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Column {
    pub(crate) name: String,
    #[serde(rename = "type")]
    pub(crate) ty: ColumnType,
}

/// Writes every batch of `reader` to an uncompressed Parquet file at `path`; returns the row count.
///
/// Reader errors are attributed to `source`; writer errors are attributed to `path`.
/// Checks `cancel` between batches and returns [`Error::Cancelled`] if signalled.
/// Output is deterministic for a given `mallops-std` version: the same batches always produce
/// the same bytes.
pub(crate) fn write_parquet(
    reader: impl RecordBatchReader,
    path: &Path,
    source: &Path,
    cancel: &CancelToken,
) -> Result<u64, Error> {
    let out_err = |e: &dyn std::fmt::Display| Error::Runtime(format!("{}: {e}", path.display()));
    let src_err = |e: &dyn std::fmt::Display| Error::Runtime(format!("{}: {e}", source.display()));
    let file = File::create(path).map_err(|e| out_err(&e))?;
    let props = WriterProperties::builder()
        .set_created_by(CREATED_BY.into())
        .set_compression(Compression::UNCOMPRESSED)
        .build();
    let mut writer =
        ArrowWriter::try_new(file, reader.schema(), Some(props)).map_err(|e| out_err(&e))?;
    let mut rows = 0;
    for batch in reader {
        if cancel.is_cancelled() {
            return Err(Error::Cancelled);
        }
        let batch = batch.map_err(|e| src_err(&e))?;
        rows += batch.num_rows() as u64;
        writer.write(&batch).map_err(|e| out_err(&e))?;
    }
    writer.close().map_err(|e| out_err(&e))?;
    Ok(rows)
}

/// Reads a Parquet file into a single concatenated [`RecordBatch`], checking `cancel` between
/// batches.
#[cfg(any(feature = "preprocess-tabular", feature = "train-smartcore"))]
pub(crate) fn read_parquet_batch(path: &Path, cancel: &CancelToken) -> Result<RecordBatch, Error> {
    let err = |e: &dyn std::fmt::Display| Error::Runtime(format!("{}: {e}", path.display()));
    let file = File::open(path)
        .map_err(|e| Error::Runtime(format!("cannot read {}: {e}", path.display())))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file).map_err(|e| err(&e))?;
    let schema = builder.schema().clone();
    let reader = builder.build().map_err(|e| err(&e))?;
    let mut batches = Vec::new();
    for batch in reader {
        if cancel.is_cancelled() {
            return Err(Error::Cancelled);
        }
        batches.push(batch.map_err(|e| err(&e))?);
    }
    concat_batches(&schema, &batches).map_err(|e| err(&e))
}

/// BLAKE3 digest of the file content at `path`, read as a stream.
pub(crate) fn hash_file(path: &Path) -> Result<Digest, Error> {
    let mut hasher = blake3::Hasher::new();
    File::open(path)
        .and_then(|f| hasher.update_reader(f).map(|_| ()))
        .map_err(|e| Error::Runtime(format!("cannot read {}: {e}", path.display())))?;
    Ok(Digest::from_bytes(*hasher.finalize().as_bytes()))
}

/// Runs blocking `f` (file I/O, conversion, hashing) on the tokio blocking pool.
pub(crate) async fn blocking<T, F>(f: F) -> Result<T, Error>
where
    F: FnOnce() -> Result<T, Error> + Send + 'static,
    T: Send + 'static,
{
    tokio::task::spawn_blocking(f)
        .await
        .map_err(|e| Error::Runtime(format!("blocking task failed: {e}")))?
}

/// Cache policy of a local-file source: deterministic, invalidated when the file content changes.
pub(crate) async fn source_policy(path: PathBuf) -> Result<CachePolicy, Error> {
    let digest = blocking(move || hash_file(&path)).await?;
    Ok(CachePolicy::Deterministic {
        extra_hash: Some(digest),
    })
}

/// Output port shared by the ingest blocks: `dataset` (`kind: dataset`, `format: parquet`).
pub(crate) fn dataset_port() -> Port {
    Port::new(OUTPUT, ArtifactKind::Dataset).with_format("parquet")
}

/// `Produced` outcome with the Parquet file at `file` as the `dataset` output.
pub(crate) fn dataset_outcome(file: PathBuf) -> BlockOutcome {
    let draft = OutputDraft::new(
        Manifest::new(ArtifactKind::Dataset, "parquet"),
        DraftPayload::File(file),
    );
    BlockOutcome::Produced(OutputDrafts::new().with(OUTPUT, draft))
}

/// Parses a block `with:` config into `T` (serde), mapping errors to [`Error::Config`].
pub(crate) fn parse_config<T: serde::de::DeserializeOwned>(
    config: &serde_json::Value,
) -> Result<T, Error> {
    T::deserialize(config).map_err(|e| Error::Config(e.to_string()))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::{Int64Array, RecordBatch, RecordBatchIterator, StringArray};
    use mallops_core::CancelSource;

    use super::*;

    fn batch() -> RecordBatch {
        RecordBatch::try_from_iter([
            (
                "id",
                Arc::new(Int64Array::from(vec![1, 2])) as arrow_array::ArrayRef,
            ),
            ("name", Arc::new(StringArray::from(vec!["a", "b"])) as _),
        ])
        .unwrap()
    }

    #[test]
    fn ingest_column_deserializes_lowercase_types() {
        let c: Vec<Column> =
            parse_config(&serde_json::json!([{"name": "x", "type": "float64"}])).unwrap();
        assert_eq!(
            c,
            vec![Column {
                name: "x".to_owned(),
                ty: ColumnType::Float64,
            }]
        );
        let bad = parse_config::<Vec<Column>>(&serde_json::json!([{"name": "x", "type": "int"}]));
        assert!(matches!(bad, Err(Error::Config(_))));
    }

    #[test]
    fn ingest_write_parquet_is_deterministic_and_counts_rows() {
        let dir = tempfile::tempdir().unwrap();
        let cancel = CancelSource::new().token();
        let write = |name: &str| {
            let b = batch();
            let path = dir.path().join(name);
            let rows = write_parquet(
                RecordBatchIterator::new([Ok(b.clone())], b.schema()),
                &path,
                Path::new("input"),
                &cancel,
            )
            .unwrap();
            (rows, std::fs::read(path).unwrap())
        };
        let (rows, a) = write("a.parquet");
        let (_, b) = write("b.parquet");
        assert_eq!(rows, 2);
        assert_eq!(a, b);
    }

    #[test]
    fn ingest_hash_file_tracks_content() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("f");
        std::fs::write(&p, "a").unwrap();
        let h1 = hash_file(&p).unwrap();
        assert_eq!(h1, hash_file(&p).unwrap());
        std::fs::write(&p, "b").unwrap();
        assert_ne!(h1, hash_file(&p).unwrap());
        assert!(hash_file(&dir.path().join("missing")).is_err());
    }

    #[test]
    fn ingest_dataset_outcome_has_dataset_kind_and_parquet_format() {
        let p = PathBuf::from("/tmp/data.parquet");
        let BlockOutcome::Produced(drafts) = dataset_outcome(p.clone()) else {
            panic!("expected Produced");
        };
        let draft = drafts.get(OUTPUT).expect("dataset draft");
        assert_eq!(draft.manifest.kind, ArtifactKind::Dataset);
        assert_eq!(draft.manifest.format, "parquet");
        assert!(dataset_port().accepts_manifest(&draft.manifest));
        assert_eq!(draft.payload, DraftPayload::File(p));
    }

    #[tokio::test]
    async fn ingest_source_policy_tracks_content_and_missing_file() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("src");
        std::fs::write(&p, "v1").unwrap();

        let p1 = source_policy(p.clone()).await.unwrap();
        assert_eq!(p1, source_policy(p.clone()).await.unwrap());
        let CachePolicy::Deterministic {
            extra_hash: Some(h1),
            ..
        } = p1
        else {
            panic!("expected Deterministic with extra_hash");
        };

        std::fs::write(&p, "v2").unwrap();
        let CachePolicy::Deterministic {
            extra_hash: Some(h2),
            ..
        } = source_policy(p).await.unwrap()
        else {
            panic!("expected Deterministic with extra_hash");
        };
        assert_ne!(h1, h2);

        let err = source_policy(dir.path().join("missing")).await.unwrap_err();
        assert!(matches!(err, Error::Runtime(_)));
    }
}
