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
use arrow_array::RecordBatchReader;
use arrow_schema::{DataType, Field, Schema};
use mallops_core::{
    ArtifactKind, BlockOutcome, CachePolicy, Digest, DraftPayload, Error, Manifest, OutputDraft,
    OutputDrafts, Port,
};
use serde::Deserialize;

/// Name of the output port of every ingest block.
pub(crate) const OUTPUT: &str = "dataset";

/// File name of the Parquet payload inside the output artifact.
pub(crate) const DATA_FILE: &str = "data.parquet";

/// Column type accepted in an expected `schema:` (v0.1 subset).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub(crate) enum ColumnType {
    Bool,
    Int64,
    Float64,
    String,
    Date32,
}

impl ColumnType {
    fn data_type(self) -> DataType {
        match self {
            Self::Bool => DataType::Boolean,
            Self::Int64 => DataType::Int64,
            Self::Float64 => DataType::Float64,
            Self::String => DataType::Utf8,
            Self::Date32 => DataType::Date32,
        }
    }
}

/// One expected column: `{name: <column>, type: <ColumnType>}`.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Column {
    pub(crate) name: String,
    #[serde(rename = "type")]
    pub(crate) ty: ColumnType,
}

/// Arrow schema (all columns nullable) of an expected column list.
#[cfg_attr(not(feature = "ingest-csv"), allow(dead_code))] // only `std/ingest.csv` parses with it
pub(crate) fn arrow_schema(columns: &[Column]) -> Schema {
    Schema::new(
        columns
            .iter()
            .map(|c| Field::new(&c.name, c.ty.data_type(), true))
            .collect::<Vec<_>>(),
    )
}

/// Checks that `actual` has exactly the `expected` column names and types, in order
/// (nullability is ignored).
#[cfg_attr(not(feature = "ingest-parquet"), allow(dead_code))] // CSV checks header names only
pub(crate) fn check_schema(expected: &[Column], actual: &Schema) -> Result<(), Error> {
    let found: Vec<(&str, &DataType)> = actual
        .fields()
        .iter()
        .map(|f| (f.name().as_str(), f.data_type()))
        .collect();
    let want: Vec<(&str, DataType)> = expected
        .iter()
        .map(|c| (c.name.as_str(), c.ty.data_type()))
        .collect();
    let same = found.len() == want.len()
        && found
            .iter()
            .zip(&want)
            .all(|(f, w)| f.0 == w.0 && *f.1 == w.1);
    if same {
        return Ok(());
    }
    let show = |cols: Vec<String>| format!("[{}]", cols.join(", "));
    Err(Error::Runtime(format!(
        "schema mismatch: expected {}, found {}",
        show(want.iter().map(|(n, t)| format!("{n}: {t}")).collect()),
        show(found.iter().map(|(n, t)| format!("{n}: {t}")).collect()),
    )))
}

/// Writes every batch of `reader` to an uncompressed Parquet file at `path`; returns the row count.
///
/// Output is deterministic: the same batches always produce the same bytes.
pub(crate) fn write_parquet(reader: impl RecordBatchReader, path: &Path) -> Result<u64, Error> {
    let err = |e: &dyn std::fmt::Display| Error::Runtime(format!("{}: {e}", path.display()));
    let file = File::create(path).map_err(|e| err(&e))?;
    let mut writer = ArrowWriter::try_new(file, reader.schema(), None).map_err(|e| err(&e))?;
    let mut rows = 0;
    for batch in reader {
        let batch = batch.map_err(|e| err(&e))?;
        rows += batch.num_rows() as u64;
        writer.write(&batch).map_err(|e| err(&e))?;
    }
    writer.close().map_err(|e| err(&e))?;
    Ok(rows)
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

    use super::*;

    fn cols(spec: &[(&str, ColumnType)]) -> Vec<Column> {
        spec.iter()
            .map(|(n, t)| Column {
                name: (*n).to_owned(),
                ty: *t,
            })
            .collect()
    }

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
        assert_eq!(c, cols(&[("x", ColumnType::Float64)]));
        let bad = parse_config::<Vec<Column>>(&serde_json::json!([{"name": "x", "type": "int"}]));
        assert!(matches!(bad, Err(Error::Config(_))));
    }

    #[test]
    fn ingest_check_schema_matches_names_types_and_order() {
        let actual = batch().schema();
        let ok = cols(&[("id", ColumnType::Int64), ("name", ColumnType::String)]);
        check_schema(&ok, &actual).unwrap();
        assert_eq!(arrow_schema(&ok).fields().len(), 2);

        for bad in [
            cols(&[("id", ColumnType::Float64), ("name", ColumnType::String)]),
            cols(&[("name", ColumnType::String), ("id", ColumnType::Int64)]),
            cols(&[("id", ColumnType::Int64)]),
        ] {
            let err = check_schema(&bad, &actual).unwrap_err().to_string();
            assert!(err.contains("schema mismatch"), "{err}");
        }
    }

    #[test]
    fn ingest_write_parquet_is_deterministic_and_counts_rows() {
        let dir = tempfile::tempdir().unwrap();
        let write = |name: &str| {
            let b = batch();
            let path = dir.path().join(name);
            let rows = write_parquet(RecordBatchIterator::new([Ok(b.clone())], b.schema()), &path)
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
}
