//! Standard blocks of mallops (`std/…`), one Cargo feature per block (ADR-0001, ADR-0002).
//!
//! Blocks are registered explicitly with [`register_std`] (ADR-0007 D8): only the blocks whose
//! feature is enabled are compiled and registered.

#[cfg(feature = "_ingest")]
// Without an ingest block only the shared Arrow/Parquet helpers are used (by preprocess).
#[cfg_attr(
    not(any(feature = "ingest-csv", feature = "ingest-parquet")),
    allow(dead_code)
)]
mod ingest;
#[cfg(feature = "preprocess-tabular")]
mod preprocess;
#[cfg(feature = "train-external")]
// ponytail: skeleton of 13.0, the allow goes away once 13.1/13.2 wire every item.
#[allow(dead_code)]
mod train;

#[cfg(feature = "preprocess-tabular")]
pub use preprocess::transform::{FittedTransform, TRANSFORM_FILE};

use mallops_core::{BlockRegistry, Error};

/// Registers every feature-enabled standard block into `registry`.
///
/// # Errors
///
/// [`Error::Validation`] if a standard block id is already registered.
#[allow(unused_variables)] // with `--no-default-features` no block is registered
pub fn register_std(registry: &mut BlockRegistry) -> Result<(), Error> {
    #[cfg(feature = "ingest-csv")]
    registry.register(std::sync::Arc::new(ingest::csv::IngestCsv::new()))?;
    #[cfg(feature = "ingest-parquet")]
    registry.register(std::sync::Arc::new(ingest::parquet::IngestParquet::new()))?;
    #[cfg(feature = "preprocess-tabular")]
    registry.register(std::sync::Arc::new(preprocess::PreprocessTabular::new()))?;
    #[cfg(feature = "train-external")]
    registry.register(std::sync::Arc::new(train::external::TrainExternal::new()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use mallops_core::{BlockRef, BlockRegistry, Error};

    use super::register_std;

    #[test]
    fn register_std_only_enabled() {
        let mut registry = BlockRegistry::new();
        register_std(&mut registry).unwrap();

        let csv: BlockRef = "std/ingest.csv".parse().unwrap();
        let parquet: BlockRef = "std/ingest.parquet".parse().unwrap();
        let preprocess: BlockRef = "std/preprocess.tabular".parse().unwrap();
        let train_external: BlockRef = "std/train.external".parse().unwrap();

        assert_eq!(registry.contains(&csv), cfg!(feature = "ingest-csv"));
        assert_eq!(
            registry.contains(&parquet),
            cfg!(feature = "ingest-parquet")
        );
        assert_eq!(
            registry.contains(&preprocess),
            cfg!(feature = "preprocess-tabular")
        );
        assert_eq!(
            registry.contains(&train_external),
            cfg!(feature = "train-external")
        );
    }

    #[test]
    fn register_std_twice_fails() {
        let mut registry = BlockRegistry::new();
        register_std(&mut registry).unwrap();
        let second = register_std(&mut registry);
        if cfg!(any(
            feature = "ingest-csv",
            feature = "ingest-parquet",
            feature = "preprocess-tabular",
            feature = "train-external"
        )) {
            assert!(matches!(second, Err(Error::Validation(_))), "{second:?}");
        } else {
            assert!(second.is_ok(), "{second:?}");
        }
    }
}
