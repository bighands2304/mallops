//! Standard blocks of mallops (`std/…`), one Cargo feature per block (ADR-0001, ADR-0002).
//!
//! Blocks are registered explicitly with [`register_std`] (ADR-0007 D8): only the blocks whose
//! feature is enabled are compiled and registered.

#[cfg(feature = "_ingest")]
mod ingest;

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

        assert_eq!(registry.contains(&csv), cfg!(feature = "ingest-csv"));
        assert_eq!(
            registry.contains(&parquet),
            cfg!(feature = "ingest-parquet")
        );
    }

    #[test]
    fn register_std_twice_fails() {
        let mut registry = BlockRegistry::new();
        register_std(&mut registry).unwrap();
        let second = register_std(&mut registry);
        if cfg!(any(feature = "ingest-csv", feature = "ingest-parquet")) {
            assert!(matches!(second, Err(Error::Validation(_))), "{second:?}");
        } else {
            assert!(second.is_ok(), "{second:?}");
        }
    }
}
