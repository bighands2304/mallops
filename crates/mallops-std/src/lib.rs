//! Standard blocks of mallops (`std/…`), one Cargo feature per block (ADR-0001, ADR-0002).
//!
//! Blocks are registered explicitly with [`register_std`] (ADR-0007 D8): only the blocks whose
//! feature is enabled are compiled and registered.

#[cfg(feature = "evaluate")]
mod evaluate;
#[cfg(feature = "external")]
mod external;
#[cfg(feature = "_ingest")]
// Without an ingest block only the shared Arrow/Parquet helpers are used (by preprocess / train.smartcore / evaluate).
#[cfg_attr(
    not(any(feature = "ingest-csv", feature = "ingest-parquet")),
    allow(dead_code)
)]
mod ingest;
#[cfg(feature = "package")]
mod package;
#[cfg(feature = "predictor")]
pub mod predictor;
#[cfg(feature = "preprocess-tabular")]
mod preprocess;
#[cfg(any(
    feature = "evaluate",
    all(feature = "serve-http", feature = "preprocess-tabular")
))]
mod rows;
#[cfg(feature = "serve-http")]
pub mod serve;
#[cfg(any(feature = "train-smartcore", feature = "predict-smartcore"))]
mod smartcore_model;
#[cfg(feature = "train-smartcore")]
mod train;

#[cfg(feature = "package")]
pub use package::Package;
#[cfg(feature = "preprocess-tabular")]
pub use preprocess::transform::{FittedTransform, TRANSFORM_FILE};

use mallops_core::{BlockRegistry, Error};

/// Whether any string (key or value) in `value` contains `${secret:`; JSON never escapes `$`,
/// `{` or `:`.
#[cfg(any(
    feature = "external",
    feature = "train-smartcore",
    feature = "evaluate"
))]
pub(crate) fn has_secret_ref(value: &impl serde::Serialize) -> bool {
    serde_json::to_string(value).is_ok_and(|json| json.contains("${secret:"))
}

/// Registers every feature-enabled standard block into `registry`.
///
/// # Errors
///
/// [`Error::Validation`] if a standard block id is already registered.
#[allow(unused_variables)] // with `--no-default-features` no block is registered
pub fn register_std(registry: &mut BlockRegistry) -> Result<(), Error> {
    #[allow(unused_imports)] // only the blocks that share state need it
    use std::sync::Arc;

    #[cfg(feature = "ingest-csv")]
    registry.register(Arc::new(ingest::csv::IngestCsv::new()))?;
    #[cfg(feature = "ingest-parquet")]
    registry.register(Arc::new(ingest::parquet::IngestParquet::new()))?;
    #[cfg(feature = "preprocess-tabular")]
    registry.register(Arc::new(preprocess::PreprocessTabular::new()))?;
    #[cfg(feature = "external")]
    registry.register(Arc::new(external::External::new()))?;
    #[cfg(feature = "train-smartcore")]
    registry.register(Arc::new(train::smartcore::TrainSmartcore::new()))?;
    #[cfg(feature = "evaluate")]
    {
        // One predictor registry shared by both evaluate blocks. Without a `predict-*` feature
        // the blocks register but fail at run time with "no predictor for flavor".
        let mut predictors = predictor::PredictorRegistry::new();
        predictor::register_std_predictors(&mut predictors)?;
        let predictors = Arc::new(predictors);
        registry.register(Arc::new(evaluate::Evaluate::classification(Arc::clone(
            &predictors,
        ))))?;
        registry.register(Arc::new(evaluate::Evaluate::regression(predictors)))?;
    }
    #[cfg(feature = "package")]
    registry.register(Arc::new(package::PackageBlock::new()))?;
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
        let external: BlockRef = "std/external".parse().unwrap();
        let train_smartcore: BlockRef = "std/train.smartcore".parse().unwrap();
        let eval_cls: BlockRef = "std/evaluate.classification".parse().unwrap();
        let eval_reg: BlockRef = "std/evaluate.regression".parse().unwrap();
        let package: BlockRef = "std/package".parse().unwrap();

        assert_eq!(registry.contains(&csv), cfg!(feature = "ingest-csv"));
        assert_eq!(
            registry.contains(&parquet),
            cfg!(feature = "ingest-parquet")
        );
        assert_eq!(
            registry.contains(&preprocess),
            cfg!(feature = "preprocess-tabular")
        );
        assert_eq!(registry.contains(&external), cfg!(feature = "external"));
        assert_eq!(
            registry.contains(&train_smartcore),
            cfg!(feature = "train-smartcore")
        );
        assert_eq!(registry.contains(&eval_cls), cfg!(feature = "evaluate"));
        assert_eq!(registry.contains(&eval_reg), cfg!(feature = "evaluate"));
        assert_eq!(registry.contains(&package), cfg!(feature = "package"));
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
            feature = "external",
            feature = "train-smartcore",
            feature = "evaluate",
            feature = "package"
        )) {
            assert!(matches!(second, Err(Error::Validation(_))), "{second:?}");
        } else {
            assert!(second.is_ok(), "{second:?}");
        }
    }
}
