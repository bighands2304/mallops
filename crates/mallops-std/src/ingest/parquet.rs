//! `std/ingest.parquet`: reads a local Parquet file into a Parquet `dataset` (task 11.2).

use mallops_core::{Block, BlockOutcome, BlockSpec, BoxFuture, Context, Error, Inputs, Stage};

use super::dataset_port;

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

    fn run<'a>(
        &'a self,
        _ctx: &'a Context,
        _inputs: Inputs,
    ) -> BoxFuture<'a, Result<BlockOutcome, Error>> {
        Box::pin(async { Err(Error::Runtime("std/ingest.parquet: todo (11.2)".into())) })
    }
}
