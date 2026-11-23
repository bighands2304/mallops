//! `std/ingest.csv`: reads a local CSV file into a Parquet `dataset` (task 11.1).

use mallops_core::{Block, BlockOutcome, BlockSpec, BoxFuture, Context, Error, Inputs, Stage};

use super::dataset_port;

use arrow_csv as _; // ponytail: placeholder until 11.1 reads CSV

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

    fn run<'a>(
        &'a self,
        _ctx: &'a Context,
        _inputs: Inputs,
    ) -> BoxFuture<'a, Result<BlockOutcome, Error>> {
        Box::pin(async { Err(Error::Runtime("std/ingest.csv: todo (11.1)".into())) })
    }
}
