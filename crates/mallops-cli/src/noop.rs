//! Built-in `core/noop` block: lets `init && validate && run` work before `mallops-std` (#11).

use mallops_core::{
    ArtifactKind, Block, BlockOutcome, BlockSpec, BoxFuture, Context, DraftPayload, Error, Inputs,
    Manifest, OutputDraft, OutputDrafts, Port, Stage,
};

/// Optional input `in`, one `out` blob (`format: noop`, empty payload).
///
/// Test affordances in `with:`:
/// - `fail: "<message>"` makes the node fail with `Error::Runtime`.
/// - `skip: "<reason>"` returns `BlockOutcome::Skipped` with no outputs.
pub(crate) struct Noop(BlockSpec);

impl Noop {
    pub(crate) fn new() -> Self {
        Self(
            BlockSpec::new("core/noop", "1.0.0", Stage::Custom)
                .with_input(Port::new("in", ArtifactKind::Blob).optional())
                .with_output(Port::new("out", ArtifactKind::Blob).with_format("noop")),
        )
    }
}

impl Block for Noop {
    fn spec(&self) -> &BlockSpec {
        &self.0
    }

    fn run<'a>(
        &'a self,
        ctx: &'a Context,
        _inputs: Inputs,
    ) -> BoxFuture<'a, Result<BlockOutcome, Error>> {
        Box::pin(async move {
            if let Some(msg) = ctx.config().get("fail").and_then(|v| v.as_str()) {
                return Err(Error::Runtime(format!("core/noop: {msg}")));
            }
            if let Some(reason) = ctx.config().get("skip").and_then(|v| v.as_str()) {
                return Ok(BlockOutcome::Skipped {
                    reason: reason.to_owned(),
                    outputs: OutputDrafts::new(),
                });
            }
            let draft = OutputDraft::new(
                Manifest::new(ArtifactKind::Blob, "noop"),
                DraftPayload::Bytes(Vec::new()),
            );
            Ok(BlockOutcome::Produced(
                OutputDrafts::new().with("out", draft),
            ))
        })
    }
}
