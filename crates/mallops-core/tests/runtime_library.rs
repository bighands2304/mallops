//! Library usage without YAML: `PipelineBuilder` + `BlockRegistry` + `Runtime` on `MemoryStore`.
//!
//! Runs with `cargo test -p mallops-core --no-default-features runtime_library`.

use std::sync::Arc;
use std::time::Duration;

use mallops_core::*;

/// Emits `out` with the given kind; the payload lists the kinds of its inputs.
struct Step {
    spec: BlockSpec,
    out: ArtifactKind,
}

impl Block for Step {
    fn spec(&self) -> &BlockSpec {
        &self.spec
    }

    fn run<'a>(
        &'a self,
        _ctx: &'a Context,
        inputs: Inputs,
    ) -> BoxFuture<'a, Result<BlockOutcome, Error>> {
        Box::pin(async move {
            let seen: Vec<String> = inputs
                .iter()
                .map(|(port, a)| format!("{port}={}", a.manifest().kind))
                .collect();
            let draft = OutputDraft::new(
                Manifest::new(self.out, "text"),
                DraftPayload::Bytes(seen.join(",").into_bytes()),
            );
            Ok(BlockOutcome::Produced(
                OutputDrafts::new().with("out", draft),
            ))
        })
    }
}

fn registry() -> BlockRegistry {
    let port = Port::new;
    let blocks = [
        (
            BlockSpec::new("lib/ingest", "1", Stage::Ingest),
            ArtifactKind::Dataset,
        ),
        (
            BlockSpec::new("lib/train", "1", Stage::Train)
                .with_input(port("data", ArtifactKind::Dataset)),
            ArtifactKind::Model,
        ),
        (
            BlockSpec::new("lib/package", "1", Stage::Package)
                .with_input(port("model", ArtifactKind::Model))
                .with_input(port("data", ArtifactKind::Dataset)),
            ArtifactKind::Package,
        ),
    ];
    let mut reg = BlockRegistry::new();
    for (spec, out) in blocks {
        let spec = spec.with_output(port("out", out));
        reg.register(Arc::new(Step { spec, out })).unwrap();
    }
    reg
}

#[tokio::test]
async fn runtime_library_runs_builder_pipeline_end_to_end() {
    let mut b = PipelineBuilder::new("library");
    b.block("ingest", "lib/ingest".parse().unwrap());
    b.block("train", "lib/train".parse().unwrap())
        .input("data", "ingest.out");
    b.block("package", "lib/package".parse().unwrap())
        .input("model", "train.out")
        .input("data", "ingest.out");
    let pipeline = b.build(&registry()).unwrap();

    let store = Arc::new(MemoryStore::new());
    let mut config = RuntimeConfig::new(Arc::clone(&store) as Arc<dyn ArtifactStore>);
    config.max_jobs = 2;
    let summary =
        tokio::time::timeout(Duration::from_secs(10), Runtime::new(config).run(&pipeline))
            .await
            .expect("run timed out")
            .unwrap();

    assert!(summary.is_success());
    let out = |node: &str| {
        let n = &summary.nodes[node];
        assert_eq!(n.status, Status::Succeeded, "{node}: {n:?}");
        n.outputs["out"]
    };
    let (data, model, package) = (out("ingest"), out("train"), out("package"));

    let artifact = |id: ArtifactId| {
        let store = Arc::clone(&store);
        async move { store.get(&id).await.unwrap().expect("committed") }
    };
    assert!(artifact(data).await.manifest().parents.is_empty());
    assert_eq!(artifact(model).await.manifest().parents, [data]);
    let pkg = artifact(package).await;
    assert_eq!(pkg.manifest().kind, ArtifactKind::Package);
    let mut parents = vec![data, model];
    parents.sort();
    assert_eq!(pkg.manifest().parents, parents);

    // The block saw both inputs, bound to their ports.
    let dir = tempfile::tempdir().unwrap();
    let path = store.fetch(&package, dir.path()).await.unwrap();
    let payload = std::fs::read_dir(&path)
        .unwrap()
        .map(|e| std::fs::read_to_string(e.unwrap().path()).unwrap())
        .collect::<String>();
    assert_eq!(payload, "data=dataset,model=model");
}
