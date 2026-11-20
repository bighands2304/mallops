# 0006 — Artifact store and lineage: contracts in the core, implementations pluggable

- Status: Accepted · 2026-11-21
- Refines: [0001](0001-modular-workspace.md)

## Context
The first draft put the artifact store (on top of `object_store`) and the lineage store inside `mallops-core`. That contradicts the "small core, optional everything else" principle and drags cloud dependencies into every build.

Two alternatives were considered:
1. **Make them blocks.** Rejected: blocks are *nodes* of the DAG, while store and lineage act on *every edge*. If storage were an optional node, the runtime could not hand data from A to B, and caching/resume would depend on the user remembering to add it. Lineage needs the whole graph, which only the runtime sees; a block would have to rebuild it.
2. **Keep full implementations in the core.** Rejected: forces `object_store` and a lineage backend on every user.

The draft also mixed two different things: the *data storage* lifecycle stage (persisting user data) and the internal *artifact store*.

## Decision
- The core defines the **contracts**: `ArtifactStore` and `LineageSink` traits, plus the `LineageEvent` type emitted by the runtime.
- The core ships only minimal, dependency-free implementations: `LocalStore` (filesystem, default), `MemoryStore` (ephemeral), `JsonSink` (one `lineage.json` per run).
- Remote stores (S3, GCS, Azure via `object_store`) and other lineage sinks (SQLite, OpenLineage, MLflow) are `mallops-std` features or plugins.
- Lineage supports zero or more sinks; `lineage: off` disables it. Sink failures are warnings, never run failures.
- "Data storage" as a lifecycle stage is an ordinary optional block that writes user data to external systems; it is unrelated to the artifact store.

## Consequences
- `mallops-core` has no cloud or database dependency.
- Caching, resume and data hand-off always work, because the store mechanism cannot be removed — only its backend swapped.
- Caching relies on artifact hashes, not lineage, so disabling lineage loses audit history but not performance.
- `ArtifactStore`, `LineageSink` and `LineageEvent` become public API and must stay stable from 1.0.
