# 0008 — Edge build profiles and on-device online learning

- Status: Accepted · 2026-11-22
- Refines: [0001](0001-modular-workspace.md), [0007](0007-core-contracts-revision.md) (D5, D6)

## Context
The target scenario is "train on a server, deploy to edge devices, keep learning on the device". [ADR-0007](0007-core-contracts-revision.md) D5 already separates the two sides: `mallops run` executes the batch DAG and produces a `package`, and `mallops serve` consumes it. [ADR-0001](0001-modular-workspace.md) makes every standard block a feature of `mallops-std`.

Three things are still undefined:
1. `ARCHITECTURE.md` says `mallops-edge` depends on "std (subset)" but never names the subset. Without a defined profile, an edge build can pull in training dependencies (polars, external trainer) by accident.
2. Training on the device is mentioned only for federated learning (FR-EDG-4). On-device **online learning**, where the model adapts to local data and no aggregator is involved, is not covered.
3. It is not specified how a model that changes on the device stays consistent with D5 (serving is not a DAG node) and with the immutable store.

## Decision

### D1 — Named feature profiles in `mallops-std`
`mallops-std` exposes profile features that only combine existing per-block features:

| Profile | Contents | Excludes |
|---|---|---|
| `profile-server` | ingest, preprocess, `train-*`, evaluate, package, `serve-http` | — |
| `profile-edge` | `serve-http`, pure-Rust predictors (`predict-linfa`, `predict-tract`), transform runtime, local metrics buffer | every `train-*`, `train-external`, `predict-external` |

Profiles are shortcuts. A user can always enable individual features instead. `predict-tract` adds the `tract` dependency, which needs approval when it is implemented.

### D2 — `mallops-edge` is inference-only by default
- Default features: `profile-edge`. A default build contains no training code, so there is nothing to disable at runtime.
- `online`: on-device online learning (D3). It enables only the `train-*` features of backends that support incremental updates (D4).
- `fl`: federated client (local training, sends only weight deltas to the aggregator, FR-EDG-4).

### D3 — On-device online learning is a finite DAG, never a mutation inside serving
On a device, online learning (and each federated round) runs as an ordinary **finite DAG** in the device `LocalStore`. `mallops-edge` starts it on a trigger (schedule, number of new labelled samples, aggregator round):

```
ingest(new labelled samples, e.g. std/ingest.mqtt, ADR-0009) → preprocess(reuse transform) → train(base = model of the current package) → evaluate(gate on local holdout) → package
```

- **`base` input port.** A train block that supports incremental updates (`partial_fit`, warm start) declares an optional input port named `base` (kind `model`). The name is a convention of `mallops-std`, opt-in per block: a block that cannot update incrementally does not declare it, so wiring `base` into it fails at validation (`blocks.<id>.in.base`, undeclared input port). No core change is needed.
- Every update produces a **new immutable package** with normal lineage (its `parents` include the previous package's model). The model being served is never changed in place.
- **Gate and hot-swap.** The local serving loop switches to the new package only if the evaluate gate passes. Otherwise the run ends with the gate failure and the current package stays active.
- **Rollback** goes back to the last package that passed the gate, or to the base package pulled from the registry.
- **Retention.** The device keeps the active package, the previous accepted one and the base package; older accepted packages are garbage-collected (ties to FR-STO-5).
- **Reporting.** By default the device reports only the metrics of each update (FR-EDG-3 buffer). Pushing accepted packages back to the registry is opt-in, because it sends a model derived from local data off the device.
- Under `fl`, the aggregator round replaces the `package` step: the device sends the delta and, once aggregation completes, pulls the new global package.

### D4 — Portable flavors for edge
Packages meant for edge should use non-code-executing, pure-Rust-servable flavors (`onnx` via tract, `linfa-*`). Online learning on the device additionally needs a flavor whose train block declares `base` (in practice `linfa-*` models with incremental fitting; `onnx` is inference-only). `predict-external` (Python worker) is not part of `profile-edge`. ADR-0007 D7 still applies: code-executing flavors are refused unless explicitly opted in.

## Consequences
- An edge binary is inference-only by construction. CI adds a check that `cargo tree -p mallops-edge -e normal` with default features contains no `train-*` dependency, plus an `aarch64`/`armv7` cross-check of the default build.
- D5 still holds: serving never becomes a DAG node, and on-device learning reuses the runtime, store, lineage, validation and gates unchanged.
- Device storage is bounded by the retention rule of D3.
- Incremental training becomes an optional capability of train blocks (`base` input). Not every backend supports it, and the validator reports misuse.
- `SPEC.md` (§3.13, roadmap) and `ARCHITECTURE.md` (§2, §8) are updated together with this ADR.
