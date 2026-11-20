# 0001 — Modular Cargo workspace

- Status: Accepted · 2026-11-20

## Context
mallops must be usable as a library, with plugins, or standalone. Users should pay (compile time, binary size, dependencies) only for the blocks they use.

## Decision
A Cargo workspace with a small number of crates: `mallops-core` (no ML deps), `mallops-std` (standard blocks, each behind a feature flag), `mallops-cli`, and later `mallops-edge`. Crates are created only when they have code.

## Consequences
- `cargo add mallops-std --features train-linfa,serve-http` gives exactly what is needed.
- Fewer crates = simpler releases. If `mallops-std` grows too heavy, per-stage crates can be split out without breaking users (re-exports).
