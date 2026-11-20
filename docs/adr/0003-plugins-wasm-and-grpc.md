# 0003 — Plugins via WASM components and external processes

- Status: Accepted · 2026-11-20

## Context
Users need to add blocks without recompiling mallops, often in languages other than Rust (Python dominates ML). Rust has no stable ABI, so `dylib` plugins break across compiler versions.

## Decision
Three plugin kinds, all exposed to the runtime through the same `Block` trait:
1. **Native**: Rust crates registered at compile time (`inventory`).
2. **WASM components** (wasmtime, WIT interface) for portable, sandboxed blocks.
3. **External processes** via a file-based CLI contract (`inputs.json`/`config.json` → `outputs.json`) and, for long-lived workers, a gRPC protocol.

## Consequences
- Any language works through the external contract; a thin Python SDK is planned.
- No `dylib` loading, no ABI issues.
- The WIT and protobuf definitions become public API and are versioned with the core.
