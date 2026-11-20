# 0002 — Arrow/Parquet as default (not mandatory) data interchange

- Status: Accepted · 2026-11-20

## Context
Standard blocks need a common tabular representation to interoperate. At the same time mallops must not lock users into one data format.

## Decision
Standard tabular blocks exchange data as Parquet files (on disk) and Arrow `RecordBatch` (in memory). This is a convention of `mallops-std`, **not** of the core: a dataset artifact can have any `format`, and blocks declare which formats they accept.

## Consequences
- Zero-copy interop with Polars, DataFusion, DuckDB, pandas/pyarrow in external blocks.
- Non-tabular data (images, audio, text corpora) flows as plain files with a declared format.
- Format conversion is an explicit block, never implicit.
