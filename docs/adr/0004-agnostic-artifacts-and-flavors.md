# 0004 — Format-agnostic artifacts with flavors

- Status: Accepted · 2026-11-20

## Context
ML teams use many model formats (ONNX, safetensors, TorchScript, pickle/joblib, GGUF, XGBoost JSON, SavedModel, custom). Picking one (e.g. ONNX) would exclude models that cannot be converted and force lossy conversions.

## Decision
An artifact is opaque payload files plus a `manifest.json` with `kind`, `format`, `flavor` and free-form metadata. The core never reads payloads. Model consumers (serving, evaluation, edge) select an implementation of the `Predictor` trait by `flavor`. Backends (ort, tract, candle, burn, linfa, llama.cpp, external Python workers…) are all equal plugins.

## Consequences
- Any model, any library, any language.
- Validation is limited to declared metadata; a block that receives an unsupported flavor fails early at `mallops validate`.
- Conversions (e.g. torch → onnx) are explicit model-engineering blocks.
