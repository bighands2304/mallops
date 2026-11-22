# 0009 — MQTT blocks and MQTT serving transport

- Status: Accepted · 2026-11-22
- Refines: [0006](0006-store-and-lineage-contracts.md), [0007](0007-core-contracts-revision.md) (D5, D7), [0008](0008-edge-build-profiles-and-on-device-training.md)

## Context
Edge and IoT deployments usually move data over an MQTT broker. The reference scenario: on an edge device a sensor publishes readings to a topic, mallops reads them, the model predicts, and the predictions are published to another topic. The same data, once labelled, feeds on-device online learning (ADR-0008). FR-ING-3 already lists MQTT as an ingest source. mallops needs to:
1. read messages from MQTT topics into a pipeline (training data, online learning);
2. write pipeline results (datasets, predictions, metrics) to MQTT topics;
3. serve inference over MQTT: request topic in, prediction topic out.

MQTT is an unbounded stream, while the `mallops run` DAG is finite and batch, and serving is not a DAG node (ADR-0007 D5). So MQTT appears in two places: bounded blocks inside the DAG, and a serving transport outside it.

## Decision

### D1 — `std/ingest.mqtt`: bounded read into a dataset
- Subscribes to one or more topic filters and stops at the **first** bound reached: `max_messages`, `max_duration`, or `idle_timeout`. At least one bound is required (`validate_config`), otherwise the run would never finish.
- Produces a `dataset` artifact (`format: jsonl` by default, or `parquet` when the payload schema is declared) and records topic, timestamp and QoS per message.
- `cache_policy` returns `Uncacheable`: the same config reads different messages each time. Once ingested, the data is content-addressed like any other dataset, so downstream nodes cache normally and the run is reproducible from the stored dataset.
- **Delivery.** QoS 1 with a persistent session (`clean_session: false`, fixed `client_id`), so messages published while no run is active are kept by the broker. The block acknowledges messages only after its output draft is complete, just before returning. This gives **at-least-once up to the block's return**: a crash or a cancelled run before that point redelivers the messages on the next run. Blocks cannot observe the runtime commit (ADR-0007), so if the commit itself fails after the block returned, that batch is lost. This window is accepted and documented; closing it needs a post-commit hook in the core and is left to a later ADR if real deployments need it. Exactly-once is not offered: consumers dedupe by message id or payload.

### D2 — `std/storage.mqtt`: publish an artifact to a topic
- An ordinary optional data-output block with `Stage::DataStorage` (the same category as FR-STO-1, not the artifact store; see ADR-0006). No new stage is introduced.
- Publishes one message per row of a `dataset`/`features` artifact (or one per artifact for `metrics`) to a topic template (e.g. `devices/{device_id}/predictions`) with the configured QoS and `retain`; with QoS 1 it waits for every `PUBACK` before returning.
- Publishing is an external side effect: `cache_policy` returns `Uncacheable`, so a cache hit never silently skips publishing.
- Output: a small `metrics` artifact (messages published, failures), so the node fits the DAG and the outcome is recorded in lineage.

### D3 — MQTT as a serving transport, not a block
- `mallops serve` and the `mallops-edge` serving loop get an `mqtt` transport next to `http`. It subscribes to a request topic, runs each message through the package's `LoadedPredictor`, and publishes the prediction to a response topic. This covers the sensor → predict → publish scenario without any DAG.
- Configured in the top-level `serve:` section and never started by `mallops run`:

  ```yaml
  serve:
    package: package.bundle
    transport: mqtt              # default: http
    mqtt:
      url: mqtts://broker.local:8883
      client_id: edge-01
      request_topic: sensors/+/readings
      response_topic: predictions/{topic.1}   # or the MQTT v5 response_topic of the request
      qos: 1
      password: ${secret:MQTT_PASSWORD}
  ```

- **Delivery.** Requests are read with QoS 1 and acknowledged only after the prediction has been published (and acknowledged by the broker at QoS 1), so every reading gets at least one prediction. Each response carries the request's MQTT v5 `correlation_data` when present, or its topic and message id, so consumers can dedupe.
- A reading that fails validation or prediction produces an error message on `{response_topic}/error` (configurable) instead of being dropped silently.
- Under `mallops-edge`, the same connection can optionally push the buffered metrics (FR-EDG-3) to a topic. gRPC stays the default for the fleet control plane and FL.

### D4 — Shared connection config, secrets and security
- A single `mqtt` connection schema is used by D1, D2 and D3: `url` (`mqtt://` / `mqtts://`), `client_id`, `keep_alive`, `tls` (CA, client cert), `username`, `password`.
- `password` and the client key accept only `${secret:NAME}` and are resolved through `ctx.secret()` (or the same secret provider in `serve`). A literal value is rejected. Secrets never appear in `config.json`, cache keys or lineage.
- `mqtts://` is recommended. A plaintext `mqtt://` connection to a non-loopback host emits a warning.
- Incoming payloads are untrusted input: size limit (`max_payload_bytes`) and schema validation (the package `io_schema` in D3) before the data reaches the predictor or the dataset.

### D5 — Implementation and dependencies
- Lives in `mallops-std` behind features `ingest-mqtt`, `storage-mqtt`, `serve-mqtt`. `serve-mqtt` is part of `profile-edge` (ADR-0008), since sensor-driven inference is the main edge scenario.
- Proposed client: `rumqttc` (pure Rust, tokio, MQTT 3.1.1 + 5, rustls). It must be approved as a new dependency and pass `cargo deny` when implemented. `mallops-core` gains no dependency; only `ServeConfig` grows the `transport` and `mqtt` fields.

## Consequences
- MQTT data enters and leaves the pipeline as ordinary artifacts, so lineage, hashing and caching downstream are unchanged.
- Re-running a pipeline with `ingest.mqtt` is not reproducible from config alone; reproducibility comes from the stored dataset artifact, as for any live source.
- Duplicates are possible on every MQTT path (at-least-once), and D1 has a documented loss window on commit failure; both are stated in the user docs.
- Integration tests need a broker: an in-process broker (e.g. `rumqttd` as a dev-dependency) in CI.
- `SPEC.md` (FR-ING-3, FR-STO-1, §3.10, §3.13, roadmap) and `ARCHITECTURE.md` (§7, §8) are updated together with this ADR.
- A future streaming runtime (continuous DAG) could replace the bounded ingest of D1; this ADR deliberately does not introduce one.
