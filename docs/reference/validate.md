# `std/validate.tabular`

Tabular data validation block with a built-in quality gate. Checks a Parquet dataset against
declared schema, row-count, and per-column expectations before it reaches preprocessing or
training. Behind the optional `validate` Cargo feature of `mallops-std` (not in `default`, no new
crate dependency; enabled in `mallops-cli`).
See [SPEC FR-PRE-4](../SPEC.md#33-preprocessing-data-features).

```yaml
blocks:
  validate:
    use: std/validate.tabular
    in: { data: raw.dataset }
    with:
      schema:
        - { name: age, type: int64 }
        - { name: income, type: float64 }
        - { name: plan, type: string }
        - { name: churned, type: bool }
      allow_extra_columns: false
      min_rows: 100
      checks:
        age:
          max_null_fraction: 0.05
          min: 18
          max: 120
        income:
          max_null_fraction: 0.10
          min: 0.0
        plan:
          allowed: ["free", "pro", "enterprise"]
        customer_id:
          unique: true
      on_fail: fail
```

To skip downstream nodes instead of failing the run when validation fails:

```yaml
blocks:
  validate:
    use: std/validate.tabular
    in: { data: raw.dataset }
    with:
      min_rows: 100
      checks:
        churned:
          max_null_fraction: 0.0
      on_fail: skip
```

## Configuration (`with:`)

Unknown fields are rejected (`deny_unknown_fields`). `validate_config` uses the same parser as
`run`, so `mallops validate` catches configuration errors before execution.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `schema` | list of `{ name, type }` | none | Expected columns and their Arrow types (`bool` \| `int64` \| `float64` \| `string` \| `date32`). |
| `allow_extra_columns` | `bool` | `true` | Whether columns present in the dataset but not listed in `schema` are allowed. Only has an effect when `schema` is set. |
| `min_rows` | non-negative integer | none | Minimum number of rows required in the dataset (`rows >= min_rows`). |
| `checks` | map `<column>` → object | none | Per-column data quality expectations (see below). |
| `on_fail` | `fail` \| `skip` | `fail` | Action when one or more checks fail. **Note:** unlike [`std/evaluate`](evaluate.md) (where the default is `skip`), `std/validate.tabular` defaults to `fail`. |

### `checks.<column>` fields

| Key | Type | Constraints |
|---|---|---|
| `max_null_fraction` | `f64` | Finite value in `[0.0, 1.0]`. Passes iff `null_count / row_count <= max_null_fraction`. |
| `min` | `f64` | Finite inclusive lower bound (`value >= min`) for numeric columns (`int64`, `float64`). |
| `max` | `f64` | Finite inclusive upper bound (`value <= max`) for numeric columns (`int64`, `float64`). |
| `allowed` | list of JSON scalars | Set of permitted non-null values. Supported only on `string`, `int64`, and `bool` columns. |
| `unique` | `bool` | When `true`, all non-null values in the column must be distinct. |

Validation rules (`validate_config`):

- Unknown fields anywhere in `with:` are rejected (`deny_unknown_fields`).
- `min > max` on the same column is rejected.
- `max_null_fraction` outside `[0.0, 1.0]` or non-finite (`NaN`, `±inf`) is rejected.
- Non-finite `min` or `max` values (`NaN`, `±inf`) are rejected.
- An empty check object for a column in `checks` (no check keys configured) is rejected.
- Strings containing `${secret:` anywhere in the config are rejected (as in
  [`std/external`](external.md) and [`std/evaluate`](evaluate.md)).

## Ports

`version: "0.1.0"`, `Stage::Preprocess`:

- **Block ID:** `std/validate.tabular`
- **Input:**
  - `data` (required): `kind: dataset`, `format: parquet`.
- **Outputs:**
  - `data`: `kind: dataset`, `format: parquet` — carries the exact same payload and manifest
    content as the input `data`, but receives a **new `ArtifactId`** because the runtime sets
    `manifest.parents` to the input `ArtifactId` on commit
    ([ADR-0007 D3](../adr/0007-core-contracts-revision.md)).
  - `report`: `kind: metrics`, `format: json` — validation report written to `report.json`.

## Execution and check semantics

- **Data problems are failed checks, not errors:** any mismatch between the dataset and the
  configured expectations produces a failed check entry in `report.json` rather than aborting with
  `Error::Runtime`. This includes:
  - a column declared in `schema` or referenced in `checks` that is missing from the dataset;
  - a column whose Arrow type does not match the type declared in `schema`;
  - extra columns present when `allow_extra_columns: false`;
  - `min` or `max` configured on a non-numeric column;
  - `allowed` configured on a `float64` (or other unsupported) column.
- **Exhaustive evaluation:** all configured checks are evaluated on the dataset; validation does
  not stop at the first failure.
- **Null values:** nulls count **only** toward `max_null_fraction`. Nulls are ignored by `min`,
  `max`, `allowed`, and `unique`.
- **Empty dataset (`0` rows):** the observed null fraction is `0.0` (so `max_null_fraction`
  passes). `min` and `max` pass on an empty dataset; `min_rows` fails if `min_rows > 0`.
- **All-null columns:** `min` and `max` pass when every value in the column is null.
- **`NaN` values:** `NaN` always fails `min` and `max` checks (same rule as
  [`std/evaluate`](evaluate.md)).
- **`allowed` types:** `allowed` is valid only on `string`, `int64`, and `bool` columns. Applying
  `allowed` to a `float64` (or `date32`) column records a failed check.
- **In-memory loading:** the entire input Parquet file is loaded into memory via the shared
  `ingest` Parquet reader.

## Output (`report.json`)

The `report` artifact payload is a single file, `report.json`, with a fixed, deterministic schema
and check ordering:

```json
{
  "v": 1,
  "passed": false,
  "checks": [
    {
      "check": "schema",
      "column": "age",
      "passed": true,
      "observed": "int64",
      "expected": "int64"
    },
    {
      "check": "min_rows",
      "passed": true,
      "observed": 250,
      "expected": 100
    },
    {
      "check": "min",
      "column": "age",
      "passed": false,
      "observed": 12,
      "expected": 18
    },
    {
      "check": "allowed",
      "column": "plan",
      "passed": false,
      "observed": { "count": 2, "examples": ["legacy", "trial"] },
      "expected": ["free", "pro", "enterprise"]
    }
  ]
}
```

- `v`: report schema version, always `1`.
- `passed`: `true` iff every entry in `checks` has `passed: true`.
- `checks`: array of `{ "check", "column"?, "passed", "observed", "expected" }` in a **fixed
  deterministic order**:
  1. `schema` checks (declared columns and, when `allow_extra_columns: false`, extra-column check),
  2. `min_rows`,
  3. per-column `checks`, with columns sorted **alphabetically**.
- **Non-finite floats:** `NaN`, `+inf`, and `-inf` are serialized as JSON strings `"NaN"`, `"inf"`,
  and `"-inf"` so `report.json` is always valid JSON.
- **`observed` for `allowed` and `unique`:** reports the violation count together with at most
  **5** offending examples.
- No timestamps, run IDs, or block-written `parents` (the runtime injects `parents` on commit).

## Quality gate semantics (`on_fail`)

- **All checks passed (`passed == true`):** returns `BlockOutcome::Produced` with both `data` and
  `report`.
- **One or more checks failed with `on_fail: fail` (default):**
  - Returns `Error::Validation` containing one `ValidationError` item per failed check.
  - The error paths (`schema.<col>`, `min_rows`, `checks.<col>.<check>`) refer to the **validation
    report**, not to the YAML pipeline definition.
  - The runtime marks the node `Failed`, cancels all descendants, does **not** commit `data` or
    `report`, leaves `runs/latest.json` unchanged, and `mallops run` exits `1`.
- **One or more checks failed with `on_fail: skip`:**
  - Returns `BlockOutcome::Skipped` (committing `report` so the failed report can be inspected).
  - **Downstream blocking:** the runtime skips downstream nodes that consume `validate.data` (or
    `validate.report`) as a **required** input. `report` reaches only downstream nodes that wire it
    as an **optional** input.
  - **`runs/latest.json` and exit code:** when any node is `Skipped` (and no node failed),
    `mallops run` leaves `runs/latest.json` unchanged and exits `0`.
  - **Caching:** `Skipped` nodes are never cached by the runtime, so a failed validation is
    re-evaluated on the next run.

## Cache

`CachePolicy::Deterministic { extra_hash: None }`.

- Same input `data`, config, node `seed:`, and block version hit the cache when the previous run
  `Succeeded`. `Skipped` and `Failed` runs are never cached.
- Changing check semantics or report format requires bumping `BlockSpec.version`.

## Errors

| Situation | Error |
|---|---|
| Unknown field, `min > max`, `max_null_fraction` outside `[0.0, 1.0]` or non-finite, non-finite `min`/`max`, empty check entry, `${secret:` in config | `Error::Config` (in `mallops validate`, aggregated into `Error::Validation` under `blocks.<id>.with`) |
| One or more data checks failed with `on_fail: fail` | `Error::Validation` (one item per failed check; report paths `schema.<col>`, `min_rows`, `checks.<col>.<check>`) |
| Unreadable or corrupted input Parquet file | `Error::Runtime` |
| Cancellation via `ctx.cancel()` | `Error::Cancelled` |

Missing columns, type mismatches, `min`/`max` on non-numeric columns, and `allowed` on `float64`
are **failed checks** (handled by the `on_fail` gate), not `Error::Runtime`.

## Known limitations

- **In-memory dataset:** the entire input Parquet file is loaded into memory. Streaming /
  out-of-core validation is planned for v0.2
  ([SPEC NFR-8](../SPEC.md#4-non-functional-requirements)).
- **No statistical drift detection in v0.1:** comparing distributions across datasets (PSI, KS
  test) is planned for v0.3 ([SPEC FR-MON-3](../SPEC.md#311-monitoring-and-logging)).
