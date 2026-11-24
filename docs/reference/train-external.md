# `std/train.external`

Runs any command (Python, R, Julia, a container…) as a training step. mallops materializes the
input artifacts into a working directory, writes `inputs.json` and `config.json`, runs the
command, and commits what the command declares in `outputs.json`. No Rust bindings are needed.
See SPEC FR-TRN-3 and [ADR-0003](../adr/0003-plugins-wasm-and-grpc.md).

```yaml
blocks:
  model:
    use: std/train.external
    in: { train: features.train, val: features.validation }
    with:
      command: ["python", "./train.py"]
      flavor: sklearn-pickle
      params: { n_estimators: 200 }
      secrets: [MLFLOW_TOKEN]
      timeout_secs: 3600
```

A runnable reference script is in `examples/train-external/train.py`.

## Configuration (`with:`)

Unknown fields are rejected.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `command` | list of strings | required | argv, executed directly (never through a shell). Not empty; `command[0]` not empty. |
| `flavor` | string | none | Flavor of every port of kind `model` (inputs and outputs); fills model outputs that do not set their own. |
| `params` | object | `{}` | Free-form, written to `config.json`. |
| `env` | list of names | `[]` | Host environment variables passed to the child; names unset on the host are skipped. |
| `secrets` | list of names | `[]` | Resolved with `ctx.secret(name)` and passed as env vars of the same name. |
| `timeout_secs` | integer > 0 | none (no timeout) | Kill the process group after this many seconds. |
| `log_cap_bytes` | integer > 0 | 1 MiB | Max bytes kept per stream (stdout, stderr); the rest is drained and dropped. |
| `watch_files` | list of paths | `[]` | Files that make the block cacheable (see [Cache](#cache)). |
| `inputs` | map port → kind | `train` (required) + `val` (optional), kind `features` | **Replaces** the default input ports. |
| `outputs` | map port → kind | `model`, kind `model` | **Replaces** the default output port. |

Validation (`mallops validate`, which never runs the command):

- `env` / `secrets` names match `[A-Za-z_][A-Za-z0-9_]*`, are disjoint from each other and differ
  from `PATH`, `SYSTEMROOT` and `MALLOPS_*`.
- Port names match `[a-z][a-z0-9_]*` (also re-checked at run time for input names given through
  the library API).
- Strings containing `${secret:` in `command`, `flavor`, `params`, `env`, `secrets` or
  `watch_files` are rejected: use `secrets:`.

## Ports

Default inputs: `train` (required) and `val` (optional), kind `features`. Default output: `model`,
kind `model`. Custom `inputs` / `outputs` replace these entirely. Unknown ports, kind mismatches and
missing required outputs are checked by the runtime when it commits the outputs.

## Contract v1

mallops runs the command with the node workdir as cwd (mode `0700` on Unix; removed with the run).
Every contract file carries `"protocol_version": 1`; a command must refuse any other value.

```text
<workdir>/
  inputs.json        written by mallops
  config.json        written by mallops
  inputs/<port>/…    read-only copies of the input files
  outputs.json       written by the command
  …                  whatever the command writes (outputs live here)
```

Input files are materialized read-only (directories stay writable) and made writable again when
the node ends, whatever the outcome, so the run workdir can be removed (Windows refuses to delete
read-only files).
All paths in the contract use `/` and are relative to the workdir.

### `inputs.json`

```json
{
  "protocol_version": 1,
  "inputs": {
    "train": {
      "id": "<artifact id, hex>",
      "path": "inputs/train",
      "manifest": {
        "schema_version": 1,
        "kind": "features",
        "format": "parquet",
        "metadata": {},
        "parents": []
      }
    }
  }
}
```

Optional ports that are not wired are absent from `inputs`. `manifest` is the artifact manifest;
`flavor` and `io_schema` appear only when set.

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "type": "object",
  "required": ["protocol_version", "inputs"],
  "additionalProperties": false,
  "properties": {
    "protocol_version": { "const": 1 },
    "inputs": {
      "type": "object",
      "additionalProperties": {
        "type": "object",
        "required": ["id", "path", "manifest"],
        "additionalProperties": false,
        "properties": {
          "id": { "type": "string" },
          "path": { "type": "string" },
          "manifest": { "type": "object" }
        }
      }
    }
  }
}
```

### `config.json`

```json
{
  "protocol_version": 1,
  "flavor": "sklearn-pickle",
  "params": { "n_estimators": 200 },
  "run_id": "<run id>"
}
```

`flavor` is `null` when not configured; `params` is `{}` when absent. **Secrets are never written
here.**

```json
{
  "type": "object",
  "required": ["protocol_version", "flavor", "params", "run_id"],
  "additionalProperties": false,
  "properties": {
    "protocol_version": { "const": 1 },
    "flavor": { "type": ["string", "null"] },
    "params": { "type": "object" },
    "run_id": { "type": "string" }
  }
}
```

### `outputs.json`

```json
{
  "protocol_version": 1,
  "status": "produced",
  "outputs": {
    "model": {
      "path": "out/model.pkl",
      "kind": "model",
      "format": "pickle",
      "flavor": "sklearn-pickle",
      "io_schema": null,
      "metadata": { "n_estimators": 200 }
    }
  }
}
```

`status` is `"produced"` or `"skipped"`; a skip may carry `"skip_reason": "…"` and no outputs.
Per output: `path` and `kind` and `format` are required; `flavor`, `io_schema` and `metadata` are
optional. An explicit `flavor` is accepted on any port; when absent, outputs of kind `model` get the
configured `with.flavor`.

```json
{
  "type": "object",
  "required": ["protocol_version", "status"],
  "additionalProperties": false,
  "properties": {
    "protocol_version": { "const": 1 },
    "status": { "enum": ["produced", "skipped"] },
    "skip_reason": { "type": ["string", "null"] },
    "outputs": {
      "type": "object",
      "additionalProperties": {
        "type": "object",
        "required": ["path", "kind", "format"],
        "additionalProperties": false,
        "properties": {
          "path": { "type": "string" },
          "kind": { "enum": ["dataset", "features", "transform", "model", "metrics", "package", "blob"] },
          "format": { "type": "string" },
          "flavor": { "type": ["string", "null"] },
          "io_schema": {},
          "metadata": { "type": "object" }
        }
      }
    }
  }
}
```

mallops validates each `path`: it must be a valid relative path (no `..`, not absolute), no
component under the workdir may be a symlink, its canonical path must stay inside the workdir,
and it must be a regular file (committed as a file artifact) or a directory (committed as a
directory artifact). A non-zero exit is an error even if
`outputs.json` exists.

## Environment of the child

The inherited environment is **cleared**. The child always gets:

- `PATH` (and `SYSTEMROOT` on Windows),
- `MALLOPS_PIPELINE_DIR`: mallops' current directory, absolute (the directory of `mallops.yaml`:
  the CLI changes into it),
- the host variables listed in `env` that are set on the host (unset ones are skipped), and the
  `secrets` (same name as in the config).

Nothing else leaks in: list what you need (`HOME`, `CUDA_VISIBLE_DEVICES`, `VIRTUAL_ENV`…).

## Secrets

Declare names in `secrets:`; the value comes from `ctx.secret(name)` and reaches the child **only
as an environment variable**. It never appears in `config.json`, the cache key or lineage, and its
value is redacted from captured stdout/stderr (including the stderr tail in error messages).

## Relative paths in `command`

The child's cwd is the workdir, so a relative script path would not resolve. Every `command`
argument starting with `./` or `../` is made absolute against mallops' current directory (the
`mallops.yaml` directory when run from the CLI). Bare words (`python`, `train.py`, `-m`) are left
unchanged and looked up through `PATH`. Hence `["python", "./train.py"]`, not `["python", "train.py"]`.

## Cache

- Without `watch_files` the block is **uncacheable**: it runs every time.
- With `watch_files` it is cacheable: the key adds a BLAKE3 hash of every watched file's path and
  content, in a fixed order. A missing file or a directory is an error.

Not part of the key: values of host environment variables, the binary or interpreter, and the
Python (or other) dependencies. Add the script and a lock file to `watch_files`, and change `params`
when you bump the environment.

## Errors

| Situation | Error |
|---|---|
| Timeout | `Error::Runtime("timed out after Ns")`, process group killed |
| Cancellation | `Error::Cancelled`, process group killed |
| Non-zero exit or killed by signal | `Error::Runtime` with the tail of stderr (secrets redacted) |
| Missing/malformed `outputs.json`, wrong `protocol_version`, bad path or symlink | `Error::Runtime` naming the problem |
| Unknown port, wrong kind, missing required output | rejected when the runtime commits outputs |

## Security

The command is trusted code that runs **with your user's privileges**; this is not a sandbox
(ADR-0007 D7, `mallops.yaml` is trusted). Process-group kill, a cleared environment and read-only
inputs are guard rails against accidents, not isolation. Flavors that execute code on load (e.g.
`sklearn-pickle`, joblib, torchscript) additionally require an explicit opt-in to be **served**.
`mallops validate` never launches the command.

## What mallops does not manage

Python itself, interpreter versions, virtualenvs, dependencies and containers are yours. Point
`command` at the interpreter you want (`["./.venv/bin/python", "./train.py"]`), or use a container:

```yaml
command: ["docker", "run", "--rm", "-e", "MLFLOW_TOKEN", "my-trainer:1.2"]
secrets: [MLFLOW_TOKEN]
```

Forward env vars explicitly with `-e NAME`. The container only sees the workdir (inputs, contract
files, outputs) if you mount it, and its path is not known statically: use a small wrapper script
(`["./run-in-docker.sh"]`) that runs `docker run -v "$PWD:/work" -w /work …`.

## Platform notes

- **Library use:** the block spawns processes and timers, so the host needs a tokio runtime built
  with `enable_all()`.
- **Windows:** only the direct child is killed on timeout or cancellation; there is no process-group
  kill, so grandchildren may survive. On Unix the whole group is killed.
