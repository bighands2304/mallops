#!/usr/bin/env python3
"""Reference trainer for mallops `std/train.external` (contract v1), standard library only.

mallops runs this script with the node workdir as cwd and writes `inputs.json` and `config.json`
there. The script checks `protocol_version == 1`, "trains" a fake model (it just stores `params`
and the input description) and declares it in `outputs.json`. Non-zero exit = failure.

Wire it in `mallops.yaml` (the `./` makes the path absolute against the mallops.yaml directory):

    model:
      use: std/train.external
      in: { train: features.train }
      with:
        command: ["python3", "./examples/train-external/train.py"]
        flavor: demo-json
        params: { alpha: 0.5 }
        watch_files: [examples/train-external/train.py]   # optional: makes the block cacheable

Full contract: docs/reference/train-external.md
"""

import json
import sys
from pathlib import Path


def load(name):
    doc = json.loads(Path(name).read_text())
    if doc.get("protocol_version") != 1:
        sys.exit(f"{name}: unsupported protocol_version {doc.get('protocol_version')!r}")
    return doc


def main():
    inputs = load("inputs.json")["inputs"]
    config = load("config.json")

    # Real trainers read the files under inputs[port]["path"]; here we only record them.
    model = {
        "params": config["params"],
        "run_id": config["run_id"],
        "trained_on": {port: entry["path"] for port, entry in inputs.items()},
    }
    Path("out").mkdir(exist_ok=True)
    Path("out/model.json").write_text(json.dumps(model, indent=2))

    outputs = {
        "protocol_version": 1,
        "status": "produced",
        "outputs": {
            "model": {
                "path": "out/model.json",
                "kind": "model",
                "format": "json",
                "flavor": config["flavor"],  # null -> mallops falls back to with.flavor
                "metadata": {"n_inputs": len(inputs)},
            }
        },
    }
    Path("outputs.json").write_text(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
