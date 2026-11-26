"""Reference mallops prediction worker for `sklearn-pickle` models (NDJSON protocol v1).

Started by mallops' ExternalPythonPredictor as `python3 -c <this file>` with the payload
directory as cwd (removed from sys.path) and a cleared environment plus:

    MALLOPS_MODEL_FILE  absolute path of the pickled model (unpickling runs its code: trusted)
    MALLOPS_FEATURES    optional JSON array of feature names, the model's column order

Protocol v1: one UTF-8 JSON object per line on stdin (host) and stdout (worker).

    worker -> {"v":1,"type":"ready"}                            once, after loading the model
    host   -> {"id":n,"type":"predict","input":J}
    host   -> {"id":n,"type":"predict_batch","inputs":[J,...]}
    worker -> {"id":n,"ok":true,"output":J}                     reply to predict
    worker -> {"id":n,"ok":true,"outputs":[J,...]}              reply to predict_batch, same order
    worker -> {"id":n,"ok":false,"error":"..."}                 request failed, worker keeps going

Requests are answered one at a time, in order; a line that is not JSON gets an error reply with
"id": null. stdout carries only protocol lines, logs go to stderr. EOF on stdin ends the worker.

An input J is an object {feature: value}, ordered by MALLOPS_FEATURES (else the model's
feature_names_in_), or a list of values already in column order. An output J is
{"prediction": X}, plus "probability": P(positive class) when the model has predict_proba and
exactly two classes. NumPy values are converted to plain JSON types. Only the standard library
is imported here; scikit-learn is needed at runtime by the pickle itself.
"""

import os
import sys

# `python -c` puts '' (the cwd, which is the payload dir) first on sys.path, so a payload file
# named like a module (json.py, pickle.py, numpy.py...) would shadow the real one on import.
# Drop it, and the cwd itself, before any other import.
_cwd = os.getcwd()
sys.path[:] = [p for p in sys.path if p and os.path.abspath(p) != _cwd]

import json  # noqa: E402
import pickle  # noqa: E402


def to_json(x):
    """Converts NumPy scalars and arrays to plain Python values."""
    if hasattr(x, "tolist"):
        return x.tolist()
    if hasattr(x, "item"):
        return x.item()
    return x


def row(names, j):
    if isinstance(j, list):
        return j
    if not isinstance(j, dict):
        raise ValueError("input must be an object or a list")
    if names is None:
        raise ValueError("object input needs MALLOPS_FEATURES or model.feature_names_in_")
    missing = [n for n in names if n not in j]
    if missing:
        raise ValueError(f"missing features {missing}")
    return [j[n] for n in names]


def predict(model, names, inputs):
    x = [row(names, j) for j in inputs]
    out = [{"prediction": to_json(p)} for p in model.predict(x)]
    if hasattr(model, "predict_proba") and len(getattr(model, "classes_", ())) == 2:
        for o, p in zip(out, model.predict_proba(x)):
            o["probability"] = to_json(p[1])
    if len(out) != len(inputs):
        raise ValueError(f"model returned {len(out)} predictions for {len(inputs)} inputs")
    return out


def main():
    out = sys.stdout
    sys.stdout = sys.stderr  # stray prints (model code included) must not corrupt the protocol

    def send(msg):
        out.write(json.dumps(msg, allow_nan=False) + "\n")
        out.flush()

    with open(os.environ["MALLOPS_MODEL_FILE"], "rb") as f:
        model = pickle.load(f)
    names = os.environ.get("MALLOPS_FEATURES")
    if names is not None:
        names = json.loads(names)
    elif hasattr(model, "feature_names_in_"):
        names = [str(n) for n in model.feature_names_in_]
    send({"v": 1, "type": "ready"})

    for line in sys.stdin:
        if not line.strip():
            continue
        rid = None
        try:
            req = json.loads(line)
            rid = req.get("id")
            kind = req.get("type")
            if kind == "predict":
                send({"id": rid, "ok": True, "output": predict(model, names, [req["input"]])[0]})
            elif kind == "predict_batch":
                send({"id": rid, "ok": True, "outputs": predict(model, names, req["inputs"])})
            else:
                raise ValueError(f"unknown request type {kind!r}")
        except Exception as e:  # one failed request never stops the worker
            print(f"request {rid}: {type(e).__name__}: {e}", file=sys.stderr)
            send({"id": rid, "ok": False, "error": f"{type(e).__name__}: {e}"})


main()
