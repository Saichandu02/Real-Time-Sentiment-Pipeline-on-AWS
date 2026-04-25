"""SageMaker scikit-learn container entrypoint.

Implements the four standard hooks (``model_fn``, ``input_fn``, ``predict_fn``,
``output_fn``) and is robust to both single-string and batch payloads. Designed
for an *async* inference endpoint, so payloads arrive as ``application/json``
documents in S3 and responses are written back to S3 by the runtime.

Accepted input shapes::

    "single text string"
    {"text": "single text string"}
    {"texts": ["a", "b", "c"]}
    {"instances": [{"text": "a"}, {"text": "b"}]}
    [{"text": "a"}, {"text": "b"}]

Response shape::

    {
      "model_version": "v1",
      "predictions": [
        {"sentiment": "positive", "label": 1, "confidence": 0.91, "scores": {"negative": 0.09, "positive": 0.91}}
      ]
    }
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, List

import joblib
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s :: %(message)s")
log = logging.getLogger("sentiment.inference")

CONTENT_TYPE_JSON = "application/json"
LABELS = ("negative", "positive")

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_HTML_RE = re.compile(r"<[^>]+>")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9\s']")
_WS_RE = re.compile(r"\s+")


def _clean(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.lower()
    s = _URL_RE.sub(" ", s)
    s = _HTML_RE.sub(" ", s)
    s = _NON_ALNUM_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    return s


def model_fn(model_dir: str):
    """Load pipeline + model_version from the SageMaker model dir."""
    model_dir_path = Path(model_dir)
    pipe = joblib.load(model_dir_path / "model.joblib")
    version_file = model_dir_path / "model_version.txt"
    version = version_file.read_text().strip() if version_file.exists() else os.environ.get(
        "MODEL_VERSION", "v1"
    )
    log.info("Loaded model version=%s from %s", version, model_dir)
    return {"pipeline": pipe, "version": version}


def _coerce_to_texts(payload: Any) -> List[str]:
    if payload is None:
        return []
    if isinstance(payload, str):
        return [payload]
    if isinstance(payload, dict):
        if "text" in payload:
            return [str(payload["text"])]
        if "texts" in payload and isinstance(payload["texts"], list):
            return [str(x) for x in payload["texts"]]
        if "instances" in payload and isinstance(payload["instances"], list):
            return [
                str(x.get("text", "")) if isinstance(x, dict) else str(x)
                for x in payload["instances"]
            ]
    if isinstance(payload, list):
        return [
            str(x.get("text", "")) if isinstance(x, dict) else str(x)
            for x in payload
        ]
    raise ValueError(f"Unsupported payload shape: {type(payload).__name__}")


def input_fn(request_body, content_type=CONTENT_TYPE_JSON):
    if content_type and content_type.split(";")[0].strip() != CONTENT_TYPE_JSON:
        raise ValueError(f"Unsupported content type: {content_type}")
    if isinstance(request_body, (bytes, bytearray)):
        request_body = request_body.decode("utf-8")
    payload = json.loads(request_body) if isinstance(request_body, str) else request_body
    texts = _coerce_to_texts(payload)
    if not texts:
        raise ValueError("Empty payload — no texts to score")
    return [_clean(t) for t in texts]


def predict_fn(inputs: List[str], model: dict) -> dict:
    pipe = model["pipeline"]
    t0 = time.perf_counter()
    proba = pipe.predict_proba(inputs)
    pred = np.argmax(proba, axis=1)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    predictions = []
    for p_row, p_idx in zip(proba, pred):
        predictions.append(
            {
                "sentiment": LABELS[int(p_idx)],
                "label": int(p_idx),
                "confidence": float(p_row[int(p_idx)]),
                "scores": {LABELS[i]: float(p_row[i]) for i in range(len(LABELS))},
            }
        )

    return {
        "model_version": model["version"],
        "inference_ms": round(elapsed_ms, 3),
        "batch_size": len(inputs),
        "predictions": predictions,
    }


def output_fn(prediction: dict, accept=CONTENT_TYPE_JSON) -> tuple:
    if accept and accept.split(";")[0].strip() != CONTENT_TYPE_JSON:
        raise ValueError(f"Unsupported accept type: {accept}")
    return json.dumps(prediction), CONTENT_TYPE_JSON
