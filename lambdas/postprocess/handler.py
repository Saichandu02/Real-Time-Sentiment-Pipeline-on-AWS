"""SageMaker async-output → DynamoDB + CloudWatch postprocess Lambda.

Triggered by ObjectCreated events on the async output bucket. Reads the
endpoint response JSON, joins it with the original request payload (if
available), persists per-prediction rows in DynamoDB, and emits CloudWatch
metrics that drive the alarms in the observability stack.

DynamoDB schema:
  PK = request_id (string)
  SK = timestamp_iso (string, ISO-8601 UTC with µs)
  attrs:
    text_hash, sentiment, confidence, model_version,
    latency_ms (end-to-end S3-PutObject → DDB write),
    inference_ms, env, source_uri

GSI: model_version-timestamp_iso-index for version-sliced reads.
"""
from __future__ import annotations

import decimal
import hashlib
import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any
from urllib.parse import unquote_plus, urlparse

import boto3
from botocore.exceptions import ClientError

log = logging.getLogger()
log.setLevel(logging.INFO)

ENV = os.environ.get("ENV", "dev")
RESULTS_TABLE = os.environ["RESULTS_TABLE"]
NAMESPACE = os.environ.get("METRIC_NAMESPACE", "Sentiment")
MODEL_VERSION_FALLBACK = os.environ.get("MODEL_VERSION", "v1")

s3 = boto3.client("s3")
cw = boto3.client("cloudwatch")
ddb = boto3.resource("dynamodb")
table = ddb.Table(RESULTS_TABLE)


def _to_decimal(v: Any) -> Any:
    if isinstance(v, float):
        return decimal.Decimal(str(round(v, 6)))
    if isinstance(v, list):
        return [_to_decimal(x) for x in v]
    if isinstance(v, dict):
        return {k: _to_decimal(x) for k, x in v.items()}
    return v


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _emit(name: str, value: float, unit: str = "Count", **dims: str) -> None:
    dimensions = [{"Name": "Env", "Value": ENV}]
    for k, v in dims.items():
        dimensions.append({"Name": k, "Value": str(v)})
    try:
        cw.put_metric_data(
            Namespace=NAMESPACE,
            MetricData=[
                {
                    "MetricName": name,
                    "Value": value,
                    "Unit": unit,
                    "Dimensions": dimensions,
                }
            ],
        )
    except ClientError as exc:
        log.warning("metric publish failed: %s", exc)


def _load_json(bucket: str, key: str) -> dict:
    obj = s3.get_object(Bucket=bucket, Key=key)
    body = obj["Body"].read()
    last_modified_epoch = obj["LastModified"].timestamp()
    payload = json.loads(body.decode("utf-8"))
    payload.setdefault("_s3", {})["last_modified_epoch"] = last_modified_epoch
    return payload


def _try_load_request_payload(response_payload: dict) -> dict | None:
    """The async runtime preserves InferenceId in the response key when configured;
    we also stash it in the payload itself when available."""
    location = response_payload.get("input_location") or response_payload.get("InputLocation")
    if not location:
        return None
    parsed = urlparse(location)
    if parsed.scheme != "s3":
        return None
    try:
        return _load_json(parsed.netloc, parsed.path.lstrip("/"))
    except ClientError as exc:
        log.warning("could not load original request payload from %s: %s", location, exc)
        return None


def _derive_request_id(response_payload: dict, key: str) -> str:
    return (
        response_payload.get("request_id")
        or response_payload.get("inference_id")
        or response_payload.get("InferenceId")
        or os.path.splitext(os.path.basename(key))[0]
    )


def _row_for(prediction: dict, *, request_id: str, idx: int, ts_iso: str,
             model_version: str, source_uri: str | None,
             latency_ms: float, inference_ms: float, instance: dict | None) -> dict:
    text_hash = (instance or {}).get("text_hash")
    if not text_hash:
        sample = (instance or {}).get("text") or prediction.get("text") or ""
        text_hash = hashlib.sha256(sample.encode("utf-8")).hexdigest()[:16] if sample else "unknown"

    return {
        "request_id": request_id,
        "timestamp_iso": ts_iso,
        "prediction_index": idx,
        "text_hash": text_hash,
        "sentiment": prediction.get("sentiment", "unknown"),
        "label": prediction.get("label"),
        "confidence": prediction.get("confidence"),
        "scores": prediction.get("scores", {}),
        "model_version": model_version,
        "latency_ms": latency_ms,
        "inference_ms": inference_ms,
        "env": ENV,
        "source_uri": source_uri,
        "ttl": int(time.time()) + 60 * 60 * 24 * 90,
    }


def _process_object(bucket: str, key: str) -> dict:
    t_start = time.perf_counter()
    response = _load_json(bucket, key)
    request_payload = _try_load_request_payload(response) or {}

    request_id = _derive_request_id(response, key)
    submitted_at = request_payload.get("submitted_at")
    submitted_epoch = None
    if submitted_at:
        try:
            submitted_epoch = datetime.fromisoformat(submitted_at).timestamp()
        except ValueError:
            submitted_epoch = None

    now = time.time()
    end_to_end_ms = round((now - submitted_epoch) * 1000.0, 3) if submitted_epoch else None
    inference_ms = float(response.get("inference_ms", 0.0))
    model_version = response.get("model_version") or request_payload.get("model_version") or MODEL_VERSION_FALLBACK
    source_uri = request_payload.get("source_uri")
    instances = request_payload.get("instances") or []
    predictions = response.get("predictions") or []

    if not predictions:
        log.warning("no predictions in response for request_id=%s", request_id)
        _emit("PostprocessEmptyResponses", 1)
        return {"request_id": request_id, "written": 0}

    written = 0
    base_ts = datetime.now(timezone.utc)
    sentiment_counts: dict[str, int] = {}
    confidences: list[float] = []

    with table.batch_writer(overwrite_by_pkeys=["request_id", "timestamp_iso"]) as batch:
        for idx, pred in enumerate(predictions):
            ts_iso = base_ts.replace(microsecond=base_ts.microsecond + idx).isoformat() \
                if base_ts.microsecond + idx < 1_000_000 else f"{base_ts.isoformat()}#{idx}"
            row = _row_for(
                pred,
                request_id=request_id,
                idx=idx,
                ts_iso=ts_iso,
                model_version=model_version,
                source_uri=source_uri,
                latency_ms=end_to_end_ms or 0.0,
                inference_ms=inference_ms,
                instance=instances[idx] if idx < len(instances) else None,
            )
            batch.put_item(Item=_to_decimal(row))
            written += 1
            sentiment_counts[row["sentiment"]] = sentiment_counts.get(row["sentiment"], 0) + 1
            if isinstance(row["confidence"], (int, float)):
                confidences.append(float(row["confidence"]))

    if end_to_end_ms is not None:
        _emit("EndToEndLatencyMs", end_to_end_ms, unit="Milliseconds", ModelVersion=model_version)
    _emit("InferenceLatencyMs", inference_ms, unit="Milliseconds", ModelVersion=model_version)
    _emit("PredictionsWritten", written, ModelVersion=model_version)
    for sentiment, count in sentiment_counts.items():
        _emit("PredictionsBySentiment", count, Sentiment=sentiment, ModelVersion=model_version)
    if confidences:
        _emit("MeanConfidence", sum(confidences) / len(confidences), unit="None", ModelVersion=model_version)
    _emit("PostprocessLatencyMs", (time.perf_counter() - t_start) * 1000.0, unit="Milliseconds")

    log.info(
        "request_id=%s written=%d e2e_ms=%s inference_ms=%.2f model=%s",
        request_id, written, end_to_end_ms, inference_ms, model_version,
    )
    return {"request_id": request_id, "written": written, "end_to_end_ms": end_to_end_ms}


def lambda_handler(event, context):
    log.info("event=%s", json.dumps(event)[:2000])
    results = []
    for record in event.get("Records", []):
        try:
            bucket = record["s3"]["bucket"]["name"]
            key = unquote_plus(record["s3"]["object"]["key"])
            results.append(_process_object(bucket, key))
        except Exception as exc:
            log.exception("postprocess failure: %s", exc)
            _emit("PostprocessErrors", 1)
            results.append({"error": str(exc), "record": record})
    return {"results": results, "count": len(results)}
