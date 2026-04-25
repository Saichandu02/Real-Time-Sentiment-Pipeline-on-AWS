"""S3 ObjectCreated → SageMaker async-invoke preprocess Lambda.

Flow:
  1. Receive an S3 event (one or more objects).
  2. Download the object, parse JSON or NDJSON or plain text.
  3. Light-weight clean / tokenize / hash.
  4. Upload a normalized payload to ``ASYNC_INPUT_BUCKET/payload/<request_id>.json``.
  5. Call ``sagemaker-runtime.invoke_endpoint_async`` with that S3 URI.
  6. The endpoint will write its response to ``ASYNC_OUTPUT_BUCKET`` which fires
     the postprocess Lambda.

Env vars (set by CloudFormation):
  SAGEMAKER_ENDPOINT_NAME  — name of the async endpoint
  ASYNC_INPUT_BUCKET       — bucket for normalized payloads
  ASYNC_OUTPUT_BUCKET      — bucket the endpoint writes responses to
  REQUESTS_TABLE           — DynamoDB table for in-flight request bookkeeping
  ENV                      — dev|staging|prod
  MODEL_VERSION            — informational
  METRIC_NAMESPACE         — CloudWatch namespace, default "Sentiment"
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
import uuid
from datetime import datetime, timezone
from typing import Iterable
from urllib.parse import unquote_plus

import boto3
from botocore.exceptions import ClientError

log = logging.getLogger()
log.setLevel(logging.INFO)

ENV = os.environ.get("ENV", "dev")
ENDPOINT = os.environ["SAGEMAKER_ENDPOINT_NAME"]
INPUT_BUCKET = os.environ["ASYNC_INPUT_BUCKET"]
OUTPUT_BUCKET = os.environ["ASYNC_OUTPUT_BUCKET"]
REQUESTS_TABLE = os.environ.get("REQUESTS_TABLE")
MODEL_VERSION = os.environ.get("MODEL_VERSION", "v1")
NAMESPACE = os.environ.get("METRIC_NAMESPACE", "Sentiment")

s3 = boto3.client("s3")
sm_async = boto3.client("sagemaker-runtime")
cw = boto3.client("cloudwatch")
ddb = boto3.resource("dynamodb")
_table = ddb.Table(REQUESTS_TABLE) if REQUESTS_TABLE else None

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_HTML_RE = re.compile(r"<[^>]+>")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9\s']")
_WS_RE = re.compile(r"\s+")

MAX_TEXT_CHARS = 10_000
MAX_BATCH = 256


def clean_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.lower()
    s = _URL_RE.sub(" ", s)
    s = _HTML_RE.sub(" ", s)
    s = _NON_ALNUM_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    return s[:MAX_TEXT_CHARS]


def text_hash(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def _parse_object(body: bytes, key: str) -> list[dict]:
    """Accept JSON dict, JSON list, NDJSON, or raw text. Returns list of records."""
    text = body.decode("utf-8", errors="replace").strip()
    if not text:
        return []

    if key.endswith(".jsonl") or "\n{" in text[:1000]:
        records = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                obj = {"text": line}
            records.append(obj)
        return records

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return [{"text": text}]

    if isinstance(parsed, dict):
        if "texts" in parsed and isinstance(parsed["texts"], list):
            return [{"text": t} for t in parsed["texts"]]
        if "instances" in parsed and isinstance(parsed["instances"], list):
            return list(parsed["instances"])
        return [parsed]
    if isinstance(parsed, list):
        return [r if isinstance(r, dict) else {"text": str(r)} for r in parsed]
    return [{"text": str(parsed)}]


def _normalize(records: Iterable[dict], source_uri: str) -> list[dict]:
    out = []
    for rec in records:
        raw = rec.get("text") or rec.get("review") or rec.get("body") or ""
        cleaned = clean_text(str(raw))
        if not cleaned:
            continue
        out.append(
            {
                "text": cleaned,
                "text_hash": text_hash(cleaned),
                "source_uri": source_uri,
                "client_request_id": rec.get("request_id") or rec.get("id"),
                "metadata": rec.get("metadata") or {},
            }
        )
        if len(out) >= MAX_BATCH:
            break
    return out


def _emit_metric(name: str, value: float, unit: str = "Count") -> None:
    try:
        cw.put_metric_data(
            Namespace=NAMESPACE,
            MetricData=[
                {
                    "MetricName": name,
                    "Value": value,
                    "Unit": unit,
                    "Dimensions": [
                        {"Name": "Env", "Value": ENV},
                        {"Name": "Stage", "Value": "preprocess"},
                    ],
                }
            ],
        )
    except ClientError as e:
        log.warning("metric publish failed: %s", e)


def _record_inflight(request_id: str, source_uri: str, batch_size: int) -> None:
    if not _table:
        return
    try:
        _table.put_item(
            Item={
                "request_id": request_id,
                "timestamp_iso": datetime.now(timezone.utc).isoformat(),
                "status": "inflight",
                "source_uri": source_uri,
                "batch_size": batch_size,
                "model_version": MODEL_VERSION,
                "env": ENV,
            }
        )
    except ClientError as e:
        log.warning("inflight bookkeeping failed: %s", e)


def _process_object(bucket: str, key: str) -> dict:
    t0 = time.perf_counter()
    obj = s3.get_object(Bucket=bucket, Key=key)
    records = _parse_object(obj["Body"].read(), key)
    source_uri = f"s3://{bucket}/{key}"
    normalized = _normalize(records, source_uri)
    if not normalized:
        log.warning("No usable records in %s", source_uri)
        return {"source_uri": source_uri, "skipped": True}

    request_id = str(uuid.uuid4())
    payload = {
        "request_id": request_id,
        "source_uri": source_uri,
        "model_version": MODEL_VERSION,
        "env": ENV,
        "submitted_at": datetime.now(timezone.utc).isoformat(),
        "preprocess_ms": round((time.perf_counter() - t0) * 1000, 3),
        "instances": normalized,
        "texts": [r["text"] for r in normalized],
    }

    payload_key = f"payload/{ENV}/{request_id}.json"
    s3.put_object(
        Bucket=INPUT_BUCKET,
        Key=payload_key,
        Body=json.dumps(payload).encode("utf-8"),
        ContentType="application/json",
        Metadata={"request_id": request_id, "env": ENV, "model_version": MODEL_VERSION},
    )

    invoke = sm_async.invoke_endpoint_async(
        EndpointName=ENDPOINT,
        InputLocation=f"s3://{INPUT_BUCKET}/{payload_key}",
        ContentType="application/json",
        Accept="application/json",
        InferenceId=request_id,
    )

    _record_inflight(request_id, source_uri, len(normalized))
    _emit_metric("PreprocessRequests", 1)
    _emit_metric("PreprocessBatchSize", len(normalized), unit="Count")
    _emit_metric(
        "PreprocessLatencyMs",
        (time.perf_counter() - t0) * 1000.0,
        unit="Milliseconds",
    )

    log.info(
        "Submitted request_id=%s batch=%d output=%s",
        request_id,
        len(normalized),
        invoke.get("OutputLocation"),
    )
    return {
        "request_id": request_id,
        "batch_size": len(normalized),
        "output_location": invoke.get("OutputLocation"),
    }


def lambda_handler(event, context):
    log.info("event=%s", json.dumps(event)[:2000])
    results = []
    for record in event.get("Records", []):
        try:
            bucket = record["s3"]["bucket"]["name"]
            key = unquote_plus(record["s3"]["object"]["key"])
            results.append(_process_object(bucket, key))
        except Exception as exc:
            log.exception("Failed processing record: %s", exc)
            _emit_metric("PreprocessErrors", 1)
            results.append({"error": str(exc), "record": record})
    return {"results": results, "count": len(results)}
