"""Unit tests for the postprocess Lambda."""
from __future__ import annotations

import importlib
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import boto3
import pytest

ROOT = Path(__file__).resolve().parents[1]
HANDLER_PATH = ROOT / "lambdas" / "postprocess" / "handler.py"

moto = pytest.importorskip("moto")


@pytest.fixture
def mocked_aws(monkeypatch):
    monkeypatch.setenv("RESULTS_TABLE", "sentiment-aws-dev-results")
    monkeypatch.setenv("MODEL_VERSION", "v1")
    monkeypatch.setenv("ENV", "dev")
    monkeypatch.setenv("METRIC_NAMESPACE", "Sentiment/dev")
    with moto.mock_aws():
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket="async-in-bucket")
        s3.create_bucket(Bucket="async-out-bucket")
        ddb = boto3.client("dynamodb", region_name="us-east-1")
        ddb.create_table(
            TableName="sentiment-aws-dev-results",
            KeySchema=[
                {"AttributeName": "request_id", "KeyType": "HASH"},
                {"AttributeName": "timestamp_iso", "KeyType": "RANGE"},
            ],
            AttributeDefinitions=[
                {"AttributeName": "request_id", "AttributeType": "S"},
                {"AttributeName": "timestamp_iso", "AttributeType": "S"},
                {"AttributeName": "model_version", "AttributeType": "S"},
            ],
            BillingMode="PAY_PER_REQUEST",
            GlobalSecondaryIndexes=[
                {
                    "IndexName": "model_version-timestamp_iso-index",
                    "KeySchema": [
                        {"AttributeName": "model_version", "KeyType": "HASH"},
                        {"AttributeName": "timestamp_iso", "KeyType": "RANGE"},
                    ],
                    "Projection": {"ProjectionType": "ALL"},
                }
            ],
        )
        yield s3


def _reload_handler():
    """Load the postprocess handler by absolute path to avoid the
    `handler` module-name collision with the preprocess Lambda."""
    sys.modules.pop("handler", None)
    sys.modules.pop("postprocess_handler", None)
    spec = importlib.util.spec_from_file_location("postprocess_handler", HANDLER_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_postprocess_writes_rows_and_metrics(mocked_aws):
    s3 = mocked_aws

    request_payload = {
        "request_id": "req-123",
        "model_version": "v1",
        "env": "dev",
        "source_uri": "s3://input-bucket/incoming/x.json",
        "submitted_at": datetime.now(timezone.utc).isoformat(),
        "instances": [
            {"text": "good", "text_hash": "abc"},
            {"text": "bad", "text_hash": "def"},
        ],
    }
    s3.put_object(Bucket="async-in-bucket", Key="payload/dev/req-123.json",
                  Body=json.dumps(request_payload).encode())

    response_payload = {
        "request_id": "req-123",
        "model_version": "v1",
        "inference_ms": 12.5,
        "input_location": "s3://async-in-bucket/payload/dev/req-123.json",
        "predictions": [
            {"sentiment": "positive", "label": 1, "confidence": 0.92,
             "scores": {"positive": 0.92, "negative": 0.08}},
            {"sentiment": "negative", "label": 0, "confidence": 0.81,
             "scores": {"positive": 0.19, "negative": 0.81}},
        ],
    }
    s3.put_object(Bucket="async-out-bucket", Key="responses/req-123.out",
                  Body=json.dumps(response_payload).encode())

    h = _reload_handler()
    event = {"Records": [{"s3": {"bucket": {"name": "async-out-bucket"},
                                 "object": {"key": "responses/req-123.out"}}}]}
    out = h.lambda_handler(event, None)

    assert out["count"] == 1
    assert out["results"][0]["written"] == 2

    ddb = boto3.resource("dynamodb", region_name="us-east-1")
    table = ddb.Table("sentiment-aws-dev-results")
    items = table.query(KeyConditionExpression=boto3.dynamodb.conditions.Key("request_id").eq("req-123"))["Items"]
    assert len(items) == 2
    sentiments = {it["sentiment"] for it in items}
    assert sentiments == {"positive", "negative"}
    for it in items:
        assert it["model_version"] == "v1"
        assert it["env"] == "dev"
        assert "ttl" in it
