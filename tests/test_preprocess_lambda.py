"""Unit tests for the preprocess Lambda using moto + a stubbed sagemaker client."""
from __future__ import annotations

import importlib
import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import patch

import boto3
import pytest

ROOT = Path(__file__).resolve().parents[1]
HANDLER_PATH = ROOT / "lambdas" / "preprocess" / "handler.py"

moto = pytest.importorskip("moto")


@pytest.fixture
def aws_env(monkeypatch):
    monkeypatch.setenv("SAGEMAKER_ENDPOINT_NAME", "sentiment-aws-dev-endpoint")
    monkeypatch.setenv("ASYNC_INPUT_BUCKET", "async-in-bucket")
    monkeypatch.setenv("ASYNC_OUTPUT_BUCKET", "async-out-bucket")
    monkeypatch.setenv("REQUESTS_TABLE", "sentiment-aws-dev-requests")
    monkeypatch.setenv("MODEL_VERSION", "v1")
    monkeypatch.setenv("ENV", "dev")
    monkeypatch.setenv("METRIC_NAMESPACE", "Sentiment/dev")


@pytest.fixture
def mocked_aws(aws_env):
    with moto.mock_aws():
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket="input-bucket")
        s3.create_bucket(Bucket="async-in-bucket")
        ddb = boto3.client("dynamodb", region_name="us-east-1")
        ddb.create_table(
            TableName="sentiment-aws-dev-requests",
            KeySchema=[
                {"AttributeName": "request_id", "KeyType": "HASH"},
                {"AttributeName": "timestamp_iso", "KeyType": "RANGE"},
            ],
            AttributeDefinitions=[
                {"AttributeName": "request_id", "AttributeType": "S"},
                {"AttributeName": "timestamp_iso", "AttributeType": "S"},
            ],
            BillingMode="PAY_PER_REQUEST",
        )
        yield s3


def _reload_handler():
    sys.modules.pop("handler", None)
    sys.modules.pop("preprocess_handler", None)
    spec = importlib.util.spec_from_file_location("preprocess_handler", HANDLER_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_clean_text_and_hash(mocked_aws):
    h = _reload_handler()
    cleaned = h.clean_text("<p>HELLO world!! https://x</p>")
    assert "<" not in cleaned and "https" not in cleaned
    assert h.text_hash("abc") == h.text_hash("abc")
    assert h.text_hash("abc") != h.text_hash("abd")


def test_parse_object_jsonl(mocked_aws):
    h = _reload_handler()
    body = b'{"text":"a"}\n{"text":"b"}\n'
    recs = h._parse_object(body, "x.jsonl")
    assert len(recs) == 2
    assert recs[0]["text"] == "a"


def test_parse_object_dict(mocked_aws):
    h = _reload_handler()
    recs = h._parse_object(b'{"texts":["x","y"]}', "x.json")
    assert recs == [{"text": "x"}, {"text": "y"}]


def test_lambda_handler_invokes_endpoint(mocked_aws):
    s3 = mocked_aws
    key = "incoming/sample.json"
    s3.put_object(Bucket="input-bucket", Key=key,
                  Body=json.dumps({"texts": ["I love this", "I hate this"]}).encode())

    h = _reload_handler()

    with patch.object(h.sm_async, "invoke_endpoint_async",
                      return_value={"OutputLocation": "s3://async-out-bucket/responses/x.json"}) as inv:
        event = {"Records": [{"s3": {"bucket": {"name": "input-bucket"},
                                     "object": {"key": key}}}]}
        out = h.lambda_handler(event, None)

    assert out["count"] == 1
    inv.assert_called_once()
    kwargs = inv.call_args.kwargs
    assert kwargs["EndpointName"] == "sentiment-aws-dev-endpoint"
    assert kwargs["InputLocation"].startswith("s3://async-in-bucket/payload/dev/")
    body = json.loads(s3.get_object(
        Bucket="async-in-bucket",
        Key=kwargs["InputLocation"].split("async-in-bucket/", 1)[1],
    )["Body"].read())
    assert body["env"] == "dev"
    assert len(body["instances"]) == 2
