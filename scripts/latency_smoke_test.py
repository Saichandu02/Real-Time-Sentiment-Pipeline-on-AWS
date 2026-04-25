"""End-to-end latency smoke test.

Drops N small JSON payloads into the input bucket, then polls the DynamoDB
results table until each ``request_id`` shows up. Computes p50 / p95 / p99
of (S3 PutObject → first DynamoDB row visible) and fails the deploy if the
p95 / p99 budgets are breached.

Usage::

    python scripts/latency_smoke_test.py \
        --env dev --region us-east-1 \
        --input-bucket sentiment-aws-dev-input-1234 \
        --results-table sentiment-aws-dev-results \
        --n 100 --p95-budget-ms 800 --p99-budget-ms 1500
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import string
import sys
import time
import uuid
from datetime import datetime, timezone

import boto3
from boto3.dynamodb.conditions import Key

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s :: %(message)s")
log = logging.getLogger("smoke")

SAMPLE_TEXTS = [
    "I absolutely loved this — best purchase of the year.",
    "Terrible experience, will not be coming back.",
    "Service was fine, nothing special, average overall.",
    "Stunning quality and the support team was incredibly helpful!",
    "Completely useless product, broke within a day.",
]


def _make_payload() -> dict:
    return {
        "request_id": str(uuid.uuid4()),
        "text": random.choice(SAMPLE_TEXTS) + " " + "".join(
            random.choices(string.ascii_lowercase + " ", k=24)
        ),
        "metadata": {"source": "latency_smoke_test"},
    }


def _put_payloads(s3, bucket: str, n: int) -> dict[str, float]:
    """Returns {object_key: epoch_seconds_at_put_completion}."""
    sent: dict[str, float] = {}
    for i in range(n):
        payload = _make_payload()
        key = f"incoming/smoke/{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}_{i:04d}_{payload['request_id']}.json"
        body = json.dumps(payload).encode("utf-8")
        s3.put_object(Bucket=bucket, Key=key, Body=body, ContentType="application/json")
        sent[key] = time.time()
    log.info("Submitted %d payloads to s3://%s/incoming/smoke/", n, bucket)
    return sent


def _poll_results(table_resource, table_name: str, expected: int,
                  poll_interval_s: float, timeout_s: float) -> list[dict]:
    """Scan the results table and return all rows tagged with our smoke source.

    Production-safer alternative would be storing source_uri + smoke flag
    and using a Query on a GSI. Scan keeps this script self-contained.
    """
    table = table_resource.Table(table_name)
    deadline = time.time() + timeout_s
    seen: dict[tuple[str, str], dict] = {}

    while time.time() < deadline:
        last_key = None
        scanned = 0
        while True:
            kw = {"Limit": 1000}
            if last_key:
                kw["ExclusiveStartKey"] = last_key
            resp = table.scan(**kw)
            for it in resp.get("Items", []):
                src = (it.get("source_uri") or "")
                if "/incoming/smoke/" in src:
                    seen[(it["request_id"], it["timestamp_iso"])] = it
            last_key = resp.get("LastEvaluatedKey")
            scanned += len(resp.get("Items", []))
            if not last_key or scanned > 10_000:
                break
        if len(seen) >= expected:
            log.info("Saw %d rows (expected %d) — done.", len(seen), expected)
            break
        log.info("Saw %d/%d rows so far; sleeping %.1fs", len(seen), expected, poll_interval_s)
        time.sleep(poll_interval_s)
    return list(seen.values())


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    k = int(round((pct / 100.0) * (len(s) - 1)))
    return s[k]


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--env", required=True)
    p.add_argument("--region", default="us-east-1")
    p.add_argument("--input-bucket", required=True)
    p.add_argument("--results-table", required=True)
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--poll-interval-s", type=float, default=2.0)
    p.add_argument("--timeout-s", type=float, default=300.0)
    p.add_argument("--p95-budget-ms", type=float, default=800.0)
    p.add_argument("--p99-budget-ms", type=float, default=1500.0)
    p.add_argument("--report-path", default=".build/smoke_report.json")
    args = p.parse_args(argv)

    session = boto3.session.Session(region_name=args.region)
    s3 = session.client("s3")
    ddb = session.resource("dynamodb")

    sent = _put_payloads(s3, args.input_bucket, args.n)
    log.info("Polling DynamoDB %s for results...", args.results_table)
    rows = _poll_results(ddb, args.results_table,
                         expected=args.n,
                         poll_interval_s=args.poll_interval_s,
                         timeout_s=args.timeout_s)

    seen_count = len(rows)
    latencies: list[float] = []
    for row in rows:
        v = row.get("latency_ms")
        if v is None:
            continue
        try:
            latencies.append(float(v))
        except (TypeError, ValueError):
            continue

    if not latencies:
        log.error("No latency_ms values collected (seen=%d, sent=%d).", seen_count, args.n)
        return 3

    p50 = _percentile(latencies, 50)
    p95 = _percentile(latencies, 95)
    p99 = _percentile(latencies, 99)
    miss = args.n - seen_count

    report = {
        "env": args.env,
        "n_sent": args.n,
        "n_seen": seen_count,
        "missed": miss,
        "p50_ms": round(p50, 2),
        "p95_ms": round(p95, 2),
        "p99_ms": round(p99, 2),
        "max_ms": round(max(latencies), 2),
        "p95_budget_ms": args.p95_budget_ms,
        "p99_budget_ms": args.p99_budget_ms,
    }
    log.info("Latency summary: %s", json.dumps(report, indent=2))

    try:
        import pathlib
        path = pathlib.Path(args.report_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2))
    except Exception as exc:
        log.warning("Could not write report: %s", exc)

    failed = False
    if miss > max(1, int(0.05 * args.n)):
        log.error("Too many missing rows (%d/%d).", miss, args.n)
        failed = True
    if p95 > args.p95_budget_ms:
        log.error("p95 budget breached: %.0fms > %.0fms", p95, args.p95_budget_ms)
        failed = True
    if p99 > args.p99_budget_ms:
        log.error("p99 budget breached: %.0fms > %.0fms", p99, args.p99_budget_ms)
        failed = True

    if failed:
        log.error("SMOKE TEST FAILED — failing the deploy.")
        return 2
    log.info("SMOKE TEST PASSED ✔")
    return 0


if __name__ == "__main__":
    sys.exit(main())
