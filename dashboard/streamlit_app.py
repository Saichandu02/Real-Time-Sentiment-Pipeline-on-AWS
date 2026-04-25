"""Streamlit observability dashboard for the sentiment pipeline.

Polls DynamoDB on a 5s loop (with Streamlit's `st.cache_data` TTL so the
dashboard is cheap with multiple viewers), and renders:

  • Rolling sentiment trend (last 1h / 6h / 24h, configurable).
  • Prediction-distribution histogram + summary KPIs.
  • Drift panel — PSI between the most-recent N predictions and a snapshot
    of the training distribution loaded from a JSON sidecar in S3 / disk.
  • Model-version comparison table (counts, mean confidence, mean latency).
  • Live alerts feed pulled from CloudWatch alarm history.

Configuration is read from environment (set in the runtime) or the sidebar:

  AWS_REGION              — default us-east-1
  RESULTS_TABLE           — DynamoDB results table name (or read SSM)
  ENV                     — dev|staging|prod (used in alarm prefix lookups)
  PROJECT_NAME            — default sentiment-aws
  TRAIN_DIST_JSON_PATH    — local file or s3://bucket/key

Run::
    streamlit run dashboard/streamlit_app.py
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any

import boto3
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from boto3.dynamodb.conditions import Key

REGION = os.environ.get("AWS_REGION", "us-east-1")
ENV_DEFAULT = os.environ.get("ENV", "dev")
PROJECT_DEFAULT = os.environ.get("PROJECT_NAME", "sentiment-aws")
TRAIN_DIST_PATH = os.environ.get("TRAIN_DIST_JSON_PATH", "")
RESULTS_TABLE_DEFAULT = os.environ.get("RESULTS_TABLE", "")

st.set_page_config(
    page_title="Sentiment Pipeline · Observability",
    page_icon="🟦",
    layout="wide",
)


# ──────────────────────────────────────────────────────────────────────
# AWS clients (cached, region-bound)
# ──────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _aws(region: str):
    session = boto3.session.Session(region_name=region)
    return {
        "ddb": session.resource("dynamodb"),
        "ssm": session.client("ssm"),
        "cw": session.client("cloudwatch"),
        "s3": session.client("s3"),
    }


def _resolve_table_name(env: str, project: str, override: str | None) -> str:
    if override:
        return override
    if RESULTS_TABLE_DEFAULT:
        return RESULTS_TABLE_DEFAULT
    try:
        ssm = _aws(REGION)["ssm"]
        return ssm.get_parameter(Name=f"/sentiment/{env}/results_table")["Parameter"]["Value"]
    except Exception:
        return f"{project}-{env}-results"


# ──────────────────────────────────────────────────────────────────────
# Data fetching
# ──────────────────────────────────────────────────────────────────────
def _decimal_to_native(obj: Any) -> Any:
    if isinstance(obj, Decimal):
        return float(obj)
    if isinstance(obj, list):
        return [_decimal_to_native(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _decimal_to_native(v) for k, v in obj.items()}
    return obj


@st.cache_data(ttl=5, show_spinner=False)
def fetch_recent(table_name: str, region: str, hours: float, max_items: int = 5000) -> pd.DataFrame:
    """Query the model_version GSI across known versions, then filter by time.

    For real workloads at scale, replace the scan-fallback with a per-version
    Query loop driven by SSM-discovered model versions.
    """
    table = _aws(region)["ddb"].Table(table_name)
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    cutoff_iso = cutoff.isoformat()

    items: list[dict] = []
    last_key = None
    pages = 0
    while pages < 20 and len(items) < max_items:
        scan_kwargs: dict[str, Any] = {"Limit": 1000}
        if last_key:
            scan_kwargs["ExclusiveStartKey"] = last_key
        resp = table.scan(**scan_kwargs)
        items.extend(resp.get("Items", []))
        last_key = resp.get("LastEvaluatedKey")
        pages += 1
        if not last_key:
            break

    if not items:
        return pd.DataFrame()

    df = pd.DataFrame([_decimal_to_native(it) for it in items])
    if "timestamp_iso" in df.columns:
        df["timestamp"] = pd.to_datetime(
            df["timestamp_iso"].str.split("#").str[0], errors="coerce", utc=True
        )
        df = df[df["timestamp"] >= pd.Timestamp(cutoff_iso)]
        df = df.sort_values("timestamp")
    return df.reset_index(drop=True)


@st.cache_data(ttl=30, show_spinner=False)
def fetch_alarm_history(region: str, project: str, env: str, max_items: int = 50) -> pd.DataFrame:
    cw = _aws(region)["cw"]
    prefix = f"{project}-{env}-"
    try:
        alarms_resp = cw.describe_alarms(AlarmNamePrefix=prefix)
        alarm_names = [a["AlarmName"] for a in alarms_resp.get("MetricAlarms", [])]
    except Exception:
        alarm_names = []

    rows: list[dict] = []
    for name in alarm_names:
        try:
            history = cw.describe_alarm_history(
                AlarmName=name,
                HistoryItemType="StateUpdate",
                MaxRecords=10,
            )
            for h in history.get("AlarmHistoryItems", []):
                rows.append(
                    {
                        "alarm": name,
                        "severity": "P1" if "-P1-" in name else "P2" if "-P2-" in name else "P3",
                        "timestamp": h.get("Timestamp"),
                        "summary": h.get("HistorySummary"),
                    }
                )
        except Exception:
            continue
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).sort_values("timestamp", ascending=False).head(max_items)
    return df


@st.cache_data(ttl=300, show_spinner=False)
def load_training_distribution(path: str) -> dict | None:
    if not path:
        return None
    try:
        if path.startswith("s3://"):
            _, _, rest = path.partition("s3://")
            bucket, _, key = rest.partition("/")
            obj = _aws(REGION)["s3"].get_object(Bucket=bucket, Key=key)
            return json.loads(obj["Body"].read())
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception as exc:
        st.warning(f"Could not load training distribution from {path}: {exc}")
        return None


# ──────────────────────────────────────────────────────────────────────
# Drift / PSI
# ──────────────────────────────────────────────────────────────────────
def population_stability_index(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Standard PSI on a single 1D distribution (e.g. predicted P(positive))."""
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    if expected.size == 0 or actual.size == 0:
        return float("nan")

    edges = np.linspace(0.0, 1.0, bins + 1)
    e_hist, _ = np.histogram(expected, bins=edges)
    a_hist, _ = np.histogram(actual, bins=edges)

    e_pct = np.where(e_hist == 0, 1e-6, e_hist / e_hist.sum())
    a_pct = np.where(a_hist == 0, 1e-6, a_hist / a_hist.sum())
    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))


def _confidence_to_signed_score(df: pd.DataFrame) -> np.ndarray:
    """Return P(positive) per row when scores are present, else best-effort."""
    if "scores" in df.columns and df["scores"].notna().any():
        out = []
        for s in df["scores"]:
            if isinstance(s, dict) and "positive" in s:
                out.append(float(s["positive"]))
            else:
                out.append(np.nan)
        arr = np.array(out, dtype=float)
        if np.isfinite(arr).any():
            return arr[np.isfinite(arr)]
    if {"sentiment", "confidence"}.issubset(df.columns):
        c = df["confidence"].astype(float).fillna(0.5)
        signed = np.where(df["sentiment"].eq("positive"), c, 1.0 - c)
        return signed.to_numpy()
    return np.array([])


# ──────────────────────────────────────────────────────────────────────
# Sidebar / config
# ──────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Connection")
    env = st.selectbox("Environment", ["dev", "staging", "prod"],
                       index=["dev", "staging", "prod"].index(ENV_DEFAULT) if ENV_DEFAULT in ("dev","staging","prod") else 0)
    project = st.text_input("Project name", value=PROJECT_DEFAULT)
    table_override = st.text_input("Results table override", value="")
    table_name = _resolve_table_name(env, project, table_override or None)
    st.caption(f"Reading: `{table_name}`  ({REGION})")

    st.header("Window")
    window_label = st.radio("Lookback", ["1h", "6h", "24h"], index=0, horizontal=True)
    window_hours = {"1h": 1, "6h": 6, "24h": 24}[window_label]

    st.header("Refresh")
    auto_refresh = st.toggle("Auto-refresh every 5s", value=True)
    if st.button("Force refresh now"):
        fetch_recent.clear()
        fetch_alarm_history.clear()

    st.header("Drift")
    drift_window_n = st.number_input("PSI sample size (last N predictions)", 100, 10_000, 1000, 100)
    train_path = st.text_input("Training distribution JSON", value=TRAIN_DIST_PATH)


# ──────────────────────────────────────────────────────────────────────
# Header KPIs
# ──────────────────────────────────────────────────────────────────────
st.title("Sentiment Pipeline · Real-Time Observability")
st.caption(f"env=`{env}` · table=`{table_name}` · region=`{REGION}` · last refresh {datetime.now().strftime('%H:%M:%S')}")

df = fetch_recent(table_name, REGION, window_hours)

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Predictions in window", f"{len(df):,}")
if not df.empty and "confidence" in df.columns:
    k2.metric("Mean confidence", f"{df['confidence'].astype(float).mean():.3f}")
else:
    k2.metric("Mean confidence", "—")
if not df.empty and "latency_ms" in df.columns:
    lat = pd.to_numeric(df["latency_ms"], errors="coerce")
    k3.metric("p95 e2e latency (ms)", f"{lat.quantile(0.95):.0f}")
    k4.metric("p99 e2e latency (ms)", f"{lat.quantile(0.99):.0f}")
else:
    k3.metric("p95 e2e latency (ms)", "—")
    k4.metric("p99 e2e latency (ms)", "—")
if not df.empty and "model_version" in df.columns:
    k5.metric("Active versions", df["model_version"].nunique())
else:
    k5.metric("Active versions", "—")

st.divider()

# ──────────────────────────────────────────────────────────────────────
# Trend + distribution
# ──────────────────────────────────────────────────────────────────────
left, right = st.columns([3, 2])
with left:
    st.subheader(f"Rolling sentiment trend · last {window_label}")
    if df.empty:
        st.info("No predictions in window yet.")
    else:
        bucket = "1min" if window_hours <= 1 else "5min" if window_hours <= 6 else "15min"
        trend = (
            df.assign(t=df["timestamp"].dt.floor(bucket))
              .groupby(["t", "sentiment"]).size().reset_index(name="count")
        )
        fig = px.area(trend, x="t", y="count", color="sentiment",
                      groupnorm="fraction",
                      color_discrete_map={"positive": "#16a34a", "negative": "#dc2626"})
        fig.update_layout(yaxis_title="share", xaxis_title="", height=360, margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("Prediction distribution")
    if df.empty:
        st.info("—")
    else:
        dist = df["sentiment"].value_counts().reset_index()
        dist.columns = ["sentiment", "count"]
        fig = px.bar(dist, x="sentiment", y="count",
                     color="sentiment",
                     color_discrete_map={"positive": "#16a34a", "negative": "#dc2626"})
        fig.update_layout(height=360, margin=dict(l=0,r=0,t=10,b=0), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

# ──────────────────────────────────────────────────────────────────────
# Drift panel
# ──────────────────────────────────────────────────────────────────────
st.subheader("Drift signal · PSI(last predictions vs training)")
train_dist = load_training_distribution(train_path)

if df.empty:
    st.info("No data yet.")
else:
    actual_scores = _confidence_to_signed_score(df.tail(drift_window_n))
    expected_scores = None
    if train_dist:
        if "p_positive_samples" in train_dist:
            expected_scores = np.array(train_dist["p_positive_samples"], dtype=float)
        elif "histogram" in train_dist and "edges" in train_dist:
            edges = np.array(train_dist["edges"], dtype=float)
            counts = np.array(train_dist["histogram"], dtype=float)
            mids = (edges[:-1] + edges[1:]) / 2
            expected_scores = np.repeat(mids, counts.astype(int))

    if expected_scores is not None and len(actual_scores) > 10:
        psi = population_stability_index(expected_scores, actual_scores, bins=10)
        c1, c2 = st.columns([1, 3])
        c1.metric(
            "PSI",
            f"{psi:.3f}",
            help="<0.10 stable · 0.10–0.25 minor · >0.25 significant drift",
        )
        edges = np.linspace(0, 1, 11)
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=expected_scores, xbins=dict(start=0, end=1, size=0.1),
                                   name="training", opacity=0.55, histnorm="probability"))
        fig.add_trace(go.Histogram(x=actual_scores, xbins=dict(start=0, end=1, size=0.1),
                                   name="recent", opacity=0.55, histnorm="probability"))
        fig.update_layout(barmode="overlay", height=320,
                          xaxis_title="P(positive)", yaxis_title="probability",
                          margin=dict(l=0, r=0, t=10, b=0))
        c2.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Drift unavailable — load a training distribution JSON or wait for more predictions.")

# ──────────────────────────────────────────────────────────────────────
# Model version comparison
# ──────────────────────────────────────────────────────────────────────
st.subheader("Model versions")
if df.empty or "model_version" not in df.columns:
    st.info("—")
else:
    grp = df.groupby("model_version").agg(
        n=("sentiment", "size"),
        positive_share=("sentiment", lambda s: float((s == "positive").mean())),
        mean_confidence=("confidence", lambda s: float(pd.to_numeric(s, errors="coerce").mean())),
        p50_latency_ms=("latency_ms", lambda s: float(pd.to_numeric(s, errors="coerce").quantile(0.5))),
        p95_latency_ms=("latency_ms", lambda s: float(pd.to_numeric(s, errors="coerce").quantile(0.95))),
        p99_latency_ms=("latency_ms", lambda s: float(pd.to_numeric(s, errors="coerce").quantile(0.99))),
    ).reset_index()
    st.dataframe(grp, use_container_width=True, hide_index=True)

# ──────────────────────────────────────────────────────────────────────
# Live alerts feed
# ──────────────────────────────────────────────────────────────────────
st.subheader("Alerts")
alerts = fetch_alarm_history(REGION, project, env)
if alerts.empty:
    st.success("No alarm-history entries — clean window.")
else:
    def _badge(sev: str) -> str:
        return {"P1": "🔴 P1", "P2": "🟠 P2", "P3": "🟡 P3"}.get(sev, sev)
    alerts = alerts.assign(severity=alerts["severity"].map(_badge))
    st.dataframe(alerts, use_container_width=True, hide_index=True)

# ──────────────────────────────────────────────────────────────────────
# Auto-refresh loop
# ──────────────────────────────────────────────────────────────────────
if auto_refresh:
    time.sleep(5)
    st.rerun()
