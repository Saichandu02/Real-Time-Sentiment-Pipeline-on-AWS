"""Shared pytest fixtures.

Adds the source directories to ``sys.path`` so tests can import the
Lambda handlers and the model module without packaging.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

for sub in ("model", "scripts", "dashboard"):
    p = ROOT / sub
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))


@pytest.fixture(autouse=True)
def _aws_safe_env(monkeypatch):
    """Default a region + dummy creds so boto3 imports cleanly in unit tests."""
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "testing")
    yield
