"""Pure-function tests for the dashboard's drift / PSI math."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dashboard"))


def _load_psi():
    import importlib.util
    spec = importlib.util.spec_from_file_location("streamlit_app", ROOT / "dashboard" / "streamlit_app.py")
    # Importing the streamlit module would trigger top-level Streamlit calls;
    # we extract the function via source compilation instead.
    src = (ROOT / "dashboard" / "streamlit_app.py").read_text()
    ns: dict = {"np": np}
    start = src.index("def population_stability_index")
    end = src.index("def _confidence_to_signed_score")
    exec(src[start:end], ns)
    return ns["population_stability_index"]


def test_psi_zero_for_identical_distributions():
    psi = _load_psi()
    rng = np.random.default_rng(0)
    a = rng.uniform(0, 1, 10_000)
    assert psi(a, a) < 1e-6


def test_psi_high_for_shifted_distribution():
    psi = _load_psi()
    rng = np.random.default_rng(1)
    base = rng.beta(2, 5, 5_000)
    shifted = rng.beta(5, 2, 5_000)
    assert psi(base, shifted) > 0.5


def test_psi_handles_empty():
    psi = _load_psi()
    assert np.isnan(psi(np.array([]), np.array([0.5])))
