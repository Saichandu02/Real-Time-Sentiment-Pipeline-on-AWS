"""Sanity tests for the model training/inference modules."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model"))

import inference  # noqa: E402
import train  # noqa: E402


def test_clean_text_normalizes():
    s = "<p>Hello WORLD! Visit https://x.com</p>"
    out = train.clean_text(s)
    assert "<" not in out and ">" not in out
    assert "https" not in out
    assert out == out.lower()
    assert "  " not in out


def test_clean_text_handles_non_strings():
    assert train.clean_text(None) == ""  # type: ignore[arg-type]
    assert train.clean_text(42) == ""  # type: ignore[arg-type]


def test_pipeline_trains_and_predicts(tmp_path: Path):
    """Train on a tiny synthetic corpus and assert non-degenerate behavior."""
    pos_words = ["amazing", "fantastic", "love", "great", "excellent", "perfect", "wonderful"]
    neg_words = ["terrible", "awful", "bad", "worst", "horrible", "useless", "dreadful"]
    rows = []
    for w in pos_words:
        for _ in range(20):
            rows.append({"text": f"this {w} product is {w} and {w}, really {w}", "label": 1})
    for w in neg_words:
        for _ in range(20):
            rows.append({"text": f"completely {w} experience, {w} and {w}, very {w}", "label": 0})
    df = pd.DataFrame(rows)
    df.to_csv(tmp_path / "imdb.csv", index=False)

    model_dir = tmp_path / "model"
    output_dir = tmp_path / "out"
    train.main(
        [
            "--dataset", "imdb",
            "--data-dir", str(tmp_path),
            "--model-dir", str(model_dir),
            "--output-dir", str(output_dir),
            "--max-features", "200",
            "--ngram-max", "1",
            "--sample-size", "0",
            "--test-size", "0.25",
            "--seed", "7",
        ]
    )

    metrics = json.loads((output_dir / "metrics.json").read_text())
    assert metrics["accuracy"] >= 0.9

    pipe = joblib.load(model_dir / "model.joblib")
    proba = pipe.predict_proba([
        "i absolutely love this great wonderful amazing product",
        "this was a terrible awful horrible dreadful experience",
    ])
    assert proba.shape == (2, 2)
    assert proba[0, 1] > proba[0, 0]
    assert proba[1, 0] > proba[1, 1]


def test_inference_input_fn_shapes():
    assert inference._coerce_to_texts("hello") == ["hello"]
    assert inference._coerce_to_texts({"text": "hi"}) == ["hi"]
    assert inference._coerce_to_texts({"texts": ["a", "b"]}) == ["a", "b"]
    assert inference._coerce_to_texts([{"text": "a"}, "b"]) == ["a", "b"]
    assert inference._coerce_to_texts({"instances": [{"text": "x"}]}) == ["x"]
    with pytest.raises(ValueError):
        inference._coerce_to_texts(123)


def test_inference_predict_fn_smoke(tmp_path: Path):
    rows = (
        [{"text": "great wonderful amazing fantastic excellent", "label": 1}] * 30
        + [{"text": "awful terrible bad horrible dreadful", "label": 0}] * 30
    )
    df = pd.DataFrame(rows)
    df.to_csv(tmp_path / "imdb.csv", index=False)

    model_dir = tmp_path / "m"
    out_dir = tmp_path / "o"
    train.main(
        ["--dataset", "imdb", "--data-dir", str(tmp_path),
         "--model-dir", str(model_dir), "--output-dir", str(out_dir),
         "--max-features", "100", "--ngram-max", "1",
         "--sample-size", "0", "--test-size", "0.25", "--seed", "1"]
    )

    model = inference.model_fn(str(model_dir))
    out = inference.predict_fn(["love it"], model)
    assert "predictions" in out and out["predictions"][0]["sentiment"] in {"positive", "negative"}
    body, ctype = inference.output_fn(out)
    assert ctype == "application/json"
    assert json.loads(body)["model_version"] == out["model_version"]
