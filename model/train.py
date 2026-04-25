"""SageMaker training entrypoint for TF-IDF + Logistic Regression sentiment model.

Trains on IMDB or Sentiment140 (subsampled to ~200k rows) and writes a
``model.tar.gz``-ready directory in ``SM_MODEL_DIR``.

The script honors the standard SageMaker contract:

  * Hyperparameters are passed via CLI flags (or ``SM_HPS`` env var).
  * Training data is read from ``SM_CHANNEL_TRAINING``.
  * The serialized model is written to ``SM_MODEL_DIR``.

Local invocation (outside SageMaker)::

    python model/train.py --data-dir ./data --model-dir ./artifacts \\
        --dataset imdb --max-features 20000 --sample-size 200000
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import tarfile
import time
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("sentiment.train")


_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_HTML_RE = re.compile(r"<[^>]+>")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9\s']")
_WS_RE = re.compile(r"\s+")


def clean_text(s: str) -> str:
    """Light, deterministic text normalization shared with inference/preprocess."""
    if not isinstance(s, str):
        return ""
    s = s.lower()
    s = _URL_RE.sub(" ", s)
    s = _HTML_RE.sub(" ", s)
    s = _NON_ALNUM_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    return s


def _load_imdb(data_dir: Path) -> pd.DataFrame:
    """Expects either a single CSV ``imdb.csv`` (text,label) or aclImdb/ tree."""
    csv_path = data_dir / "imdb.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        df.columns = [c.lower() for c in df.columns]
        if "review" in df.columns and "text" not in df.columns:
            df = df.rename(columns={"review": "text"})
        if "sentiment" in df.columns and "label" not in df.columns:
            df["label"] = (df["sentiment"].astype(str).str.lower() == "positive").astype(int)
        return df[["text", "label"]].dropna()

    rows = []
    for split in ("train", "test"):
        for label_name, label in (("pos", 1), ("neg", 0)):
            d = data_dir / "aclImdb" / split / label_name
            if not d.exists():
                continue
            for fp in d.glob("*.txt"):
                rows.append({"text": fp.read_text(encoding="utf-8", errors="ignore"), "label": label})
    if not rows:
        raise FileNotFoundError(f"No IMDB data found under {data_dir}")
    return pd.DataFrame(rows)


def _load_sentiment140(data_dir: Path) -> pd.DataFrame:
    """Sentiment140 CSV: label,id,date,query,user,text  (label in {0,2,4})."""
    csv_path = next(data_dir.glob("*sentiment140*.csv"), None) or next(
        data_dir.glob("training.1600000.processed.noemoticon.csv"), None
    )
    if csv_path is None:
        raise FileNotFoundError(f"No sentiment140 CSV found under {data_dir}")
    df = pd.read_csv(
        csv_path,
        encoding="latin-1",
        header=None,
        names=["label", "id", "date", "query", "user", "text"],
    )
    df = df[df["label"].isin([0, 4])].copy()
    df["label"] = (df["label"] == 4).astype(int)
    return df[["text", "label"]]


def load_dataset(name: str, data_dir: Path, sample_size: int, seed: int) -> pd.DataFrame:
    name = name.lower()
    if name == "imdb":
        df = _load_imdb(data_dir)
    elif name in {"sentiment140", "s140"}:
        df = _load_sentiment140(data_dir)
    else:
        raise ValueError(f"Unknown dataset {name!r}; expected imdb | sentiment140")

    df = df.dropna(subset=["text", "label"]).reset_index(drop=True)
    if sample_size and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=seed).reset_index(drop=True)
    log.info("Loaded %s rows from %s", len(df), name)
    return df


def build_pipeline(max_features: int, ngram_max: int, C: float) -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=max_features,
                    ngram_range=(1, ngram_max),
                    sublinear_tf=True,
                    min_df=2,
                    strip_accents="unicode",
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    C=C,
                    solver="liblinear",
                    max_iter=1000,
                    class_weight="balanced",
                ),
            ),
        ]
    )


def evaluate(pipe: Pipeline, X: pd.Series, y: np.ndarray) -> dict:
    proba = pipe.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred)),
        "roc_auc": float(roc_auc_score(y, proba)),
        "support": int(len(y)),
    }


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default=os.environ.get("DATASET", "imdb"))
    p.add_argument(
        "--data-dir",
        default=os.environ.get("SM_CHANNEL_TRAINING", "./data"),
        type=Path,
    )
    p.add_argument(
        "--model-dir",
        default=os.environ.get("SM_MODEL_DIR", "./artifacts"),
        type=Path,
    )
    p.add_argument("--output-dir", default=os.environ.get("SM_OUTPUT_DATA_DIR", "./output"), type=Path)
    p.add_argument("--max-features", type=int, default=20000)
    p.add_argument("--ngram-max", type=int, default=2)
    p.add_argument("--C", type=float, default=1.0)
    p.add_argument("--sample-size", type=int, default=200_000)
    p.add_argument("--test-size", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model-version", default=os.environ.get("MODEL_VERSION", "v1"))
    p.add_argument(
        "--package-tar",
        action="store_true",
        help="Also emit model.tar.gz (SageMaker does this automatically when run as a job).",
    )
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    args.model_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    df = load_dataset(args.dataset, args.data_dir, args.sample_size, args.seed)
    df["text"] = df["text"].astype(str).map(clean_text)

    X_train, X_val, y_train, y_val = train_test_split(
        df["text"],
        df["label"].to_numpy(),
        test_size=args.test_size,
        random_state=args.seed,
        stratify=df["label"],
    )
    log.info("Train=%d  Val=%d", len(X_train), len(X_val))

    pipe = build_pipeline(args.max_features, args.ngram_max, args.C)
    pipe.fit(X_train, y_train)

    metrics = evaluate(pipe, X_val, y_val)
    train_secs = time.time() - t0
    metrics["train_seconds"] = round(train_secs, 2)
    metrics["model_version"] = args.model_version
    metrics["dataset"] = args.dataset
    metrics["hyperparameters"] = {
        "max_features": args.max_features,
        "ngram_range": [1, args.ngram_max],
        "C": args.C,
        "sample_size": args.sample_size,
    }

    log.info("Validation metrics: %s", json.dumps(metrics, indent=2))
    log.info(
        "Detailed report:\n%s",
        classification_report(y_val, pipe.predict(X_val), digits=4),
    )

    model_path = args.model_dir / "model.joblib"
    joblib.dump(pipe, model_path)
    (args.model_dir / "model_version.txt").write_text(args.model_version)
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (args.model_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    log.info("Wrote model to %s (%.1f KB)", model_path, model_path.stat().st_size / 1024)

    if args.package_tar:
        tar_path = args.output_dir / "model.tar.gz"
        with tarfile.open(tar_path, "w:gz") as tar:
            for f in args.model_dir.iterdir():
                tar.add(f, arcname=f.name)
        log.info("Packaged %s", tar_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
