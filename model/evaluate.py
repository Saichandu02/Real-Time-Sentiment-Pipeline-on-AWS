"""Offline / promotion-gate evaluation.

Loads a serialized scikit-learn pipeline (``model.joblib``) plus a labeled
held-out CSV (``text,label``) and emits a JSON report. The accompanying
``scripts/promote.sh`` reads the JSON to gate staging→prod promotion on a
configurable accuracy floor (default 0.78).

CLI::

    python model/evaluate.py \\
        --model-path artifacts/model.joblib \\
        --eval-csv data/holdout.csv \\
        --report-path artifacts/eval_report.json \\
        --min-accuracy 0.78
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

from train import clean_text  # type: ignore  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("sentiment.evaluate")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", type=Path, required=True)
    p.add_argument("--eval-csv", type=Path, required=True)
    p.add_argument("--report-path", type=Path, default=Path("eval_report.json"))
    p.add_argument("--min-accuracy", type=float, default=0.78)
    p.add_argument("--text-col", default="text")
    p.add_argument("--label-col", default="label")
    p.add_argument(
        "--fail-on-gate",
        action="store_true",
        help="Exit non-zero if accuracy < --min-accuracy (used by promote.sh).",
    )
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    pipe = joblib.load(args.model_path)
    df = pd.read_csv(args.eval_csv)
    if args.text_col not in df.columns or args.label_col not in df.columns:
        raise SystemExit(
            f"eval csv must contain columns {args.text_col!r} and {args.label_col!r}; "
            f"got {list(df.columns)}"
        )

    df[args.text_col] = df[args.text_col].astype(str).map(clean_text)
    X = df[args.text_col]
    y = df[args.label_col].astype(int).to_numpy()

    proba = pipe.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)

    cm = confusion_matrix(y, pred).tolist()
    report = {
        "accuracy": float(accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred)),
        "roc_auc": float(roc_auc_score(y, proba)) if len(np.unique(y)) > 1 else None,
        "support": int(len(y)),
        "min_accuracy_gate": args.min_accuracy,
        "passed_gate": float(accuracy_score(y, pred)) >= args.min_accuracy,
        "confusion_matrix": cm,
        "classification_report": classification_report(y, pred, output_dict=True, digits=4),
    }

    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text(json.dumps(report, indent=2))

    log.info("Report written to %s", args.report_path)
    log.info(
        "accuracy=%.4f  f1=%.4f  gate=%.2f  passed=%s",
        report["accuracy"],
        report["f1"],
        args.min_accuracy,
        report["passed_gate"],
    )

    if args.fail_on_gate and not report["passed_gate"]:
        log.error("Promotion gate FAILED (accuracy %.4f < %.2f)", report["accuracy"], args.min_accuracy)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
