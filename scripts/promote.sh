#!/usr/bin/env bash
# Promote staging → prod, gated on accuracy ≥ threshold (default 0.78).
#
# Workflow:
#   1. Resolve the staging stack's model artifact key.
#   2. Download model.tar.gz, extract, run model/evaluate.py with the
#      held-out evaluation CSV (path: $EVAL_CSV) — must produce
#      accuracy >= MIN_ACCURACY or the script exits non-zero.
#   3. Copy the validated artifact into the prod artifacts bucket.
#   4. Invoke scripts/deploy.sh prod.
#
# Required env / args:
#   AWS_REGION       — default us-east-1
#   EVAL_CSV         — path to labeled holdout (text,label)
#   MIN_ACCURACY     — default 0.78
#   PROD_ART_BUCKET  — prod artifacts bucket (overrides parameters/prod.json)
#
# Usage:
#   scripts/promote.sh
set -euo pipefail

REGION="${AWS_REGION:-us-east-1}"
PROJECT="sentiment-aws"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="$( cd "${SCRIPT_DIR}/.." && pwd )"
EVAL_CSV="${EVAL_CSV:-${ROOT_DIR}/tests/data/holdout.csv}"
MIN_ACCURACY="${MIN_ACCURACY:-0.78}"
WORK_DIR="$(mktemp -d)"
trap 'rm -rf "${WORK_DIR}"' EXIT

if [[ ! -f "${EVAL_CSV}" ]]; then
  echo "!! EVAL_CSV not found: ${EVAL_CSV}" >&2
  exit 1
fi

STAGING_STACK="${PROJECT}-staging"
PROD_PARAMS="${ROOT_DIR}/infra/stacks/parameters/prod.json"

STAGING_ART_BUCKET="$(jq -r '.[] | select(.ParameterKey=="ArtifactsBucket") | .ParameterValue' \
  "${ROOT_DIR}/infra/stacks/parameters/staging.json")"
STAGING_MODEL_KEY="$(jq -r '.[] | select(.ParameterKey=="ModelArtifactKey") | .ParameterValue' \
  "${ROOT_DIR}/infra/stacks/parameters/staging.json")"
PROD_ART_BUCKET_DEFAULT="$(jq -r '.[] | select(.ParameterKey=="ArtifactsBucket") | .ParameterValue' \
  "${PROD_PARAMS}")"
PROD_ART_BUCKET="${PROD_ART_BUCKET:-${PROD_ART_BUCKET_DEFAULT}}"

echo ">> downloading staging model artifact s3://${STAGING_ART_BUCKET}/${STAGING_MODEL_KEY}"
aws s3 cp "s3://${STAGING_ART_BUCKET}/${STAGING_MODEL_KEY}" "${WORK_DIR}/model.tar.gz" --region "${REGION}"

mkdir -p "${WORK_DIR}/model"
tar -xzf "${WORK_DIR}/model.tar.gz" -C "${WORK_DIR}/model"

echo ">> running accuracy gate (>= ${MIN_ACCURACY})"
python "${ROOT_DIR}/model/evaluate.py" \
  --model-path "${WORK_DIR}/model/model.joblib" \
  --eval-csv "${EVAL_CSV}" \
  --report-path "${WORK_DIR}/eval_report.json" \
  --min-accuracy "${MIN_ACCURACY}" \
  --fail-on-gate

ACCURACY="$(jq -r '.accuracy' "${WORK_DIR}/eval_report.json")"
echo ">> accuracy=${ACCURACY} (gate ${MIN_ACCURACY}) — PASSED"

echo ">> copying validated artifact into prod bucket: s3://${PROD_ART_BUCKET}/${STAGING_MODEL_KEY}"
aws s3 cp "${WORK_DIR}/model.tar.gz" "s3://${PROD_ART_BUCKET}/${STAGING_MODEL_KEY}" --region "${REGION}"
aws s3 cp "${WORK_DIR}/eval_report.json" \
  "s3://${PROD_ART_BUCKET}/eval-reports/$(date -u +%Y%m%dT%H%M%S)_accuracy_${ACCURACY}.json" --region "${REGION}"

echo ">> deploying prod stack"
"${SCRIPT_DIR}/deploy.sh" prod --artifacts-bucket "${PROD_ART_BUCKET}"
