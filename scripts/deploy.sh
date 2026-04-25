#!/usr/bin/env bash
# Deploy the sentiment-aws nested stack for a given environment.
#
# Steps:
#   1. Package preprocess + postprocess Lambda zips.
#   2. Upload nested templates + Lambda zips to the artifacts bucket.
#   3. Run `aws cloudformation deploy` against infra/root.yaml with the
#      per-env parameter overrides from infra/stacks/parameters/<env>.json.
#   4. Run scripts/latency_smoke_test.py and FAIL THE DEPLOY if SLOs are
#      breached (p95 ≤ 800ms / p99 ≤ 1500ms by default).
#
# Usage:
#   scripts/deploy.sh <env> [--skip-smoke] [--artifacts-bucket NAME]
#
# Required env / args:
#   AWS_REGION  (default us-east-1)
#   AWS_PROFILE (optional)
set -euo pipefail

ENV="${1:-}"
shift || true

if [[ -z "${ENV}" || ! "${ENV}" =~ ^(dev|staging|prod)$ ]]; then
  echo "Usage: $0 <dev|staging|prod> [--skip-smoke] [--artifacts-bucket NAME]" >&2
  exit 1
fi

SKIP_SMOKE=0
ART_BUCKET_OVERRIDE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-smoke)        SKIP_SMOKE=1 ;;
    --artifacts-bucket)  shift; ART_BUCKET_OVERRIDE="${1}" ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
  shift
done

REGION="${AWS_REGION:-us-east-1}"
PROJECT="sentiment-aws"
STACK_NAME="${PROJECT}-${ENV}"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="$( cd "${SCRIPT_DIR}/.." && pwd )"
PARAMS_FILE="${ROOT_DIR}/infra/stacks/parameters/${ENV}.json"
ROOT_TEMPLATE="${ROOT_DIR}/infra/root.yaml"
STACKS_DIR="${ROOT_DIR}/infra/stacks"
BUILD_DIR="${ROOT_DIR}/.build/${ENV}"

echo ">> Region=${REGION}  Env=${ENV}  Stack=${STACK_NAME}"

ART_BUCKET="${ART_BUCKET_OVERRIDE}"
if [[ -z "${ART_BUCKET}" ]]; then
  ART_BUCKET="$(jq -r '.[] | select(.ParameterKey=="ArtifactsBucket") | .ParameterValue' "${PARAMS_FILE}")"
fi
if [[ -z "${ART_BUCKET}" || "${ART_BUCKET}" == REPLACE-ME-* ]]; then
  echo "!! ArtifactsBucket is unset (got '${ART_BUCKET}'). Override with --artifacts-bucket or update ${PARAMS_FILE}." >&2
  exit 1
fi
echo ">> Artifacts bucket: ${ART_BUCKET}"

aws s3api head-bucket --bucket "${ART_BUCKET}" --region "${REGION}" 2>/dev/null || {
  echo ">> creating artifacts bucket"
  aws s3api create-bucket --bucket "${ART_BUCKET}" --region "${REGION}" \
    $( [[ "${REGION}" != "us-east-1" ]] && echo "--create-bucket-configuration LocationConstraint=${REGION}" )
  aws s3api put-bucket-versioning --bucket "${ART_BUCKET}" \
    --versioning-configuration Status=Enabled
  aws s3api put-bucket-encryption --bucket "${ART_BUCKET}" \
    --server-side-encryption-configuration '{"Rules":[{"ApplyServerSideEncryptionByDefault":{"SSEAlgorithm":"AES256"}}]}'
}

mkdir -p "${BUILD_DIR}"

package_lambda() {
  local name="$1"
  local src="${ROOT_DIR}/lambdas/${name}"
  local zip="${BUILD_DIR}/${name}.zip"
  echo ">> packaging lambda: ${name}"
  rm -f "${zip}"
  ( cd "${src}" && zip -qr "${zip}" handler.py )
  echo "${zip}"
}

PREPROCESS_ZIP="$(package_lambda preprocess)"
POSTPROCESS_ZIP="$(package_lambda postprocess)"

echo ">> uploading templates + lambdas to s3://${ART_BUCKET}/"
aws s3 cp "${STACKS_DIR}/storage.yaml"        "s3://${ART_BUCKET}/templates/storage.yaml"        --region "${REGION}"
aws s3 cp "${STACKS_DIR}/compute.yaml"        "s3://${ART_BUCKET}/templates/compute.yaml"        --region "${REGION}"
aws s3 cp "${STACKS_DIR}/observability.yaml"  "s3://${ART_BUCKET}/templates/observability.yaml"  --region "${REGION}"
aws s3 cp "${PREPROCESS_ZIP}"                 "s3://${ART_BUCKET}/lambdas/preprocess.zip"        --region "${REGION}"
aws s3 cp "${POSTPROCESS_ZIP}"                "s3://${ART_BUCKET}/lambdas/postprocess.zip"       --region "${REGION}"

# Build flat key=value overrides for `aws cloudformation deploy`
PARAM_OVERRIDES=()
while IFS= read -r line; do
  k="$(jq -r '.ParameterKey' <<<"$line")"
  v="$(jq -r '.ParameterValue' <<<"$line")"
  [[ "$k" == "ArtifactsBucket" ]] && v="${ART_BUCKET}"
  PARAM_OVERRIDES+=("${k}=${v}")
done < <(jq -c '.[]' "${PARAMS_FILE}")

echo ">> aws cloudformation deploy"
aws cloudformation deploy \
  --region "${REGION}" \
  --stack-name "${STACK_NAME}" \
  --template-file "${ROOT_TEMPLATE}" \
  --capabilities CAPABILITY_NAMED_IAM CAPABILITY_AUTO_EXPAND \
  --parameter-overrides "${PARAM_OVERRIDES[@]}" \
  --tags Project="${PROJECT}" Env="${ENV}" ManagedBy=cloudformation \
  --no-fail-on-empty-changeset

INPUT_BUCKET="$(aws cloudformation describe-stacks --region "${REGION}" --stack-name "${STACK_NAME}" \
  --query "Stacks[0].Outputs[?OutputKey=='InputBucketName'].OutputValue" --output text)"
RESULTS_TABLE="$(aws cloudformation describe-stacks --region "${REGION}" --stack-name "${STACK_NAME}" \
  --query "Stacks[0].Outputs[?OutputKey=='ResultsTableName'].OutputValue" --output text)"

echo ">> deploy complete"
echo "   InputBucket  : ${INPUT_BUCKET}"
echo "   ResultsTable : ${RESULTS_TABLE}"

if [[ "${SKIP_SMOKE}" -eq 1 ]]; then
  echo ">> --skip-smoke set, skipping latency smoke test"
  exit 0
fi

echo ">> running latency smoke test (n=100)"
python "${ROOT_DIR}/scripts/latency_smoke_test.py" \
  --env "${ENV}" \
  --region "${REGION}" \
  --input-bucket "${INPUT_BUCKET}" \
  --results-table "${RESULTS_TABLE}" \
  --n 100 \
  --p95-budget-ms 800 \
  --p99-budget-ms 1500
