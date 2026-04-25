#!/usr/bin/env bash
# One-command teardown for a sentiment-aws environment.
#
# Empties Retain-policy buckets, removes SSM params, then deletes the
# CloudFormation stack. Will refuse to run on prod unless --force is set.
#
# Usage:
#   scripts/teardown.sh <dev|staging|prod> [--force] [--keep-buckets]
set -euo pipefail

ENV="${1:-}"
shift || true

if [[ -z "${ENV}" || ! "${ENV}" =~ ^(dev|staging|prod)$ ]]; then
  echo "Usage: $0 <dev|staging|prod> [--force] [--keep-buckets]" >&2
  exit 1
fi

FORCE=0
KEEP_BUCKETS=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --force)        FORCE=1 ;;
    --keep-buckets) KEEP_BUCKETS=1 ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
  shift
done

if [[ "${ENV}" == "prod" && "${FORCE}" -ne 1 ]]; then
  echo "!! prod teardown requires --force" >&2
  exit 1
fi

REGION="${AWS_REGION:-us-east-1}"
PROJECT="sentiment-aws"
STACK_NAME="${PROJECT}-${ENV}"

echo ">> Region=${REGION}  Stack=${STACK_NAME}"

empty_bucket() {
  local b="$1"
  aws s3api head-bucket --bucket "${b}" --region "${REGION}" 2>/dev/null || { echo "   (bucket ${b} not present)"; return 0; }
  echo "   emptying ${b}"
  aws s3 rm "s3://${b}" --recursive --region "${REGION}" >/dev/null || true
  aws s3api list-object-versions --bucket "${b}" --region "${REGION}" \
    --output=json --query='{Objects: Versions[].{Key:Key,VersionId:VersionId}}' 2>/dev/null \
    | jq '. | select(.Objects != null)' \
    | aws s3api delete-objects --bucket "${b}" --region "${REGION}" --delete file:///dev/stdin >/dev/null 2>&1 || true
  aws s3api list-object-versions --bucket "${b}" --region "${REGION}" \
    --output=json --query='{Objects: DeleteMarkers[].{Key:Key,VersionId:VersionId}}' 2>/dev/null \
    | jq '. | select(.Objects != null)' \
    | aws s3api delete-objects --bucket "${b}" --region "${REGION}" --delete file:///dev/stdin >/dev/null 2>&1 || true
}

if [[ "${KEEP_BUCKETS}" -ne 1 ]]; then
  for kind in input async-in async-out; do
    BUCKETS="$(aws s3api list-buckets --query "Buckets[?starts_with(Name, '${PROJECT}-${ENV}-${kind}-')].Name" --output text 2>/dev/null || true)"
    for b in ${BUCKETS}; do
      empty_bucket "${b}"
      aws s3api delete-bucket --bucket "${b}" --region "${REGION}" 2>/dev/null || true
    done
  done
fi

echo ">> deleting SSM parameters"
aws ssm delete-parameters --region "${REGION}" \
  --names "/sentiment/${ENV}/endpoint_name" "/sentiment/${ENV}/results_table" \
          "/sentiment/${ENV}/model_version" "/sentiment/${ENV}/input_bucket" \
          "/sentiment/${ENV}/accuracy_gate" 2>/dev/null || true

echo ">> deleting stack ${STACK_NAME}"
aws cloudformation delete-stack --region "${REGION}" --stack-name "${STACK_NAME}"
aws cloudformation wait stack-delete-complete --region "${REGION}" --stack-name "${STACK_NAME}" || {
  echo "!! stack-delete-complete wait failed; check the console." >&2
  exit 1
}
echo ">> teardown complete"
