# sentiment-aws

A fully serverless real-time sentiment classification pipeline on AWS,
provisioned via CloudFormation **nested stacks**, with a Streamlit
observability dashboard.

```text
[Client] → [S3 input] → [Lambda preprocess] → [SageMaker async endpoint]
                                                    │
                                                    ▼
                       [Lambda postprocess] → [DynamoDB] → [Streamlit dashboard]
                                  │
                                  ▼
                          [CloudWatch] → [SNS P1/P2/P3]
```

See `docs/architecture.md` for the sequence diagram and end-to-end latency
budget.

## Repo layout

```
sentiment-aws/
  model/                       # TF-IDF + LogReg training, evaluation, SageMaker entrypoint
  lambdas/
    preprocess/                # S3 ObjectCreated → SM async invoke
    postprocess/               # SM async response → DynamoDB + CloudWatch metrics
  infra/
    root.yaml                  # nested-stack root template
    stacks/
      storage.yaml             # S3 buckets + DynamoDB
      compute.yaml             # Lambdas + SM async endpoint + autoscaling
      observability.yaml       # SNS P1/P2/P3 + alarms + dashboard
      parameters/{dev,staging,prod}.json
  dashboard/streamlit_app.py   # observability dashboard
  scripts/
    deploy.sh                  # cfn deploy + smoke test gate
    promote.sh                 # staging → prod with accuracy gate
    latency_smoke_test.py      # 100-input p95/p99 enforcement
    teardown.sh                # one-command teardown
  tests/                       # pytest + moto-based unit tests
  .github/workflows/ci-cd.yml  # lint → tests → deploy dev → staging → prod
  docs/architecture.md
```

## Quickstart

### 1. Train the model

```bash
pip install -r model/requirements.txt
mkdir -p data
# place IMDB CSV (text,label) at data/imdb.csv  OR Sentiment140 file
python model/train.py --dataset imdb --data-dir ./data \
  --model-dir ./artifacts --output-dir ./artifacts --package-tar
```

This produces `artifacts/model.tar.gz` (along with `model.joblib`,
`model_version.txt`, and `metrics.json`).

### 2. Upload model + bootstrap deploy

Edit `infra/stacks/parameters/dev.json` to set:
- `ArtifactsBucket` — your bootstrap S3 bucket (created by `deploy.sh` if absent).
- `AlarmEmailP1`, `AlarmEmailP2`, optionally `AlarmEmailP3`.
- `RunbookBaseUrl` — your wiki / runbooks URL prefix.

Then upload the model to the artifacts bucket and run the deploy:

```bash
aws s3 cp artifacts/model.tar.gz s3://<ArtifactsBucket>/models/v1/model.tar.gz
bash scripts/deploy.sh dev
```

`deploy.sh`:
1. Packages `lambdas/preprocess` and `lambdas/postprocess` into zips.
2. Uploads the nested templates and Lambda zips to the artifacts bucket.
3. Runs `aws cloudformation deploy` against `infra/root.yaml` with the
   per-env parameter overrides.
4. Runs `scripts/latency_smoke_test.py` and **fails the deploy** if
   `p95 > 800ms` or `p99 > 1500ms`.

### 3. Run the Streamlit dashboard

```bash
pip install -r dashboard/requirements.txt
ENV=dev RESULTS_TABLE=$(aws ssm get-parameter --name /sentiment/dev/results_table \
  --query Parameter.Value --output text) \
  streamlit run dashboard/streamlit_app.py
```

### 4. Tests

```bash
pip install -r tests/requirements.txt
pytest -q tests
```

### 5. Promote to prod

```bash
EVAL_CSV=tests/data/holdout.csv MIN_ACCURACY=0.78 bash scripts/promote.sh
```

`promote.sh` downloads the staging artifact, re-runs `model/evaluate.py`
with `--fail-on-gate`, copies the validated artifact to the prod bucket,
then invokes `deploy.sh prod`.

### 6. Teardown

```bash
bash scripts/teardown.sh dev
bash scripts/teardown.sh prod --force
```

## CI/CD

`.github/workflows/ci-cd.yml`:
1. `lint` — `ruff` + `cfn-lint` on the root and nested templates.
2. `unit-tests` — `pytest` over model, lambdas, dashboard PSI, and CFN
   sanity tests.
3. `deploy-dev` (push to `main`) — `deploy.sh dev` + integration smoke.
4. `deploy-staging` — `deploy.sh staging` + accuracy-gate evaluation.
5. `deploy-prod` — protected GitHub environment (manual approval) →
   `promote.sh` (re-runs the accuracy gate before promoting).

GitHub OIDC is used for AWS auth; provide:
- `AWS_DEPLOY_ROLE_DEV`, `AWS_DEPLOY_ROLE_STAGING`, `AWS_DEPLOY_ROLE_PROD`
- `AWS_STAGING_ARTIFACTS_BUCKET`

## Observability

- **CloudWatch Dashboard**: created in-stack, named `sentiment-aws-<env>`.
- **Alarms (severity-tiered)**:

  | Severity | Trigger                                                   |
  |----------|-----------------------------------------------------------|
  | P1       | endpoint 5xx > 5% for 2min, or DDB throttles > 10/min     |
  | P2       | p99 e2e > 1500ms for 5min, or rolling accuracy < 0.78     |
  | P3       | rolling-window accuracy delta > 3pp (drift)               |

  All alarms carry `Severity`, `Service`, `Env` tags and a runbook URL in
  the description.

- **Streamlit dashboard panels**:
  - Rolling sentiment trend (1h / 6h / 24h, configurable).
  - Prediction-distribution histogram + KPIs (p95/p99 latency, mean confidence).
  - Drift panel (PSI of last N predictions vs training distribution).
  - Model-version comparison table.
  - Live alerts feed from `DescribeAlarmHistory`.
