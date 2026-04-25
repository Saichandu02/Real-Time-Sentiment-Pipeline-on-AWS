"""Lightweight CloudFormation sanity checks (in addition to `cfn-lint` in CI)."""
from __future__ import annotations

from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

ROOT = Path(__file__).resolve().parents[1]
INFRA = ROOT / "infra"


class _CfnLoader(yaml.SafeLoader):
    """SafeLoader that tolerates CloudFormation's `!Ref`, `!Sub`, `!GetAtt`, …"""


def _cfn_constructor(loader, tag_suffix, node):
    if isinstance(node, yaml.ScalarNode):
        return {f"Fn::{tag_suffix}": loader.construct_scalar(node)}
    if isinstance(node, yaml.SequenceNode):
        return {f"Fn::{tag_suffix}": loader.construct_sequence(node, deep=True)}
    if isinstance(node, yaml.MappingNode):
        return {f"Fn::{tag_suffix}": loader.construct_mapping(node, deep=True)}
    raise TypeError(f"unexpected node {type(node)!r} for tag {tag_suffix}")


_CfnLoader.add_multi_constructor("!", _cfn_constructor)


def _load(p: Path) -> dict:
    return yaml.load(p.read_text(), Loader=_CfnLoader)


def test_root_template_has_three_nested_stacks():
    tpl = _load(INFRA / "root.yaml")
    res = tpl["Resources"]
    assert {"StorageStack", "ComputeStack", "ObservabilityStack"}.issubset(res.keys())
    for k in ("StorageStack", "ComputeStack", "ObservabilityStack"):
        assert res[k]["Type"] == "AWS::CloudFormation::Stack"


def test_storage_emits_required_outputs():
    tpl = _load(INFRA / "stacks" / "storage.yaml")
    outputs = set(tpl["Outputs"].keys())
    required = {
        "InputBucketName", "AsyncInputBucketName", "AsyncOutputBucketName",
        "ResultsTableName", "RequestsTableName", "ResultsTableStreamArn",
    }
    assert required.issubset(outputs)


def test_compute_has_lambdas_and_endpoint():
    tpl = _load(INFRA / "stacks" / "compute.yaml")
    res = tpl["Resources"]
    assert "PreprocessLambda" in res
    assert "PostprocessLambda" in res
    assert "Endpoint" in res
    assert res["EndpointConfig"]["Properties"].get("AsyncInferenceConfig") is not None


def test_observability_has_three_topics_and_alarms():
    tpl = _load(INFRA / "stacks" / "observability.yaml")
    res = tpl["Resources"]
    assert {"TopicP1", "TopicP2", "TopicP3"}.issubset(res.keys())
    alarms = [k for k, v in res.items() if v.get("Type") == "AWS::CloudWatch::Alarm"]
    assert len(alarms) >= 4
    for a in alarms:
        desc = res[a]["Properties"].get("AlarmDescription", "")
        if isinstance(desc, dict):
            desc = next(iter(desc.values()))
        if isinstance(desc, list):
            desc = " ".join(str(x) for x in desc)
        assert "Runbook" in str(desc), f"alarm {a} missing runbook URL"


def test_per_env_param_files_match_root_parameters():
    root = _load(INFRA / "root.yaml")
    declared = set(root["Parameters"].keys())
    import json
    for env in ("dev", "staging", "prod"):
        data = json.loads((INFRA / "stacks" / "parameters" / f"{env}.json").read_text())
        provided = {p["ParameterKey"] for p in data}
        unknown = provided - declared
        assert not unknown, f"{env}.json has unknown keys: {unknown}"
