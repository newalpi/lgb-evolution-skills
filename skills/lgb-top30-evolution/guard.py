from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import yaml


class GuardViolation(RuntimeError):
    pass


SKILL_DIR = Path(__file__).resolve().parent

ALLOWED_GROUPS = {
    "baseline",
    "lgb_params",
    "factor_subset",
    "factor_processing",
    "portfolio_rule",
    "exposure_constraint",
    "position_rule",
}

GROUP_ALLOWED_PATHS = {
    "baseline": (),
    "lgb_params": ("experiment.model.lgb_params_overlay",),
    "factor_subset": ("experiment.features.include", "experiment.features.exclude"),
    "factor_processing": ("experiment.features.processing",),
    "portfolio_rule": (
        "experiment.portfolio.topk",
        "experiment.portfolio.n_drop",
        "experiment.portfolio.weight_method",
        "experiment.portfolio.mvo_optimizer",
    ),
    "exposure_constraint": ("experiment.exposure_constraints",),
    "position_rule": ("experiment.position_rule",),
}

DIFF_IGNORED_PREFIXES = (
    "accepted_version",
    "current",
    "paths",
    "data_snapshot",
    "artifact_paths",
    "log_tail_path",
    "baseline_metrics",
    "candidate_metrics",
)

LOCKED_PATHS = (
    "rolling_model_start",
    "rolling_model_end",
    "oot_validation_start",
    "oot_validation_end",
    "test_review_start",
    "test_review_end",
    "data.label.horizon",
    "data.label.type",
    "model.rolling.purge_days",
    "backtest.strategy.topk",
    "backtest.strategy.rebalance_freq",
    "backtest.costs.commission_rate",
    "backtest.costs.slippage",
    "backtest.costs.stamp_tax",
    "features.base_features.custom.processing.filter_st",
    "features.base_features.custom.processing.filter_limit",
    "features.base_features.custom.processing.filter_new_share",
    "risk_control.filters.filter_st",
    "risk_control.filters.filter_limit",
    "score_neutralization.enabled",
    "score_neutralization.mode",
)

PROHIBITED_READ_PATTERNS = (
    "base_scores.parquet",
    "raw_",
    "final_",
    "holdings_full.csv",
    "portfolio_positions.csv",
    "period_returns.csv",
    "equity_curve.csv",
    "rank_bucket_period_returns.csv",
)


def load_yaml(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _get(config: dict, dotted: str) -> Any:
    value: Any = config
    for part in dotted.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            out.update(_flatten(child, child_prefix))
        return out
    return {prefix: value}


def _is_ignored_path(path: str) -> bool:
    return any(path == prefix or path.startswith(prefix + ".") for prefix in DIFF_IGNORED_PREFIXES)


def _is_allowed_path(path: str, allowed_prefixes: Iterable[str]) -> bool:
    return any(path == prefix or path.startswith(prefix + ".") for prefix in allowed_prefixes)


def _resolve_skill_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else SKILL_DIR / path


def _load_accepted_config(skill_config: dict) -> dict | None:
    explicit = skill_config.get("paths", {}).get("accepted_config")
    if explicit:
        path = _resolve_skill_path(explicit)
    else:
        versions_dir = _resolve_skill_path(skill_config.get("paths", {}).get("versions_dir", "versions"))
        path = versions_dir / "accepted" / "skill_config.yaml"
    return load_yaml(path) if path.exists() else None


def validate_locked_boundaries(candidate_config: dict, locked_config: dict) -> None:
    for dotted in LOCKED_PATHS:
        cand = _get(candidate_config, dotted)
        locked = _get(locked_config, dotted)
        if cand != locked:
            raise GuardViolation(f"locked boundary changed: {dotted} candidate={cand!r} locked={locked!r}")


def validate_results_split(context: dict) -> None:
    split = str(context.get("split", "validation")).lower()
    if split == "test" and not context.get("manual_review", False):
        raise GuardViolation("test split cannot enter automatic results.tsv or baseline decisions")
    if split not in {"validation", "test"}:
        raise GuardViolation(f"unsupported split: {split}")


def validate_hypothesis_scope(skill_config: dict) -> None:
    current = skill_config.get("current", {})
    group = current.get("hypothesis_group")
    changed_groups = current.get("changed_groups") or []
    if isinstance(changed_groups, str):
        changed_groups = [changed_groups]
    if len(changed_groups) != 1:
        raise GuardViolation("exactly one hypothesis_group must change per iteration")
    if group not in ALLOWED_GROUPS:
        raise GuardViolation(f"unsupported hypothesis_group: {group!r}")
    if group != changed_groups[0]:
        raise GuardViolation(f"hypothesis_group mismatch: {group!r} vs changed_groups={changed_groups!r}")
    if "position_rule" in changed_groups and group != "position_rule":
        raise GuardViolation("position_rule changes must be tested in hypothesis_group=position_rule only")


def validate_hypothesis_diff(skill_config: dict, accepted_config: dict | None = None) -> None:
    current = skill_config.get("current", {})
    group = current.get("hypothesis_group")
    if group == "baseline":
        return
    accepted = accepted_config if accepted_config is not None else _load_accepted_config(skill_config)
    if accepted is None:
        raise GuardViolation("accepted config is required for non-baseline hypothesis diff validation")

    allowed = GROUP_ALLOWED_PATHS.get(str(group), ())
    left = {k: v for k, v in _flatten(accepted).items() if not _is_ignored_path(k)}
    right = {k: v for k, v in _flatten(skill_config).items() if not _is_ignored_path(k)}
    changed = sorted(path for path in set(left) | set(right) if left.get(path) != right.get(path))
    outside = [path for path in changed if not _is_allowed_path(path, allowed)]
    if outside:
        raise GuardViolation(f"changed paths outside hypothesis_group={group}: {outside}")


def validate_lgb_infrastructure_lock(skill_config: dict) -> None:
    if skill_config.get("infrastructure_locked") is not True:
        raise GuardViolation("LightGBM parameter injection infrastructure must be locked before strategy iteration")


def validate_read_contract(paths: Iterable[str | Path]) -> None:
    for raw_path in paths:
        text = str(raw_path)
        if any(pattern in text for pattern in PROHIBITED_READ_PATTERNS):
            raise GuardViolation(f"Codex read contract forbids heavy artifact: {text}")


def _contract_paths_from_config(skill_config: dict) -> list[str]:
    paths: list[str] = []
    artifact_paths = skill_config.get("artifact_paths", {})
    if isinstance(artifact_paths, dict):
        paths.extend(str(path) for path in artifact_paths.values() if path)
    if skill_config.get("log_tail_path"):
        paths.append(str(skill_config["log_tail_path"]))
    return paths


def run_guard(
    skill_config: dict,
    runtime_config: dict | None = None,
    locked_config: dict | None = None,
    run_context: dict | None = None,
    accepted_config: dict | None = None,
) -> dict:
    validate_lgb_infrastructure_lock(skill_config)
    validate_hypothesis_scope(skill_config)
    validate_results_split(run_context or skill_config.get("current", {}))
    validate_hypothesis_diff(skill_config, accepted_config=accepted_config)
    validate_read_contract(_contract_paths_from_config(skill_config))
    if runtime_config is not None:
        validate_locked_boundaries(runtime_config, locked_config or skill_config.get("locked_boundaries", {}))
    return {"passed": True, "violations": []}


def guard_result_or_errors(
    skill_config: dict,
    runtime_config: dict | None = None,
    run_context: dict | None = None,
    accepted_config: dict | None = None,
) -> dict:
    try:
        return run_guard(
            skill_config,
            runtime_config=runtime_config,
            run_context=run_context,
            accepted_config=accepted_config,
        )
    except GuardViolation as exc:
        return {"passed": False, "violations": [str(exc)]}


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate LGB Top30 evolution guard rails.")
    parser.add_argument("--skill-config", default="skill_config.yaml")
    parser.add_argument("--runtime-config", default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    skill_config = load_yaml(args.skill_config)
    runtime_config = load_yaml(args.runtime_config) if args.runtime_config else None
    result = guard_result_or_errors(skill_config, runtime_config=runtime_config)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print("PASS" if result["passed"] else "FAIL")
        for violation in result.get("violations", []):
            print(violation)
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
