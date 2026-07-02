from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import yaml


class GuardViolation(RuntimeError):
    pass


LOCKED_PATHS = (
    "data.train_start",
    "data.train_end",
    "data.valid_start",
    "data.valid_end",
    "data.test_start",
    "data.test_end",
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
    if group != changed_groups[0]:
        raise GuardViolation(f"hypothesis_group mismatch: {group!r} vs changed_groups={changed_groups!r}")
    if "position_rule" in changed_groups and group != "position_rule":
        raise GuardViolation("position_rule changes must be tested in hypothesis_group=position_rule only")


def validate_lgb_infrastructure_lock(skill_config: dict) -> None:
    if skill_config.get("infrastructure_locked") is not True:
        raise GuardViolation("LightGBM parameter injection infrastructure must be locked before strategy iteration")


def validate_read_contract(paths: Iterable[str | Path]) -> None:
    for raw_path in paths:
        text = str(raw_path)
        if any(pattern in text for pattern in PROHIBITED_READ_PATTERNS):
            raise GuardViolation(f"Codex read contract forbids heavy artifact: {text}")


def run_guard(skill_config: dict, runtime_config: dict | None = None, locked_config: dict | None = None) -> dict:
    validate_lgb_infrastructure_lock(skill_config)
    validate_hypothesis_scope(skill_config)
    validate_results_split(skill_config.get("current", {}))
    if runtime_config is not None:
        validate_locked_boundaries(runtime_config, locked_config or skill_config.get("locked_boundaries", {}))
    return {"passed": True, "violations": []}


def guard_result_or_errors(skill_config: dict, runtime_config: dict | None = None) -> dict:
    try:
        return run_guard(skill_config, runtime_config=runtime_config)
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
