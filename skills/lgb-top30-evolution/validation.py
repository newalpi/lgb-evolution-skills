from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from guard import guard_result_or_errors, load_yaml


RESULTS_FIELDS = [
    "version",
    "parent_version",
    "status",
    "hypothesis_group",
    "hypothesis",
    "score_delta",
    "decision",
    "config_hash",
    "code_hash",
    "data_snapshot",
    "summary_path",
    "timestamp",
]


def _float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, str):
        value = value.strip()
        if value.endswith("%"):
            return float(value[:-1]) / 100.0
    return float(value)


def evaluate_candidate(candidate: dict, baseline: dict) -> dict:
    d_rankic = _float(candidate.get("mean_rank_ic")) - _float(baseline.get("mean_rank_ic"))
    d_top30_sharpe = _float(candidate.get("top30_sharpe")) - _float(baseline.get("top30_sharpe"))
    d_mdd = _float(candidate.get("max_drawdown")) - _float(baseline.get("max_drawdown"))
    d_turnover = _float(candidate.get("avg_turnover")) - _float(baseline.get("avg_turnover"))
    d_industry_excess = _float(candidate.get("industry_top5_excess")) - _float(baseline.get("industry_top5_excess"))

    score_delta = (
        0.35 * (d_rankic / 0.01)
        + 0.30 * (d_top30_sharpe / 0.25)
        - 0.15 * (d_mdd / 0.05)
        - 0.10 * (d_turnover / 0.10)
        - 0.10 * (d_industry_excess / 0.10)
    )

    failed: list[str] = []
    base_turnover = _float(baseline.get("avg_turnover"))
    if d_rankic < -0.002:
        failed.append("d_rankic")
    if d_top30_sharpe < -0.05:
        failed.append("d_top30_sharpe")
    if d_mdd > 0.02:
        failed.append("d_mdd")
    if d_turnover > max(0.02, 0.15 * base_turnover):
        failed.append("d_turnover")
    if d_industry_excess > 0.03:
        failed.append("d_industry_excess")

    hard_gates = {"passed": not failed, "failed": failed}
    decision = "keep" if score_delta >= 0.05 and hard_gates["passed"] else "rollback"
    return {
        "score_delta": score_delta,
        "decision": decision,
        "deltas": {
            "d_rankic": d_rankic,
            "d_top30_sharpe": d_top30_sharpe,
            "d_mdd": d_mdd,
            "d_turnover": d_turnover,
            "d_industry_excess": d_industry_excess,
        },
        "hard_gates": hard_gates,
    }


def compute_industry_exposure_metrics(positions: pd.DataFrame, candidate_pool: pd.DataFrame) -> dict:
    if positions.empty:
        return {"industry_top5_weight_raw": 0.0, "industry_top5_excess": 0.0}

    pos = positions.copy()
    pool = candidate_pool.copy()
    pos["rebalance_date"] = pd.to_datetime(pos["rebalance_date"])
    pool["rebalance_date"] = pd.to_datetime(pool["rebalance_date"])
    weight_col = "effective_weight" if "effective_weight" in pos.columns else "weight"
    pos = pos[pos[weight_col] > 0].copy()
    if pos.empty:
        return {"industry_top5_weight_raw": 0.0, "industry_top5_excess": 0.0}

    raw_values: list[float] = []
    excess_values: list[float] = []

    for date, day_pos in pos.groupby("rebalance_date"):
        weight_sum = float(day_pos[weight_col].sum())
        if weight_sum <= 0:
            continue
        port_ind = (day_pos.groupby("industry")[weight_col].sum() / weight_sum).sort_values(ascending=False)
        top_industries = port_ind.head(5).index
        raw_values.append(float(port_ind.reindex(top_industries).sum()))

        day_pool = pool[pool["rebalance_date"] == date]
        if day_pool.empty:
            pool_ind = pd.Series(dtype=float)
        elif "candidate_weight" in day_pool.columns:
            pool_weight_sum = float(day_pool["candidate_weight"].sum())
            pool_ind = (
                day_pool.groupby("industry")["candidate_weight"].sum() / pool_weight_sum
                if pool_weight_sum > 0
                else pd.Series(dtype=float)
            )
        else:
            pool_ind = day_pool["industry"].value_counts(normalize=True)

        excess = 0.0
        for industry, port_weight in port_ind.reindex(top_industries).items():
            excess += max(float(port_weight) - float(pool_ind.get(industry, 0.0)), 0.0)
        excess_values.append(excess)

    return {
        "industry_top5_weight_raw": float(pd.Series(raw_values).mean()) if raw_values else 0.0,
        "industry_top5_excess": float(pd.Series(excess_values).mean()) if excess_values else 0.0,
    }


def stable_hash(obj: Any) -> str:
    payload = json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def code_hash(paths: list[str | Path] | None = None) -> str:
    if paths:
        digest = hashlib.sha256()
        for path in sorted(Path(p) for p in paths if Path(p).exists()):
            digest.update(str(path).encode("utf-8"))
            digest.update(path.read_bytes())
        return digest.hexdigest()[:16]
    try:
        tree = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        diff = subprocess.check_output(["git", "diff", "--", "workflow.py", "scripts/pipeline/train_rolling.py", "utils/lightgbm_params.py", "skills/lgb-top30-evolution"], text=True)
        return hashlib.sha256((tree + "\n" + diff).encode("utf-8")).hexdigest()[:16]
    except Exception:
        return ""


def build_summary(skill_config: dict, guard_result: dict | None = None) -> dict:
    current = skill_config.get("current", {})
    baseline = skill_config.get("baseline_metrics", {})
    candidate = skill_config.get("candidate_metrics", {})
    rule = evaluate_candidate(candidate, baseline)
    summary_path = skill_config.get("paths", {}).get("latest_summary", "latest_summary.json")
    summary = {
        "version": current.get("version"),
        "parent_version": current.get("parent_version"),
        "hypothesis_group": current.get("hypothesis_group"),
        "hypothesis": current.get("hypothesis"),
        "decision_inputs": {
            "baseline": baseline,
            "candidate": candidate,
            "deltas": rule["deltas"],
        },
        "rule_result": {
            "score_delta": rule["score_delta"],
            "decision": rule["decision"],
            "hard_gates": rule["hard_gates"],
        },
        "guard_result": guard_result if guard_result is not None else guard_result_or_errors(skill_config),
        "decision": rule["decision"],
        "config_hash": stable_hash(skill_config),
        "code_hash": code_hash(),
        "data_snapshot": skill_config.get("data_snapshot", {}),
        "artifact_paths": skill_config.get("artifact_paths", {}),
        "log_tail_path": skill_config.get("log_tail_path"),
        "status": "evaluated",
        "split": current.get("split", "validation"),
        "summary_path": summary_path,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if not summary["guard_result"].get("passed", False):
        summary["decision"] = "rollback"
        summary["rule_result"]["decision"] = "rollback"
    return summary


def append_results_tsv(results_path: str | Path, summary: dict) -> None:
    if str(summary.get("split", "validation")).lower() != "validation":
        raise ValueError("test review summaries must not be appended to automatic results.tsv")

    path = Path(results_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    row = {
        "version": summary.get("version"),
        "parent_version": summary.get("parent_version"),
        "status": summary.get("status", "evaluated"),
        "hypothesis_group": summary.get("hypothesis_group"),
        "hypothesis": summary.get("hypothesis"),
        "score_delta": f"{_float(summary.get('score_delta', summary.get('rule_result', {}).get('score_delta', 0.0))):.10f}",
        "decision": summary.get("decision"),
        "config_hash": summary.get("config_hash"),
        "code_hash": summary.get("code_hash"),
        "data_snapshot": json.dumps(summary.get("data_snapshot", {}), ensure_ascii=False, sort_keys=True),
        "summary_path": summary.get("summary_path"),
        "timestamp": summary.get("timestamp"),
    }
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULTS_FIELDS, delimiter="\t")
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def write_summary(path: str | Path, summary: dict) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build LGB Top30 validation summary.")
    parser.add_argument("--skill-config", default="skill_config.yaml")
    parser.add_argument("--latest-summary", default=None)
    parser.add_argument("--results", default=None)
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--append-results", action="store_true")
    args = parser.parse_args()

    skill_config = load_yaml(args.skill_config)
    guard_result = guard_result_or_errors(skill_config)
    summary = build_summary(skill_config, guard_result=guard_result)

    latest_summary = Path(args.latest_summary or skill_config.get("paths", {}).get("latest_summary", "latest_summary.json"))
    results_path = Path(args.results or skill_config.get("paths", {}).get("results_tsv", "results.tsv"))
    write_summary(latest_summary, summary)
    if args.append_results and not args.summary_only:
        append_results_tsv(results_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if guard_result.get("passed", False) else 2


if __name__ == "__main__":
    raise SystemExit(main())
