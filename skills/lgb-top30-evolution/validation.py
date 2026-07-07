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

from guard import GuardViolation, guard_result_or_errors, load_yaml, validate_read_contract


SKILL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SKILL_DIR.parents[1]

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
    "oot_validation_start",
    "oot_validation_end",
    "summary_path",
    "timestamp",
]

METRIC_KEYS = ("mean_rank_ic", "top30_sharpe", "annual_return", "sharpe_ratio", "max_drawdown")

DEFAULT_HASH_PATHS = [
    "scripts/oot_validation.py",
    "configs/base.yaml",
    "configs/features.yaml",
    "configs/model_lgb.yaml",
    "configs/strategy.yaml",
    "workflow.py",
    "scripts/pipeline/train_rolling.py",
    "utils/lightgbm_params.py",
    "skills/lgb-top30-evolution",
]


def _float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, str):
        value = value.strip()
        if value.endswith("%"):
            return float(value[:-1]) / 100.0
    return float(value)


def _metric_float(value: Any, default: float = 0.0) -> float:
    try:
        return _float(value, default=default)
    except (TypeError, ValueError):
        return default


def _drawdown_float(value: Any, default: float = 0.0) -> float:
    return abs(_metric_float(value, default=default))


def _resolve_skill_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else SKILL_DIR / path


def _resolve_latest_oot_run(oot_path: str | Path) -> Path:
    base = Path(oot_path)
    if not base.is_absolute():
        base = _resolve_skill_path(base)
    if (base / "performance_summary.yaml").exists():
        return base

    candidates = [p.parent for p in base.glob("*/performance_summary.yaml")]
    if not candidates:
        raise FileNotFoundError(f"未找到 OOT performance_summary.yaml: {base}")
    return max(candidates, key=lambda p: (p / "performance_summary.yaml").stat().st_mtime)


def _read_yaml_dict(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML 摘要必须是 dict: {path}")
    return data


def _mean_csv_column(path: Path, column: str) -> float | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if column not in df.columns:
        return None
    values = pd.to_numeric(df[column], errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.mean())


def _first_summary_metric(summary: dict, keys: list[str], default: float = 0.0) -> float:
    for key in keys:
        if key in summary:
            return _metric_float(summary.get(key), default=default)
    return default


def _first_summary_drawdown(summary: dict, keys: list[str], default: float = 0.0) -> float:
    for key in keys:
        if key in summary:
            return _drawdown_float(summary.get(key), default=default)
    return default


def evaluate_candidate(candidate: dict, baseline: dict, keep_threshold: float = 0.05) -> dict:
    d_rankic = _float(candidate.get("mean_rank_ic")) - _float(baseline.get("mean_rank_ic"))
    d_top30_sharpe = _float(candidate.get("top30_sharpe")) - _float(baseline.get("top30_sharpe"))
    d_annual_return = _float(candidate.get("annual_return")) - _float(baseline.get("annual_return"))
    d_sharpe_ratio = _float(candidate.get("sharpe_ratio")) - _float(baseline.get("sharpe_ratio"))
    d_mdd = _drawdown_float(candidate.get("max_drawdown")) - _drawdown_float(baseline.get("max_drawdown"))

    score_delta = (
        0.30 * (d_rankic / 0.01)
        + 0.20 * (d_top30_sharpe / 0.25)
        + 0.20 * (d_annual_return / 0.10)
        + 0.20 * (d_sharpe_ratio / 0.25)
        - 0.10 * (d_mdd / 0.05)
    )

    failed: list[str] = []
    if d_rankic < -0.002:
        failed.append("d_rankic")
    if d_top30_sharpe < -0.05:
        failed.append("d_top30_sharpe")
    if d_mdd > 0.02:
        failed.append("d_mdd")

    hard_gates = {"passed": not failed, "failed": failed}
    decision = "keep" if score_delta >= keep_threshold and hard_gates["passed"] else "rollback"
    return {
        "score_delta": score_delta,
        "keep_threshold": keep_threshold,
        "decision": decision,
        "deltas": {
            "d_rankic": d_rankic,
            "d_top30_sharpe": d_top30_sharpe,
            "d_annual_return": d_annual_return,
            "d_sharpe_ratio": d_sharpe_ratio,
            "d_mdd": d_mdd,
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


def _extract_industry_metrics(run_dir: Path) -> tuple[dict, bool]:
    positions_path = run_dir / "portfolio_positions.csv"
    candidate_pool_path = run_dir / "candidate_pool.csv"
    if not positions_path.exists() or not candidate_pool_path.exists():
        return {"industry_top5_weight_raw": 0.0, "industry_top5_excess": 0.0}, False

    positions = pd.read_csv(positions_path)
    candidate_pool = pd.read_csv(candidate_pool_path)
    if not {"rebalance_date", "industry"}.issubset(positions.columns):
        return {"industry_top5_weight_raw": 0.0, "industry_top5_excess": 0.0}, False
    if not {"rebalance_date", "industry"}.issubset(candidate_pool.columns):
        return {"industry_top5_weight_raw": 0.0, "industry_top5_excess": 0.0}, False
    if positions.columns.intersection({"effective_weight", "weight"}).empty:
        return {"industry_top5_weight_raw": 0.0, "industry_top5_excess": 0.0}, False

    return compute_industry_exposure_metrics(positions, candidate_pool), True


def extract_oot_metrics(oot_path: str | Path, output_path: str | Path | None = None) -> dict:
    """从 OOT 产物目录提取小型决策指标。"""
    run_dir = _resolve_latest_oot_run(oot_path)
    summary_path = run_dir / "performance_summary.yaml"
    period_path = run_dir / "period_returns.csv"
    daily_rank_ic_path = run_dir / "daily_rank_ic.csv"
    summary = _read_yaml_dict(summary_path)

    mean_rank_ic = _first_summary_metric(summary, ["mean_rank_ic", "daily_mean_rank_ic"])
    if mean_rank_ic == 0.0:
        csv_rank_ic = _mean_csv_column(daily_rank_ic_path, "rank_ic")
        if csv_rank_ic is None:
            csv_rank_ic = _mean_csv_column(period_path, "rank_ic")
        if csv_rank_ic is not None:
            mean_rank_ic = csv_rank_ic

    avg_turnover = _first_summary_metric(summary, ["avg_turnover"])
    if avg_turnover == 0.0:
        csv_turnover = _mean_csv_column(period_path, "turnover")
        if csv_turnover is not None:
            avg_turnover = csv_turnover

    industry_metrics, industry_available = _extract_industry_metrics(run_dir)
    metrics = {
        "mean_rank_ic": mean_rank_ic,
        "top30_sharpe": _first_summary_metric(summary, ["top30_sharpe", "sharpe_ratio"]),
        "annual_return": _first_summary_metric(summary, ["annual_return"]),
        "sharpe_ratio": _first_summary_metric(summary, ["sharpe_ratio"]),
        "max_drawdown": _first_summary_drawdown(summary, ["max_drawdown", "top30_max_drawdown"]),
        "avg_turnover": avg_turnover,
        "industry_top5_excess": float(industry_metrics["industry_top5_excess"]),
        "industry_top5_weight_raw": float(industry_metrics["industry_top5_weight_raw"]),
        "industry_metrics_available": industry_available,
        "num_periods": int(_metric_float(summary.get("num_periods"), default=0.0)),
        "source_dir": str(run_dir),
        "artifact_paths": {
            "performance_summary": str(summary_path),
        },
    }

    if output_path is not None:
        write_summary(_resolve_skill_path(output_path), metrics)
    return metrics


def stable_hash(obj: Any) -> str:
    payload = json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _iter_hash_files(paths: list[str | Path]) -> list[Path]:
    files: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            for child in path.rglob("*"):
                rel_parts = set(child.relative_to(path).parts)
                if rel_parts.intersection({"__pycache__", "versions", "test_reviews"}):
                    continue
                if child.name in {"latest_summary.json", "results.tsv"}:
                    continue
                if child.is_file():
                    files.append(child)
    return sorted(set(files))


def code_hash(paths: list[str | Path] | None = None) -> str:
    selected = paths or DEFAULT_HASH_PATHS
    files = _iter_hash_files(selected)
    if files:
        digest = hashlib.sha256()
        try:
            tree = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
            digest.update(tree.encode("utf-8"))
        except Exception:
            pass
        for path in files:
            digest.update(str(path.relative_to(PROJECT_ROOT) if path.is_relative_to(PROJECT_ROOT) else path).encode("utf-8"))
            digest.update(path.read_bytes())
        return digest.hexdigest()[:16]
    try:
        tree = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        diff = subprocess.check_output(["git", "diff", "--", "workflow.py", "scripts/pipeline/train_rolling.py", "utils/lightgbm_params.py", "skills/lgb-top30-evolution"], text=True)
        return hashlib.sha256((tree + "\n" + diff).encode("utf-8")).hexdigest()[:16]
    except Exception:
        return ""


def _candidate_metrics_from_oot(skill_config: dict) -> dict | None:
    current = skill_config.get("current", {})
    if str(current.get("split", "validation")).lower() != "validation":
        return None

    version = current.get("version", "v000")
    paths = skill_config.get("paths", {})
    versions_dir = _resolve_skill_path(paths.get("versions_dir", "versions"))
    explicit_oot = current.get("oot_output_dir") or skill_config.get("artifact_paths", {}).get("oot_output_dir")
    oot_path = Path(explicit_oot) if explicit_oot else versions_dir / version / "oot_validation"
    if explicit_oot and not oot_path.is_absolute():
        oot_path = _resolve_skill_path(oot_path)

    try:
        return extract_oot_metrics(oot_path, output_path=versions_dir / version / "oot_metrics.json")
    except FileNotFoundError:
        return None


def _oot_window(skill_config: dict) -> tuple[str, str]:
    locked = skill_config.get("locked_boundaries", {})
    return str(locked.get("oot_validation_start", "")), str(locked.get("oot_validation_end", ""))


def _versions_dir(skill_config: dict) -> Path:
    return _resolve_skill_path(skill_config.get("paths", {}).get("versions_dir", "versions"))


def accepted_metrics_path(skill_config: dict) -> Path:
    explicit = skill_config.get("paths", {}).get("accepted_metrics")
    if explicit:
        return _resolve_skill_path(explicit)
    return _versions_dir(skill_config) / "accepted" / "accepted_metrics.json"


def load_accepted_metrics(skill_config: dict) -> dict | None:
    path = accepted_metrics_path(skill_config)
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return {key: data.get(key, 0.0) for key in METRIC_KEYS}


def _zero_metrics() -> dict:
    return {key: 0.0 for key in METRIC_KEYS}


def trial_count_for_window(skill_config: dict) -> int:
    results_path = _resolve_skill_path(skill_config.get("paths", {}).get("results_tsv", "results.tsv"))
    if not results_path.exists():
        return 0
    start, end = _oot_window(skill_config)
    with results_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        return sum(
            1
            for row in reader
            if row.get("oot_validation_start") == start and row.get("oot_validation_end") == end
        )


def keep_threshold_for_trials(skill_config: dict) -> float:
    n_trials = trial_count_for_window(skill_config)
    if n_trials >= 31:
        return 0.10
    if n_trials >= 10:
        return 0.08
    return 0.05


def _summary_contract_paths(summary: dict) -> list[str]:
    paths: list[str] = []
    artifact_paths = summary.get("artifact_paths", {})
    if isinstance(artifact_paths, dict):
        paths.extend(str(path) for path in artifact_paths.values() if path)
    oot_metrics = summary.get("oot_metrics", {})
    if isinstance(oot_metrics, dict):
        metric_artifacts = oot_metrics.get("artifact_paths", {})
        if isinstance(metric_artifacts, dict):
            paths.extend(str(path) for path in metric_artifacts.values() if path)
    if summary.get("log_tail_path"):
        paths.append(str(summary["log_tail_path"]))
    return paths


def _blocked_summary(skill_config: dict, guard_result: dict | None, status: str, oot_metrics: dict | None = None, baseline: dict | None = None) -> dict:
    current = skill_config.get("current", {})
    start, end = _oot_window(skill_config)
    summary_path = skill_config.get("paths", {}).get("latest_summary", "latest_summary.json")
    return {
        "version": current.get("version"),
        "parent_version": current.get("parent_version"),
        "hypothesis_group": current.get("hypothesis_group"),
        "hypothesis": current.get("hypothesis"),
        "decision_inputs": {"baseline": baseline or {}, "candidate": {}, "deltas": {}},
        "rule_result": {"score_delta": 0.0, "keep_threshold": keep_threshold_for_trials(skill_config), "decision": "rollback", "hard_gates": {"passed": False, "failed": [status]}},
        "guard_result": guard_result if guard_result is not None else guard_result_or_errors(skill_config),
        "decision": "rollback",
        "config_hash": stable_hash(skill_config),
        "code_hash": code_hash(),
        "data_snapshot": skill_config.get("data_snapshot", {}),
        "artifact_paths": skill_config.get("artifact_paths", {}),
        "oot_metrics": oot_metrics,
        "log_tail_path": skill_config.get("log_tail_path"),
        "status": status,
        "split": current.get("split", "validation"),
        "oot_validation_start": start,
        "oot_validation_end": end,
        "summary_path": summary_path,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def build_summary(skill_config: dict, guard_result: dict | None = None) -> dict:
    current = skill_config.get("current", {})
    start, end = _oot_window(skill_config)
    split = str(current.get("split", "validation")).lower()
    if split != "validation":
        return _blocked_summary(skill_config, guard_result, "manual_test_review_only")

    oot_metrics = _candidate_metrics_from_oot(skill_config)
    if oot_metrics is None:
        return _blocked_summary(skill_config, guard_result, "blocked_missing_oot_metrics")

    candidate = {key: oot_metrics[key] for key in METRIC_KEYS}
    if current.get("hypothesis_group") == "baseline":
        baseline = {}
        rule = {
            "score_delta": 0.0,
            "decision": "keep",
            "deltas": {},
            "hard_gates": {"passed": True, "failed": []},
        }
    else:
        baseline = load_accepted_metrics(skill_config)
        if baseline is None:
            return _blocked_summary(skill_config, guard_result, "blocked_missing_accepted_metrics", oot_metrics=oot_metrics)
        threshold = keep_threshold_for_trials(skill_config)
        rule = evaluate_candidate(candidate, baseline, keep_threshold=threshold)
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
        "oot_metrics": oot_metrics,
        "log_tail_path": skill_config.get("log_tail_path"),
        "status": "evaluated",
        "split": split,
        "oot_validation_start": start,
        "oot_validation_end": end,
        "summary_path": summary_path,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if not summary["guard_result"].get("passed", False):
        summary["decision"] = "rollback"
        summary["rule_result"]["decision"] = "rollback"
    try:
        validate_read_contract(_summary_contract_paths(summary))
    except GuardViolation as exc:
        summary["decision"] = "rollback"
        summary["rule_result"]["decision"] = "rollback"
        summary["status"] = "blocked_read_contract"
        summary["rule_result"]["hard_gates"] = {"passed": False, "failed": [str(exc)]}
    return summary


def append_results_tsv(results_path: str | Path, summary: dict) -> None:
    if str(summary.get("split", "validation")).lower() != "validation":
        raise ValueError("test review summaries must not be appended to automatic results.tsv")
    if summary.get("status") != "evaluated":
        raise ValueError("only evaluated validation summaries may be appended to automatic results.tsv")

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
        "oot_validation_start": summary.get("oot_validation_start"),
        "oot_validation_end": summary.get("oot_validation_end"),
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

    latest_summary = _resolve_skill_path(args.latest_summary or skill_config.get("paths", {}).get("latest_summary", "latest_summary.json"))
    results_path = _resolve_skill_path(args.results or skill_config.get("paths", {}).get("results_tsv", "results.tsv"))
    write_summary(latest_summary, summary)
    if args.append_results and not args.summary_only:
        append_results_tsv(results_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if guard_result.get("passed", False) and summary.get("status") == "evaluated" else 2


if __name__ == "__main__":
    raise SystemExit(main())
