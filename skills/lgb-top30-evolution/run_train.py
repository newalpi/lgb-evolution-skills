from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import yaml

from guard import GuardViolation, guard_result_or_errors, load_yaml
from validation import code_hash, stable_hash, write_summary


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SKILL_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIGS = [
    "configs/base.yaml",
    "configs/features.yaml",
    "configs/model_lgb.yaml",
    "configs/strategy.yaml",
]


def deep_merge(dest: dict, src: dict) -> dict:
    for key, value in (src or {}).items():
        if isinstance(value, dict) and isinstance(dest.get(key), dict):
            deep_merge(dest[key], value)
        else:
            dest[key] = value
    return dest


def load_repo_config() -> dict:
    merged: dict = {}
    for rel in DEFAULT_CONFIGS:
        with (PROJECT_ROOT / rel).open("r", encoding="utf-8") as f:
            deep_merge(merged, yaml.safe_load(f) or {})
    return merged


def apply_skill_config(runtime_config: dict, skill_config: dict, version_dir: Path) -> dict:
    experiment = skill_config.get("experiment", {})
    model_exp = experiment.get("model", {})
    feature_exp = experiment.get("features", {})
    portfolio_exp = experiment.get("portfolio", {})
    locked = skill_config.get("locked_boundaries", {})

    model_cfg = runtime_config.setdefault("model", {})
    rolling_cfg = model_cfg.setdefault("rolling", {})
    if locked.get("rolling_model_start"):
        rolling_cfg["start_date"] = str(locked["rolling_model_start"])
    if locked.get("rolling_model_end"):
        rolling_cfg["end_date"] = str(locked["rolling_model_end"])
    locked_sample_weight = locked.get("model", {}).get("sample_weight")
    if isinstance(locked_sample_weight, dict):
        deep_merge(model_cfg.setdefault("sample_weight", {}), locked_sample_weight)
    if model_exp.get("lgb_params_overlay"):
        model_cfg["lgb_params_overlay"] = model_exp["lgb_params_overlay"]

    for key in (
        "rolling_model_start",
        "rolling_model_end",
        "oot_validation_start",
        "oot_validation_end",
        "test_review_start",
        "test_review_end",
    ):
        if key in locked:
            runtime_config[key] = locked[key]

    data_cfg = runtime_config.setdefault("data", {})
    if locked.get("oot_validation_start"):
        data_cfg["test_start"] = str(locked["oot_validation_start"])
    if locked.get("oot_validation_end"):
        data_cfg["test_end"] = str(locked["oot_validation_end"])

    custom = runtime_config.setdefault("features", {}).setdefault("base_features", {}).setdefault("custom", {})
    if feature_exp.get("include") is not None:
        custom["filter_mode"] = "include"
        custom["include"] = feature_exp["include"]
    if feature_exp.get("exclude") is not None:
        custom["exclude"] = feature_exp["exclude"]
    if feature_exp.get("processing"):
        custom.setdefault("processing", {}).update(feature_exp["processing"])

    backtest_cfg = runtime_config.setdefault("backtest", {})
    if locked.get("oot_validation_start"):
        backtest_cfg["start_date"] = str(locked["oot_validation_start"])
    if locked.get("oot_validation_end"):
        backtest_cfg["end_date"] = str(locked["oot_validation_end"])

    strategy = backtest_cfg.setdefault("strategy", {})
    for src_key, dst_key in (("topk", "topk"), ("n_drop", "n_drop"), ("weight_method", "weight_method")):
        if src_key in portfolio_exp:
            strategy[dst_key] = portfolio_exp[src_key]
    if portfolio_exp.get("mvo_optimizer"):
        backtest_cfg.setdefault("mvo_optimizer", {}).update(portfolio_exp["mvo_optimizer"])

    runtime_config.setdefault("output", {})
    runtime_config["output"]["model_dir"] = str(version_dir / "models")
    runtime_config["output"]["predictions_dir"] = str(version_dir / "predictions")
    return runtime_config


def version_dir_for(skill_config: dict, resume: bool = False) -> Path:
    current = skill_config.get("current", {})
    version = current.get("version", "v000")
    version_dir = SKILL_DIR / "versions" / version
    if version_dir.exists() and not resume:
        raise GuardViolation(f"version directory already exists: {version_dir}; pass --resume to reuse it")
    version_dir.mkdir(parents=True, exist_ok=True)
    return version_dir


def render_runtime_config(skill_config: dict, resume: bool = False) -> tuple[dict, Path, Path]:
    version_dir = version_dir_for(skill_config, resume=resume)
    runtime_config = apply_skill_config(load_repo_config(), skill_config, version_dir)
    out_path = version_dir / "runtime_config.yaml"
    with out_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(runtime_config, f, allow_unicode=True, sort_keys=False)
    return runtime_config, out_path, version_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Run guarded LGB Top30 rolling training.")
    parser.add_argument("--skill-config", default="skill_config.yaml")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    skill_config = load_yaml(args.skill_config)
    try:
        runtime_config, runtime_path, version_dir = render_runtime_config(skill_config, resume=args.resume)
    except GuardViolation as exc:
        summary = {"stage": "train", "guard_result": {"passed": False, "violations": [str(exc)]}}
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 2
    guard_result = guard_result_or_errors(skill_config, runtime_config=runtime_config)
    command = [
        "/home/newalpi/miniforge3/envs/quant/bin/python",
        "scripts/pipeline/train_rolling.py",
        "--config",
        str(runtime_path),
    ]
    summary = {
        "stage": "train",
        "version": skill_config.get("current", {}).get("version"),
        "guard_result": guard_result,
        "command": command,
        "runtime_config": str(runtime_path),
        "config_hash": stable_hash(runtime_config),
        "code_hash": code_hash(),
    }
    write_summary(version_dir / "train_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.dry_run:
        return 0 if guard_result.get("passed", False) else 2
    if not guard_result.get("passed", False):
        return 2
    return subprocess.call(command, cwd=PROJECT_ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
