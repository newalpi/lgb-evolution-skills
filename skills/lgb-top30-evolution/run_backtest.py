from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from guard import GuardViolation, guard_result_or_errors, load_yaml
from validation import code_hash, stable_hash, write_summary


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def build_command(skill_config: dict, split: str, manual_review: bool) -> tuple[list[str], Path]:
    current = skill_config.get("current", {})
    paths = skill_config.get("paths", {})
    version = current.get("version", "v000")
    if split == "test" and not manual_review:
        raise GuardViolation("test split requires --manual-review and must not write results.tsv")

    skill_dir = Path(__file__).resolve().parent
    versions_dir = Path(paths.get("versions_dir", "versions"))
    if not versions_dir.is_absolute():
        versions_dir = skill_dir / versions_dir
    version_dir = versions_dir / version

    review_dir = Path(paths.get("test_reviews_dir", "test_reviews"))
    if not review_dir.is_absolute():
        review_dir = skill_dir / review_dir

    model_path = current.get("model_path") or skill_config.get("artifact_paths", {}).get("model_path")
    if not model_path:
        start_ym = str(skill_config.get("locked_boundaries", {}).get("rolling_model_start", "YYYY-MM"))
        model_path = str(version_dir / "models" / f"lgb_{start_ym}.txt")

    locked = skill_config.get("locked_boundaries", {})
    if split == "test":
        start_date = locked.get("test_review_start")
        end_date = locked.get("test_review_end")
        output_dir = review_dir / version
    else:
        start_date = locked.get("oot_validation_start")
        end_date = locked.get("oot_validation_end")
        output_dir = version_dir / "oot_validation"

    command = [
        "/home/newalpi/miniforge3/envs/quant/bin/python",
        "scripts/oot_validation.py",
        "--model",
        model_path,
        "--config",
        str(version_dir / "runtime_config.yaml"),
        "--start-date",
        str(start_date),
        "--end-date",
        str(end_date),
        "--top-n",
        "30",
        "--n-drop",
        str(skill_config.get("experiment", {}).get("portfolio", {}).get("n_drop", 30)),
        "--output-dir",
        str(output_dir),
    ]
    return command, output_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Run validation-only OOT backtest for LGB Top30 evolution.")
    parser.add_argument("--skill-config", default="skill_config.yaml")
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--manual-review", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    skill_config = load_yaml(args.skill_config)
    command, output_dir = build_command(skill_config, args.split, args.manual_review)
    guard_result = guard_result_or_errors(
        skill_config,
        run_context={"split": args.split, "manual_review": args.manual_review},
    )
    summary = {
        "stage": "backtest",
        "version": skill_config.get("current", {}).get("version"),
        "split": args.split,
        "manual_review": args.manual_review,
        "guard_result": guard_result,
        "command": command,
        "output_dir": str(output_dir),
        "config_hash": stable_hash(skill_config),
        "code_hash": code_hash(),
    }
    summary_path = output_dir.parent / "backtest_summary.json" if args.split == "validation" else output_dir / "backtest_summary.json"
    write_summary(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.dry_run:
        return 0 if guard_result.get("passed", False) else 2
    if not guard_result.get("passed", False):
        return 2
    return subprocess.call(command, cwd=PROJECT_ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
