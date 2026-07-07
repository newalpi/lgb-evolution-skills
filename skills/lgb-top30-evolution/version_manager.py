from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import yaml


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _write_yaml(path: Path, data: dict) -> None:
    path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")


def _resolve_skill_path(skill_config_path: Path, path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else skill_config_path.parent / path


def _copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def accept_version(skill_config_path: str | Path) -> dict:
    config_path = Path(skill_config_path)
    skill_config = _load_yaml(config_path)
    version = skill_config.get("current", {}).get("version")
    if not version:
        raise ValueError("current.version is required")

    versions_dir = _resolve_skill_path(config_path, skill_config.get("paths", {}).get("versions_dir", "versions"))
    version_dir = versions_dir / str(version)
    accepted_dir = versions_dir / "accepted"
    latest_summary_path = _resolve_skill_path(
        config_path,
        skill_config.get("paths", {}).get("latest_summary", "latest_summary.json"),
    )
    if not latest_summary_path.exists():
        raise FileNotFoundError(f"latest summary not found: {latest_summary_path}")

    summary = json.loads(latest_summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "evaluated" or summary.get("decision") != "keep":
        raise ValueError("only evaluated keep summaries can be accepted")

    metrics: dict[str, Any] = summary.get("decision_inputs", {}).get("candidate", {})
    if not metrics:
        raise ValueError("accepted summary missing candidate metrics")

    accepted_dir.mkdir(parents=True, exist_ok=True)
    (accepted_dir / "accepted_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (accepted_dir / "accepted_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    _copy_if_exists(version_dir / "runtime_config.yaml", accepted_dir / "runtime_config.yaml")
    _copy_if_exists(version_dir / "oot_metrics.json", accepted_dir / "oot_metrics.json")

    skill_config["accepted_version"] = str(version)
    _write_yaml(accepted_dir / "skill_config.yaml", skill_config)
    _write_yaml(config_path, skill_config)
    return {"accepted_version": str(version), "accepted_dir": str(accepted_dir)}


def rollback_version(skill_config_path: str | Path) -> dict:
    config_path = Path(skill_config_path)
    skill_config = _load_yaml(config_path)
    accepted_version = skill_config.get("accepted_version")
    if not accepted_version:
        raise ValueError("accepted_version is required for rollback")

    current = skill_config.setdefault("current", {})
    current["version"] = str(accepted_version)
    current["parent_version"] = str(accepted_version)
    current["split"] = "validation"
    _write_yaml(config_path, skill_config)
    return {"current_version": str(accepted_version), "parent_version": str(accepted_version)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Manage lgb-top30-evolution accepted version pointer.")
    parser.add_argument("command", choices=["accept", "rollback"])
    parser.add_argument("--skill-config", default=str(Path(__file__).resolve().parent / "skill_config.yaml"))
    args = parser.parse_args()

    if args.command == "accept":
        result = accept_version(args.skill_config)
    else:
        result = rollback_version(args.skill_config)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
