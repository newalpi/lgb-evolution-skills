---
name: lgb-top30-evolution
description: Guarded LightGBM Top30 A-share strategy evolution protocol. Use when iterating LightGBM params, factor subsets, factor preprocessing, TopK/Dropout/MVO/equal weighting, exposure constraints, or position rules for the validation-only LGB Top30 research loop in /home/newalpi/quant.
---

# LGB Top30 Evolution

## Strategy Kernel

Use one LightGBM model to predict A-share 10 trading-day forward returns and build a Top30 portfolio.

Locked investment contract:

- Label: `Ref($vwap, -11) / Ref($vwap, -1) - 1`.
- Buy/sell assumption: T+1 VWAP buy to T+11 VWAP sell.
- Rolling model window: `rolling_model_start` to `rolling_model_end` controls which monthly `lgb_YYYY-MM.txt` models are trained.
- Outer OOT validation window: `oot_validation_start` to `oot_validation_end` controls automatic keep/rollback decisions.
- Training internal validation is still the rolling split inside `scripts/pipeline/train_rolling.py` (`train_months`, `valid_ratio`, `purge_days`); it is not the same thing as the skill's outer OOT validation window.
- Purge gap: `purge_days=11`.
- Trading costs, ST/new-share/suspension/limit filters, and OOT/backtest settlement must stay aligned with the repository shared scripts.

Current local protocol:

- Automatic validation uses the pre-2026 OOT window `2025-01-01` to `2025-12-31`.
- Manual review uses `2026-01-01` to `2026-07-01` and must be run as `split=test --manual-review`.
- Market-similarity sample weighting is anchored with `anchor_reference=split_start` for the automatic validation protocol, so the pre-2026 validation window does not use post-window market states as its anchor.

## Codex Read Budget

For each automatic iteration, read only:

- `SKILL.md`
- `skill_config.yaml`
- `results.tsv`
- `latest_summary.json`
- `versions/<version>/oot_metrics.json` if present
- the last 100 lines of an error log if `latest_summary.json.log_tail_path` points to one

Do not read full market data, full factor matrices, complete predictions, complete trade details, full holding history, or long training logs. Local scripts must summarize heavy artifacts before Codex decides. `validation.py` may read OOT artifacts locally and write a compact `oot_metrics.json`; Codex should read that summary instead of raw OOT CSVs.

## Allowed Changes

Change exactly one main `hypothesis_group` per iteration:

- `baseline` only when establishing a new accepted baseline or resetting the validation protocol
- `lgb_params`
- `factor_subset`
- `factor_processing`
- `portfolio_rule`
- `exposure_constraint`
- `position_rule`

Position rules must be tested alone with `hypothesis_group: position_rule`. Do not mix position-rule changes with model, factor, or portfolio changes.

## Forbidden Changes

Do not change the strategy kernel, label, rolling/OOT/test review boundaries, purge logic, transaction costs, trade filters, OOT/backtest settlement, or `score_neutralization.enabled=false` during ordinary candidate iterations. Boundary changes require an explicit `hypothesis_group: baseline` protocol reset and a new accepted baseline before further automatic evolution.

Do not use test results to update `results.tsv`, accepted baseline, or automatic keep/rollback decisions. Test reviews must stay under `test_reviews/`.

The LightGBM parameter injection hook is a one-time infrastructure change. After `infrastructure_locked: true`, future iterations may only use `model.lgb_params_overlay` through `skill_config.yaml`.

## Workflow

1. Edit `skill_config.yaml` for one hypothesis.
2. Run `guard.py` before training.
3. Run `run_train.py`; let local scripts read heavy data and logs.
4. Run `run_backtest.py` on validation only.
5. Run `validation.py`; it extracts validation OOT metrics locally, writes `versions/<version>/oot_metrics.json` and `latest_summary.json`, and appends validation-only `results.tsv`. Missing OOT metrics fail closed and cannot be accepted.
6. Codex reads the small summary and decides `keep` or `rollback`.

Use `version_manager.py accept` to copy small manifests and summaries to `versions/accepted/`. Use `version_manager.py rollback` to restore the skill-local accepted pointer only; never use destructive git commands.
