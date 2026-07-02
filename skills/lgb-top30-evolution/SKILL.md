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
- Sample split: use the repository runtime config as the fixed boundary.
- Purge gap: `purge_days=11`.
- Trading costs, ST/new-share/suspension/limit filters, and OOT/backtest settlement must stay aligned with the repository shared scripts.

## Codex Read Budget

For each automatic iteration, read only:

- `SKILL.md`
- `skill_config.yaml`
- `results.tsv`
- `latest_summary.json`
- the last 100 lines of an error log if `latest_summary.json.log_tail_path` points to one

Do not read full market data, full factor matrices, complete predictions, complete trade details, full holding history, or long training logs. Local scripts must summarize heavy artifacts before Codex decides.

## Allowed Changes

Change exactly one main `hypothesis_group` per iteration:

- `lgb_params`
- `factor_subset`
- `factor_processing`
- `portfolio_rule`
- `exposure_constraint`
- `position_rule`

Position rules must be tested alone with `hypothesis_group: position_rule`. Do not mix position-rule changes with model, factor, or portfolio changes.

## Forbidden Changes

Do not change the strategy kernel, label, split boundaries, purge logic, transaction costs, trade filters, OOT/backtest settlement, or `score_neutralization.enabled=false`.

Do not use test results to update `results.tsv`, accepted baseline, or automatic keep/rollback decisions. Test reviews must stay under `test_reviews/`.

The LightGBM parameter injection hook is a one-time infrastructure change. After `infrastructure_locked: true`, future iterations may only use `model.lgb_params_overlay` through `skill_config.yaml`.

## Workflow

1. Edit `skill_config.yaml` for one hypothesis.
2. Run `guard.py` before training.
3. Run `run_train.py`; let local scripts read heavy data and logs.
4. Run `run_backtest.py` on validation only.
5. Run `validation.py`; it writes `latest_summary.json` and appends validation-only `results.tsv`.
6. Codex reads the small summary and decides `keep` or `rollback`.

Keep copies small manifests and summaries to `versions/accepted/`. Rollback restores the skill-local accepted pointer only; never use destructive git commands.
