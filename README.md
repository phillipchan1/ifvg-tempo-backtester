# IFVG Tempo Backtester

This repository is being evolved into a **Python-first backtesting lab** for your IFVG entry model.

## Current + Planned Workflow

1. Keep transcript extraction as your qualitative data source.
2. Convert transcript observations into structured trade records.
3. Run backtests and permutation tests against market/context variables.
4. Compare results by **profit factor, consistency, and narrative scenario**.
5. Version your rules so model changes are explicit and auditable over time.

## New Structure

- `src/ifvg_backtester/`: core package modules.
- `configs/rules/`: version-controlled rule definitions (`*.yaml`).
- `docs/repo_structure.md`: target architecture and ownership of folders.
- `transcriptions/`: retained as-is for your source transcripts.

## Rule Versioning (Key Requirement)

Your entry model rules now live in `configs/rules/` with one file per revision.

Example lifecycle:
- Start with `ifvg_entry_v001.yaml`.
- Duplicate and increment when you refine logic (`v002`, `v003`, ...).
- Backtest each version and compare metrics side-by-side.

This gives you true model governance: every ruleset is diffable in git and tied to results.

## Immediate Next Steps

- Move existing extraction scripts under `scripts/` in a follow-up pass.
- Implement backtest engine in `src/ifvg_backtester/backtest/`.
- Add permutation runner for scenario/narrative sweeps.
- Add tests around rule parsing and simulation invariants.

## Existing Utilities (still available)

- `extract_journal.py`
- `market_metrics.py`
- `analyze_trades.py`
- `run.sh`

These can be gradually migrated into the new package + script layout.
