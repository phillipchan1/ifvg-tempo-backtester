# Repository Structure (Target)

- `src/ifvg_backtester/`: reusable package code
  - `rules/`: rule schema + loaders
  - `backtest/`: simulation and permutation engine
  - `analysis/`: performance and consistency analytics
  - `io/`: data loading/parsing utilities
- `configs/rules/`: **version-controlled trading rule definitions**
- `data/raw/`: immutable input datasets
- `data/processed/`: derived datasets and features
- `transcriptions/`: source transcripts (existing)
- `scripts/`: CLI entrypoints for extraction/backtests
- `tests/`: automated tests
- `notebooks/`: ad hoc research
