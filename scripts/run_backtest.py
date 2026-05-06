"""Run v001 IFVG backtest on a date range. Smoke-test runner.

Usage:
    python scripts/run_backtest.py --start 2026-03-01 --end 2026-03-26 \\
        --csv data/nq-front-month.ohlcv-1m.csv \\
        --out runs/v001_2026-03.csv
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from dataclasses import asdict
from datetime import time as dtime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ifvg_backtester.engine.data import load_1m_ny  # noqa: E402
from ifvg_backtester.engine.levels import (  # noqa: E402
    SweepLevel,
    active_htf_fvg_levels,
    build_session_levels,
    daily_pdmid,
    detect_htf_fvgs,
    resample_to_htf,
)
from ifvg_backtester.engine.simulator import run_session  # noqa: E402


HTF_TIMEFRAMES = (("1H", "1h"), ("4H", "4h"))   # (label, pandas-rule)
HTF_PROXIMITY_PTS = 300.0                        # filter HTF FVGs to within ±300pt of session open


def _prior_trading_dates(all_dates: list[pd.Timestamp]) -> dict[pd.Timestamp, pd.Timestamp]:
    """date_ny → prior trading-date_ny in the dataset (skips weekends/holidays)."""
    sorted_dates = sorted(all_dates)
    out = {}
    for i in range(1, len(sorted_dates)):
        out[sorted_dates[i]] = sorted_dates[i - 1]
    return out


def _friday_close_and_monday_open(
    date_ny: pd.Timestamp, by_date: dict[pd.Timestamp, pd.DataFrame]
) -> tuple[float | None, float | None]:
    """For a Monday, return (prior Friday's RTH close, this Monday's RTH open).
    Returns (None, None) for non-Mondays or missing data."""
    if date_ny.dayofweek != 0:
        return None, None
    fri_dates = [d for d in by_date if d.dayofweek == 4 and d < date_ny]
    if not fri_dates:
        return None, None
    fri_df = by_date[max(fri_dates)]
    fri_rth = fri_df[
        ((fri_df.index.hour == 9) & (fri_df.index.minute >= 30))
        | ((fri_df.index.hour > 9) & (fri_df.index.hour < 16))
    ]
    if fri_rth.empty:
        return None, None
    fri_close = float(fri_rth.iloc[-1]["close"])

    mon_df = by_date[date_ny]
    mon_open_bar = mon_df[(mon_df.index.hour == 9) & (mon_df.index.minute >= 30)]
    if mon_open_bar.empty:
        return None, None
    mon_open = float(mon_open_bar.iloc[0]["open"])
    return fri_close, mon_open


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="data/nq-front-month.ohlcv-1m.csv")
    p.add_argument("--start", required=True, help="NY date inclusive, YYYY-MM-DD")
    p.add_argument("--end", required=True, help="NY date exclusive, YYYY-MM-DD")
    p.add_argument("--out", default=None, help="Optional CSV path for trade log")
    args = p.parse_args()

    print(f"Loading 1m bars  csv={args.csv}  range=[{args.start}, {args.end})")
    # Buffer of prior days for prior-day levels and HTF FVG history.
    buffered_start = (pd.Timestamp(args.start) - pd.Timedelta(days=14)).strftime("%Y-%m-%d")
    df = load_1m_ny(args.csv, start_date_ny=buffered_start, end_date_ny=args.end)
    print(f"  bars loaded: {len(df):,}  ({df.index.min()} → {df.index.max()})")

    print("Building prior-day midpoint table (R02 bias)")
    pdmid = daily_pdmid(df)

    print("Detecting HTF FVGs and tracking close-through inversion")
    htf_fvgs_by_tf: dict[str, list[dict]] = {}
    for tf_label, rule in HTF_TIMEFRAMES:
        htf_bars = resample_to_htf(df, rule)
        fvgs = detect_htf_fvgs(htf_bars)
        htf_fvgs_by_tf[tf_label] = fvgs
        n_active = sum(1 for f in fvgs if f["inverted_at"] is None)
        print(f"  {tf_label}: {len(fvgs):,} total FVGs ({n_active:,} never inverted)")

    print(f"Running sessions {args.start} → {args.end}")
    by_date_dict = {d: g for d, g in df.groupby(df.index.normalize())}
    prior_map = _prior_trading_dates(list(by_date_dict.keys()))

    cutoff_start = pd.Timestamp(args.start, tz="America/New_York").normalize()
    cutoff_end = pd.Timestamp(args.end, tz="America/New_York").normalize()

    all_trades = []
    sessions_run = 0
    sessions_skipped = 0
    sweep_groups_seen: Counter = Counter()
    sweep_groups_per_session: list[int] = []

    for date_ny, day_df in sorted(by_date_dict.items()):
        if not (cutoff_start <= date_ny < cutoff_end):
            continue
        if date_ny.dayofweek >= 5:
            continue
        if date_ny not in pdmid.index or pd.isna(pdmid.loc[date_ny, "pdh"]):
            sessions_skipped += 1
            continue

        prior_date = prior_map.get(date_ny)
        prior_day_df = by_date_dict.get(prior_date) if prior_date is not None else None

        fri_close, mon_open = _friday_close_and_monday_open(date_ny, by_date_dict)

        # Session H/L levels
        sweep_levels = build_session_levels(
            date_ny, day_df, prior_day_df, friday_close=fri_close, monday_open=mon_open
        )

        # HTF FVGs active at session start, filtered to nearby ones
        sess_open_bar = day_df[(day_df.index.hour == 9) & (day_df.index.minute >= 30)]
        if not sess_open_bar.empty:
            ref_price = float(sess_open_bar.iloc[0]["open"])
            sess_start_ts = sess_open_bar.index[0]
            for tf_label, _rule in HTF_TIMEFRAMES:
                sweep_levels.extend(
                    active_htf_fvg_levels(
                        htf_fvgs_by_tf[tf_label],
                        tf_label,
                        as_of=sess_start_ts,
                        proximity_pts=HTF_PROXIMITY_PTS,
                        reference_price=ref_price,
                    )
                )

        # Tally for the summary
        for L in sweep_levels:
            sweep_groups_seen[L.group] += 1
        sweep_groups_per_session.append(len(sweep_levels))

        trades = run_session(
            day_df,
            date_ny,
            float(pdmid.loc[date_ny, "pd_mid"]),
            sweep_levels,
        )
        all_trades.extend(trades)
        sessions_run += 1

    print(f"  sessions run: {sessions_run}  skipped (no prior-day levels): {sessions_skipped}")
    if sweep_groups_per_session:
        avg_levels = sum(sweep_groups_per_session) / len(sweep_groups_per_session)
        print(f"  avg sweep levels per session: {avg_levels:.1f}")
        print(f"  level pool by group: {dict(sweep_groups_seen.most_common())}")
    print(f"  trades generated: {len(all_trades)}")

    if not all_trades:
        print("\nNo trades. Check rule defaults vs market action in this window.")
        return 0

    rows = [asdict(t) for t in all_trades]
    out_df = pd.DataFrame(rows)

    wins = (out_df["outcome"] == "win").sum()
    losses = (out_df["outcome"] == "loss").sum()
    closes = (out_df["outcome"] == "eod_close").sum()
    n = len(out_df)
    win_rate = wins / n if n else 0.0
    total_r = out_df["r_multiple"].sum()
    avg_r = out_df["r_multiple"].mean()
    expectancy_r = total_r / n if n else 0.0

    print("\n=== SUMMARY ===")
    print(f"  trades:      {n}")
    print(f"  wins:        {wins}")
    print(f"  losses:      {losses}")
    print(f"  EOD closes:  {closes}")
    print(f"  win rate:    {win_rate:.1%}")
    print(f"  total R:     {total_r:+.2f}")
    print(f"  avg R/trade: {avg_r:+.3f}")
    print(f"  expectancy:  {expectancy_r:+.3f} R")

    # Per-group breakdown — useful given the wider sweep target set
    out_df["sweep_group"] = out_df["swept_level_name"].astype(str).str.replace(
        r"_\d{8}_\d{4}$", "", regex=True
    )
    print("\n=== BY SWEEP GROUP ===")
    grp = out_df.groupby("sweep_group").agg(
        n=("outcome", "size"),
        wins=("outcome", lambda s: (s == "win").sum()),
        total_R=("r_multiple", "sum"),
    )
    grp["WR_pct"] = (grp["wins"] / grp["n"] * 100).round(1)
    print(grp.to_string())

    print("\n=== TRADES ===")
    cols = [
        "session_date", "direction", "swept_level_name", "sweep_ts",
        "target_gap_size", "bars_to_inversion", "entry_ts", "entry_price",
        "risk_points", "exit_ts", "outcome", "pnl_points", "r_multiple",
    ]
    show = out_df[cols].copy()
    show["session_date"] = show["session_date"].dt.strftime("%Y-%m-%d")
    show["sweep_ts"] = show["sweep_ts"].dt.strftime("%H:%M")
    show["entry_ts"] = show["entry_ts"].dt.strftime("%H:%M")
    show["exit_ts"] = show["exit_ts"].dt.strftime("%Y-%m-%d %H:%M")
    with pd.option_context("display.max_rows", None, "display.width", 220):
        print(show.to_string(index=False))

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_df.drop(columns=["sweep_group"]).to_csv(out_path, index=False)
        print(f"\nTrade log written → {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
