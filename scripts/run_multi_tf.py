"""Run v002 across multiple execution timeframes on the same date range.

Levels (HTF FVGs, swing-based session H/L, prior-day midpoint) are built ONCE from
1m data — so the level pool is identical across TF runs. Only the execution layer
(sweep first-touch, intra-session FVG detection, inversion close-through, forward-walk
exit) changes per TF.

Usage:
    python scripts/run_multi_tf.py --start 2026-03-01 --end 2026-03-26 \\
        --tfs 30s 1m 2m 3m 5m
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ifvg_backtester.engine.data import load_1m_ny  # noqa: E402  (loader works for any TF csv)
from ifvg_backtester.engine.levels import (  # noqa: E402
    active_htf_fvg_levels,
    build_session_levels,
    daily_pdmid,
    detect_htf_fvgs,
    resample_to_htf,
)
from ifvg_backtester.engine.simulator import run_session  # noqa: E402


TF_CSV = {
    "15s": "data/nq-front-month.ohlcv-15s.csv",
    "30s": "data/nq-front-month.ohlcv-30s.csv",
    "1m":  "data/nq-front-month.ohlcv-1m.csv",
    "2m":  "data/nq-front-month.ohlcv-2m.csv",
    "3m":  "data/nq-front-month.ohlcv-3m.csv",
    "5m":  "data/nq-front-month.ohlcv-5m.csv",
}

HTF_TIMEFRAMES = (("1H", "1h"), ("4H", "4h"))
HTF_PROXIMITY_PTS = 300.0


def _friday_close_and_monday_open(
    date_ny: pd.Timestamp, by_date: dict[pd.Timestamp, pd.DataFrame]
) -> tuple[float | None, float | None]:
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


def run_one_tf(
    tf_label: str,
    exec_csv: str,
    df_1m: pd.DataFrame,
    pdmid: pd.DataFrame,
    htf_fvgs_by_tf: dict[str, list[dict]],
    cutoff_start: pd.Timestamp,
    cutoff_end: pd.Timestamp,
    buffered_start: str,
    end_str: str,
) -> tuple[list, dict]:
    exec_df = load_1m_ny(exec_csv, start_date_ny=buffered_start, end_date_ny=end_str)

    by_date_1m = {d: g for d, g in df_1m.groupby(df_1m.index.normalize())}
    by_date_exec = {d: g for d, g in exec_df.groupby(exec_df.index.normalize())}

    all_trades = []
    sessions_run = 0
    sweep_groups_seen: Counter = Counter()

    for date_ny, exec_day_df in sorted(by_date_exec.items()):
        if not (cutoff_start <= date_ny < cutoff_end):
            continue
        if date_ny.dayofweek >= 5:
            continue
        if date_ny not in pdmid.index or pd.isna(pdmid.loc[date_ny, "pdh"]):
            continue

        # Build levels from 1m
        day_df_1m = by_date_1m.get(date_ny)
        if day_df_1m is None or day_df_1m.empty:
            continue
        prior_1m_dates = sorted([d for d in by_date_1m if d < date_ny])
        prior_df_1m = by_date_1m[prior_1m_dates[-1]] if prior_1m_dates else None

        fri_close, mon_open = _friday_close_and_monday_open(date_ny, by_date_1m)

        sweep_levels = build_session_levels(
            date_ny, day_df_1m, prior_df_1m,
            friday_close=fri_close, monday_open=mon_open,
        )

        # HTF FVG levels — refer to exec_day_df's session open price
        sess_open_bar = exec_day_df[(exec_day_df.index.hour == 9) & (exec_day_df.index.minute >= 30)]
        if not sess_open_bar.empty:
            ref_price = float(sess_open_bar.iloc[0]["open"])
            sess_start_ts = sess_open_bar.index[0]
            for tf_l, _rule in HTF_TIMEFRAMES:
                sweep_levels.extend(
                    active_htf_fvg_levels(
                        htf_fvgs_by_tf[tf_l], tf_l, sess_start_ts,
                        proximity_pts=HTF_PROXIMITY_PTS, reference_price=ref_price,
                    )
                )

        for L in sweep_levels:
            sweep_groups_seen[L.group] += 1

        trades = run_session(
            exec_day_df,
            date_ny,
            float(pdmid.loc[date_ny, "pd_mid"]),
            sweep_levels,
        )
        all_trades.extend(trades)
        sessions_run += 1

    return all_trades, {"sessions": sessions_run, "groups": dict(sweep_groups_seen)}


def summary_row(tf: str, trades: list) -> dict:
    n = len(trades)
    wins = sum(1 for t in trades if t.outcome == "win")
    losses = sum(1 for t in trades if t.outcome == "loss")
    closes = sum(1 for t in trades if t.outcome == "eod_close")
    total_r = sum(t.r_multiple for t in trades)
    return {
        "tf": tf,
        "n": n,
        "wins": wins,
        "losses": losses,
        "eod": closes,
        "WR%": round(100 * wins / n, 1) if n else 0.0,
        "total_R": round(total_r, 2),
        "exp_R": round(total_r / n, 3) if n else 0.0,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--tfs", nargs="+", default=["30s", "1m", "2m", "3m", "5m"])
    p.add_argument("--out", default="runs/v002/2026-03_multi_tf.csv")
    args = p.parse_args()

    buffered_start = (pd.Timestamp(args.start) - pd.Timedelta(days=14)).strftime("%Y-%m-%d")

    print("Loading 1m for level construction (used by all TF runs)")
    df_1m = load_1m_ny(TF_CSV["1m"], start_date_ny=buffered_start, end_date_ny=args.end)
    print(f"  {len(df_1m):,} 1m bars  ({df_1m.index.min()} → {df_1m.index.max()})")

    print("Building prior-day midpoint table")
    pdmid = daily_pdmid(df_1m)

    print("Detecting HTF FVGs (1H/4H) on 1m-resampled bars")
    htf_fvgs_by_tf: dict[str, list[dict]] = {}
    for tf_label, rule in HTF_TIMEFRAMES:
        htf_bars = resample_to_htf(df_1m, rule)
        htf_fvgs_by_tf[tf_label] = detect_htf_fvgs(htf_bars)
        n_active = sum(1 for f in htf_fvgs_by_tf[tf_label] if f["inverted_at"] is None)
        print(f"  {tf_label}: {len(htf_fvgs_by_tf[tf_label])} total ({n_active} never inverted)")

    cutoff_start = pd.Timestamp(args.start, tz="America/New_York").normalize()
    cutoff_end = pd.Timestamp(args.end, tz="America/New_York").normalize()

    all_runs: dict[str, list] = {}
    for tf in args.tfs:
        if tf not in TF_CSV:
            print(f"Unknown TF: {tf} (have: {list(TF_CSV)})")
            continue
        print(f"\n--- Running TF={tf} ---")
        trades, info = run_one_tf(
            tf, TF_CSV[tf], df_1m, pdmid, htf_fvgs_by_tf,
            cutoff_start, cutoff_end, buffered_start, args.end,
        )
        all_runs[tf] = trades
        print(f"  sessions={info['sessions']}  trades={len(trades)}")

    print("\n=== SUMMARY by TF ===")
    rows = [summary_row(tf, trades) for tf, trades in all_runs.items()]
    summary_df = pd.DataFrame(rows)
    print(summary_df.to_string(index=False))

    print("\n=== ALL TRADES (chronological per TF) ===")
    rows = []
    for tf, trades in all_runs.items():
        for t in trades:
            r = asdict(t)
            r["exec_tf"] = tf
            rows.append(r)
    if not rows:
        print("(none)")
        return 0

    out_df = pd.DataFrame(rows)
    out_df = out_df[[
        "exec_tf", "session_date", "direction", "swept_level_name",
        "sweep_ts", "target_gap_size", "bars_to_inversion",
        "entry_ts", "entry_price", "risk_points", "exit_ts",
        "outcome", "pnl_points", "r_multiple",
    ]]
    out_df["session_date"] = pd.to_datetime(out_df["session_date"]).dt.strftime("%Y-%m-%d")
    out_df["sweep_ts"] = pd.to_datetime(out_df["sweep_ts"]).dt.strftime("%H:%M:%S")
    out_df["entry_ts"] = pd.to_datetime(out_df["entry_ts"]).dt.strftime("%H:%M:%S")
    out_df["exit_ts"] = pd.to_datetime(out_df["exit_ts"]).dt.strftime("%m-%d %H:%M:%S")
    out_df = out_df.sort_values(["session_date", "exec_tf", "sweep_ts"])
    with pd.option_context("display.max_rows", None, "display.width", 240):
        print(out_df.to_string(index=False))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"\nMulti-TF trade log → {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
