"""Trace why a specific setup picked a specific target gap.

Usage:
    python scripts/trace_setup.py --date 2026-03-04 --sweep-time 10:36 --direction short --swept-price 25068
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ifvg_backtester.engine.data import load_1m_ny
from ifvg_backtester.engine.fvg import detect_fvgs, fvg_inverted_before
from ifvg_backtester.engine.simulator import (
    GAP_SIZE_MIN, GAP_SIZE_MAX, ELIGIBILITY_PROX_POINTS, MAX_GAPS_BETWEEN,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True)
    ap.add_argument("--sweep-time", required=True, help="HH:MM")
    ap.add_argument("--direction", required=True, choices=["short", "long"])
    ap.add_argument("--swept-price", type=float, required=True)
    args = ap.parse_args()

    sweep_ts = pd.Timestamp(f"{args.date} {args.sweep_time}", tz="America/New_York")
    next_day = (pd.Timestamp(args.date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    df = load_1m_ny("data/nq-front-month.ohlcv-1m.csv", start_date_ny=args.date, end_date_ny=next_day)
    sess_end = pd.Timestamp(f"{args.date} 11:00", tz="America/New_York")
    bars = df[df.index < sess_end]
    sess_open_ts = pd.Timestamp(f"{args.date} 09:30", tz="America/New_York")

    fvgs = detect_fvgs(bars)

    sweep_bar = bars.loc[sweep_ts]
    current_price = float(sweep_bar["close"])
    print(f"=== {args.date} {args.sweep_time}  {args.direction.upper()}  swept @ {args.swept_price:.2f}  close@sweep = {current_price:.2f} ===\n")

    if args.direction == "short":
        same_dir = [f for f in fvgs if f.direction == "bullish"]
        between = [f for f in same_dir if f.high <= current_price]
    else:
        same_dir = [f for f in fvgs if f.direction == "bearish"]
        between = [f for f in same_dir if f.low >= current_price]

    in_size = [f for f in between if GAP_SIZE_MIN <= f.size <= GAP_SIZE_MAX]
    timely = [f for f in in_size if f.formed_at <= sweep_ts]
    unfilled = [f for f in timely if not fvg_inverted_before(f, bars, sweep_ts)]

    def near_edge(g):
        return g.high if g.direction == "bullish" else g.low

    print(f"Same direction (opposite of sweep): {len(same_dir)}")
    print(f"  → between price and ... (right side): {len(between)}")
    print(f"  → size band {GAP_SIZE_MIN}-{GAP_SIZE_MAX}pt: {len(in_size)}")
    print(f"  → formed at/before sweep: {len(timely)}")
    print(f"  → not inverted before sweep: {len(unfilled)}")

    print(f"\nAll size-band, in-direction, unfilled gaps formed before sweep:")
    print(f"{'formed_at':<10} {'session':<8} {'low':>10} {'high':>10} {'size':>6} {'dist_lvl':>9} {'in_50pt':>8} {'in_RTH':>7}")
    for f in sorted(unfilled, key=lambda x: x.formed_at):
        d_level = abs(args.swept_price - near_edge(f))
        prox_ok = d_level <= ELIGIBILITY_PROX_POINTS
        in_rth = f.formed_at >= sess_open_ts
        sess_label = "RTH" if in_rth else "pre-mkt"
        print(f"{f.formed_at.strftime('%H:%M'):<10} {sess_label:<8} {f.low:>10.2f} {f.high:>10.2f} {f.size:>6.2f} {d_level:>9.2f} {'YES' if prox_ok else 'no':>8} {'YES' if in_rth else 'no':>7}")

    eligible = [f for f in unfilled if abs(args.swept_price - near_edge(f)) <= ELIGIBILITY_PROX_POINTS]
    print(f"\nCURRENT RULE eligible (proximity ≤{ELIGIBILITY_PROX_POINTS}pt): {len(eligible)}")
    for f in sorted(eligible, key=lambda x: x.formed_at):
        print(f"  {f.formed_at.strftime('%H:%M')} ({f.low:.0f},{f.high:.0f}) size={f.size:.1f}")

    if len(eligible) == 0:
        print(f"  → SKIP (0 gaps)")
    elif len(eligible) > MAX_GAPS_BETWEEN:
        print(f"  → SKIP (>{MAX_GAPS_BETWEEN} gaps)")
    elif len(eligible) == 1:
        print(f"  → TAKE the single gap")
    else:
        target = max(eligible, key=lambda g: g.formed_at)
        print(f"  → TAKE most_recent: {target.formed_at.strftime('%H:%M')} ({target.low:.0f},{target.high:.0f})")

    rth_eligible = [f for f in eligible if f.formed_at >= sess_open_ts]
    print(f"\nv2 PROPOSED (RTH-formed only, then most_recent regardless of count): {len(rth_eligible)}")
    for f in sorted(rth_eligible, key=lambda x: x.formed_at):
        print(f"  {f.formed_at.strftime('%H:%M')} ({f.low:.0f},{f.high:.0f}) size={f.size:.1f}")
    if rth_eligible:
        target = max(rth_eligible, key=lambda g: g.formed_at)
        print(f"  → TAKE most_recent RTH: {target.formed_at.strftime('%H:%M')} ({target.low:.0f},{target.high:.0f}) size={target.size:.1f}")
    else:
        print(f"  → SKIP")


if __name__ == "__main__":
    main()
