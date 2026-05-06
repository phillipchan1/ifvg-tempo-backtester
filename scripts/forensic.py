"""Per-day forensic: for every session, show every sweep event and what happened.

For each sweep:
  - Which level, which bar, sweep close vs pd_mid (bias)
  - Eligible gaps (or why none): size-band rejects, proximity rejects, fill rejects
  - If selected, the target gap and bar-by-bar over next 30 bars: did price
    ever close through it? What was the closest miss?
"""

from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ifvg_backtester.engine.data import load_1m_ny
from ifvg_backtester.engine.levels import daily_levels, london_session_levels
from ifvg_backtester.engine.fvg import detect_fvgs, fvg_filled_at
from ifvg_backtester.engine.simulator import (
    GAP_SIZE_MIN, GAP_SIZE_MAX, ELIGIBILITY_PROX_POINTS,
    MAX_GAPS_BETWEEN, TIME_TO_INVERSION_BARS,
    _bias_allows, _eligible_gaps_for_direction, _select_target_gap,
)

START, END = "2026-03-01", "2026-03-26"
df = load_1m_ny("data/nq-front-month.ohlcv-1m.csv",
                start_date_ny=(pd.Timestamp(START) - pd.Timedelta(days=4)).strftime("%Y-%m-%d"),
                end_date_ny=END)
pd_levels = daily_levels(df)
london = london_session_levels(df)

cutoff_start = pd.Timestamp(START, tz="America/New_York").normalize()
cutoff_end = pd.Timestamp(END, tz="America/New_York").normalize()


def explain_zero_gaps(direction, all_fvgs, current_price, swept_level_price, bars, as_of):
    """Why did this sweep produce 0 eligible gaps?"""
    if direction == "short":
        same_dir = [f for f in all_fvgs if f.direction == "bullish"]
    else:
        same_dir = [f for f in all_fvgs if f.direction == "bearish"]
    n0 = len(same_dir)

    # between price and swept level
    if direction == "short":
        in_range = [f for f in same_dir if f.high <= current_price]
    else:
        in_range = [f for f in same_dir if f.low >= current_price]
    n1 = len(in_range)

    # size band
    in_size = [f for f in in_range if GAP_SIZE_MIN <= f.size <= GAP_SIZE_MAX]
    n2 = len(in_size)

    # formed at/before sweep
    timely = [f for f in in_size if f.formed_at <= as_of]
    n3 = len(timely)

    # unfilled
    unfilled = [f for f in timely if not fvg_filled_at(f, bars, as_of)]
    n4 = len(unfilled)

    # proximity
    def near_edge(g): return g.high if g.direction == "bullish" else g.low
    in_prox = [f for f in unfilled if abs(swept_level_price - near_edge(f)) <= ELIGIBILITY_PROX_POINTS]
    n5 = len(in_prox)

    return f"all={n0} between_price_and_level={n1} size_7_20={n2} unfilled={n4} prox_50pt={n5}"


def trace_inversion(target, direction, sess_bars, sweep_idx, scan_bars=30):
    """Walk forward from sweep+1 for `scan_bars` bars. Report:
    - first close-through bar and r-multiple if it fired
    - closest miss (smallest gap to trigger close) if it didn't
    """
    n = len(sess_bars)
    fire_at = None
    closest_miss = None  # (bar_offset, distance)
    for j in range(sweep_idx + 1, min(sweep_idx + 1 + scan_bars, n)):
        close = sess_bars.iloc[j]["close"]
        ts = sess_bars.index[j]
        if direction == "short":
            triggered = close < target.low
            dist = close - target.low  # negative when triggered
        else:
            triggered = close > target.high
            dist = target.high - close  # negative when triggered
        if triggered and fire_at is None:
            fire_at = (j - sweep_idx, ts.strftime("%H:%M"), close)
        if closest_miss is None or dist < closest_miss[1]:
            closest_miss = (j - sweep_idx, dist, ts.strftime("%H:%M"), close)
    return fire_at, closest_miss


for date_ny, day_df in df.groupby(df.index.normalize()):
    if not (cutoff_start <= date_ny < cutoff_end): continue
    if date_ny.dayofweek >= 5: continue
    if date_ny not in pd_levels.index: continue
    row = pd_levels.loc[date_ny]
    if pd.isna(row["pdh"]): continue

    pdh, pdl, pd_mid = float(row["pdh"]), float(row["pdl"]), float(row["pd_mid"])
    lh = float(london.loc[date_ny, "london_high"]) if date_ny in london.index else None
    ll = float(london.loc[date_ny, "london_low"])  if date_ny in london.index else None

    sess_start = pd.Timestamp(f"{date_ny.date()} 09:30", tz="America/New_York")
    sess_end_pdmid = pd.Timestamp(f"{date_ny.date()} 11:00", tz="America/New_York")
    pre_session = day_df[day_df.index < sess_end_pdmid]
    sess_bars = day_df[(day_df.index >= sess_start) & (day_df.index < sess_end_pdmid)]
    if len(pre_session) < 3 or sess_bars.empty: continue

    all_fvgs = detect_fvgs(pre_session)
    high_lvls = [("PDH", pdh)] + ([("LondonHigh", lh)] if lh is not None else [])
    low_lvls  = [("PDL", pdl)] + ([("LondonLow", ll)]  if ll is not None else [])

    lh_str = f"{lh:.0f}" if lh else "n/a"
    ll_str = f"{ll:.0f}" if ll else "n/a"
    print(f"\n{'='*80}")
    print(f"{date_ny.strftime('%Y-%m-%d %a')}  PDH={pdh:.0f}  PDL={pdl:.0f}  PDmid={pd_mid:.0f}  "
          f"LH={lh_str}  LL={ll_str}")
    sess_high = sess_bars["high"].max()
    sess_low  = sess_bars["low"].min()
    print(f"  Session range 09:30–11:00: {sess_low:.0f} – {sess_high:.0f}  "
          f"(open={sess_bars.iloc[0]['open']:.0f}, close@11={sess_bars.iloc[-1]['close']:.0f})")

    state = {n: "fresh" for n, _ in high_lvls + low_lvls}

    for idx, (ts, bar) in enumerate(sess_bars.iterrows()):
        # Short side
        for name, lvl in high_lvls:
            if state[name] != "fresh" or bar["high"] < lvl: continue
            state[name] = "swept"
            t = ts.strftime("%H:%M")
            bias_ok = _bias_allows("short", bar["close"], pd_mid)
            label = f"  ↑ SHORT? {name}@{lvl:.0f} swept @{t}  close={bar['close']:.0f} (mid={pd_mid:.0f})"
            if not bias_ok:
                print(f"{label}  → BIAS-BLOCK (close ≤ mid)")
                continue
            cands = _eligible_gaps_for_direction("short", all_fvgs, bar["close"], lvl, pre_session, ts)
            if not cands:
                why = explain_zero_gaps("short", all_fvgs, bar["close"], lvl, pre_session, ts)
                print(f"{label}  → 0 GAPS [{why}]")
                continue
            if len(cands) > MAX_GAPS_BETWEEN:
                gap_descs = ", ".join([f"({g.low:.0f},{g.high:.0f},{g.size:.1f})" for g in cands])
                print(f"{label}  → SKIP {len(cands)} gaps: {gap_descs}")
                continue
            tgt = _select_target_gap(cands, lvl, ts, pre_session)
            print(f"{label}  → SELECT gap=({tgt.low:.0f},{tgt.high:.0f}) size={tgt.size:.1f}")
            fire, miss = trace_inversion(tgt, "short", sess_bars, idx, scan_bars=30)
            if fire:
                bar_off, fts, fclose = fire
                in_window = bar_off <= TIME_TO_INVERSION_BARS
                tag = "FIRED" if in_window else f"FIRED@{bar_off} (LATE — past {TIME_TO_INVERSION_BARS})"
                print(f"      → {tag}: close@{fts}={fclose:.0f} (target<{tgt.low:.0f})")
            else:
                bar_off, dist, mts, mclose = miss
                print(f"      → NO TRIGGER in 30 bars. Closest: bar+{bar_off} @{mts} close={mclose:.0f} "
                      f"(dist to target {tgt.low:.0f} = {dist:+.1f}pt)")

        # Long side
        for name, lvl in low_lvls:
            if state[name] != "fresh" or bar["low"] > lvl: continue
            state[name] = "swept"
            t = ts.strftime("%H:%M")
            bias_ok = _bias_allows("long", bar["close"], pd_mid)
            label = f"  ↓ LONG?  {name}@{lvl:.0f} swept @{t}  close={bar['close']:.0f} (mid={pd_mid:.0f})"
            if not bias_ok:
                print(f"{label}  → BIAS-BLOCK (close ≥ mid)")
                continue
            cands = _eligible_gaps_for_direction("long", all_fvgs, bar["close"], lvl, pre_session, ts)
            if not cands:
                why = explain_zero_gaps("long", all_fvgs, bar["close"], lvl, pre_session, ts)
                print(f"{label}  → 0 GAPS [{why}]")
                continue
            if len(cands) > MAX_GAPS_BETWEEN:
                gap_descs = ", ".join([f"({g.low:.0f},{g.high:.0f},{g.size:.1f})" for g in cands])
                print(f"{label}  → SKIP {len(cands)} gaps: {gap_descs}")
                continue
            tgt = _select_target_gap(cands, lvl, ts, pre_session)
            print(f"{label}  → SELECT gap=({tgt.low:.0f},{tgt.high:.0f}) size={tgt.size:.1f}")
            fire, miss = trace_inversion(tgt, "long", sess_bars, idx, scan_bars=30)
            if fire:
                bar_off, fts, fclose = fire
                in_window = bar_off <= TIME_TO_INVERSION_BARS
                tag = "FIRED" if in_window else f"FIRED@bar+{bar_off} (LATE — past {TIME_TO_INVERSION_BARS})"
                print(f"      → {tag}: close@{fts}={fclose:.0f} (target>{tgt.high:.0f})")
            else:
                bar_off, dist, mts, mclose = miss
                print(f"      → NO TRIGGER in 30 bars. Closest: bar+{bar_off} @{mts} close={mclose:.0f} "
                      f"(dist to target {tgt.high:.0f} = {dist:+.1f}pt)")
