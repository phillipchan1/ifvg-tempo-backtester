"""Funnel diagnostic — reflects current simulator semantics (post-iteration #1)."""

from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ifvg_backtester.engine.data import load_1m_ny
from ifvg_backtester.engine.levels import daily_levels, london_session_levels
from ifvg_backtester.engine.fvg import detect_fvgs
from ifvg_backtester.engine.simulator import (
    SESSION_START, SESSION_END, TIME_TO_INVERSION_BARS,
    _bias_allows, _eligible_gaps_for_direction, _select_target_gap,
)  # type: ignore

START, END = "2026-03-01", "2026-03-26"
df = load_1m_ny("data/nq-front-month.ohlcv-1m.csv",
                start_date_ny=(pd.Timestamp(START) - pd.Timedelta(days=4)).strftime("%Y-%m-%d"),
                end_date_ny=END)
pd_levels = daily_levels(df)
london = london_session_levels(df)

stats = {
    "sessions": 0,
    "level_swept": 0,
    "bias_pass": 0,
    "had_eligible_gaps": 0,
    "skip_too_many_gaps": 0,
    "skip_zero_gaps": 0,
    "selected": 0,
    "trade_fired": 0,
    "expired_no_trigger": 0,
}

cutoff_start = pd.Timestamp(START, tz="America/New_York").normalize()
cutoff_end = pd.Timestamp(END, tz="America/New_York").normalize()

per_day = []

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
    sess_end = pd.Timestamp(f"{date_ny.date()} 11:00", tz="America/New_York")
    pre_session = day_df[day_df.index < sess_end]
    sess_bars = day_df[(day_df.index >= sess_start) & (day_df.index < sess_end)]
    if len(pre_session) < 3 or sess_bars.empty: continue

    all_fvgs = detect_fvgs(pre_session)
    high_lvls = [("PDH", pdh)] + ([("LondonHigh", lh)] if lh is not None else [])
    low_lvls  = [("PDL", pdl)] + ([("LondonLow", ll)]  if ll is not None else [])

    stats["sessions"] += 1
    notes = []
    state = {n: "fresh" for n, _ in high_lvls + low_lvls}

    def _check_inversion_window(target, direction, sweep_idx, day_df, sess_bars):
        for j in range(sweep_idx + 1, min(sweep_idx + 1 + TIME_TO_INVERSION_BARS, len(sess_bars))):
            close = sess_bars.iloc[j]["close"]
            if direction == "short" and close < target.low:
                return j - sweep_idx
            if direction == "long" and close > target.high:
                return j - sweep_idx
        return None

    for idx, (ts, bar) in enumerate(sess_bars.iterrows()):
        for name, lvl in high_lvls:
            if state[name] != "fresh": continue
            if bar["high"] < lvl: continue
            state[name] = "swept"
            stats["level_swept"] += 1
            if not _bias_allows("short", bar["close"], pd_mid):
                notes.append(f"{name}@{lvl:.0f} swept@{ts.strftime('%H:%M')} BIAS-block")
                continue
            stats["bias_pass"] += 1
            cands = _eligible_gaps_for_direction("short", all_fvgs, bar["close"], lvl, pre_session, ts)
            if len(cands) == 0:
                stats["skip_zero_gaps"] += 1
                notes.append(f"{name}@{lvl:.0f} swept@{ts.strftime('%H:%M')} 0 gaps")
                continue
            stats["had_eligible_gaps"] += 1
            if len(cands) > 2:
                stats["skip_too_many_gaps"] += 1
                notes.append(f"{name}@{lvl:.0f} swept@{ts.strftime('%H:%M')} {len(cands)} gaps (skip)")
                continue
            tgt = _select_target_gap(cands, lvl, ts, pre_session)
            stats["selected"] += 1
            inv = _check_inversion_window(tgt, "short", idx, day_df, sess_bars)
            if inv is None:
                stats["expired_no_trigger"] += 1
                notes.append(f"SHORT {name}@{lvl:.0f} sel@{ts.strftime('%H:%M')} gap=({tgt.low:.0f},{tgt.high:.0f}) EXPIRED")
            else:
                stats["trade_fired"] += 1
                notes.append(f"SHORT {name}@{lvl:.0f} sel@{ts.strftime('%H:%M')} gap=({tgt.low:.0f},{tgt.high:.0f}) FIRED@bar+{inv}")

        for name, lvl in low_lvls:
            if state[name] != "fresh": continue
            if bar["low"] > lvl: continue
            state[name] = "swept"
            stats["level_swept"] += 1
            if not _bias_allows("long", bar["close"], pd_mid):
                notes.append(f"{name}@{lvl:.0f} swept@{ts.strftime('%H:%M')} BIAS-block")
                continue
            stats["bias_pass"] += 1
            cands = _eligible_gaps_for_direction("long", all_fvgs, bar["close"], lvl, pre_session, ts)
            if len(cands) == 0:
                stats["skip_zero_gaps"] += 1
                notes.append(f"{name}@{lvl:.0f} swept@{ts.strftime('%H:%M')} 0 gaps")
                continue
            stats["had_eligible_gaps"] += 1
            if len(cands) > 2:
                stats["skip_too_many_gaps"] += 1
                notes.append(f"{name}@{lvl:.0f} swept@{ts.strftime('%H:%M')} {len(cands)} gaps (skip)")
                continue
            tgt = _select_target_gap(cands, lvl, ts, pre_session)
            stats["selected"] += 1
            inv = _check_inversion_window(tgt, "long", idx, day_df, sess_bars)
            if inv is None:
                stats["expired_no_trigger"] += 1
                notes.append(f"LONG {name}@{lvl:.0f} sel@{ts.strftime('%H:%M')} gap=({tgt.low:.0f},{tgt.high:.0f}) EXPIRED")
            else:
                stats["trade_fired"] += 1
                notes.append(f"LONG {name}@{lvl:.0f} sel@{ts.strftime('%H:%M')} gap=({tgt.low:.0f},{tgt.high:.0f}) FIRED@bar+{inv}")

    if notes:
        per_day.append((date_ny.strftime("%Y-%m-%d"), notes))

print("=== FUNNEL ===")
for k, v in stats.items():
    print(f"  {k:30s} {v}")

print("\n=== PER DAY ===")
for d, notes in per_day:
    print(f"\n{d}:")
    for n in notes:
        print(f"  {n}")
