"""Bar-by-bar simulator implementing v001 defaults.

LOCKED v001 defaults (no permutation in this run):
    R01 session window: 09:30–11:00 ET
    R02 bias reference: prior-day mid
    R03 liquidity targets: London H/L + PDH/PDL
    R04 sweep validity: wick_touch (high reaches level AND close < level for highs;
                        low reaches level AND close > level for lows)
    R05 FVG: wick FVG
    R06 gap size: 7–20 points
    R07 target gap selection: max_gaps=2, prox 3 candles / 15 pts, target=most_recent
    R08 time-to-inversion: 5 candles
    R09 inversion trigger: close_through
    R10 entry: market on close of inversion candle
    R11 stop: target_gap.{far edge} ± 5pt buffer
    R12 take profit: full at 1R
    R13 re-entry: 1 trade per session per direction
    R14 news filter: OFF for v001 smoke test (no calendar wired in yet)
    R15 timeframe: 1m
"""

from dataclasses import dataclass
from typing import List, Optional
import pandas as pd

from .models import FVG, Trade
from .fvg import detect_fvgs, fvg_filled_at


# v002 defaults --- inline so the simulator stays readable.
SESSION_START = (9, 30)
SESSION_END = (11, 0)
GAP_SIZE_MIN, GAP_SIZE_MAX = 7.0, 20.0
MAX_GAPS_BETWEEN = 2
# Eligibility: gap's near edge within X pts of swept level. The inversion target
# must be part of the recent rally context — gaps far from the swept level
# aren't structural speed bumps.
ELIGIBILITY_PROX_POINTS = 50.0
# v002: only inversion-target FVGs formed during the RTH session are eligible.
# Pre-market gaps reflect overnight structure, not the killzone reversal context.
RTH_ONLY_INVERSION_TARGETS = True
# v002: allow STRADDLING gaps (bullish FVG with top above current price as long
# as bottom is below). Close-through trigger still fires on close < gap.low.
ALLOW_STRADDLING_GAPS = True
TIME_TO_INVERSION_BARS = 10
STOP_BUFFER_PTS = 5.0
RR_TAKE_PROFIT = 1.0
MAX_TRADES_PER_SESSION_PER_DIR = 1


@dataclass
class _ActiveSetup:
    direction: str  # 'short' | 'long'
    sweep_ts: pd.Timestamp
    sweep_bar_idx: int  # session-bar index of sweep
    swept_level_name: str
    swept_level_price: float
    target: FVG


def _bias_allows(direction: str, price: float, pd_mid: float) -> bool:
    """Above mid → premium → shorts only. Below mid → discount → longs only."""
    if direction == "short":
        return price > pd_mid
    return price < pd_mid


def _select_target_gap(
    candidates: List[FVG],
    swept_level_price: float,
    sweep_ts: pd.Timestamp,
    bars: pd.DataFrame,
) -> Optional[FVG]:
    """Apply R07 selection. Proximity, RTH window, and inversion-not-fired-yet
    already enforced at eligibility time.

    v002 selection: always pick the most recent eligible gap (closest to the sweep
    in time), as long as 1 ≤ n ≤ MAX_GAPS_BETWEEN. Skip otherwise.
    """
    n = len(candidates)
    if n == 0 or n > MAX_GAPS_BETWEEN:
        return None
    return max(candidates, key=lambda g: g.formed_at)


def _eligible_gaps_for_direction(
    direction: str,
    all_fvgs: List[FVG],
    current_price: float,
    swept_level_price: float,
    bars: pd.DataFrame,
    as_of: pd.Timestamp,
) -> List[FVG]:
    """Opposite-direction unfilled FVGs sitting between current price and the swept level."""
    if direction == "short":
        # need a bullish FVG below current price (so close-through fires on close < gap.low)
        candidates = [f for f in all_fvgs if f.direction == "bullish"]
        if ALLOW_STRADDLING_GAPS:
            # v002: gap eligible if its BOTTOM is below close (top can straddle).
            candidates = [f for f in candidates if f.low < current_price]
        else:
            # v001 strict-below
            candidates = [f for f in candidates if f.high <= current_price]
    else:
        candidates = [f for f in all_fvgs if f.direction == "bearish"]
        if ALLOW_STRADDLING_GAPS:
            # v002: gap eligible if its TOP is above close (bottom can straddle).
            candidates = [f for f in candidates if f.high > current_price]
        else:
            candidates = [f for f in candidates if f.low >= current_price]

    # gap must be within R06 size band
    candidates = [f for f in candidates if GAP_SIZE_MIN <= f.size <= GAP_SIZE_MAX]
    # must be formed at/before sweep
    candidates = [f for f in candidates if f.formed_at <= as_of]
    # v002: RTH-only — gap must form at/after 09:30 NY on the session date.
    # Pre-market gaps reflect overnight structure, not the killzone reversal context.
    if RTH_ONLY_INVERSION_TARGETS:
        sess_start = pd.Timestamp(
            f"{as_of.tz_convert('America/New_York').date()} "
            f"{SESSION_START[0]:02d}:{SESSION_START[1]:02d}",
            tz="America/New_York",
        )
        candidates = [f for f in candidates if f.formed_at >= sess_start]
    # must be unfilled at sweep time (this implicitly enforces "speed bump" semantics —
    # a gap that's already been filled by the rally is not a remaining bump)
    candidates = [f for f in candidates if not fvg_filled_at(f, bars, as_of)]
    # eligibility: proximity to swept level — gap's near edge within X points
    def _near_edge(g: FVG) -> float:
        return g.high if g.direction == "bullish" else g.low
    candidates = [
        f for f in candidates
        if abs(swept_level_price - _near_edge(f)) <= ELIGIBILITY_PROX_POINTS
    ]
    return candidates


def _simulate_exit(
    bars_after: pd.DataFrame,
    direction: str,
    entry: float,
    stop: float,
    tp: float,
) -> tuple[Optional[pd.Timestamp], float, str]:
    """Walk forward bar-by-bar after entry. Returns (exit_ts, exit_price, outcome).

    Conservative tie-break: if both stop and TP would hit on the same bar, assume STOP first.
    """
    for ts, bar in bars_after.iterrows():
        hit_stop = bar["high"] >= stop if direction == "short" else bar["low"] <= stop
        hit_tp = bar["low"] <= tp if direction == "short" else bar["high"] >= tp
        if hit_stop and hit_tp:
            return ts, stop, "loss"
        if hit_stop:
            return ts, stop, "loss"
        if hit_tp:
            return ts, tp, "win"
    # never resolved within available bars → close at last bar
    if not bars_after.empty:
        last = bars_after.iloc[-1]
        return bars_after.index[-1], float(last["close"]), "eod_close"
    return None, entry, "eod_close"


def run_session(
    day_df: pd.DataFrame,
    date_ny: pd.Timestamp,
    pd_mid: float,
    sweep_levels: list,  # list[SweepLevel] — see engine.levels.SweepLevel
) -> List[Trade]:
    """Run one session day. Return list of Trade records (0+).

    `sweep_levels` is the unified list of qualifying sweep targets for this date —
    session H/L (PDH/PDL, London, Asia, …), HTF FVG zones, etc. Each carries a
    `side` of 'resistance' (sweep from below → short candidate) or 'support'
    (sweep from above → long candidate).
    """
    sess_start = pd.Timestamp(
        f"{date_ny.date()} {SESSION_START[0]:02d}:{SESSION_START[1]:02d}",
        tz="America/New_York",
    )
    sess_end = pd.Timestamp(
        f"{date_ny.date()} {SESSION_END[0]:02d}:{SESSION_END[1]:02d}",
        tz="America/New_York",
    )

    # Build FVGs over pre-market AND session up to sess_end. We need pre-market
    # bars in the detection sweep so that 3-candle FVG patterns spanning the
    # 09:28→09:30 boundary are correctly identified. The RTH-only inversion-target
    # filter (R05 v002) is applied AFTER detection: only gaps with formed_at >=
    # sess_start are kept as inversion candidates.
    pre_and_session_bars = day_df[day_df.index < sess_end]
    if len(pre_and_session_bars) < 3:
        return []
    all_fvgs = detect_fvgs(pre_and_session_bars)
    if RTH_ONLY_INVERSION_TARGETS:
        all_fvgs = [f for f in all_fvgs if f.formed_at >= sess_start]

    # session bars only — we walk these for sweep + trigger logic
    sess_bars = day_df[(day_df.index >= sess_start) & (day_df.index < sess_end)]
    if sess_bars.empty:
        return []

    # Split sweep targets into resistance (sweep-from-below = short candidate) and
    # support (sweep-from-above = long candidate). Each tuple is (name, price).
    #
    # Filter to levels on the CORRECT side of the session-open price. A "resistance"
    # level below the open is already broken (price gapped through it overnight) —
    # it can't be swept in the ICT sense. Same for "support" levels above the open.
    # This is a lightweight stand-in for fvgc-backtest's swept_pre_rth tracking.
    sess_open = float(sess_bars.iloc[0]["open"])
    high_levels: List[tuple[str, float]] = [
        (L.name, float(L.price))
        for L in sweep_levels
        if L.side == "resistance" and L.price > sess_open
    ]
    low_levels: List[tuple[str, float]] = [
        (L.name, float(L.price))
        for L in sweep_levels
        if L.side == "support" and L.price < sess_open
    ]

    # Per-level lifecycle: each level can be swept once per session (first-touch only).
    # Sweep = ANY first-touch (wick or close-through — same to us). The inversion
    # mechanic does the real reversal filtering, not the sweep candle itself.
    level_state: dict[str, str] = {name: "fresh" for name, _ in high_levels + low_levels}
    active: dict[str, Optional[_ActiveSetup]] = {"short": None, "long": None}
    trades_this_session = {"short": 0, "long": 0}
    trades: List[Trade] = []

    for idx, (ts, bar) in enumerate(sess_bars.iterrows()):
        # 1) Detect sweeps THIS bar — first-touch only, no wick/close distinction.
        for name, lvl in high_levels:
            if level_state[name] != "fresh" or active["short"] is not None:
                continue
            if bar["high"] < lvl:
                continue  # level not reached this bar
            level_state[name] = "swept"
            # bias still required: shorts only above prior-day mid
            if not _bias_allows("short", bar["close"], pd_mid):
                continue
            cands = _eligible_gaps_for_direction(
                "short", all_fvgs, bar["close"], lvl, pre_and_session_bars, ts
            )
            target = _select_target_gap(cands, lvl, ts, pre_and_session_bars)
            if target is not None and trades_this_session["short"] < MAX_TRADES_PER_SESSION_PER_DIR:
                active["short"] = _ActiveSetup("short", ts, idx, name, lvl, target)

        for name, lvl in low_levels:
            if level_state[name] != "fresh" or active["long"] is not None:
                continue
            if bar["low"] > lvl:
                continue
            level_state[name] = "swept"
            if not _bias_allows("long", bar["close"], pd_mid):
                continue
            cands = _eligible_gaps_for_direction(
                "long", all_fvgs, bar["close"], lvl, pre_and_session_bars, ts
            )
            target = _select_target_gap(cands, lvl, ts, pre_and_session_bars)
            if target is not None and trades_this_session["long"] < MAX_TRADES_PER_SESSION_PER_DIR:
                active["long"] = _ActiveSetup("long", ts, idx, name, lvl, target)

        # 2) Check inversion trigger for each active setup (skip the sweep bar itself)
        for direction in ("short", "long"):
            setup = active[direction]
            if setup is None or ts == setup.sweep_ts:
                continue
            bars_since = idx - setup.sweep_bar_idx
            if bars_since > TIME_TO_INVERSION_BARS:
                active[direction] = None
                continue

            tg = setup.target
            triggered = (
                bar["close"] < tg.low if direction == "short" else bar["close"] > tg.high
            )
            if not triggered:
                continue

            # ENTRY
            entry = float(bar["close"])
            if direction == "short":
                stop = tg.high + STOP_BUFFER_PTS
                risk = stop - entry
                tp = entry - RR_TAKE_PROFIT * risk
            else:
                stop = tg.low - STOP_BUFFER_PTS
                risk = entry - stop
                tp = entry + RR_TAKE_PROFIT * risk

            # exit simulation: from next bar through end-of-day
            after = day_df[day_df.index > ts]
            exit_ts, exit_price, outcome = _simulate_exit(after, direction, entry, stop, tp)

            if direction == "short":
                pnl = entry - exit_price
            else:
                pnl = exit_price - entry
            r_mult = pnl / risk if risk > 0 else 0.0

            trades.append(
                Trade(
                    session_date=date_ny,
                    direction=direction,
                    swept_level_name=setup.swept_level_name,
                    swept_level_price=setup.swept_level_price,
                    sweep_ts=setup.sweep_ts,
                    target_gap_high=tg.high,
                    target_gap_low=tg.low,
                    target_gap_size=tg.size,
                    target_gap_formed_at=tg.formed_at,
                    bars_to_inversion=bars_since,
                    entry_ts=ts,
                    entry_price=entry,
                    stop_price=stop,
                    tp_price=tp,
                    risk_points=risk,
                    exit_ts=exit_ts,
                    exit_price=exit_price,
                    outcome=outcome,
                    pnl_points=pnl,
                    r_multiple=r_mult,
                )
            )
            trades_this_session[direction] += 1
            active[direction] = None

    return trades
