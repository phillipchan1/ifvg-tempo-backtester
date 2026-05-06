"""Sweep-level construction.

Patterns adapted from fvgc-backtest/data/levels/build_session_levels.py and
build_htf_fvgs.py — same level definitions and time windows. We diverge from
their pipeline on ONE point: HTF FVG activation uses close-through inversion
(matches our IFVG semantic), not near-edge wick mitigation.

Public API:
    SweepLevel                               — dataclass for a sweep target
    build_session_levels(...)                — per-day session H/L levels
    resample_to_htf(df_1m, rule)             — 1m → HTF bars
    detect_htf_fvgs(htf_bars)                — FVGs with close-through inversion tracked
    active_htf_fvg_levels(fvgs, tf, as_of)   — currently-active HTF FVG sweep targets
    daily_pdmid(df_1m)                       — prior-day midpoint (R02 bias filter input)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import time as dtime
from typing import List, Optional, Sequence, Tuple

import pandas as pd


# Swing-point detection — used by all session H/L computations.
# A "swing high" is a confirmed pivot: a bar whose high is the strict max of
# the [i-N, i+N] window. Same for swing low. We resample 1m bars to SWING_TF
# before scanning so the swings reflect what's visible on a typical execution
# chart — a 1m N=2 fractal is usually noise.
SWING_TF = "15min"
SWING_N = 2


# ----------------------------------------------------------------------------
# Bias filter input — keep daily_pdmid for R02 ("prior day midpoint")
# ----------------------------------------------------------------------------

def daily_pdmid(df_1m: pd.DataFrame) -> pd.DataFrame:
    """Per-NY-calendar-day H/L/mid, shifted so each row holds the *prior* day's values."""
    df = df_1m.copy()
    df["date_ny"] = df.index.normalize()
    daily = df.groupby("date_ny").agg(high=("high", "max"), low=("low", "min"))
    daily["mid"] = (daily["high"] + daily["low"]) / 2
    return daily.shift(1).rename(columns={"high": "pdh", "low": "pdl", "mid": "pd_mid"})


# Backwards-compat — old name, same function.
daily_levels = daily_pdmid


# ----------------------------------------------------------------------------
# SweepLevel
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class SweepLevel:
    name: str        # unique id for logging
    group: str       # 'prev_day' | 'london' | 'asia' | 'overnight' | '6am' | 'nwog' | 'htf_fvg_1H' | ...
    side: str        # 'resistance' (price approached from below) | 'support' (from above)
    price: float


# ----------------------------------------------------------------------------
# Swing-point detection
# ----------------------------------------------------------------------------

def find_swings(bars: pd.DataFrame, n: int = SWING_N) -> List[dict]:
    """Confirmed swing highs and lows.

    A bar at index i is a swing high if its high is the strict max of the
    [i-n, i+n] window AND strictly greater than the i-1 high. Same for lows.
    Requires `bars` indexed chronologically with at least `2n+1` rows.
    """
    out: List[dict] = []
    if len(bars) < 2 * n + 1:
        return out
    arr_h = bars["high"].to_numpy()
    arr_l = bars["low"].to_numpy()
    times = bars.index
    for i in range(n, len(bars) - n):
        win_h = arr_h[i - n : i + n + 1]
        win_l = arr_l[i - n : i + n + 1]
        if arr_h[i] == win_h.max() and arr_h[i] > arr_h[i - 1]:
            out.append({"idx": i, "ts": times[i], "price": float(arr_h[i]), "type": "high"})
        if arr_l[i] == win_l.min() and arr_l[i] < arr_l[i - 1]:
            out.append({"idx": i, "ts": times[i], "price": float(arr_l[i]), "type": "low"})
    return out


def _resample_for_swings(bars_1m: pd.DataFrame, tf: str) -> pd.DataFrame:
    """1m → swing-detection TF (right-labelled, right-closed)."""
    if tf == "1min":
        return bars_1m
    return (
        bars_1m.resample(tf, label="right", closed="right")
        .agg(open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"))
        .dropna(how="all")
    )


def session_swing_high_low(
    bars_1m: pd.DataFrame,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    *,
    swing_tf: str = SWING_TF,
    n: int = SWING_N,
) -> Tuple[Optional[float], Optional[float]]:
    """Highest swing high and lowest swing low within `[start_ts, end_ts)`.

    Resamples 1m → `swing_tf` for swing detection. Returns (None, None) if the
    window has insufficient bars to confirm any swing. Falls back to raw extremes
    only if no swings are confirmed at all (rare).
    """
    window_1m = bars_1m[(bars_1m.index >= start_ts) & (bars_1m.index < end_ts)]
    if window_1m.empty:
        return None, None

    swing_bars = _resample_for_swings(window_1m, swing_tf)
    swings = find_swings(swing_bars, n=n)
    highs = [s["price"] for s in swings if s["type"] == "high"]
    lows = [s["price"] for s in swings if s["type"] == "low"]

    if not highs and not lows:
        # No confirmed swings (very short window or perfectly trending). Fall
        # back to raw extreme so the level isn't silently dropped.
        return float(window_1m["high"].max()), float(window_1m["low"].min())

    swing_high = max(highs) if highs else float(window_1m["high"].max())
    swing_low = min(lows) if lows else float(window_1m["low"].min())
    return swing_high, swing_low


# ----------------------------------------------------------------------------
# Session levels (one call per session date)
# ----------------------------------------------------------------------------

def _safe_concat(parts: Sequence[pd.DataFrame]) -> pd.DataFrame:
    nonempty = [p for p in parts if p is not None and not p.empty]
    if not nonempty:
        return pd.DataFrame()
    return pd.concat(nonempty)


def _swing_levels_for_window(
    bars_1m: pd.DataFrame,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    high_name: str,
    low_name: str,
    group: str,
) -> List[SweepLevel]:
    """Helper: return [high, low] SweepLevel pair for a session window using
    swing-point detection. Empty list if window has no bars."""
    high, low = session_swing_high_low(bars_1m, start_ts, end_ts)
    out: List[SweepLevel] = []
    if high is not None:
        out.append(SweepLevel(high_name, group, "resistance", float(high)))
    if low is not None:
        out.append(SweepLevel(low_name, group, "support", float(low)))
    return out


def build_session_levels(
    date_ny: pd.Timestamp,
    day_df: pd.DataFrame,
    prior_day_df: Optional[pd.DataFrame],
    friday_close: Optional[float] = None,
    monday_open: Optional[float] = None,
) -> List[SweepLevel]:
    """All session-based H/L sweep levels available at the start of `date_ny`.

    Highs and lows are SWING POINTS (confirmed pivots on a 5m N=2 fractal),
    not raw session extremes. This is intentional — a single 1m wick is not a
    real liquidity pool. See `session_swing_high_low`.

    Window definitions:
        prev_day:  prior trading day RTH 09:30–16:00 NY
        london:    02:00–08:00 NY (current session date)
        asia:      19:00 prior calendar day → 02:00 NY today
        overnight: 18:00 prior calendar day → 09:29 NY today
        6am:       04:00–08:00 NY (current session date)
        nwog:      Friday's RTH close ↔ Monday's RTH open (Mondays only — no swings)
    """
    levels: List[SweepLevel] = []
    date_only = date_ny.date()

    # We need the union of prior_day_df and day_df to compute multi-day windows
    # (asia, overnight) without losing any bars at the seam.
    pieces: list[pd.DataFrame] = []
    if prior_day_df is not None and not prior_day_df.empty:
        pieces.append(prior_day_df)
    pieces.append(day_df)
    bars_1m = _safe_concat(pieces).sort_index()
    if bars_1m.empty:
        return levels

    # prev_day RTH 09:30–16:00 (PRIOR trading day)
    if prior_day_df is not None and not prior_day_df.empty:
        prior_date = prior_day_df.index[0].date()
        pd_start = pd.Timestamp(f"{prior_date} 09:30", tz="America/New_York")
        pd_end = pd.Timestamp(f"{prior_date} 16:00", tz="America/New_York")
        levels.extend(_swing_levels_for_window(
            bars_1m, pd_start, pd_end, "prev_day_high", "prev_day_low", "prev_day"
        ))

    # london 02:00–08:00 NY (today)
    levels.extend(_swing_levels_for_window(
        bars_1m,
        pd.Timestamp(f"{date_only} 02:00", tz="America/New_York"),
        pd.Timestamp(f"{date_only} 08:00", tz="America/New_York"),
        "london_high", "london_low", "london",
    ))

    # asia 19:00 prev → 02:00 today
    if prior_day_df is not None and not prior_day_df.empty:
        prior_date = prior_day_df.index[0].date()
        levels.extend(_swing_levels_for_window(
            bars_1m,
            pd.Timestamp(f"{prior_date} 19:00", tz="America/New_York"),
            pd.Timestamp(f"{date_only} 02:00", tz="America/New_York"),
            "asia_high", "asia_low", "asia",
        ))

    # overnight 18:00 prev → 09:30 today
    if prior_day_df is not None and not prior_day_df.empty:
        prior_date = prior_day_df.index[0].date()
        levels.extend(_swing_levels_for_window(
            bars_1m,
            pd.Timestamp(f"{prior_date} 18:00", tz="America/New_York"),
            pd.Timestamp(f"{date_only} 09:30", tz="America/New_York"),
            "overnight_high", "overnight_low", "overnight",
        ))

    # 6am 04:00–08:00 NY (today)
    levels.extend(_swing_levels_for_window(
        bars_1m,
        pd.Timestamp(f"{date_only} 04:00", tz="America/New_York"),
        pd.Timestamp(f"{date_only} 08:00", tz="America/New_York"),
        "6am_high", "6am_low", "6am",
    ))

    # NWOG (Mondays only) — two prices, not swings
    if date_ny.dayofweek == 0 and friday_close is not None and monday_open is not None:
        levels.append(SweepLevel("nwog_high", "nwog", "resistance", float(max(friday_close, monday_open))))
        levels.append(SweepLevel("nwog_low",  "nwog", "support",    float(min(friday_close, monday_open))))

    return levels


# ----------------------------------------------------------------------------
# HTF FVG levels — detect on resampled bars, track close-through inversion
# ----------------------------------------------------------------------------

def resample_to_htf(df_1m: pd.DataFrame, rule: str) -> pd.DataFrame:
    """1m → HTF OHLC. Right-labelled, right-closed (matches fvgc-backtest)."""
    g = df_1m.resample(rule, label="right", closed="right").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
    ).dropna(how="all")
    return g


def detect_htf_fvgs(htf_bars: pd.DataFrame) -> list[dict]:
    """Detect 3-candle FVGs on HTF bars; track close-through inversion forward.

    Returns list of dicts:
        direction: 'bullish' | 'bearish'
        top: float            top edge of the gap
        bottom: float         bottom edge
        formed_at: Timestamp  close-time of the 3rd candle
        inverted_at: Timestamp | None   first HTF bar that closed through

    Inversion (matches our IFVG semantic):
        bullish FVG inverted when an HTF bar CLOSES <= bottom
        bearish FVG inverted when an HTF bar CLOSES >= top
    """
    if len(htf_bars) < 3:
        return []

    arr_h = htf_bars["high"].to_numpy()
    arr_l = htf_bars["low"].to_numpy()
    arr_c = htf_bars["close"].to_numpy()
    times = htf_bars.index

    fvgs: list[dict] = []
    pending: list[dict] = []  # not yet inverted

    for i in range(len(htf_bars)):
        c = float(arr_c[i])
        # Process pending — check if this bar's close inverts any.
        still: list[dict] = []
        for f in pending:
            if f["direction"] == "bullish" and c <= f["bottom"]:
                f["inverted_at"] = times[i]
            elif f["direction"] == "bearish" and c >= f["top"]:
                f["inverted_at"] = times[i]
            else:
                still.append(f)
        pending = still

        # Detect new FVG using bars [i-2, i-1, i].
        if i >= 2:
            c1_h = float(arr_h[i - 2])
            c1_l = float(arr_l[i - 2])
            c3_h = float(arr_h[i])
            c3_l = float(arr_l[i])
            if c1_h < c3_l:
                f = {
                    "direction": "bullish",
                    "top": c3_l,
                    "bottom": c1_h,
                    "formed_at": times[i],
                    "inverted_at": None,
                }
                fvgs.append(f)
                pending.append(f)
            elif c1_l > c3_h:
                f = {
                    "direction": "bearish",
                    "top": c1_l,
                    "bottom": c3_h,
                    "formed_at": times[i],
                    "inverted_at": None,
                }
                fvgs.append(f)
                pending.append(f)

    return fvgs


def active_htf_fvg_levels(
    fvgs: list[dict],
    tf_label: str,
    as_of: pd.Timestamp,
    proximity_pts: float | None = None,
    reference_price: float | None = None,
) -> List[SweepLevel]:
    """Convert HTF FVGs that are active (formed and not yet inverted) at `as_of`
    into SweepLevel objects.

    If `proximity_pts` and `reference_price` are given, filter to FVGs whose near
    edge is within `proximity_pts` of `reference_price` — keeps the sweep target
    set scoped to "could plausibly be reached today."

    Bullish FVG (support): sweep level = top  (near edge from above)
    Bearish FVG (resistance): sweep level = bottom (near edge from below)
    """
    out: List[SweepLevel] = []
    for f in fvgs:
        if f["formed_at"] >= as_of:
            continue
        inv = f.get("inverted_at")
        if inv is not None and inv < as_of:
            continue
        if f["direction"] == "bullish":
            near_edge = f["top"]
            side = "support"
        else:
            near_edge = f["bottom"]
            side = "resistance"
        if proximity_pts is not None and reference_price is not None:
            if abs(near_edge - reference_price) > proximity_pts:
                continue
        formed_str = f["formed_at"].strftime("%Y%m%d_%H%M")
        name = f"htf_fvg_{tf_label}_{f['direction']}_{formed_str}"
        out.append(SweepLevel(name=name, group=f"htf_fvg_{tf_label}", side=side, price=float(near_edge)))
    return out


# ----------------------------------------------------------------------------
# Compatibility shim — the old london_session_levels per-day frame, kept so
# anything still calling it doesn't break. New code should use build_session_levels.
# ----------------------------------------------------------------------------

def london_session_levels(df_1m: pd.DataFrame, start_hour: int = 2, end_hour: int = 8) -> pd.DataFrame:
    mask = (df_1m.index.hour >= start_hour) & (df_1m.index.hour < end_hour)
    london = df_1m[mask].copy()
    london["date_ny"] = london.index.normalize()
    return london.groupby("date_ny").agg(
        london_high=("high", "max"),
        london_low=("low", "min"),
    )
