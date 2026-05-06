"""FVG detection (R05 wick-FVG default) and fill-state tracking."""

from typing import List
import pandas as pd

from .models import FVG


def detect_fvgs(bars: pd.DataFrame) -> List[FVG]:
    """Walk 3-candle windows over `bars`. Return all wick-FVGs in time order.

    Bullish FVG: bars[i-2].high < bars[i].low → gap = (bars[i-2].high, bars[i].low).
    Bearish FVG: bars[i-2].low  > bars[i].high → gap = (bars[i].high,  bars[i-2].low).

    `formed_at` is the timestamp of the 3rd candle (i) — the candle that confirms the gap.
    """
    out: List[FVG] = []
    if len(bars) < 3:
        return out

    arr = bars[["high", "low"]].to_numpy()
    times = bars.index
    for i in range(2, len(arr)):
        c2_h, c2_l = arr[i - 2]
        ci_h, ci_l = arr[i]
        if c2_h < ci_l:
            out.append(FVG("bullish", times[i], high=ci_l, low=c2_h))
        elif c2_l > ci_h:
            out.append(FVG("bearish", times[i], high=c2_l, low=ci_h))
    return out


def fvg_inverted_before(fvg: FVG, bars: pd.DataFrame, as_of: pd.Timestamp) -> bool:
    """Has this FVG already been INVERTED (closed-through) on/before `as_of`?

    Inversion semantics for the IFVG model: a gap is "used" when a candle closes
    through it (the same close-through that would fire R09's entry trigger). A
    gap merely wicked into (mitigated) but not closed through is STILL a valid
    inversion candidate — the close-through trigger hasn't fired yet.

    - Bullish FVG: inverted when a later candle CLOSES below fvg.low.
    - Bearish FVG: inverted when a later candle CLOSES above fvg.high.

    This differs from the sister project's mitigation logic (which is correct for
    a continuation model that wants fresh untouched gaps as draws). For IFVG we
    only invalidate on prior inversion.
    """
    later = bars[(bars.index > fvg.formed_at) & (bars.index <= as_of)]
    if later.empty:
        return False
    if fvg.direction == "bullish":
        return bool((later["close"] <= fvg.low).any())
    else:
        return bool((later["close"] >= fvg.high).any())


# Backwards-compat alias — call sites use the older name.
fvg_filled_at = fvg_inverted_before
