"""Plain dataclasses used across the engine."""

from dataclasses import dataclass, field
from typing import Optional
import pandas as pd


@dataclass
class FVG:
    direction: str  # 'bullish' or 'bearish'
    formed_at: pd.Timestamp  # close time of the 3rd candle (i)
    high: float  # top edge of gap
    low: float   # bottom edge of gap
    size: float = 0.0
    filled: bool = False
    fill_at: Optional[pd.Timestamp] = None

    def __post_init__(self):
        self.size = self.high - self.low


@dataclass
class Trade:
    session_date: pd.Timestamp
    direction: str  # 'short' | 'long'
    swept_level_name: str
    swept_level_price: float
    sweep_ts: pd.Timestamp
    target_gap_high: float
    target_gap_low: float
    target_gap_size: float
    target_gap_formed_at: pd.Timestamp
    bars_to_inversion: int
    entry_ts: pd.Timestamp
    entry_price: float
    stop_price: float
    tp_price: float
    risk_points: float
    exit_ts: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    outcome: str = ""  # 'win' | 'loss' | 'eod_close'
    pnl_points: float = 0.0
    r_multiple: float = 0.0
