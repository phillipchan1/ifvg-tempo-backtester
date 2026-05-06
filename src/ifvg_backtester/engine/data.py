"""1m bar loading. Index = NY-tz timestamps."""

from pathlib import Path
import pandas as pd


def load_1m_ny(
    csv_path: str | Path,
    start_date_ny: str | None = None,
    end_date_ny: str | None = None,
) -> pd.DataFrame:
    """Load 1m OHLCV CSV and return a DataFrame indexed by NY-time bar-open timestamps.

    Date filters are inclusive of start, exclusive of end (YYYY-MM-DD strings).
    """
    df = pd.read_csv(csv_path, usecols=["timestamp_ny", "open", "high", "low", "close", "volume"])
    df["ts_ny"] = pd.to_datetime(df["timestamp_ny"], utc=True).dt.tz_convert("America/New_York")
    df = df.drop(columns=["timestamp_ny"]).set_index("ts_ny").sort_index()

    if start_date_ny:
        df = df[df.index >= pd.Timestamp(start_date_ny, tz="America/New_York")]
    if end_date_ny:
        df = df[df.index < pd.Timestamp(end_date_ny, tz="America/New_York")]
    return df
