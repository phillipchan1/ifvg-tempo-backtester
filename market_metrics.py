#!/usr/bin/env python3
"""
Market Metrics Calculator

Processes OHLCV data to calculate daily volume and range metrics
for correlation analysis with trade performance.
"""

import pandas as pd
from datetime import datetime, time
from pathlib import Path
from typing import Dict, Optional, Tuple
import pytz


# Timezone definitions
UTC = pytz.UTC
ET = pytz.timezone('America/New_York')  # Handles DST automatically

# NY Session times (ET)
NY_OPEN_TIME = time(9, 30)  # 9:30 AM ET
NY_CLOSE_TIME = time(16, 0)  # 4:00 PM ET

# Time periods for metrics
PERIOD_45MIN_END = time(10, 15)  # 10:15 AM ET
PERIOD_90MIN_END = time(11, 0)   # 11:00 AM ET


def normalize_symbol(symbol: str) -> str:
    """
    Normalize futures symbol to underlying asset.
    
    Examples:
    - NQZ0, NQH1, NQM1 → NQ
    - ESZ0, ESH1 → ES
    - GCZ0, GCH1 → GC
    
    Args:
        symbol: Futures contract symbol
        
    Returns:
        Normalized asset symbol (NQ, ES, GC) or original if no match
    """
    symbol_upper = symbol.upper()
    
    if symbol_upper.startswith('NQ'):
        return 'NQ'
    elif symbol_upper.startswith('ES'):
        return 'ES'
    elif symbol_upper.startswith('GC'):
        return 'GC'
    else:
        return symbol_upper


def is_futures_contract(symbol: str) -> bool:
    """
    Check if symbol is an actual futures contract (not a spread).
    
    Spread contracts contain '-' (e.g., 'NQZ0-NQH1')
    Actual contracts are single symbols (e.g., 'NQZ0', 'ESZ0', 'GCZ0')
    
    Args:
        symbol: Symbol string
        
    Returns:
        True if it's a futures contract, False if it's a spread
    """
    return '-' not in symbol.upper()


def load_ohlcv_data(file_path: Path, assets: list = None) -> pd.DataFrame:
    """
    Load and filter OHLCV data efficiently.
    
    Args:
        file_path: Path to OHLCV CSV file
        assets: List of assets to filter (e.g., ['NQ', 'ES', 'GC']). 
                If None, loads all NQ, ES, GC symbols.
                
    Returns:
        DataFrame with filtered OHLCV data
    """
    print(f"Loading OHLCV data from {file_path}...")
    
    # Default assets if not specified
    if assets is None:
        assets = ['NQ', 'ES', 'GC']
    
    # Read in chunks for memory efficiency
    chunk_size = 100000
    chunks = []
    
    for chunk in pd.read_csv(file_path, chunksize=chunk_size):
        # Filter for relevant symbols
        chunk = chunk[chunk['symbol'].str.upper().str.startswith(tuple(assets))]
        
        # Filter out spread contracts - only keep actual futures contracts
        chunk = chunk[chunk['symbol'].apply(is_futures_contract)]
        
        if len(chunk) > 0:
            chunks.append(chunk)
    
    if not chunks:
        raise ValueError(f"No data found for assets: {assets}")
    
    df = pd.concat(chunks, ignore_index=True)
    
    print(f"Loaded {len(df):,} rows")
    
    # Convert timestamp to datetime
    df['ts_event'] = pd.to_datetime(df['ts_event'], utc=True)
    
    # Normalize symbols to underlying assets
    df['asset'] = df['symbol'].apply(normalize_symbol)
    
    # Filter to only NQ, ES, GC
    df = df[df['asset'].isin(['NQ', 'ES', 'GC'])]
    
    # Convert UTC to ET
    df['ts_et'] = df['ts_event'].dt.tz_convert(ET)
    
    # Extract date and time components
    df['date'] = df['ts_et'].dt.date
    df['time_et'] = df['ts_et'].dt.time
    
    print(f"Filtered to {len(df):,} rows for assets: {df['asset'].unique().tolist()}")
    
    return df


def calculate_daily_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate volume and range metrics for each trading day and asset.
    
    Args:
        df: DataFrame with OHLCV data (must have columns from load_ohlcv_data)
        
    Returns:
        DataFrame with daily metrics indexed by (date, asset)
    """
    print("Calculating daily metrics...")
    
    # Filter for NY session (9:30 AM - 4:00 PM ET)
    ny_mask = (
        (df['time_et'] >= NY_OPEN_TIME) & 
        (df['time_et'] <= NY_CLOSE_TIME)
    )
    ny_data = df[ny_mask].copy()
    
    if len(ny_data) == 0:
        raise ValueError("No NY session data found")
    
    # Group by date and asset
    results = []
    
    for (date, asset), group in ny_data.groupby(['date', 'asset']):
        # Sort by time
        group = group.sort_values('time_et')
        
        # Find the most active contract (highest volume) for this day
        contract_volumes = group.groupby('symbol')['volume'].sum()
        if len(contract_volumes) == 0:
            continue
        
        most_active_contract = contract_volumes.idxmax()
        active_contract_data = group[group['symbol'] == most_active_contract].copy()
        
        # Full session metrics (using most active contract)
        full_volume = group['volume'].sum()  # Sum across all contracts
        full_range = active_contract_data['high'].max() - active_contract_data['low'].min()
        
        # First 45 minutes (9:30 AM - 10:15 AM ET)
        period_45min_all = group[
            (group['time_et'] >= NY_OPEN_TIME) & 
            (group['time_et'] <= PERIOD_45MIN_END)
        ]
        period_45min = period_45min_all[period_45min_all['symbol'] == most_active_contract]
        
        vol_45min = period_45min_all['volume'].sum() if len(period_45min_all) > 0 else 0
        range_45min = (
            period_45min['high'].max() - period_45min['low'].min() 
            if len(period_45min) > 0 else None
        )
        
        # First 90 minutes (9:30 AM - 11:00 AM ET)
        period_90min_all = group[
            (group['time_et'] >= NY_OPEN_TIME) & 
            (group['time_et'] <= PERIOD_90MIN_END)
        ]
        period_90min = period_90min_all[period_90min_all['symbol'] == most_active_contract]
        
        vol_90min = period_90min_all['volume'].sum() if len(period_90min_all) > 0 else 0
        range_90min = (
            period_90min['high'].max() - period_90min['low'].min() 
            if len(period_90min) > 0 else None
        )
        
        results.append({
            'date': date,
            'asset': asset,
            'NY_First_45min_Volume': vol_45min,
            'NY_First_45min_Range': range_45min,
            'NY_First_90min_Volume': vol_90min,
            'NY_First_90min_Range': range_90min,
            'NY_Full_Session_Volume': full_volume,
            'NY_Full_Session_Range': full_range,
        })
    
    metrics_df = pd.DataFrame(results)
    
    print(f"Calculated metrics for {len(metrics_df)} date-asset combinations")
    
    return metrics_df


def get_metrics_for_trade(
    metrics_df: pd.DataFrame, 
    trade_date: Optional[str], 
    asset: str
) -> Optional[Dict]:
    """
    Get market metrics for a specific trade.
    
    Args:
        metrics_df: DataFrame with daily metrics (from calculate_daily_metrics)
        trade_date: Trade date as string (YYYY-MM-DD) or None
        asset: Asset symbol (NQ, ES, GC)
        
    Returns:
        Dictionary with metrics or None if not found
    """
    if trade_date is None or pd.isna(trade_date) or trade_date == '':
        return None
    
    try:
        # Parse date
        if isinstance(trade_date, str):
            date_obj = datetime.strptime(trade_date, '%Y-%m-%d').date()
        else:
            date_obj = trade_date
        
        # Normalize asset
        asset_normalized = normalize_symbol(asset)
        
        # Lookup metrics
        match = metrics_df[
            (metrics_df['date'] == date_obj) & 
            (metrics_df['asset'] == asset_normalized)
        ]
        
        if len(match) == 0:
            return None
        
        # Return as dictionary (exclude date and asset)
        row = match.iloc[0]
        return {
            'NY_First_45min_Volume': row['NY_First_45min_Volume'],
            'NY_First_45min_Range': row['NY_First_45min_Range'],
            'NY_First_90min_Volume': row['NY_First_90min_Volume'],
            'NY_First_90min_Range': row['NY_First_90min_Range'],
            'NY_Full_Session_Volume': row['NY_Full_Session_Volume'],
            'NY_Full_Session_Range': row['NY_Full_Session_Range'],
        }
    except Exception as e:
        print(f"Error getting metrics for {trade_date}, {asset}: {e}")
        return None


def load_and_calculate_metrics(data_file: Path) -> pd.DataFrame:
    """
    Convenience function to load data and calculate metrics.
    
    Args:
        data_file: Path to OHLCV CSV file
        
    Returns:
        DataFrame with daily metrics
    """
    df = load_ohlcv_data(data_file)
    metrics_df = calculate_daily_metrics(df)
    return metrics_df


if __name__ == "__main__":
    # Test the module
    data_file = Path("data/glbx-mdp3-20200927-20250926.ohlcv-1m.csv")
    
    if data_file.exists():
        print("Testing market_metrics module...")
        metrics_df = load_and_calculate_metrics(data_file)
        print("\nSample metrics:")
        print(metrics_df.head(10))
        print(f"\nTotal date-asset combinations: {len(metrics_df)}")
    else:
        print(f"Data file not found: {data_file}")

