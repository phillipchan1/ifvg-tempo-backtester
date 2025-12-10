#!/usr/bin/env python3
"""
Trade Correlation Analysis

Analyzes trade journal data to identify factors that correlate with success.
Calculates win rates and average R-multiples by various factors.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple


def load_and_clean_data(csv_path: str = "trade_journal_master.csv") -> pd.DataFrame:
    """
    Load trade data and perform initial cleaning.
    
    Args:
        csv_path: Path to the trade journal CSV file
        
    Returns:
        Cleaned DataFrame
    """
    print("Loading trade data...")
    df = pd.read_csv(csv_path)
    
    print(f"Loaded {len(df)} trades")
    
    # Filter out trades with missing critical data
    initial_count = len(df)
    df = df[df['Outcome'].notna()].copy()
    removed = initial_count - len(df)
    if removed > 0:
        print(f"  Removed {removed} trades with missing Outcome")
    
    print(f"Analyzing {len(df)} trades with valid outcomes")
    print()
    
    return df


def parse_confluences(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse comma-separated confluences and create individual factor columns.
    
    Args:
        df: DataFrame with Confluences column
        
    Returns:
        DataFrame with additional confluence indicator columns
    """
    # Common confluences to track
    confluence_types = [
        'SMT', 'Order Block', 'Liquidity Grab', 'BPR', 
        'Market Structure Break', 'V Shape Reversal',
        'Fair Value Gap', 'Equal Highs', 'Equal Lows',
        'Kill Zone', 'Discount', 'Premium', 'Breaker Block',
        'Mitigation Block', 'Displacement'
    ]
    
    df = df.copy()
    
    # Create indicator columns for each confluence type
    for confluence in confluence_types:
        col_name = f'Has_{confluence.replace(" ", "_")}'
        # Check if confluence appears in the Confluences string (case-insensitive)
        df[col_name] = df['Confluences'].fillna('').str.contains(
            confluence, case=False, na=False, regex=False
        )
    
    # Count total number of confluences
    df['Num_Confluences'] = df['Confluences'].fillna('').str.split(',').apply(
        lambda x: len([c.strip() for c in x if c.strip()]) if isinstance(x, list) else 0
    )
    
    return df


def bucket_gap_size(gap_size: float) -> str:
    """Bucket gap size into categories."""
    if pd.isna(gap_size):
        return "Unknown"
    elif gap_size < 10:
        return "<10"
    elif gap_size < 15:
        return "10-15"
    elif gap_size < 20:
        return "15-20"
    elif gap_size < 25:
        return "20-25"
    else:
        return ">25"


def bucket_range(range_val: float) -> str:
    """Bucket range values into categories."""
    if pd.isna(range_val):
        return "Unknown"
    elif range_val < 50:
        return "<50"
    elif range_val < 100:
        return "50-100"
    elif range_val < 150:
        return "100-150"
    elif range_val < 200:
        return "150-200"
    else:
        return ">200"


def calculate_win_rate(df: pd.DataFrame, group_col: str, min_samples: int = 1) -> pd.DataFrame:
    """
    Calculate win rate by grouping column.
    
    Args:
        df: DataFrame with Outcome column
        group_col: Column to group by
        min_samples: Minimum sample size to include in results
        
    Returns:
        DataFrame with win rate statistics
    """
    # Filter out missing values
    valid_df = df[df[group_col].notna()].copy()
    
    if len(valid_df) == 0:
        return pd.DataFrame()
    
    # Calculate statistics
    results = []
    for value in valid_df[group_col].unique():
        subset = valid_df[valid_df[group_col] == value]
        
        if len(subset) < min_samples:
            continue
        
        wins = len(subset[subset['Outcome'] == 'Win'])
        losses = len(subset[subset['Outcome'] == 'Loss'])
        break_even = len(subset[subset['Outcome'] == 'Break-Even'])
        total = len(subset)
        
        win_rate = wins / total if total > 0 else 0
        
        # Calculate average R-multiple
        avg_r = subset['PnL_R_Multiple'].mean() if 'PnL_R_Multiple' in subset.columns else None
        
        results.append({
            'Factor': value,
            'Total_Trades': total,
            'Wins': wins,
            'Losses': losses,
            'Break_Even': break_even,
            'Win_Rate': win_rate,
            'Avg_R_Multiple': avg_r
        })
    
    result_df = pd.DataFrame(results)
    result_df = result_df.sort_values('Win_Rate', ascending=False)
    
    return result_df


def analyze_by_factor(df: pd.DataFrame, factor_name: str, min_samples: int = 5) -> None:
    """
    Analyze win rate and R-multiple by a specific factor.
    
    Args:
        df: DataFrame with trade data
        factor_name: Name of the factor column to analyze
        min_samples: Minimum sample size to include (default 5 for statistical significance)
    """
    if factor_name not in df.columns:
        print(f"  ⚠ Factor '{factor_name}' not found in data")
        return
    
    print(f"\n{'='*70}")
    print(f"Analysis by {factor_name}")
    print(f"{'='*70}")
    
    stats = calculate_win_rate(df, factor_name, min_samples)
    
    if len(stats) == 0:
        print(f"  ⚠ No data available with minimum {min_samples} trades for {factor_name}")
        # Try with min_samples=1 to show what we have
        stats_all = calculate_win_rate(df, factor_name, min_samples=1)
        if len(stats_all) > 0:
            print(f"  (Showing all data with any trades, but sample sizes may be too small):")
            stats = stats_all
        else:
            return
    
    # Print results
    print(f"\n{'Rank':<6} {'Factor':<28} {'Trades':<8} {'Win Rate':<12} {'Avg R':<10} {'W':<4} {'L':<4} {'BE':<4}")
    print("-" * 70)
    
    for rank, (_, row) in enumerate(stats.iterrows(), 1):
        factor = str(row['Factor'])[:26]  # Truncate if too long
        trades = int(row['Total_Trades'])
        wr = row['Win_Rate']
        avg_r = row['Avg_R_Multiple'] if pd.notna(row['Avg_R_Multiple']) else 0
        wins = int(row['Wins'])
        losses = int(row['Losses'])
        be = int(row['Break_Even'])
        
        # Mark small sample sizes
        sample_warning = " ⚠" if trades < 5 else "  "
        
        print(f"{rank:<4}{sample_warning} {factor:<28} {trades:<8} {wr*100:>6.1f}%      {avg_r:>6.2f}    {wins:<4} {losses:<4} {be:<4}")
    
    # Highlight best and worst (only if multiple options)
    if len(stats) > 1:
        best = stats.iloc[0]
        worst = stats.iloc[-1]
        print(f"\n  ✅ BEST: {best['Factor']} - {best['Win_Rate']*100:.1f}% WR, Avg R={best['Avg_R_Multiple']:.2f} ({int(best['Total_Trades'])} trades)")
        print(f"  ❌ WORST: {worst['Factor']} - {worst['Win_Rate']*100:.1f}% WR, Avg R={worst['Avg_R_Multiple']:.2f} ({int(worst['Total_Trades'])} trades)")


def analyze_confluences(df: pd.DataFrame) -> None:
    """Analyze individual confluences."""
    print(f"\n{'='*70}")
    print("Analysis by Individual Confluences")
    print(f"{'='*70}")
    
    # Get all confluence indicator columns
    confluence_cols = [col for col in df.columns if col.startswith('Has_')]
    
    if len(confluence_cols) == 0:
        print("  No confluence data available")
        return
    
    results = []
    
    for col in confluence_cols:
        confluence_name = col.replace('Has_', '').replace('_', ' ')
        
        # Trades with this confluence
        with_confluence = df[df[col] == True]
        without_confluence = df[df[col] == False]
        
        if len(with_confluence) == 0:
            continue
        
        # Calculate stats
        wr_with = len(with_confluence[with_confluence['Outcome'] == 'Win']) / len(with_confluence)
        avg_r_with = with_confluence['PnL_R_Multiple'].mean() if 'PnL_R_Multiple' in with_confluence.columns else None
        
        wr_without = len(without_confluence[without_confluence['Outcome'] == 'Win']) / len(without_confluence) if len(without_confluence) > 0 else None
        avg_r_without = without_confluence['PnL_R_Multiple'].mean() if len(without_confluence) > 0 and 'PnL_R_Multiple' in without_confluence.columns else None
        
        results.append({
            'Confluence': confluence_name,
            'Trades_With': len(with_confluence),
            'WR_With': wr_with,
            'Avg_R_With': avg_r_with,
            'Trades_Without': len(without_confluence),
            'WR_Without': wr_without,
            'Avg_R_Without': avg_r_without
        })
    
    if len(results) == 0:
        print("  No confluence data available")
        return
    
    result_df = pd.DataFrame(results)
    result_df = result_df.sort_values('WR_With', ascending=False)
    
    print(f"\n{'Confluence':<25} {'With':<15} {'Without':<15} {'Diff':<10}")
    print(f"{'':<25} {'Trades WR% R':<15} {'Trades WR% R':<15} {'WR%':<10}")
    print("-" * 70)
    
    for _, row in result_df.iterrows():
        conf = row['Confluence'][:23]
        trades_w = int(row['Trades_With'])
        wr_w = row['WR_With'] * 100
        r_w = row['Avg_R_With'] if pd.notna(row['Avg_R_With']) else 0
        
        trades_wo = int(row['Trades_Without']) if pd.notna(row['Trades_Without']) else 0
        wr_wo = row['WR_Without'] * 100 if pd.notna(row['WR_Without']) else 0
        r_wo = row['Avg_R_Without'] if pd.notna(row['Avg_R_Without']) else 0
        
        diff = wr_w - wr_wo if pd.notna(wr_wo) else 0
        
        print(f"{conf:<25} {trades_w:<4} {wr_w:>5.1f}% {r_w:>4.2f}  {trades_wo:<4} {wr_wo:>5.1f}% {r_wo:>4.2f}  {diff:>+6.1f}%")


def analyze_liquidity_sweeps(df: pd.DataFrame) -> None:
    """Analyze performance by liquidity sweep type."""
    print(f"\n{'='*70}")
    print("Analysis by Liquidity Swept (Ranked)")
    print(f"{'='*70}")
    
    if 'Liquidity_Swept' not in df.columns:
        print("  ⚠ Liquidity_Swept column not found")
        return
    
    # Filter out missing values
    valid_df = df[df['Liquidity_Swept'].notna()].copy()
    valid_df = valid_df[valid_df['Liquidity_Swept'] != '']
    
    if len(valid_df) == 0:
        print("  ⚠ No liquidity sweep data available")
        return
    
    # Parse comma-separated liquidity sweeps
    all_sweeps = set()
    for sweeps_str in valid_df['Liquidity_Swept']:
        if pd.notna(sweeps_str) and sweeps_str.strip():
            sweeps = [s.strip() for s in str(sweeps_str).split(',')]
            all_sweeps.update(sweeps)
    
    if len(all_sweeps) == 0:
        print("  ⚠ No liquidity sweep data available")
        return
    
    results = []
    
    for sweep in sorted(all_sweeps):
        # Find trades with this liquidity sweep
        has_sweep = valid_df[valid_df['Liquidity_Swept'].str.contains(sweep, case=False, na=False)]
        
        if len(has_sweep) < 3:  # Minimum 3 trades
            continue
        
        wins = len(has_sweep[has_sweep['Outcome'] == 'Win'])
        losses = len(has_sweep[has_sweep['Outcome'] == 'Loss'])
        break_even = len(has_sweep[has_sweep['Outcome'] == 'Break-Even'])
        total = len(has_sweep)
        
        win_rate = wins / total if total > 0 else 0
        avg_r = has_sweep['PnL_R_Multiple'].mean() if 'PnL_R_Multiple' in has_sweep.columns else None
        
        results.append({
            'Liquidity_Sweep': sweep,
            'Total_Trades': total,
            'Wins': wins,
            'Losses': losses,
            'Break_Even': break_even,
            'Win_Rate': win_rate,
            'Avg_R_Multiple': avg_r
        })
    
    if len(results) == 0:
        print("  ⚠ No liquidity sweeps with sufficient sample size (min 3 trades)")
        return
    
    result_df = pd.DataFrame(results)
    result_df = result_df.sort_values('Win_Rate', ascending=False)
    
    print(f"\n{'Rank':<6} {'Liquidity Sweep':<30} {'Trades':<8} {'Win Rate':<12} {'Avg R':<10} {'W':<4} {'L':<4} {'BE':<4}")
    print("-" * 70)
    
    for rank, (_, row) in enumerate(result_df.iterrows(), 1):
        sweep = str(row['Liquidity_Sweep'])[:28]
        trades = int(row['Total_Trades'])
        wr = row['Win_Rate']
        avg_r = row['Avg_R_Multiple'] if pd.notna(row['Avg_R_Multiple']) else 0
        wins = int(row['Wins'])
        losses = int(row['Losses'])
        be = int(row['Break_Even'])
        
        sample_warning = " ⚠" if trades < 5 else "  "
        
        print(f"{rank:<4}{sample_warning} {sweep:<30} {trades:<8} {wr*100:>6.1f}%      {avg_r:>6.2f}    {wins:<4} {losses:<4} {be:<4}")
    
    if len(result_df) > 1:
        best = result_df.iloc[0]
        worst = result_df.iloc[-1]
        print(f"\n  ✅ BEST: {best['Liquidity_Sweep']} - {best['Win_Rate']*100:.1f}% WR, Avg R={best['Avg_R_Multiple']:.2f} ({int(best['Total_Trades'])} trades)")
        print(f"  ❌ WORST: {worst['Liquidity_Sweep']} - {worst['Win_Rate']*100:.1f}% WR, Avg R={worst['Avg_R_Multiple']:.2f} ({int(worst['Total_Trades'])} trades)")


def analyze_market_conditions(df: pd.DataFrame) -> None:
    """Analyze performance vs market conditions."""
    print(f"\n{'='*70}")
    print("Analysis by Market Conditions")
    print(f"{'='*70}")
    
    # Check if we have market data
    market_cols = [
        'NY_First_45min_Range', 'NY_First_90min_Range', 'NY_Full_Session_Range',
        'NY_First_45min_Volume', 'NY_First_90min_Volume', 'NY_Full_Session_Volume'
    ]
    
    has_market_data = any(df[col].notna().sum() > 0 for col in market_cols)
    
    if not has_market_data:
        print("  ⚠ No market condition data available")
        return
    
    # Analyze by range buckets
    for range_col in ['NY_First_45min_Range', 'NY_First_90min_Range', 'NY_Full_Session_Range']:
        if df[range_col].notna().sum() == 0:
            continue
        
        print(f"\n{range_col}:")
        df_with_range = df[df[range_col].notna()].copy()
        df_with_range['Range_Bucket'] = df_with_range[range_col].apply(bucket_range)
        
        analyze_by_factor(df_with_range, 'Range_Bucket', min_samples=1)


def print_overall_summary(df: pd.DataFrame) -> None:
    """Print overall statistics summary."""
    print(f"\n{'='*70}")
    print("OVERALL SUMMARY")
    print(f"{'='*70}")
    
    total_trades = len(df)
    wins = len(df[df['Outcome'] == 'Win'])
    losses = len(df[df['Outcome'] == 'Loss'])
    break_even = len(df[df['Outcome'] == 'Break-Even'])
    
    win_rate = wins / total_trades if total_trades > 0 else 0
    
    # Average R-multiple
    valid_r = df['PnL_R_Multiple'].dropna()
    avg_r = valid_r.mean() if len(valid_r) > 0 else None
    
    print(f"\nTotal Trades: {total_trades}")
    print(f"  Wins: {wins} ({wins/total_trades*100:.1f}%)")
    print(f"  Losses: {losses} ({losses/total_trades*100:.1f}%)")
    print(f"  Break-Even: {break_even} ({break_even/total_trades*100:.1f}%)")
    print(f"\nOverall Win Rate: {win_rate*100:.1f}%")
    
    if avg_r is not None:
        print(f"Average R-Multiple: {avg_r:.2f}")
        print(f"  (based on {len(valid_r)} trades with R-multiple data)")
    
    # Best and worst trades
    if 'PnL_R_Multiple' in df.columns:
        best_trade = df.loc[df['PnL_R_Multiple'].idxmax()] if len(df[df['PnL_R_Multiple'].notna()]) > 0 else None
        worst_trade = df.loc[df['PnL_R_Multiple'].idxmin()] if len(df[df['PnL_R_Multiple'].notna()]) > 0 else None
        
        if best_trade is not None:
            print(f"\nBest Trade: R={best_trade['PnL_R_Multiple']:.2f}, {best_trade.get('Confluences', 'N/A')}")
        if worst_trade is not None:
            print(f"Worst Trade: R={worst_trade['PnL_R_Multiple']:.2f}, {worst_trade.get('Confluences', 'N/A')}")


def main():
    """Main analysis function."""
    print("="*70)
    print("TRADE CORRELATION ANALYSIS")
    print("="*70)
    print()
    
    # Load and clean data
    df = load_and_clean_data()
    
    # Parse confluences
    df = parse_confluences(df)
    
    # Print overall summary
    print_overall_summary(df)
    
    # Analyze by various factors
    print("\n" + "="*70)
    print("DETAILED FACTOR ANALYSIS")
    print("="*70)
    
    # Analyze by basic factors (with minimum sample sizes)
    analyze_by_factor(df, 'Bias', min_samples=5)
    analyze_by_factor(df, 'Chart_Timeframe', min_samples=5)  # Ranked timeframes
    analyze_by_factor(df, 'Management_Style', min_samples=5)
    analyze_by_factor(df, 'Asset_Ticked', min_samples=5)
    
    # Analyze gap size (bucketed) - ranked
    if 'Gap_Size_Points' in df.columns:
        df['Gap_Size_Bucket'] = df['Gap_Size_Points'].apply(bucket_gap_size)
        analyze_by_factor(df, 'Gap_Size_Bucket', min_samples=3)  # Ranked gap sizes
    
    # Analyze number of gaps inversed - ranked
    if 'Number_of_Gaps_Inversed' in df.columns:
        analyze_by_factor(df, 'Number_of_Gaps_Inversed', min_samples=3)  # Ranked number of gaps
    
    # Analyze entry execution grade (note: higher grade = better execution, so 100% WR makes sense)
    if 'Entry_Execution_Grade' in df.columns:
        analyze_by_factor(df, 'Entry_Execution_Grade', min_samples=5)
    
    # Analyze number of confluences
    if 'Num_Confluences' in df.columns:
        analyze_by_factor(df, 'Num_Confluences', min_samples=5)
    
    # Analyze liquidity sweeps - ranked
    analyze_liquidity_sweeps(df)
    
    # Analyze individual confluences
    analyze_confluences(df)
    
    # Analyze market conditions
    analyze_market_conditions(df)
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()

