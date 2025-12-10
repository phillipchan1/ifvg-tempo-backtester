# Trade Transcript Extractor & Journal Analyzer

A Python application that processes trade transcript files, extracts structured trading data using Azure OpenAI API, and enriches it with market metrics for correlation analysis.

## Project Overview

This tool automates the extraction of trading data from video transcripts, structures it into a standardized format, and correlates trade performance with daily market conditions (volume and range). It's designed specifically for Smart Money Concepts (SMC) and ICT (Inner Circle Trader) methodology traders who use IFVG (Institutional Fair Value Gap) setups.

## What This Project Does

1. **Transcript Processing**: Reads trade transcript files from the `transcriptions/` folder
2. **AI-Powered Extraction**: Uses Azure OpenAI API to extract structured trade data including:
   - Trade dates, times, and assets
   - Setup details (confluences, bias, gap sizes)
   - Trade outcomes (PnL in points, R-multiples)
   - Liquidity sweeps and targets
   - Management style and execution grades
   - Key takeaways and lessons learned

3. **Market Metrics Enrichment**: Correlates trades with daily market conditions:
   - First 45 minutes of NY session (9:30 AM - 10:15 AM ET)
   - First 90 minutes of NY session (9:30 AM - 11:00 AM ET)
   - Full NY session (9:30 AM - 4:00 PM ET)
   - For each period: Volume and Range (in points)

4. **CSV Output**: Generates `trade_journal_master.csv` with all trades and metrics for analysis

## Features

- **Intelligent Date Extraction**: Automatically extracts trade dates from filenames when not explicitly stated in transcripts
- **Common-Sense Validation**: Validates PnL points using gap size logic (IFVG gaps are typically 5-25 points)
- **Multi-Asset Support**: Handles NQ, ES, and GC futures contracts
- **Contract Normalization**: Automatically uses the most active contract for accurate range calculations
- **Error Handling**: Continues processing even if individual files fail
- **Data-Friendly Output**: CSV format optimized for correlation analysis

## Setup

### Prerequisites

- Python 3.10+
- Azure OpenAI API access
- OHLCV market data file (1-minute bars)

### Installation

1. **Clone or navigate to the project directory**

2. **Create and activate virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**:
   
   Create a `.env` file with your Azure OpenAI credentials:
   ```
   AZURE_OPENAI_API_KEY=your_api_key_here
   AZURE_OPENAI_BASE_URL=https://your-resource.openai.azure.com/openai/deployments/gpt-4o/
   AZURE_OPENAI_API_VERSION=2024-02-15-preview
   ```

5. **Prepare market data**:
   
   Place your OHLCV data file in the `data/` folder:
   - File should be named: `glbx-mdp3-20200927-20250926.ohlcv-1m.csv`
   - Format: CSV with columns: `ts_event`, `symbol`, `open`, `high`, `low`, `close`, `volume`
   - Timestamps should be in UTC

## Usage

### Basic Usage

```bash
# Activate virtual environment
source venv/bin/activate

# Run the extractor
python extract_journal.py
```

Or use the convenience script:
```bash
./run.sh
```

### File Organization

- **Input**: Place transcript files (`.txt`) in the `transcriptions/` folder
- **Output**: Generated CSV saved as `trade_journal_master.csv` in the project root

### Transcript File Naming

Filenames are used to extract dates when not explicitly stated. Supported formats:
- `2025-12-04_11-58-48.txt` → Date: 2025-12-04
- `2025-9-12.txt` → Date: 2025-09-12 (normalized)
- `2020-12-04.txt` → Date: 2020-12-04

## Output Format

The generated CSV contains the following columns:

### Trade Information
- `Date_of_Trade`: Trade date (YYYY-MM-DD)
- `Time_of_Trade`: Entry time
- `Video_Title`: Source transcript filename
- `Asset_Ticked`: Asset symbol (NQ, ES, RTY, CL, Other)

### Setup Details
- `Chart_Timeframe`: Chart timeframe used (30s, 1m, 2m, 3m, 5m, etc.)
- `Gap_Size_Points`: Size of gap inversed (typically 5-25 points)
- `Number_of_Gaps_Inversed`: How many gaps were involved
- `Confluences`: ICT confluences (BPR, SMT, Order Block, etc.)
- `Bias`: Market bias (In Discount, In Premium, N/A)

### Trade Outcome
- `Outcome`: Win, Loss, or Break-Even
- `PnL_Points`: Profit/Loss in points (NOT dollars)
- `PnL_R_Multiple`: Risk-reward multiple
- `Management_Style`: Full TP, Aggressive Trim, Held Runner, Breakeven, Stop Out
- `Entry_Execution_Grade`: Quality rating (1-5)

### Liquidity Analysis
- `Liquidity_Swept`: Initial liquidity level(s) swept to trigger IFVG
- `Liquidity_Target`: Target liquidity area(s) for profit

### Market Metrics (Correlation Data)
- `NY_First_45min_Volume`: Volume in first 45 minutes
- `NY_First_45min_Range`: Range (points) in first 45 minutes
- `NY_First_90min_Volume`: Volume in first 90 minutes
- `NY_First_90min_Range`: Range (points) in first 90 minutes
- `NY_Full_Session_Volume`: Total NY session volume
- `NY_Full_Session_Range`: Total NY session range (points)

### Analysis
- `Key_Takeaway`: Detailed lessons and insights from the trade

## Data Processing Details

### Market Metrics Calculation

- **Contract Selection**: Uses the most active contract (highest volume) for each trading day
- **Spread Filtering**: Automatically excludes spread contracts (e.g., NQZ0-NQH1)
- **Time Periods**: 
  - First 45 min: 9:30 AM - 10:15 AM ET
  - First 90 min: 9:30 AM - 11:00 AM ET
  - Full session: 9:30 AM - 4:00 PM ET
- **Range Calculation**: High - Low (in points) for the most active contract

### Date Matching

- Trades are matched to market metrics using `Date_of_Trade` and `Asset_Ticked`
- If `Date_of_Trade` is missing, the date is extracted from the filename
- Metrics are only populated if the date exists in the market data file

## Use Cases

This tool is designed for:

1. **Trade Journaling**: Automatically extract and structure trade data from video transcripts
2. **Performance Analysis**: Correlate trade outcomes with market conditions
3. **Pattern Recognition**: Identify which market conditions (volume/range) lead to better trades
4. **Strategy Refinement**: Use correlation data to improve entry timing and trade selection
5. **Backtesting Preparation**: Structure historical trade data for analysis

## Troubleshooting

**"ModuleNotFoundError"**:
- Ensure virtual environment is activated: `source venv/bin/activate`
- Reinstall dependencies: `pip install -r requirements.txt`

**"AZURE_OPENAI_API_KEY not found"**:
- Check that `.env` file exists and contains required variables
- Verify variable names match exactly

**No market metrics populated**:
- Verify `Date_of_Trade` is populated (check CSV output)
- Ensure trade date exists in market data file date range
- Check that `Asset_Ticked` matches supported assets (NQ, ES, GC)

**Range values seem incorrect**:
- Range is calculated using the most active contract
- Values should be in points (not dollars)
- Typical ranges: 50-300 points for NQ

## Project Structure

```
ifvg-tempo-backtester/
├── extract_journal.py          # Main extraction script
├── market_metrics.py            # Market data processing module
├── requirements.txt             # Python dependencies
├── run.sh                      # Convenience execution script
├── README.md                   # This file
├── .env                        # Environment variables (not in git)
├── .gitignore                  # Git ignore rules
├── transcriptions/             # Input transcript files
│   └── *.txt
├── data/                       # Market data files
│   └── glbx-mdp3-*.csv
└── trade_journal_master.csv    # Generated output (not in git)
```

## Dependencies

- `openai>=1.0.0`: Azure OpenAI API client
- `pandas>=2.0.0`: Data manipulation and CSV operations
- `python-dotenv>=1.0.0`: Environment variable management
- `pydantic>=2.0.0`: Data validation
- `pytz>=2024.0`: Timezone handling for market hours

## Notes

- All PnL values are in **points**, not dollars
- Market metrics use **Eastern Time (ET)** for NY session calculations
- The system automatically handles Daylight Saving Time transitions
- Only futures contracts are used (spread contracts are filtered out)
- Date extraction from filenames is a fallback when transcripts don't explicitly state dates

## License

[Add your license here]

## Contributing

[Add contribution guidelines if applicable]
