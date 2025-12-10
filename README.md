# Trade Transcript Extractor

A Python application that processes trade transcript files, extracts structured trading data using Azure OpenAI API, and generates a consolidated CSV file.

## Setup

### 1. Create and activate virtual environment

```bash
# Create virtual environment (first time only)
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate
```

**Important:** You must activate the virtual environment before running the script each time you open a new terminal.

### 2. Install dependencies

```bash
# Make sure venv is activated (you should see (venv) in your prompt)
pip install -r requirements.txt
```

### 3. Configure environment variables

Ensure your `.env` file contains:
- `AZURE_OPENAI_API_KEY`
- `AZURE_OPENAI_BASE_URL`
- `AZURE_OPENAI_API_VERSION`

## Usage

```bash
# Activate virtual environment
source venv/bin/activate

# Run the script
python extract_journal.py
```

The script will:
1. Process all `.txt` files in the `transcriptions/` folder
2. Extract trade data using Azure OpenAI API
3. Generate `trade_journal_master.csv` with all extracted trades

## Output

The script generates `trade_journal_master.csv` with the following columns:
- Date_of_Trade
- Time_of_Trade
- Video_Title
- Asset_Ticked
- Setup_Type
- Bias
- Outcome
- PnL_Points
- PnL_R_Multiple
- Liquidity_Target
- Liquidity_Swept
- Management_Style
- Entry_Execution_Grade
- Key_Takeaway

## Troubleshooting

**"ModuleNotFoundError" or import errors:**
- Make sure the virtual environment is activated (`source venv/bin/activate`)
- Verify dependencies are installed (`pip install -r requirements.txt`)

**"AZURE_OPENAI_API_KEY not found":**
- Check that your `.env` file exists and contains the required variables



