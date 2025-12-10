#!/usr/bin/env python3
"""
Trade Transcript Extractor

Processes trade transcript files from the transcriptions folder,
extracts structured trading data using Azure OpenAI API, and
outputs a consolidated CSV file with all trades.
"""

import os
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from dotenv import load_dotenv
from openai import AzureOpenAI

import market_metrics


# Load environment variables
load_dotenv()

# System prompt for the API
SYSTEM_PROMPT = """**Role:** You are an expert financial data analyst for a Smart Money Concepts (SMC) trader specializing in ICT (Inner Circle Trader) methodology.

**Task:** Extract distinct trades from the transcript into a JSON Array. **CRITICAL:** Each trade must be a separate object in the array. If a transcript describes multiple trades, create separate entries for each one.

**Important Context:**
- All trades are IFVG (Institutional Fair Value Gap) based setups
- IFVG setups REQUIRE an initial liquidity sweep to trigger the setup
- 50% of Daily Range / 50% Level is a **liquidity target**, NOT a confluence and NOT a liquidity sweep

**CRITICAL WARNING - Common Mistake to Avoid:**
- Transcripts often mix dollars ($) and points. PnL_Points MUST be in POINTS (price movement), NEVER dollars.
- If you see "made $200" or "lost $220" but no explicit "points" mentioned, look for price levels, trim levels, or use null - DO NOT use the dollar amount.

**Rules:**

1.  **R-Multiple:** Infer the Risk Unit (R) from context. (e.g., "Lost $200" = 1R. "Made $600" = 3R). Loss is always -1.0.

2.  **PnL_Points - CRITICAL:** This field MUST contain the actual POINTS (price movement), NEVER dollars. 
   * **COMMON SENSE VALIDATION:** IFVG trades inverse gaps that are typically 5-25 points in size. Maximum loss on a stop-out is typically around 25 points. If you see a number like "220" or "200" in the context of a loss, this is almost certainly DOLLARS, not points.
   * Look for explicit mentions of "points" in the transcript (e.g., "20 points", "100 points", "lost 220 points")
   * If entry and exit prices are mentioned, calculate the difference in points
   * If the transcript only mentions dollar amounts (e.g., "$200", "made $180", "lost 220"), DO NOT use those values - they are dollars
   * If dollar amounts are mentioned but points are not explicitly stated, look for price level references or trim levels mentioned in points
   * **Validation Rule:** If a loss value exceeds 30 points, it's likely dollars - look for gap size or other context clues
   * Examples:
     - "I lost 220" with no "points" mentioned → This is likely $220, not points. Look for gap size (5-25 points) or use null
     - "I lost 220 points" → PnL_Points: -220 (explicitly stated as points)
     - "20 point trim" → This indicates 20 points of movement
     - "made $200" but "20 point trim" mentioned → Use the points value, NOT $200
     - "Got in at 602 and targeted 582" → Calculate: 602 - 582 = 20 points
     - "inversed a 15 point gap" → Loss would be around 15 points, not 220
   * If you cannot determine points from the transcript, use null - NEVER guess or use dollar amounts

3.  **Bias:** Must be "In Discount", "In Premium", or "N/A".

4.  **Missing Data:** Use null for uncertain fields.

**JSON Output Schema:**

Return a JSON object with a key "trades" containing an array of objects. Each object must have:

- "Date_of_Trade": "YYYY-MM-DD" or null
- "Time_of_Trade": String or null
- "Asset_Ticked": Enum ["NQ", "ES", "RTY", "CL", "Other"]
- "Chart_Timeframe": String - The timeframe chart used for the trade. Common values: "30s", "1m", "2m", "3m", "5m", "15m", "1H", "4H", "Daily", or null if not mentioned
- "Gap_Size_Points": Integer or null - The size of the gap (in points) that was inversed. IFVG trades typically inverse gaps ranging from 5-25 points. Extract from mentions like "15 point gap", "20 point gap", "inversed a 10 point gap", etc.
- "Number_of_Gaps_Inversed": Integer or null - How many gaps were involved. Was it just one gap inversed, or multiple gaps where the last one was inversed? Extract from context like "single gap", "multiple gaps", "last gap out of several", etc. Use 1 if only one gap mentioned, or the number if multiple gaps are discussed.
- "Confluences": Array of strings - List of ICT confluences present in the trade beyond the base IFVG setup. Common ICT confluences include:
  * "BPR" (Bullish/Bearish Premium Range)
  * "SMT" (Smart Money Trap)
  * "Order Block"
  * "Liquidity Grab"
  * "Market Structure Break"
  * Identify other ICT confluences from context (e.g., "Breaker Block", "Mitigation Block", "Displacement", "V Shape Reversal", etc.)
  * Can be empty array [] if no additional confluences beyond IFVG
- "Bias": Enum ["In Discount", "In Premium", "N/A"]
- "Outcome": Enum ["Win", "Loss", "Break-Even"]
- "PnL_Points": Integer or null - **CRITICAL: This MUST be POINTS (price movement), NEVER dollars.**
  * **COMMON SENSE CHECK:** IFVG gap inversions are typically 5-25 points. Losses rarely exceed 25-30 points. If you see "220" or "200" for a loss without "points" explicitly stated, this is DOLLARS, not points.
  * Extract from explicit "points" mentions: "20 points", "100 points", "lost 220 points"
  * Calculate from price levels if entry/exit mentioned: entry price - exit price = points
  * Look for trim levels mentioned in points: "20 point trim", "45 point trim" can indicate total move
  * Cross-reference with Gap_Size_Points - if gap was 15 points, loss should be around that range, not 220
  * If transcript mentions "$200" or "made $180" or "lost 220" (without "points"), use null - DO NOT use dollar amounts
  * Examples: 
    - "lost 220" (no "points") → null (this is $220, gap inversions don't lose 220 points)
    - "lost 220 points" → -220 (explicitly stated as points)
    - "inversed 15 point gap, lost on stop" → -15 (approximately, based on gap size)
    - "20 point trim at 45 points" → calculate total from context
    - "made $200" with no points → null
- "PnL_R_Multiple": Float (The most important metric)
- "Liquidity_Swept": Array of strings - The initial liquidity level(s) that were SWEPT to create the IFVG setup. Common options include:
  * "Asia High" or "Asia Low"
  * "London High" or "London Low"
  * "New York High" or "New York Low"
  * "Previous Day High" or "Previous Day Low"
  * "Opening Range High" or "Opening Range Low"
  * "4H High" or "4H Low"
  * "1H High" or "1H Low"
  * "930 High" or "930 Low"
  * "Data High" or "Data Low"
  * "SSL" (Sell Side Liquidity) or "BSL" (Buy Side Liquidity)
  * "5m Gap", "15m Gap", "1H Gap", "4H Gap"
  * "New Week Opening Gap"
  * Can be empty array [] if not explicitly mentioned, but IFVG setups typically require a sweep
- "Liquidity_Target": Array of strings - The liquidity area(s) being TARGETED for profit. These are separate from the initial sweep. Common options include:
  * "50% Level" or "50% of Daily Range" (this is a liquidity target, NOT a confluence)
  * "New York High" or "New York Low"
  * "London High" or "London Low"
  * "Asia High" or "Asia Low"
  * "Previous Day High" or "Previous Day Low"
  * Other specific liquidity levels mentioned
  * **Important:** Do NOT confuse liquidity targets with liquidity sweeps. The sweep happens FIRST to create the setup, targets are what you're aiming for.
- "Management_Style": Enum ["Full TP", "Aggressive Trim", "Held Runner", "Breakeven", "Stop Out"]
- "Entry_Execution_Grade": Integer (1-5)
- "Key_Takeaway": String - Provide a detailed, comprehensive takeaway. Include:
  * What worked or didn't work in the trade
  * Key lessons learned
  * Risk management observations
  * Entry/exit quality assessment
  * Market structure insights
  * Any specific details about confluences, liquidity, or execution that are noteworthy
  * Be thorough and specific, not generic
"""


def parse_azure_config() -> tuple[str, str, str, str]:
    """
    Parse Azure OpenAI configuration from environment variables.
    
    Returns:
        tuple: (api_key, endpoint, deployment_name, api_version)
    """
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    base_url = os.getenv("AZURE_OPENAI_BASE_URL")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
    
    if not api_key:
        raise ValueError("AZURE_OPENAI_API_KEY not found in environment variables")
    if not base_url:
        raise ValueError("AZURE_OPENAI_BASE_URL not found in environment variables")
    
    # Parse BASE_URL to extract endpoint and deployment
    # Format: https://philchanopenai.openai.azure.com/openai/deployments/gpt-4o/
    # Extract endpoint: https://philchanopenai.openai.azure.com
    # Extract deployment: gpt-4o
    
    # Remove trailing slash
    base_url = base_url.rstrip('/')
    
    # Extract endpoint (everything before /openai/deployments/)
    if '/openai/deployments/' in base_url:
        endpoint = base_url.split('/openai/deployments/')[0]
        deployment = base_url.split('/openai/deployments/')[1].rstrip('/')
    else:
        # Fallback: try to extract from URL structure
        match = re.match(r'(https://[^/]+)', base_url)
        if match:
            endpoint = match.group(1)
            # Try to extract deployment from path
            deployment_match = re.search(r'/deployments/([^/]+)', base_url)
            deployment = deployment_match.group(1) if deployment_match else "gpt-4o"
        else:
            raise ValueError(f"Could not parse endpoint from BASE_URL: {base_url}")
    
    # Clean up API version (handle multi-line values)
    api_version = api_version.replace('\\\n', '').replace('\n', '').strip()
    
    return api_key, endpoint, deployment, api_version


def initialize_client() -> AzureOpenAI:
    """Initialize and return Azure OpenAI client."""
    api_key, endpoint, deployment, api_version = parse_azure_config()
    
    client = AzureOpenAI(
        api_key=api_key,
        api_version=api_version,
        azure_endpoint=endpoint
    )
    
    return client, deployment


def extract_trades_from_transcript(client: AzureOpenAI, deployment: str, transcript_text: str) -> List[Dict[str, Any]]:
    """
    Send transcript to Azure OpenAI API and extract trades.
    
    Args:
        client: Azure OpenAI client instance
        deployment: Deployment name
        transcript_text: The transcript text content
        
    Returns:
        List of trade dictionaries
    """
    try:
        response = client.chat.completions.create(
            model=deployment,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": transcript_text}
            ],
            response_format={"type": "json_object"},
            temperature=0.3
        )
        
        # Extract JSON from response
        content = response.choices[0].message.content
        
        # Parse JSON
        result = json.loads(content)
        
        # Extract trades array
        trades = result.get("trades", [])
        
        if not isinstance(trades, list):
            print(f"Warning: Expected 'trades' to be a list, got {type(trades)}")
            return []
        
        return trades
        
    except json.JSONDecodeError as e:
        print(f"Error: Failed to parse JSON response: {e}")
        return []
    except Exception as e:
        print(f"Error: API call failed: {e}")
        return []


def extract_date_from_filename(filename: str) -> Optional[str]:
    """
    Extract date from filename as fallback when Date_of_Trade is missing.
    
    Supports formats:
    - 2025-12-04_11-58-48 → 2025-12-04
    - 2025-9-12 → 2025-09-12 (normalize to YYYY-MM-DD)
    - 2020-12-04 → 2020-12-04
    
    Args:
        filename: Filename (with or without extension)
        
    Returns:
        Date string in YYYY-MM-DD format or None
    """
    # Remove extension if present
    stem = Path(filename).stem
    
    # Try to extract date from various formats
    # Format 1: YYYY-MM-DD_HH-MM-SS
    match = re.match(r'(\d{4})-(\d{1,2})-(\d{1,2})', stem)
    if match:
        year, month, day = match.groups()
        # Normalize to YYYY-MM-DD
        return f"{year}-{month.zfill(2)}-{day.zfill(2)}"
    
    return None


def process_transcript_file(client: AzureOpenAI, deployment: str, file_path: Path) -> List[Dict[str, Any]]:
    """
    Process a single transcript file and return list of trades.
    
    Args:
        client: Azure OpenAI client instance
        deployment: Deployment name
        file_path: Path to the transcript file
        
    Returns:
        List of trade dictionaries with Video_Title and Date_of_Trade (from filename) added
    """
    try:
        # Read file content
        with open(file_path, 'r', encoding='utf-8') as f:
            transcript_text = f.read()
        
        # Extract filename without extension for Video_Title
        video_title = file_path.stem
        
        # Extract date from filename as fallback
        filename_date = extract_date_from_filename(video_title)
        
        # Extract trades from transcript
        trades = extract_trades_from_transcript(client, deployment, transcript_text)
        
        # Add Video_Title and Date_of_Trade (from filename if missing) to each trade
        for trade in trades:
            trade["Video_Title"] = video_title
            # Use filename date as fallback if Date_of_Trade is missing
            date_value = trade.get("Date_of_Trade")
            if not date_value or (isinstance(date_value, str) and date_value.strip() == "") or pd.isna(date_value):
                if filename_date:
                    trade["Date_of_Trade"] = filename_date
        
        return trades
        
    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
        return []
    except Exception as e:
        print(f"Error: Failed to process file {file_path}: {e}")
        return []


def convert_array_fields_to_string(trade: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert array fields to comma-separated strings for CSV compatibility.
    
    Args:
        trade: Trade dictionary
        
    Returns:
        Modified trade dictionary with array fields as strings
    """
    trade_copy = trade.copy()
    
    # Convert array fields to comma-separated strings
    for field in ["Confluences", "Liquidity_Swept", "Liquidity_Target"]:
        if field in trade_copy:
            value = trade_copy[field]
            if isinstance(value, list):
                trade_copy[field] = ", ".join(str(v) for v in value) if value else ""
            elif value is None:
                trade_copy[field] = ""
    
    return trade_copy


def main():
    """Main execution function."""
    print("Trade Transcript Extractor")
    print("=" * 50)
    
    # Initialize Azure OpenAI client
    try:
        client, deployment = initialize_client()
        print(f"✓ Azure OpenAI client initialized (deployment: {deployment})")
    except Exception as e:
        print(f"✗ Failed to initialize Azure OpenAI client: {e}")
        return
    
    # Define transcriptions folder
    transcriptions_folder = Path("transcriptions")
    
    # Check if folder exists
    if not transcriptions_folder.exists():
        print(f"✗ Folder '{transcriptions_folder}' not found")
        return
    
    # Find all .txt files
    txt_files = list(transcriptions_folder.glob("*.txt"))
    
    if not txt_files:
        print(f"✗ No .txt files found in '{transcriptions_folder}' folder")
        return
    
    print(f"✓ Found {len(txt_files)} transcript file(s)")
    print(f"Processing {min(2, len(txt_files))} files concurrently...")
    print()
    
    # Process files concurrently (2 at a time)
    all_trades = []
    
    def process_file_wrapper(file_path):
        """Wrapper function for concurrent processing."""
        try:
            trades = process_transcript_file(client, deployment, file_path)
            return file_path.name, trades, None
        except Exception as e:
            return file_path.name, [], str(e)
    
    # Use ThreadPoolExecutor to process 2 files concurrently
    with ThreadPoolExecutor(max_workers=2) as executor:
        # Submit all files for processing
        future_to_file = {
            executor.submit(process_file_wrapper, file_path): file_path 
            for file_path in sorted(txt_files)
        }
        
        # Process completed tasks as they finish
        for future in as_completed(future_to_file):
            file_path = future_to_file[future]
            try:
                filename, trades, error = future.result()
                
                if error:
                    print(f"Processing: {filename}...")
                    print(f"  ✗ Error: {error}")
                elif trades:
                    print(f"Processing: {filename}...")
                    print(f"  ✓ Extracted {len(trades)} trade(s)")
                    all_trades.extend(trades)
                else:
                    print(f"Processing: {filename}...")
                    print(f"  ⚠ No trades extracted")
            except Exception as e:
                print(f"Processing: {file_path.name}...")
                print(f"  ✗ Unexpected error: {e}")
    
    print()
    
    # Check if we have any trades
    if not all_trades:
        print("✗ No trades extracted from any files")
        return
    
    print(f"✓ Total trades extracted: {len(all_trades)}")
    
    # Convert array fields to strings
    processed_trades = [convert_array_fields_to_string(trade) for trade in all_trades]
    
    # Load market metrics
    print()
    print("Loading market metrics...")
    data_file = Path("data/glbx-mdp3-20200927-20250926.ohlcv-1m.csv")
    
    metrics_df = None
    if data_file.exists():
        try:
            metrics_df = market_metrics.load_and_calculate_metrics(data_file)
            print(f"✓ Loaded market metrics for {len(metrics_df)} date-asset combinations")
        except Exception as e:
            print(f"⚠ Warning: Failed to load market metrics: {e}")
            print("  Continuing without market metrics...")
    else:
        print(f"⚠ Warning: Market data file not found: {data_file}")
        print("  Continuing without market metrics...")
    
    # Enrich trades with market metrics
    if metrics_df is not None:
        print("Enriching trades with market metrics...")
        for trade in processed_trades:
            trade_date = trade.get("Date_of_Trade")
            asset = trade.get("Asset_Ticked")
            
            metrics = market_metrics.get_metrics_for_trade(metrics_df, trade_date, asset)
            
            if metrics:
                trade.update(metrics)
            else:
                # Add empty metrics columns
                trade.update({
                    'NY_First_45min_Volume': None,
                    'NY_First_45min_Range': None,
                    'NY_First_90min_Volume': None,
                    'NY_First_90min_Range': None,
                    'NY_Full_Session_Volume': None,
                    'NY_Full_Session_Range': None,
                })
        print("✓ Enriched trades with market metrics")
    else:
        # Add empty metrics columns if metrics not available
        for trade in processed_trades:
            trade.update({
                'NY_First_45min_Volume': None,
                'NY_First_45min_Range': None,
                'NY_First_90min_Volume': None,
                'NY_First_90min_Range': None,
                'NY_Full_Session_Volume': None,
                'NY_Full_Session_Range': None,
            })
    
    print()
    
    # Define column order
    column_order = [
        "Date_of_Trade",
        "Time_of_Trade",
        "Video_Title",
        "Asset_Ticked",
        "Chart_Timeframe",
        "Gap_Size_Points",
        "Number_of_Gaps_Inversed",
        "Confluences",
        "Bias",
        "Outcome",
        "PnL_Points",
        "PnL_R_Multiple",
        "Liquidity_Swept",
        "Liquidity_Target",
        "Management_Style",
        "Entry_Execution_Grade",
        "NY_First_45min_Volume",
        "NY_First_45min_Range",
        "NY_First_90min_Volume",
        "NY_First_90min_Range",
        "NY_Full_Session_Volume",
        "NY_Full_Session_Range",
        "Key_Takeaway"
    ]
    
    # Create DataFrame
    df = pd.DataFrame(processed_trades)
    
    # Ensure all columns exist (fill missing with empty values)
    for col in column_order:
        if col not in df.columns:
            df[col] = None
    
    # Reorder columns
    df = df[column_order]
    
    # Save to CSV
    output_file = "trade_journal_master.csv"
    df.to_csv(output_file, index=False)
    
    print(f"✓ Saved {len(all_trades)} trade(s) to '{output_file}'")
    print()
    print("Done!")


if __name__ == "__main__":
    main()

