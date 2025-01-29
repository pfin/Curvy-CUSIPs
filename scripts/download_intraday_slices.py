import sys 
from pathlib import Path
import asyncio
from datetime import datetime
from typing import Optional, Tuple
import pandas as pd
import httpx
from io import BytesIO
import zipfile
import os

# Add parent directory to path
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.append(str(project_root))

async def fetch_slice(agency: str, date: datetime, slice_num: int, product_type: str = "RATES") -> Optional[pd.DataFrame]:
    """Fetch a single intraday slice"""
    url = f"https://pddata.dtcc.com/ppd/api/report/intraday/{agency.lower()}/{agency.upper()}_SLICE_{product_type}_{date.strftime('%Y_%m_%d')}_{slice_num}.zip"
    
    print(f"Testing URL: {url}")
    
    # Test URL with curl
    try:
        import subprocess
        result = subprocess.run(['curl', '-I', url], capture_output=True, text=True)
        print("Curl response:")
        print(result.stdout)
    except Exception as e:
        print(f"Error testing URL: {e}")
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=10)
            if response.status_code == 404:
                print(f"URL returned 404 - slice does not exist")
                return None
            response.raise_for_status()
            
            # Extract CSV/Excel from zip
            with zipfile.ZipFile(BytesIO(response.content)) as zip_file:
                for info in zip_file.infolist():
                    if info.filename.endswith(('.csv', '.xlsx', '.xls')):
                        with zip_file.open(info) as f:
                            if info.filename.endswith('.csv'):
                                return pd.read_csv(f)
                            return pd.read_excel(f)
    except Exception as e:
        print(f"Error fetching slice {slice_num}: {e}")
        return None

async def download_intraday(agency: str, date: datetime, product_type: str = "RATES") -> Tuple[pd.DataFrame, int]:
    """Download all slices for a given day"""
    all_trades = []
    slice_num = 1
    total_trades = 0
    
    # Create directory for the date if it doesn't exist
    date_dir = project_root / 'db' / date.strftime('%Y%m%d')
    date_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nStarting {agency} download for {date.strftime('%Y-%m-%d')}")
    
    while True:
        # Check if the file already exists
        output_file = date_dir / f"{agency.lower()}_{product_type.lower()}_slice_{slice_num}.csv"
        if output_file.exists():
            print(f"File {output_file} already exists. Skipping download.")
            slice_num += 1
            continue
        
        print(f"\nFetching {agency} slice {slice_num}...")
        df = await fetch_slice(agency, date, slice_num, product_type)
        if df is None:
            print(f"No more slices found for {agency} (stopped at slice {slice_num-1})")
            break
            
        num_trades = len(df)
        total_trades += num_trades
        print(f"Found {num_trades} trades in slice {slice_num}")
        print(f"Current total trades: {total_trades}")
        
        # Show sample of new trades
        if not df.empty:
            print("\nSample trades from this slice:")
            # Use more generic column names that exist in both intraday and historical
            cols_to_show = []
            for col in ['Event timestamp', 'Execution Timestamp', 'Notional Amount', 
                       'Notional amount', 'Notional amount-Leg 1', 'Notional amount-Leg 2']:
                if col in df.columns:
                    cols_to_show.append(col)
            
            if cols_to_show:
                print(df[cols_to_show].head())
            else:
                print("No standard columns found in data")
        
        # Save the DataFrame to a CSV file
        df.to_csv(output_file, index=False)
        print(f"Saved slice {slice_num} to {output_file}")
        
        all_trades.append(df)
        slice_num += 1
        
    print(f"\nFinished {agency} download")
    print(f"Total slices processed: {slice_num-1}")
    print(f"Total trades found: {total_trades}")
    
    if all_trades:
        return pd.concat(all_trades, ignore_index=True), total_trades
    return pd.DataFrame(), 0

async def main():
    """Download and save intraday data"""
    today = datetime.now()
    
    # Download CFTC data
    cftc_trades, cftc_total = await download_intraday("CFTC", today, "RATES")
    if not cftc_trades.empty:
        print(f"\nSaved {cftc_total} CFTC trades to the db directory for {today.strftime('%Y%m%d')}")
    
    # Download SEC data
    sec_trades, sec_total = await download_intraday("SEC", today, "RATES")
    if not sec_trades.empty:
        print(f"\nSaved {sec_total} SEC trades to the db directory for {today.strftime('%Y%m%d')}")

if __name__ == "__main__":
    asyncio.run(main())