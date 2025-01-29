import sys
from pathlib import Path
import asyncio
from datetime import datetime
import pandas as pd

# Add parent directory to path
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.append(str(project_root))

from CurvyCUSIPs.CurveDataFetcher import CurveDataFetcher

async def test_intraday_fetching():
    """Test intraday DTCC data fetching and save raw data"""
    
    curve_data_fetcher = CurveDataFetcher()
    today = datetime.now()
    
    print(f"\nFetching DTCC intraday RATES data for {today.date()}")
    
    try:
        # Fetch CFTC data
        cftc_data = await curve_data_fetcher.dtcc_sdr_fetcher.fetch_intraday_sdr_data(
            date=today,
            agency="CFTC",
            asset_class="RATES",
            background=False
        )
        
        if cftc_data:
            # Combine all slices
            all_cftc_trades = pd.concat(cftc_data.values(), ignore_index=True)
            
            # Save raw data
            output_file = f"dtcc_cftc_rates_{today.strftime('%Y%m%d')}.csv"
            all_cftc_trades.to_csv(output_file, index=False)
            print(f"Saved {len(all_cftc_trades)} CFTC trades to {output_file}")
            
        # Fetch SEC data
        sec_data = await curve_data_fetcher.dtcc_sdr_fetcher.fetch_intraday_sdr_data(
            date=today,
            agency="SEC", 
            asset_class="RATES",
            background=False
        )
        
        if sec_data:
            # Combine all slices
            all_sec_trades = pd.concat(sec_data.values(), ignore_index=True)
            
            # Save raw data
            output_file = f"dtcc_sec_rates_{today.strftime('%Y%m%d')}.csv"
            all_sec_trades.to_csv(output_file, index=False)
            print(f"Saved {len(all_sec_trades)} SEC trades to {output_file}")

    except Exception as e:
        print(f"Error: {str(e)}")

def main():
    try:
        asyncio.run(test_intraday_fetching())
    except RuntimeError as e:
        if "asyncio.run() cannot be called from a running event loop" in str(e):
            loop = asyncio.get_event_loop()
            loop.run_until_complete(test_intraday_fetching())
        else:
            raise

if __name__ == "__main__":
    main() 