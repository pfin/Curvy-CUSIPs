import sys 
from pathlib import Path
import asyncio
from datetime import datetime
import pandas as pd
import httpx

# Add parent directory to path
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.append(str(project_root))

async def fetch_slice(agency: str, date: datetime, slice_num: int) -> Optional[pd.DataFrame]:
    """Fetch a single intraday slice"""
    url = f"https://pddata.dtcc.com/ppd/api/report/intraday/{agency}/{agency}_SLICE_RATES_{date.strftime('%Y_%m_%d')}_{slice_num}.zip"
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=10)
            if response.status_code == 404:
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

async def download_intraday(agency: str, date: datetime) -> pd.DataFrame:
    """Download all slices for a given day"""
    all_trades = []
    slice_num = 1
    
    while True:
        print(f"Fetching {agency} slice {slice_num}...")
        df = await fetch_slice(agency, date, slice_num)
        if df is None:
            break
            
        all_trades.append(df)
        slice_num += 1
        
    if all_trades:
        return pd.concat(all_trades, ignore_index=True)
    return pd.DataFrame()

async def main():
    """Download and save intraday data"""
    today = datetime.now()
    
    # Download CFTC data
    cftc_trades = await download_intraday("CFTC", today)
    if not cftc_trades.empty:
        output_file = f"dtcc_cftc_rates_{today.strftime('%Y%m%d')}.csv"
        cftc_trades.to_csv(output_file, index=False)
        print(f"Saved {len(cftc_trades)} CFTC trades to {output_file}")
    
    # Download SEC data
    sec_trades = await download_intraday("SEC", today)
    if not sec_trades.empty:
        output_file = f"dtcc_sec_rates_{today.strftime('%Y%m%d')}.csv"
        sec_trades.to_csv(output_file, index=False)
        print(f"Saved {len(sec_trades)} SEC trades to {output_file}")

if __name__ == "__main__":
    asyncio.run(main())