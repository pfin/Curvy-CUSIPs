import sys
import os
from pathlib import Path

# Add parent directory to path
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.append(str(project_root))

from datetime import datetime
import pandas as pd
import QuantLib as ql
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from pysabr import Hagan2002LognormalSABR, Hagan2002NormalSABR

from CurvyCUSIPs.CurveDataFetcher import CurveDataFetcher
from CurvyCUSIPs.S490Swaps import S490Swaps
from CurvyCUSIPs.S490Swaptions import S490Swaptions
from CurvyCUSIPs.CurveInterpolator import GeneralCurveInterpolator

# Configure plotting
plt.style.use('ggplot')
params = {
    'legend.fontsize': 'x-large',
    'figure.figsize': (18, 10),
    'axes.labelsize': 'x-large',
    'axes.titlesize': 'x-large',
    'xtick.labelsize': 'x-large',
    'ytick.labelsize': 'x-large'
}
pylab.rcParams.update(params)

def print_trade_statistics(df: pd.DataFrame, date: datetime):
    """Print detailed statistics for trades on a given date."""
    print(f"\nDate: {date}")
    print(f"Total number of trades: {len(df)}")
    print(f"Total notional: ${df['Notional Amount'].sum():,.2f}")
    
    # Option tenor statistics
    print("\nBreakdown by Option Tenor:")
    opt_tenor_stats = df.groupby('Option Tenor').agg({
        'Notional Amount': ['count', 'sum', 'mean'],
        'Option Premium per Notional': 'mean'
    }).round(4)
    opt_tenor_stats.columns = ['Trade Count', 'Total Notional', 'Avg Notional', 'Avg Premium/Notional']
    print(opt_tenor_stats.sort_values('Trade Count', ascending=False))
    
    # Underlying tenor statistics
    print("\nBreakdown by Underlying Tenor:")
    und_tenor_stats = df.groupby('Underlying Tenor').agg({
        'Notional Amount': ['count', 'sum', 'mean'],
        'Option Premium per Notional': 'mean'
    }).round(4)
    und_tenor_stats.columns = ['Trade Count', 'Total Notional', 'Avg Notional', 'Avg Premium/Notional']
    print(und_tenor_stats.sort_values('Trade Count', ascending=False))
    
    # Cross table of counts
    print("\nTrade Count by Option x Underlying Tenor:")
    tenor_matrix = pd.crosstab(df['Option Tenor'], df['Underlying Tenor'])
    print(tenor_matrix)
    
    # Cross table of total notional
    print("\nTotal Notional by Option x Underlying Tenor (millions):")
    notional_matrix = pd.crosstab(
        df['Option Tenor'], 
        df['Underlying Tenor'], 
        values=df['Notional Amount'],
        aggfunc='sum'
    ) / 1_000_000
    print(notional_matrix.round(2))
    
    # Last 5 trades of the day
    print("\nLast 5 trades of the day:")
    last_trades = df.sort_values('Execution Timestamp').tail(5)
    display_cols = [
        'Execution Timestamp', 'Option Tenor', 'Underlying Tenor', 
        'Strike Price', 'Option Premium per Notional', 'Notional Amount',
        'Direction', 'Style'
    ]
    pd.set_option('display.float_format', lambda x: '%.6f' % x)
    print(last_trades[display_cols].to_string())
    
    # Print time range
    print(f"\nTrading time range: {df['Execution Timestamp'].min()} to {df['Execution Timestamp'].max()}")

def main():
    # Initialize data fetcher
    curve_data_fetcher = CurveDataFetcher()
    
    # Set analysis dates
    as_of_start_date = datetime(2025, 1, 17)
    as_of_end_date = datetime(2025, 1, 27)

    try:
        # Fetch swaption trades directly from DTCC
        swaption_trades = curve_data_fetcher.dtcc_sdr_fetcher.fetch_historical_swaption_time_and_sales(
            start_date=as_of_start_date,
            end_date=as_of_end_date,
            underlying_swap_types=["Fixed_Float_OIS"],
            underlying_reference_floating_rates=[
                "USD-SOFR-OIS Compound",
                "USD-SOFR-COMPOUND", 
                "USD-SOFR",
                "USD-SOFR Compounded Index",
                "USD-SOFR CME Term"
            ],
            underlying_ccy="USD",
            underlying_reference_floating_rate_term_value=1,
            underlying_reference_floating_rate_term_unit="DAYS",
            underlying_notional_schedule="Constant",
            underlying_delivery_types=["CASH", "PHYS"],
            swaption_exercise_styles=["European"]
        )

        # Print detailed statistics for each day
        print("\n=== Detailed Swaption Trade Statistics ===")
        for date, df in swaption_trades.items():
            print("\n" + "="*50)
            print_trade_statistics(df, date)

        # Setup database paths
        notebook_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        db_path = os.path.join(notebook_dir, "db", "nyclose_sofr_ois")
        atm_vol_db_path = os.path.join(notebook_dir, "db", "s490_swaption_atm_vol")
        vol_cube_db_path = os.path.join(notebook_dir, "db", "s490_swaption_vol_cube")
        
        print(f"\nUsing database paths:")
        print(f"Curves DB: {db_path}")
        print(f"ATM Vol DB: {atm_vol_db_path}")
        print(f"Vol Cube DB: {vol_cube_db_path}")

        # Initialize S490 objects with all necessary DB paths
        s490_swaps = S490Swaps(s490_curve_db_path=db_path, curve_data_fetcher=curve_data_fetcher)
        s490_swaptions = S490Swaptions(
            s490_swaps=s490_swaps,
            atm_vol_timeseries_db_path=atm_vol_db_path,
            s490_vol_cube_timeseries_db_path=vol_cube_db_path
        )

        # Mark ATM vol term structure
        s490_swaptions.mark_s490_atm_vol_term_structure_markings_db(
            date=as_of_start_date,
            option_maturity="12M",
            vol_type="bp_vol",
            show_plot=True,
            gci_interp_func_str="monotone_convex",
            hard_coded_vols={2: 7.11}
        )

        # Create SABR smile using the fetched trades
        s490_swaptions.create_sabr_smile_interactive(
            swaption_time_and_sales=swaption_trades[as_of_start_date],
            option_tenor="12M",
            underlying_tenor="10Y",
            initial_beta=0.5,
            show_trades=True,
            scale_by_notional=False,
            model="normal",
            implementation="rss",
            year_day_count=252,
            ploty_height=1250,
            receiver_skew_anchor_bpvol=0.01,
            payer_skew_anchor_bpvol=0.75,
            anchor_weight=25,
            skew_offset_anchor_bps=100,
            drop_trades_idxs=[22, 23, 8, 9]
        )

    except Exception as e:
        print(f"Error in analysis: {str(e)}")
        raise

if __name__ == "__main__":
    main() 