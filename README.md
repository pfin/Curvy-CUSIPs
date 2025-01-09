# Curvy-CUSIPs

Using Open Sourced Data for Rates Trading examples in Python:

- Cash Treasuries: [FedInvest EOD Markings](https://www.treasurydirect.gov/GA-FI/FedInvest/selectSecurityPriceDate)
- Swaps/Swaptions: [DTCC SDR](https://pddata.dtcc.com/ppd/cftcdashboard), [BBG SEF](https://data.bloombergsef.com/)
- SOFR OIS Curve: [Eris SOFR Swap Futures FTP](https://files.erisfutures.com/ftp/)
- Economics/Misc: [FRED](https://fred.stlouisfed.org/), [NYFRB](https://markets.newyorkfed.org/static/docs/markets-api.html)

## To get started

Clone repo:

```bash
git clone https://github.com/yieldcurvemonkey/Curvy-CUSIPs.git
```

pip install dependencies: 

```bash
pip install -r requirements.txt
```

cd into scripts dir:

```bash
cd .\Curvy-CUSIPs\scripts
```

Init Cash Treasuries DBs: 

```py
python update_ust_cusips_db.py init
```

- This will take 30-60 minutes
- This will create 6 databases
    - ust_cusip_set
    - ust_cusip_timeseries
    - ust_eod_ct_yields
    - ust_bid_ct_yields
    - ust_mid_ct_yields
    - ust_offer_ct_yields

- Data is source from FedInvest
- Data availability: daily bid, offer, mid prices and ytms at cusip level (and cusip characteristics) starting from late 2008
- Script, by default, fetches data starting from 2018
- See below for examples

Update Cash Treasuries DBs:

```py
python update_ust_cusips_db.py
```

Init SOFR OIS Curve DB:

```py
python update_sofr_ois._db.py init
```

- This will take 30-60 minutes
- This will create one database: nyclose_sofr_ois
- Curve marking from < 2024-01-12 is from the BBG 490 Curve, newer EOD markings are sourced from the Eris FTP
- See below for examples

Update SOFR OIS Curve DB:
```
python update_sofr_ois._db.py
```

Init and update ATM Swaption Vol DB:

```py
python update_atm_swaption_vol.py
```

Init Swaption Vol Cube DB

```py
python update_swaption_vol_cube.py init
```

Update Swaption Vol Cube DB

```py
python update_swaption_vol_cube.py
```

## Checkout examples in `\notebooks`

### [usts_basics.ipynb](https://github.com/yieldcurvemonkey/Curvy-CUSIPs/blob/main/notebooks/usts_basics.ipynb)

- Yield Curve Plotting
- CT Yields Dataframe
- CUSIP look up
- CUSIP plotting
- Spread, Flies plotting

### [swaps_basics.ipynb](https://github.com/yieldcurvemonkey/Curvy-CUSIPs/blob/main/notebooks/swaps_basics.ipynb)

- SOFR OIS Curve Ploting
- SOFR OIS Swap Spreads
- Spread, Flies plotting
- Swap Fwds

### [swaption_basics.ipynb](https://github.com/yieldcurvemonkey/Curvy-CUSIPs/blob/main/notebooks/swaption_basics.ipynb) 

- similar to NSV and VCUB screens 
- Historical ATM Grids
- Historical Vol Cubes
- Plotting 3 surfaces  
    - expiry-tail
    - expiry-strike
    - tail-strike
- Vol Timeseries 

### [swap_pricer.ipynb](https://github.com/yieldcurvemonkey/Curvy-CUSIPs/blob/main/notebooks/swap_pricer.ipynb)

- Recreating the SWPM screen

![womp womp](./dump/Screenshot%202025-01-08%20160645.png)


### [swaption_pricer.ipynb](https://github.com/yieldcurvemonkey/Curvy-CUSIPs/blob/main/notebooks/swaption_pricer.ipynb)

- Recreating the SWPM screen

![womp womp](./dump/swaption_pricer_demo.png)


### [par_curves.ipynb](https://github.com/yieldcurvemonkey/Curvy-CUSIPs/blob/main/notebooks/par_curves.ipynb)

- Plot your splines/par curve model against all active CUSIPs on a historical date
- Example shown is a textbook par model: filter based on some liquidity premium then fit a cubic spline

![womp womp](./dump/Screenshot%202024-12-05%20221335.png)

### [jpm_rv_example.ipynb](https://github.com/yieldcurvemonkey/Curvy-CUSIPs/blob/main/notebooks/jpm_rv_example.ipynb)

- Thats a pretty good fit given that we are using publicly sourced data!
- The par model in the example is loosely based (same knot points) on JPM's [par curve model](https://github.com/yieldcurvemonkey/Curvy-CUSIPs/blob/main/research/Linear/JPM%20Rates%20Strategy%20The%20(par)%20curves%20they%20are%20a-changin%E2%80%99%20Making%20enhancements%20to%20our%20Treasury%20%26%20TIPS%20par%20curves.%20Tue%20Jul%2023%202024.pdf), so we should expect that our residuals are similar

![womp womp](./dump/Screenshot%202024-12-05%20225137.png)
![womp womp](./dump/Screenshot%202024-12-05%20224928.png)

### [calculating_cash_hedge_ratios.ipynb](https://github.com/yieldcurvemonkey/Curvy-CUSIPs/blob/main/notebooks/calculating_cash_hedge_ratios.ipynb)

- DV01 Neutrality, Beta and PCA risk weightings

### [calculating_swap_hedge_ratios.ipynb](https://github.com/yieldcurvemonkey/Curvy-CUSIPs/blob/main/notebooks/calculating_swap_hedge_ratios.ipynb)

- Comparing with JPM's Rates Stratgey `10s/30s swap curve steepeners paired with a 16% weighted shorts in 2s` and `Buy the belly of a 6M forward 5s/10s/30s swap yield curve butterfly (-0.5:1.0:-0.5 weighted)`
- PV01 Neutrality, Beta and PCA risk weightings

### [swaps_rv.ipynb](https://github.com/yieldcurvemonkey/Curvy-CUSIPs/blob/main/notebooks/swaps_rv.ipynb)

- Using PCA and different regressions to find RV in swap spaces

![womp womp](./dump/output.png)

### [sabr_smile.ipynb](https://github.com/yieldcurvemonkey/Curvy-CUSIPs/blob/main/notebooks/sabr_smile.ipynb)

![til](./dump/sabrsmileexample.gif)

### More examples/notebooks coming soon
