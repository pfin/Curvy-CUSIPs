import shelve
import time
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import QuantLib as ql
import scipy.interpolate
import tqdm
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import BDay, BMonthEnd, CustomBusinessDay
from termcolor import colored

import sys

sys.path.insert(0, "../")
from CurvyCUSIPs.CurveDataFetcher import CurveDataFetcher
from CurvyCUSIPs.utils.dtcc_swaps_utils import DEFAULT_SWAP_TENORS, tenor_to_years, build_ql_piecewise_curves, datetime_to_ql_date, tenor_to_ql_period
from CurvyCUSIPs.utils.ShelveDBWrapper import ShelveDBWrapper

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)

t_main = time.time()

verbose = True
DB_PATH = "../db/nyclose_sofr_ois"

if __name__ == "__main__":

    if len(sys.argv) >= 2 and sys.argv[1] == "init":
        data_fetcher = CurveDataFetcher(error_verbose=verbose)
        df = pd.read_excel("../db/xlsx/bbg_s490_data_dump.xlsx", sheet_name="data")
        columns = [
            "Date",
            "1M",
            "2M",
            "3M",
            "4M",
            "5M",
            "6M",
            "7M",
            "8M",
            "9M",
            "10M",
            "11M",
            "12M",
            "13M",
            "14M",
            "15M",
            "16M",
            "17M",
            "18M",
            "19M",
            "20M",
            "21M",
            "22M",
            "23M",
            "2Y",
            "2Y3M",
            "2Y6M",
            "2Y9M",
            "3Y",
            "4Y",
            "5Y",
            "6Y",
            "7Y",
            "8Y",
            "9Y",
            "10Y",
            "15Y",
            "20Y",
            "25Y",
            "30Y",
            "40Y",
            "50Y",
        ]
        df.columns = columns
        df = df.drop(columns=["2Y3M", "2Y6M", "2Y9M"])
        df["Date"] = pd.to_datetime(df["Date"])

        sofr_fixing_rates_df = data_fetcher.nyfrb_data_fetcher.get_sofr_fixings_df(
            start_date=datetime(2018, 4, 1), end_date=(datetime.today() - BDay(1)).to_pydatetime()
        )
        sofr_fixing_rates_df["percentRate"] = sofr_fixing_rates_df["percentRate"] / 100
        sofr_fixing_rates_df = sofr_fixing_rates_df[["effectiveDate", "percentRate"]]
        
        historical_sofr_df = pd.read_excel(
            "../db/xlsx/calculated_historical_sofr_fixings.xlsx", sheet_name="VWM Rates", header=1, usecols=[0, 1, 2, 3]
        )
        historical_sofr_df["effectiveDate"] = pd.to_datetime(historical_sofr_df["Date"], errors="coerce")
        historical_sofr_df["percentRate"] = historical_sofr_df["Secured Overnight Financing Rate"] / 100 / 100
        historical_sofr_df = historical_sofr_df[["effectiveDate", "percentRate"]]
        
        sofr_fixing_rates_df = pd.concat([historical_sofr_df, sofr_fixing_rates_df]).reset_index(drop=True)
        sofr_fixing_rates_df = sofr_fixing_rates_df.drop_duplicates(subset=["effectiveDate"])
        sofr_fixing_rates_df = sofr_fixing_rates_df.sort_values(by=["effectiveDate"])
        
        eris_dict = data_fetcher.eris_data_fetcher.fetch_eris_ftp_timeseries(
            start_date=datetime(2024, 1, 12), end_date=(datetime.today() - BDay(1)).to_pydatetime()
        )

        start_date = datetime(2018, 1, 1)
        end_date = (datetime.today() - BDay(1)).to_pydatetime()
        print_sofr_fixing_rates_df = True 
        
    else:
        if len(sys.argv) >= 2 and sys.argv[1] == "use_today":
            day_num_offset = 0
        else:
            day_num_offset = 1

        my_db = ShelveDBWrapper(DB_PATH)
        my_db.open()
        most_recent_db_dt = datetime.fromtimestamp(int(max(my_db.keys())))
        bday_offset = ((datetime.today() - BDay(day_num_offset)) - most_recent_db_dt).days

        if not (len(sys.argv) >= 2 and sys.argv[1] == "use_today"):
            if bday_offset == 0 and len(sys.argv) == 1:
                print(colored("DB is up to date - exiting...", "green"))
                sys.exit()

        data_fetcher = CurveDataFetcher(error_verbose=verbose)
        start_date = (datetime.today() - BDay(bday_offset)).to_pydatetime()
        end_date = datetime.today() if len(sys.argv) > 1 else (datetime.today() - BDay(day_num_offset)).to_pydatetime()
        print_sofr_fixing_rates_df = True 
        try:
            sofr_fixing_rates_df = data_fetcher.nyfrb_data_fetcher.get_sofr_fixings_df(start_date=start_date, end_date=end_date)
        except:
            print_sofr_fixing_rates_df = False
            print(colored("are you running this on the weekend? - NYFRB's website is under maintainence", "yellow"))
        eris_dict = data_fetcher.eris_data_fetcher.fetch_eris_ftp_timeseries(start_date=start_date, end_date=end_date)

    if verbose and print_sofr_fixing_rates_df:
        print("sofr_fixing_rates_df: ")
        print(sofr_fixing_rates_df)

    calendar = ql.UnitedStates(ql.UnitedStates.GovernmentBond)
    day_count = ql.Actual360()
    fixing_lag = 1

    errors = []
    quantlib_term_structures = [
        "logLinearDiscount",
        "logCubicDiscount",
        "linearZero",
        "cubicZero",
        "linearForward",
        "splineCubicDiscount",
    ]

    with shelve.open(DB_PATH) as db:
        for curr_bdate in tqdm.tqdm(
            pd.date_range(start=start_date, end=end_date, freq=CustomBusinessDay(calendar=USFederalHolidayCalendar())),
            desc="Writing to DB...",
        ):
            curr_bdate_dt = curr_bdate.to_pydatetime().replace(hour=0, minute=0, second=0, microsecond=0)

            try:
                print(colored(f"\n Starting {curr_bdate_dt}", "green")) if verbose else None
                t1 = time.time()

                if len(sys.argv) == 2 and sys.argv[1] == "init":
                    if curr_bdate_dt > datetime(2024, 1, 12):
                        curr_df = eris_dict[curr_bdate_dt]
                        curr_ohlc_df = pd.DataFrame(
                            {
                                "Expiry": pd.to_datetime(curr_df["MaturityDate"], errors="coerce", format="%m/%d/%Y"),
                                "Tenor": [sym[4:] for sym in curr_df["Symbol"].to_list()],
                                "Close": curr_df["Coupon (%)"].to_numpy() / 100,
                            }
                        )
                        if "1D" not in curr_ohlc_df["Tenor"].to_numpy():
                            on_rate = sofr_fixing_rates_df[sofr_fixing_rates_df["effectiveDate"].dt.date == curr_bdate_dt.date()][
                                "percentRate"
                            ].iloc[-1]
                            on_expiry: pd.Timestamp = curr_bdate_dt + BDay(2)
                            curr_ohlc_df.loc[-1] = [on_expiry.to_pydatetime(), "1D", on_rate]
                            curr_ohlc_df.index = curr_ohlc_df.index + 1
                            curr_ohlc_df.sort_index(inplace=True)
                    else:
                        curr_df = df[df["Date"].dt.date == curr_bdate_dt.date()]
                        tenors = ["1D"] + curr_df.columns[1:].to_list()
                        close_rates = [
                            sofr_fixing_rates_df[sofr_fixing_rates_df["effectiveDate"].dt.date == curr_bdate_dt.date()]["percentRate"].iloc[-1]
                        ] + list(curr_df.values[0][1:] / 100)
                        curr_ohlc_df = pd.DataFrame(
                            {
                                "Expiry": [
                                    calendar.advance(datetime_to_ql_date(curr_bdate_dt), tenor_to_ql_period(tenor)).to_date() for tenor in tenors
                                ],
                                "Tenor": tenors,
                                "Close": close_rates,
                            }
                        )

                else:
                    curr_df = eris_dict[curr_bdate_dt]
                    curr_ohlc_df = pd.DataFrame(
                        {
                            "Expiry": pd.to_datetime(curr_df["MaturityDate"], errors="coerce", format="%m/%d/%Y"),
                            "Tenor": [sym[4:] for sym in curr_df["Symbol"].to_list()],
                            "Close": curr_df["Coupon (%)"].to_numpy() / 100,
                        }
                    )
                    if "1D" not in curr_ohlc_df["Tenor"].to_numpy():
                        on_rate = (
                            sofr_fixing_rates_df[sofr_fixing_rates_df["effectiveDate"].dt.date == curr_bdate_dt.date()]["percentRate"].iloc[-1]
                            / 100
                        )
                        on_expiry: pd.Timestamp = curr_bdate_dt + BDay(2)
                        curr_ohlc_df.loc[-1] = [on_expiry.to_pydatetime(), "1D", on_rate]
                        curr_ohlc_df.index = curr_ohlc_df.index + 1
                        curr_ohlc_df.sort_index(inplace=True)

                curr_ohlc_df = curr_ohlc_df[curr_ohlc_df["Close"].notna()]
                curr_ohlc_df["Expiry"] = pd.to_datetime(curr_ohlc_df["Expiry"])
                curr_ohlc_df = curr_ohlc_df[curr_ohlc_df["Tenor"].isin(DEFAULT_SWAP_TENORS)]

                x = np.array([tenor_to_years(t) for t in curr_ohlc_df["Tenor"]])
                x_new = np.array([tenor_to_years(t) for t in DEFAULT_SWAP_TENORS])
                interp_df = pd.DataFrame({"Tenor": DEFAULT_SWAP_TENORS})

                tenor_map = {
                    tenor_to_years(t): {col: curr_ohlc_df.loc[curr_ohlc_df["Tenor"] == t, col].values[0] for col in ["Close"]}
                    for t in curr_ohlc_df["Tenor"]
                }

                def log_linear_interp1d(x, y, fill_value="extrapolate", **kwargs):
                    log_y = np.log(y)
                    interp_func = scipy.interpolate.interp1d(x, log_y, kind="linear", fill_value=fill_value, **kwargs)

                    def log_linear_interp(x_new):
                        return np.exp(interp_func(x_new))

                    return log_linear_interp

                for col in ["Close"]:
                    y = curr_ohlc_df[col].values
                    y_new = log_linear_interp1d(x, y)(x_new)
                    interp_df[col] = [
                        tenor_map[tenor_to_years(t)][col] if tenor_to_years(t) in tenor_map else y_val
                        for t, y_val in zip(DEFAULT_SWAP_TENORS, y_new)
                    ]

                settlement_date: pd.Timestamp = curr_bdate_dt + BDay(2)
                is_end_of_bmonth = BMonthEnd().rollforward(settlement_date) == settlement_date
                interp_df.insert(
                    1,
                    "Expiry",
                    [
                        ql.UnitedStates(ql.UnitedStates.GovernmentBond)
                        .advance(
                            datetime_to_ql_date(settlement_date.to_pydatetime()), tenor_to_ql_period(t), ql.ModifiedFollowing, is_end_of_bmonth
                        )
                        .to_date()
                        for t in interp_df["Tenor"]
                    ],
                )
                interp_df = interp_df.sort_values("Expiry").reset_index(drop=True)
                interp_df["Expiry"] = pd.to_datetime(interp_df["Expiry"], errors="coerce")
                interp_df["Close"] = interp_df["Close"].clip(lower=0)
                print(interp_df) if verbose else None

                try:
                    ql_pw_curves = build_ql_piecewise_curves(
                        df=interp_df,
                        as_of_date=curr_bdate_dt,
                        settlement_t_plus=2,
                        payment_lag=2,
                        is_ois=True,
                        tenor_col="Tenor",
                        fixed_rate_col="Close",
                        fixed_leg_frequency=ql.Annual,
                        fixed_leg_daycount=day_count,
                        fixed_leg_convention=ql.ModifiedFollowing,
                        fixed_leg_calendar=calendar,
                        ql_index=ql.OvernightIndex("SOFR", fixing_lag, ql.USDCurrency(), calendar, day_count, ql.YieldTermStructureHandle()),
                        logLinearDiscount=True,
                        logCubicDiscount=True,
                        linearZero=True,
                        cubicZero=True,
                        linearForward=True,
                        splineCubicDiscount=True,
                    )
                except Exception as e:
                    print(f'"fetch_historical_swaps_term_structure" --- Quantlib Error at {curr_bdate_dt} --- {str(e)}') if verbose else None
                    errors.append({"Date": curr_bdate_dt, "Error Message": str(e)})
                    ql_pw_curves = {
                        "logLinearDiscount": None,
                        "logCubicDiscount": None,
                        "linearZero": None,
                        "cubicZero": None,
                        "linearForward": None,
                        "splineCubicDiscount": None,
                    }

                interp_df["Expiry"] = interp_df["Expiry"].apply(lambda x: str(int(x.timestamp())))
                curr_term_struct_obj = {
                    "ohlc": interp_df.to_dict("records"),
                    "time_and_sales": None,
                    "logLinearDiscount": None,
                    "logCubicDiscount": None,
                    "linearZero": None,
                    "cubicZero": None,
                    "linearForward": None,
                    "splineCubicDiscount": None,
                }

                for ql_ts in quantlib_term_structures:
                    try:
                        curr_curve = ql_pw_curves[ql_ts]
                        if curr_curve is None:
                            raise ValueError("QuantLib Curve is None")

                        curr_curve.enableExtrapolation()

                        max_retries = 5
                        success = False
                        while not success and max_retries > 0:
                            try:
                                (
                                    print(colored(f"{curr_bdate_dt}-{ql_ts} Quantlib Nodes Calc --- Attempts remaining: {max_retries}", "yellow"))
                                    if verbose
                                    else None
                                )

                                if max_retries == 5:
                                    evaluation_date = datetime_to_ql_date(curr_bdate_dt)
                                    ql.Settings.instance().evaluationDate = evaluation_date

                                # idk why
                                if curr_bdate_dt.date() == datetime(2022, 12, 29).date() and ql_ts == "splineCubicDiscount":
                                    ql.Settings.instance().evaluationDate = datetime_to_ql_date(curr_bdate_dt + BDay(3))

                                nodes = curr_curve.nodes()
                                success = True
                                dates = [datetime(node[0].year(), node[0].month(), node[0].dayOfMonth()) for node in nodes]
                                discount_factors = [node[1] for node in nodes]
                                curr_term_struct_obj[ql_ts] = dict(zip(dates, discount_factors))
                            except Exception as e:
                                max_retries -= 1

                                if max_retries == 4:
                                    try:
                                        ql.Settings.instance().evaluationDate = datetime_to_ql_date(curr_bdate_dt + BDay(1))
                                    except:
                                        max_retries -= 1

                                if max_retries == 3:
                                    try:
                                        ql.Settings.instance().evaluationDate = datetime_to_ql_date(curr_bdate_dt - BDay(1))
                                    except:
                                        max_retries -= 1

                                if max_retries == 2:
                                    try:
                                        ql.Settings.instance().evaluationDate = datetime_to_ql_date(curr_bdate_dt + BDay(2))
                                    except:
                                        max_retries -= 1

                                if max_retries == 1:
                                    try:
                                        ql.Settings.instance().evaluationDate = datetime_to_ql_date(curr_bdate_dt - BDay(2))
                                    except:
                                        max_retries -= 1

                                if max_retries == 0:
                                    print(colored(f"Failed after 3 retries for {curr_bdate_dt}-{ql_ts}: {e}", "red")) if verbose else None
                                    errors.append(
                                        {"Date": curr_bdate_dt, "Error Message": f"Failed after 3 retries for {curr_bdate_dt}-{ql_ts}: {e}"}
                                    )
                                    curr_term_struct_obj[ql_ts] = None

                    except Exception as e:
                        err_mess = f"Failed: {curr_bdate_dt} --- Quantlib bootstrap error at {ql_ts} {curr_bdate_dt} --- {e}"
                        print(colored(err_mess, "red")) if verbose else None
                        errors.append({"Date": curr_bdate_dt, "Error Message": err_mess})
                        curr_term_struct_obj[ql_ts] = None

                db[str(int(curr_bdate_dt.timestamp()))] = curr_term_struct_obj
                (
                    print(colored(f"Group: {curr_bdate_dt} - {curr_bdate_dt} --- DB Write took: {time.time() - t1} seconds", "green"))
                    if verbose
                    else None
                )

            except Exception as e:
                print(colored(f"Something blew up - Error Message: {e}", "red"))
                errors.append({"Date": curr_bdate_dt, "Error Message": str(e)})

    print(
        "Error Report: ",
    )
    print(pd.DataFrame(errors))
    print("Script Took (seconds): ", time.time() - t_main)
