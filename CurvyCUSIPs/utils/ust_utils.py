import asyncio
import math
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from itertools import product
from typing import Dict, List, Literal, Optional, Tuple, TypeAlias

import aiohttp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import QuantLib as ql
import rateslib as rl
import requests
import scipy
import ujson as json
from pandas.tseries.offsets import BDay
from scipy.optimize import minimize

JSON: TypeAlias = dict[str, "JSON"] | list["JSON"] | str | int | float | bool | None


def auction_df_filterer(historical_auctions_df: pd.DataFrame):
    historical_auctions_df = historical_auctions_df.copy()
    historical_auctions_df["issue_date"] = pd.to_datetime(
        historical_auctions_df["issue_date"]
        # , errors="coerce"
    )
    historical_auctions_df["maturity_date"] = pd.to_datetime(
        historical_auctions_df["maturity_date"]
        # , errors="coerce"
    )
    historical_auctions_df["auction_date"] = pd.to_datetime(
        historical_auctions_df["auction_date"]
        # , errors="coerce"
    )
    historical_auctions_df.loc[
        historical_auctions_df["original_security_term"].str.contains("29-Year", case=False, na=False),
        "original_security_term",
    ] = "30-Year"
    historical_auctions_df.loc[
        historical_auctions_df["original_security_term"].str.contains("30-", case=False, na=False),
        "original_security_term",
    ] = "30-Year"
    historical_auctions_df = historical_auctions_df[
        (historical_auctions_df["security_type"] == "Bill")
        | (historical_auctions_df["security_type"] == "Note")
        | (historical_auctions_df["security_type"] == "Bond")
    ]
    return historical_auctions_df


def build_treasurydirect_header(
    host_str: Optional[str] = "api.fiscaldata.treasury.gov",
    cookie_str: Optional[str] = None,
    origin_str: Optional[str] = None,
    referer_str: Optional[str] = None,
):
    return {
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7,application/json",
        "Accept-Encoding": "gzip, deflate, br, zstd",
        "Accept-Language": "en-US,en;q=0.9",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "Cookie": cookie_str or "",
        "DNT": "1",
        "Host": host_str or "",
        "Origin": origin_str or "",
        "Referer": referer_str or "",
        "Pragma": "no-cache",
        "Sec-CH-UA": '"Not)A;Brand";v="99", "Google Chrome";v="127", "Chromium";v="127"',
        "Sec-CH-UA-Mobile": "?0",
        "Sec-CH-UA-Platform": '"Windows"',
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-User": "?1",
        "Upgrade-Insecure-Requests": "1",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36",
    }


# n == 0 => On-the-runs
def get_last_n_off_the_run_cusips(
    auction_json: Optional[JSON] = None,
    auctions_df: Optional[pd.DataFrame] = None,
    n=0,
    filtered=False,
    as_of_date=datetime.today(),
    use_issue_date=False,
) -> List[Dict[str, str]] | pd.DataFrame:
    if not auction_json and auctions_df is None:
        return pd.DataFrame(columns=historical_auction_cols())

    if auction_json and auctions_df is None:
        auctions_df = pd.DataFrame(auction_json)

    auctions_df = auctions_df[
        (auctions_df["security_type"] != "TIPS")
        & (auctions_df["security_type"] != "TIPS Note")
        & (auctions_df["security_type"] != "TIPS Bond")
        & (auctions_df["security_type"] != "FRN")
        & (auctions_df["security_type"] != "FRN Note")
        & (auctions_df["security_type"] != "FRN Bond")
        & (auctions_df["security_type"] != "CMB")
    ]
    # auctions_df = auctions_df.drop(
    #     auctions_df[
    #         (auctions_df["security_type"] == "Bill")
    #         & (
    #             auctions_df["original_security_term"]
    #             != auctions_df["security_term_week_year"]
    #         )
    #     ].index
    # )
    auctions_df["auction_date"] = pd.to_datetime(auctions_df["auction_date"])
    auctions_df["issue_date"] = pd.to_datetime(auctions_df["issue_date"])
    current_date = as_of_date
    auctions_df = auctions_df[auctions_df["auction_date" if not use_issue_date else "issue_date"].dt.date <= current_date.date()]
    auctions_df = auctions_df.sort_values("auction_date" if not use_issue_date else "issue_date", ascending=False)

    mapping = {
        "4-Week": 0.077,
        "8-Week": 0.15,
        "13-Week": 0.25,
        "17-Week": 0.33,
        "26-Week": 0.5,
        "52-Week": 1,
        "2-Year": 2,
        "3-Year": 3,
        "5-Year": 5,
        "7-Year": 7,
        "10-Year": 10,
        "20-Year": 20,
        "30-Year": 30,
    }

    on_the_run = auctions_df.groupby("original_security_term").first().reset_index()
    on_the_run = on_the_run[(on_the_run["security_type"] == "Note") | (on_the_run["security_type"] == "Bond")]
    on_the_run_result = on_the_run[
        [
            "original_security_term",
            "security_type",
            "cusip",
            "auction_date",
            "issue_date",
        ]
    ]

    on_the_run_bills = auctions_df.groupby("security_term").first().reset_index()
    on_the_run_bills = on_the_run_bills[on_the_run_bills["security_type"] == "Bill"]
    on_the_run_result_bills = on_the_run_bills[
        [
            "original_security_term",
            "security_type",
            "cusip",
            "auction_date",
            "issue_date",
        ]
    ]

    on_the_run = pd.concat([on_the_run_result_bills, on_the_run_result])

    if n == 0:
        return on_the_run

    off_the_run = auctions_df[~auctions_df.index.isin(on_the_run.index)]
    off_the_run_result = off_the_run.groupby("original_security_term").nth(list(range(1, n + 1))).reset_index()

    combined_result = pd.concat([on_the_run_result, off_the_run_result], ignore_index=True)
    combined_result = combined_result.sort_values(by=["original_security_term", "issue_date"], ascending=[True, False])

    combined_result["target_tenor"] = combined_result["original_security_term"].replace(mapping)
    mask = combined_result["original_security_term"].isin(mapping.keys())
    mapped_and_filtered_df = combined_result[mask]
    grouped = mapped_and_filtered_df.groupby("original_security_term")
    max_size = grouped.size().max()
    wrapper = []
    for i in range(max_size):
        sublist = []
        for _, group in grouped:
            if i < len(group):
                sublist.append(group.iloc[i].to_dict())
        sublist = sorted(sublist, key=lambda d: d["target_tenor"])
        if filtered:
            wrapper.append({auctioned_dict["target_tenor"]: auctioned_dict["cusip"] for auctioned_dict in sublist})
        else:
            wrapper.append(sublist)

    return wrapper


def get_historical_on_the_run_cusips(
    auctions_df: pd.DataFrame,
    as_of_date=datetime.today(),
    use_issue_date=False,
) -> pd.DataFrame:

    current_date = as_of_date
    auctions_df = auctions_df[auctions_df["auction_date" if not use_issue_date else "issue_date"].dt.date <= current_date.date()]
    auctions_df = auctions_df[auctions_df["maturity_date"].dt.date >= current_date.date()]
    auctions_df = auctions_df.sort_values("auction_date" if not use_issue_date else "issue_date", ascending=False)

    mapping = {
        "17-Week": 0.25,
        "26-Week": 0.5,
        "52-Week": 1,
        "2-Year": 2,
        "3-Year": 3,
        "5-Year": 5,
        "7-Year": 7,
        "10-Year": 10,
        "20-Year": 20,
        "30-Year": 30,
    }

    on_the_run_df = auctions_df.groupby("original_security_term").first().reset_index()
    on_the_run_filtered_df = on_the_run_df[
        [
            "original_security_term",
            "security_type",
            "cusip",
            "auction_date",
            "issue_date",
        ]
    ]
    on_the_run_filtered_df["target_tenor"] = on_the_run_filtered_df["original_security_term"].replace(mapping)

    return on_the_run_filtered_df


def get_active_cusips(
    auction_json: Optional[JSON] = None,
    historical_auctions_df: Optional[pd.DataFrame] = None,
    as_of_date=datetime.today(),
    use_issue_date=False,
) -> pd.DataFrame:
    if not auction_json and historical_auctions_df is None:
        return pd.DataFrame(columns=historical_auction_cols())

    if auction_json and historical_auctions_df is None:
        historical_auctions_df = pd.DataFrame(auction_json)

    historical_auctions_df = auction_df_filterer(historical_auctions_df)
    historical_auctions_df = historical_auctions_df[
        historical_auctions_df["auction_date" if not use_issue_date else "issue_date"].dt.date <= as_of_date.date()
    ]
    historical_auctions_df = historical_auctions_df[historical_auctions_df["maturity_date"] >= as_of_date]
    historical_auctions_df = historical_auctions_df.drop_duplicates(subset=["cusip"], keep="first")
    historical_auctions_df["int_rate"] = pd.to_numeric(historical_auctions_df["int_rate"], errors="coerce")
    historical_auctions_df["time_to_maturity"] = (historical_auctions_df["maturity_date"] - as_of_date).dt.days / 365
    return historical_auctions_df


def last_day_n_months_ago(given_date: datetime, n: int = 1, return_all: bool = False) -> datetime | List[datetime]:
    if return_all:
        given_date = pd.Timestamp(given_date)
        return [(given_date - pd.offsets.MonthEnd(i)).to_pydatetime() for i in range(1, n + 1)]

    given_date = pd.Timestamp(given_date)
    last_day = given_date - pd.offsets.MonthEnd(n)
    return last_day.to_pydatetime()


def cookie_string_to_dict(cookie_string):
    cookie_pairs = cookie_string.split("; ")
    cookie_dict = {pair.split("=")[0]: pair.split("=")[1] for pair in cookie_pairs if "=" in pair}
    return cookie_dict


def is_valid_ust_cusip(potential_ust_cusip: str):
    return len(potential_ust_cusip) == 9 and "912" in potential_ust_cusip


def historical_auction_cols():
    return [
        "cusip",
        "security_type",
        "auction_date",
        "issue_date",
        "maturity_date",
        "price_per100",
        "allocation_pctage",
        "avg_med_yield",
        "bid_to_cover_ratio",
        "comp_accepted",
        "comp_tendered",
        "corpus_cusip",
        "tint_cusip_1",
        "currently_outstanding",
        "direct_bidder_accepted",
        "direct_bidder_tendered",
        "est_pub_held_mat_by_type_amt",
        "fima_included",
        "fima_noncomp_accepted",
        "fima_noncomp_tendered",
        "high_discnt_rate",
        "high_investment_rate",
        "high_price",
        "high_yield",
        "indirect_bidder_accepted",
        "indirect_bidder_tendered",
        "int_rate",
        "low_investment_rate",
        "low_price",
        "low_discnt_margin",
        "low_yield",
        "max_comp_award",
        "max_noncomp_award",
        "noncomp_accepted",
        "noncomp_tenders_accepted",
        "offering_amt",
        "security_term",
        "original_security_term",
        "security_term_week_year",
        "primary_dealer_accepted",
        "primary_dealer_tendered",
        "reopening",
        "total_accepted",
        "total_tendered",
        "treas_retail_accepted",
        "treas_retail_tenders_accepted",
    ]


def ust_labeler(row: pd.Series):
    mat_date = row["maturity_date"]
    tenor = row["original_security_term"]
    if np.isnan(row["int_rate"]):
        return str(row["high_investment_rate"])[:5] + "s , " + mat_date.strftime("%b %y") + "s" + ", " + tenor
    return str(row["int_rate"]) + "s, " + mat_date.strftime("%b %y") + "s, " + tenor


def ust_sorter(term: str):
    if " " in term:
        term = term.split(" ")[0]
    num, unit = term.split("-")
    num = int(num)
    unit_multiplier = {"Year": 365, "Month": 30, "Week": 7, "Day": 1}
    return num * unit_multiplier[unit]


def get_otr_cusips_by_date(historical_auctions_df: pd.DataFrame, dates: list, use_issue_date: bool = True):
    historical_auctions_df = auction_df_filterer(historical_auctions_df)
    date_column = "issue_date" if use_issue_date else "auction_date"
    historical_auctions_df = historical_auctions_df.sort_values(by=[date_column], ascending=False)
    historical_auctions_df = historical_auctions_df.drop_duplicates(subset=["cusip"], keep="last")
    grouped = historical_auctions_df.groupby("original_security_term")
    otr_cusips_by_date = {date: [] for date in dates}
    for _, group in grouped:
        group = group.reset_index(drop=True)
        for date in dates:
            filtered_group = group[(group[date_column] <= date) & (group["maturity_date"] > date)]
            if not filtered_group.empty:
                otr_cusip = filtered_group.iloc[0]["cusip"]
                otr_cusips_by_date[date].append(otr_cusip)

    return otr_cusips_by_date


def process_cusip_otr_daterange(cusip, historical_auctions_df, date_column):
    try:
        tenor = historical_auctions_df[historical_auctions_df["cusip"] == cusip]["original_security_term"].iloc[0]
        tenor_df: pd.DataFrame = historical_auctions_df[historical_auctions_df["original_security_term"] == tenor].reset_index()
        otr_df = tenor_df[tenor_df["cusip"] == cusip]
        otr_index = otr_df.index[0]
        start_date: pd.Timestamp = otr_df[date_column].iloc[0]
        start_date = start_date.to_pydatetime()

        if otr_index == 0:
            return cusip, (start_date, datetime.today().date())

        if otr_index < len(tenor_df) - 1:
            end_date: pd.Timestamp = tenor_df[date_column].iloc[otr_index - 1]
            end_date = end_date.to_pydatetime()
        else:
            end_date = datetime.today().date()

        return cusip, {"start_date": start_date, "end_date": end_date}
    except Exception as e:
        # print(f"Something went wrong for {cusip}: {e}")
        return cusip, {"start_date": None, "end_date": None}


def get_otr_date_ranges(historical_auctions_df: pd.DataFrame, cusips: List[str], use_issue_date: bool = True) -> Dict[str, Tuple[datetime, datetime]]:

    historical_auctions_df = auction_df_filterer(historical_auctions_df)
    date_column = "issue_date" if use_issue_date else "auction_date"
    historical_auctions_df = historical_auctions_df.sort_values(by=[date_column], ascending=False)
    historical_auctions_df = historical_auctions_df[historical_auctions_df["issue_date"].dt.date < datetime.today().date()]
    historical_auctions_df = historical_auctions_df.drop_duplicates(subset=["cusip"], keep="last")

    cusip_daterange_map = {}
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(process_cusip_otr_daterange, cusip, historical_auctions_df, date_column): cusip for cusip in cusips}

        for future in as_completed(futures):
            cusip, date_range = future.result()
            cusip_daterange_map[cusip] = date_range

    return cusip_daterange_map


def pydatetime_to_quantlib_date(py_datetime: datetime) -> ql.Date:
    return ql.Date(py_datetime.day, py_datetime.month, py_datetime.year)


def quantlib_date_to_pydatetime(ql_date: ql.Date):
    return datetime(ql_date.year(), ql_date.month(), ql_date.dayOfMonth())


def get_isin_from_cusip(cusip_str, country_code: str = "US"):
    """
    >>> get_isin_from_cusip('037833100', 'US')
    'US0378331005'
    """
    isin_to_digest = country_code + cusip_str.upper()

    get_numerical_code = lambda c: str(ord(c) - 55)
    encode_letters = lambda c: c if c.isdigit() else get_numerical_code(c)
    to_digest = "".join(map(encode_letters, isin_to_digest))

    ints = [int(s) for s in to_digest[::-1]]
    every_second_doubled = [x * 2 for x in ints[::2]] + ints[1::2]

    sum_digits = lambda i: sum(divmod(i, 10))
    digit_sum = sum([sum_digits(i) for i in every_second_doubled])

    check_digit = (10 - digit_sum % 10) % 10
    return isin_to_digest + str(check_digit)


def get_cstrips_cusips(
    historical_auctions_df: pd.DataFrame,
    as_of_date: Optional[datetime] = None,
):
    historical_auctions_df = auction_df_filterer(historical_auctions_df)
    active_df = historical_auctions_df[historical_auctions_df["maturity_date"] > as_of_date]
    tint_cusip = "tint_cusip_1"
    active_df[tint_cusip] = active_df[tint_cusip].replace("null", np.nan)
    active_df = active_df[active_df[tint_cusip].notna()]
    active_df = active_df.sort_values(by=["maturity_date"]).reset_index(drop=True)
    return active_df[["maturity_date", tint_cusip]]


def original_security_term_to_ql_period(original_security_term: str):
    ost_ql_period = {
        "52-Week": ql.Period("1Y"),
        "2-Year": ql.Period("2Y"),
        "3-Year": ql.Period("3Y"),
        "5-Year": ql.Period("5Y"),
        "7-Year": ql.Period("7Y"),
        "10-Year": ql.Period("10Y"),
        "20-Year": ql.Period("20Y"),
        "30-Year": ql.Period("30Y"),
    }
    return ost_ql_period[original_security_term]


def enhanced_plotly_blue_scale():
    enhanced_blue_scale = [
        [0.0, "#0508b8"],  # Deepest blue
        [0.02, "#0508b8"],  # Extending deepest blue
        [0.04, "#0610b9"],  # Very subtle shift
        [0.06, "#0712ba"],  # Slightly lighter
        [0.08, "#0814bb"],  # Incrementally lighter
        [0.10, "#0916bc"],  # Another slight lightening
        [0.12, "#0a18bd"],  # Continuing the trend
        [0.14, "#0b1abd"],  # Gradual lightening
        [0.16, "#0c1cbe"],  # More noticeable change
        [0.18, "#0d1ebe"],  # Further lightening
        [0.20, "#0e20bf"],  # Approaching mid blues
        [0.25, "#1116e8"],  # Jumping to a lighter blue for contrast
        [0.3, "#1910d8"],  # Original Plotly3 blue
    ]

    # Transitioning back to the original Plotly3 scale, starting from a point after the enhanced blue
    transition_scale = [
        [0.35, "#3c19f0"],
        [0.40, "#6b1cfb"],
        [0.45, "#981cfd"],
        [0.50, "#bf1cfd"],
        [0.55, "#dd2bfd"],
        [0.60, "#f246fe"],
        [0.65, "#fc67fd"],
        [0.70, "#fe88fc"],
        [0.75, "#fea5fd"],
        [0.80, "#febefe"],
        [0.85, "#fec3fe"],
    ]

    # Combine the more granular blue scale with the rest of the Plotly3 colors
    combined_scale = enhanced_blue_scale + transition_scale

    return combined_scale


def to_quantlib_fixed_rate_bond_obj(bond_info: Dict[str, str], as_of_date: datetime, print_bond_info=False):
    if print_bond_info:
        print(json.dumps(bond_info, indent=4, default=str))
    maturity_date: pd.Timestamp = bond_info["maturity_date"]
    calendar = ql.UnitedStates(m=ql.UnitedStates.GovernmentBond)
    today = calendar.adjust(pydatetime_to_quantlib_date(py_datetime=as_of_date))
    ql.Settings.instance().evaluationDate = today
    t_plus = 1
    bond_settlement_date = calendar.advance(today, ql.Period(t_plus, ql.Days))

    schedule = ql.Schedule(
        bond_settlement_date,
        pydatetime_to_quantlib_date(maturity_date),
        ql.Period(ql.Semiannual),
        calendar,
        ql.ModifiedFollowing,
        ql.ModifiedFollowing,
        ql.DateGeneration.Backward,
        False,
    )
    return ql.FixedRateBond(
        t_plus,
        100.0,
        schedule,
        [bond_info["int_rate"] / 100],
        ql.ActualActual(ql.ActualActual.ISDA),
    )


class NoneReturningSpline:
    def __init__(self, *args, **kwargs):
        # Accept the same parameters as a usual spline, but do nothing with them
        pass

    def __call__(self, x):
        # Always return None for any input x
        if isinstance(x, (np.ndarray, list)):
            return [None] * len(x)  # Return a list of None for multiple inputs
        else:
            return None


def _calc_spot_rates_on_tenors(
    yield_curve: ql.DiscountCurve | ql.ZeroCurve,
    on_rate: float,
    day_count: ql.ActualActual = ql.ActualActual(ql.ActualActual.ISDA),
    price_col: Optional[str] = None,
    custom_price_col: Optional[str] = None,
    continuous_compounded_zero: Optional[bool] = False,
):
    spots = []
    tenors = []
    maturity_dates = []
    ref_date = yield_curve.referenceDate()

    dates = yield_curve.dates()
    for i, d in enumerate(dates):
        yrs = day_count.yearFraction(ref_date, d)
        if i == 0:
            tenors.append(1 / 360)
            spots.append(on_rate)
            t_plus_1_sr: pd.Timestamp = quantlib_date_to_pydatetime(d) - BDay(1)
            t_plus_1_sr = t_plus_1_sr.to_pydatetime()
            t_plus_1_sr = t_plus_1_sr.replace(hour=0, minute=0, second=0, microsecond=0)
            maturity_dates.append(t_plus_1_sr)
            continue

        if continuous_compounded_zero:
            zero_rate = yield_curve.zeroRate(yrs, ql.Continuous, True)
            eq_rate = zero_rate.equivalentRate(day_count, ql.Continuous, ql.NoFrequency, ref_date, d).rate()
        else:
            compounding = ql.Compounded
            freq = ql.Semiannual
            zero_rate = yield_curve.zeroRate(yrs, compounding, freq, True)
            eq_rate = zero_rate.equivalentRate(day_count, compounding, freq, ref_date, d).rate()

        tenors.append(yrs)
        spots.append(100 * eq_rate)
        maturity_dates.append(quantlib_date_to_pydatetime(d))

    price_col_type = price_col.split("_")[0] if price_col else None
    spot_col_name = f"{price_col_type}_spot_rate" if price_col else "spot_rate"
    if custom_price_col:
        spot_col_name = custom_price_col
    return pd.DataFrame(
        {
            "maturity_date": maturity_dates,
            "time_to_maturity": tenors,
            spot_col_name: spots,
        }
    )


def _calc_spot_rates_intep_months(
    yield_curve: ql.DiscountCurve | ql.ZeroCurve,
    on_rate: float,
    months: Optional[int] = 361,
    month_freq: Optional[float] = 1,
    custom_tenors: Optional[List[int]] = None,
    day_count=ql.ActualActual(ql.ActualActual.ISDA),
    calendar=ql.UnitedStates(m=ql.UnitedStates.GovernmentBond),
    price_col: Optional[str] = None,
    custom_price_col: Optional[str] = None,
    continuous_compounded_zero: Optional[bool] = False,
):
    spots = []
    tenors = []
    maturity_dates = []
    ref_date = yield_curve.referenceDate()
    to_iterate = custom_tenors if custom_tenors else range(0, months, month_freq)
    for month in to_iterate:
        d = calendar.advance(ref_date, ql.Period(month, ql.Months))
        yrs = month / 12.0
        if yrs == 0:
            tenors.append(1 / 360)
            spots.append(on_rate)
            t_plus_1_sr: pd.Timestamp = quantlib_date_to_pydatetime(d) - BDay(1)
            t_plus_1_sr = t_plus_1_sr.to_pydatetime()
            t_plus_1_sr = t_plus_1_sr.replace(hour=0, minute=0, second=0, microsecond=0)
            maturity_dates.append(t_plus_1_sr)
            continue

        if continuous_compounded_zero:
            zero_rate = yield_curve.zeroRate(yrs, ql.Continuous, True)
            eq_rate = zero_rate.equivalentRate(day_count, ql.Continuous, ql.NoFrequency, ref_date, d).rate()
        else:
            compounding = ql.Compounded
            freq = ql.Semiannual
            zero_rate = yield_curve.zeroRate(yrs, compounding, freq, True)
            eq_rate = zero_rate.equivalentRate(day_count, compounding, freq, ref_date, d).rate()

        tenors.append(yrs)
        spots.append(100 * eq_rate)
        maturity_dates.append(quantlib_date_to_pydatetime(d))

    price_col_type = price_col.split("_")[0] if price_col else None
    spot_col_name = f"{price_col_type}_spot_rate" if price_col else "spot_rate"
    if custom_price_col:
        spot_col_name = custom_price_col
    return pd.DataFrame(
        {
            "maturity_date": maturity_dates,
            "time_to_maturity": tenors,
            spot_col_name: spots,
        }
    )


def _calc_spot_rates_intep_days(
    yield_curve: ql.DiscountCurve | ql.ZeroCurve,
    on_rate: float,
    days: Optional[int] = 361 * 30,
    custom_tenors: Optional[List[int]] = None,
    day_count=ql.ActualActual(ql.ActualActual.ISDA),
    calendar=ql.UnitedStates(m=ql.UnitedStates.GovernmentBond),
    price_col: Optional[str] = None,
    custom_price_col: Optional[str] = None,
    continuous_compounded_zero: Optional[bool] = False,
):
    spots = []
    tenors = []
    maturity_dates = []
    ref_date = yield_curve.referenceDate()
    to_iterate = custom_tenors if custom_tenors else range(0, days, 1)
    for day in to_iterate:
        d = calendar.advance(ref_date, ql.Period(day, ql.Days))
        yrs = day / 365.0
        if yrs == 0:
            tenors.append(1 / 360)
            spots.append(on_rate)
            t_plus_1_sr: pd.Timestamp = quantlib_date_to_pydatetime(d) - BDay(1)
            t_plus_1_sr = t_plus_1_sr.to_pydatetime()
            t_plus_1_sr = t_plus_1_sr.replace(hour=0, minute=0, second=0, microsecond=0)
            maturity_dates.append(t_plus_1_sr)
            continue

        if continuous_compounded_zero:
            zero_rate = yield_curve.zeroRate(yrs, ql.Continuous, True)
            eq_rate = zero_rate.equivalentRate(day_count, ql.Continuous, ql.NoFrequency, ref_date, d).rate()
        else:
            compounding = ql.Compounded
            freq = ql.Semiannual
            zero_rate = yield_curve.zeroRate(yrs, compounding, freq, True)
            eq_rate = zero_rate.equivalentRate(day_count, compounding, freq, ref_date, d).rate()

        tenors.append(yrs)
        spots.append(100 * eq_rate)
        maturity_dates.append(quantlib_date_to_pydatetime(d))

    price_col_type = price_col.split("_")[0] if price_col else None
    spot_col_name = f"{price_col_type}_spot_rate" if price_col else "spot_rate"
    if custom_price_col:
        spot_col_name = custom_price_col
    return pd.DataFrame(
        {
            "maturity_date": maturity_dates,
            "time_to_maturity": tenors,
            spot_col_name: spots,
        }
    )


"""
Using QuantLib's Piecewise yield term structure for bootstrapping market observed prices to zeros rates at the respective ttms
- small differences between methods
- flag to take the averages of all Piecewise methods or pass in a specifc method
- passing in multiple ql_bootstrap_methods will take the average of the spot rates calculated from the different methods 
"""


def ql_piecewise_method_pretty(bs_method):
    ql_piecewise_methods_pretty_dict = {
        "ql_plld": "Piecewise Log Linear Discount",
        "ql_lcd": "Piecewise Log Cubic Discount",
        "ql_lz": "Piecewise Linear Zero",
        "ql_cz": "Piecewise Cubic Zero",
        "ql_lf": "Piecewise Linear Forward",
        "ql_spd": "Piecewise Spline Cubic Discount",
        "ql_kz": "Piecewise Kruger Zero",
        "ql_kld": "Piecewise Kruger Log Discount",
        "ql_mcf": "Piecewise Convex Monotone Forward",
        "ql_mcz": "Piecewise Convex Monotone Zero",
        "ql_ncz": "Piecewise Natural Cubic Zero",
        "ql_nlcd": "Piecewise Natural Log Cubic Discount",
        "ql_lmlcd": "Piecewise Log Mixed Linear Cubic Discount",
        "ql_pcz": "Piecewise Parabolic Cubic Zero",
        "ql_mpcz": "Piecewise Monotonic Parabolic Cubic Zero",
        "ql_lpcd": "Piecewise Log Parabolic Cubic Discount",
        "ql_mlpcd": "Piecewise Monotonic Log Parabolic Cubic Discount",
        "ql_f_ns": "Nelson-Siegel Fitting",
        "ql_f_nss": "Svensson Fitting",
        "ql_f_np": "Simple Polynomial Fitting",
        "ql_f_es": "Exponential Splines Fitting",
        "ql_f_cbs": "Cubic B-Splines Fitting",
    }
    return ql_piecewise_methods_pretty_dict[bs_method]


def get_spot_rates_bootstrapper(
    curve_set_df: pd.DataFrame,
    as_of_date: datetime,
    on_rate: float,
    ql_bootstrap_interp_methods: Optional[
        List[
            Literal[
                "ql_plld",
                "ql_lcd",
                "ql_lz",
                "ql_cz",
                "ql_lf",
                "ql_spd",
                "ql_kz",
                "ql_kld",
                "ql_mcf",
                "ql_mcz",
                "ql_ncz",
                "ql_nlcd",
                "ql_lmlcd",
                "ql_pcz",
                "ql_mpcz",
                "ql_lpcd",
                "ql_mlpcd",
            ]
        ]
    ] = ["ql_plld"],
    return_ql_zero_curve: Optional[bool] = False,
    interpolated_months_num: Optional[int] = None,
    interpolated_curve_yearly_freq: Optional[int] = 1,
    custom_yearly_tenors: Optional[List[int]] = None,
    # return_rel_val_df: Optional[bool] = False,
    daily_interpolation: Optional[bool] = False,
    return_scipy_interp_func: Optional[bool] = False,
    continuous_compounded_zero: Optional[bool] = False,
) -> Dict[str, pd.DataFrame | ql.DiscountCurve | ql.ZeroCurve]:
    ql_piecewise_methods: Dict[str, ql.DiscountCurve | ql.ZeroCurve] = {
        "ql_plld": ql.PiecewiseLogLinearDiscount,
        "ql_lcd": ql.PiecewiseLogCubicDiscount,
        "ql_lz": ql.PiecewiseLinearZero,
        "ql_cz": ql.PiecewiseCubicZero,
        "ql_lf": ql.PiecewiseLinearForward,
        "ql_spd": ql.PiecewiseSplineCubicDiscount,
        "ql_kz": ql.PiecewiseKrugerZero,
        "ql_kld": ql.PiecewiseKrugerLogDiscount,
        "ql_mcf": ql.PiecewiseConvexMonotoneForward,
        "ql_mcz": ql.PiecewiseConvexMonotoneZero,
        "ql_ncz": ql.PiecewiseNaturalCubicZero,
        "ql_nlcd": ql.PiecewiseNaturalLogCubicDiscount,
        "ql_lmlcd": ql.PiecewiseLogMixedLinearCubicDiscount,
        "ql_pcz": ql.PiecewiseParabolicCubicZero,
        "ql_mpcz": ql.PiecewiseMonotonicParabolicCubicZero,
        "ql_lpcd": ql.PiecewiseLogParabolicCubicDiscount,
        "ql_mlpcd": ql.PiecewiseMonotonicLogParabolicCubicDiscount,
        "ql_f_ns": ql.NelsonSiegelFitting,
        "ql_f_nss": ql.SvenssonFitting,
        "ql_f_np": ql.SimplePolynomialFitting,
        "ql_f_es": ql.ExponentialSplinesFitting,
        "ql_f_cbs": ql.CubicBSplinesFitting,
    }

    price_cols = ["bid_price", "offer_price", "mid_price", "eod_price"]
    required_cols = ["issue_date", "maturity_date", "int_rate"]
    price_col_exists = any(item in curve_set_df.columns for item in price_cols)
    missing_required_cols = [item for item in required_cols if item not in curve_set_df.columns]

    if not price_col_exists:
        raise ValueError(f"Build Spot Curve - Couldn't find a valid price col in your curve set df - one of {price_cols}")
    if missing_required_cols:
        raise ValueError(f"Build Spot Curve - Missing required curve set cols: {missing_required_cols}")

    price_col = next((item for item in price_cols if item in curve_set_df.columns), None)
    calendar = ql.UnitedStates(m=ql.UnitedStates.GovernmentBond)
    today = calendar.adjust(pydatetime_to_quantlib_date(py_datetime=as_of_date))
    ql.Settings.instance().evaluationDate = today

    t_plus = 1
    bond_settlement_date = calendar.advance(today, ql.Period(t_plus, ql.Days))
    frequency = ql.Semiannual
    day_count = ql.ActualActual(ql.ActualActual.ISDA)
    par = 100.0

    bond_helpers = []
    for _, row in curve_set_df.iterrows():
        maturity = pydatetime_to_quantlib_date(row["maturity_date"])
        if np.isnan(row["int_rate"]):
            quote = ql.QuoteHandle(ql.SimpleQuote(row[price_col]))
            tbill = ql.ZeroCouponBond(
                t_plus,
                calendar,
                par,
                maturity,
                ql.ModifiedFollowing,
                100.0,
                bond_settlement_date,
            )
            helper = ql.BondHelper(quote, tbill)
        else:
            schedule = ql.Schedule(
                bond_settlement_date,
                maturity,
                ql.Period(frequency),
                calendar,
                ql.ModifiedFollowing,
                ql.ModifiedFollowing,
                ql.DateGeneration.Backward,
                False,
            )
            helper = ql.FixedRateBondHelper(
                ql.QuoteHandle(ql.SimpleQuote(row[price_col])),
                t_plus,
                100.0,
                schedule,
                [row["int_rate"] / 100],
                day_count,
                ql.ModifiedFollowing,
                par,
            )

        bond_helpers.append(helper)

    spot_dfs: List[pd.DataFrame] = []
    ql_curves: Dict[str, ql.DiscountCurve | ql.ZeroCurve] = {}

    for bs_method in ql_bootstrap_interp_methods:
        if bs_method.split("_")[1] == "f":
            ql_fit_method = ql_piecewise_methods[bs_method]
            curr_curve = ql.FittedBondDiscountCurve(bond_settlement_date, bond_helpers, day_count, ql_fit_method())
            curr_curve.enableExtrapolation()
            ql_curves[bs_method] = curr_curve
        else:
            curr_curve = ql_piecewise_methods[bs_method](bond_settlement_date, bond_helpers, day_count)
            curr_curve.enableExtrapolation()
            ql_curves[bs_method] = curr_curve
        if interpolated_months_num or custom_yearly_tenors:
            if daily_interpolation:
                curr_spot_df = _calc_spot_rates_intep_days(
                    yield_curve=curr_curve,
                    on_rate=on_rate,
                    days=interpolated_months_num * 31,
                    custom_price_col=f"{bs_method}_spot_rate",
                    continuous_compounded_zero=continuous_compounded_zero,
                )
            else:
                curr_spot_df = _calc_spot_rates_intep_months(
                    yield_curve=curr_curve,
                    on_rate=on_rate,
                    months=interpolated_months_num,
                    month_freq=interpolated_curve_yearly_freq,
                    custom_tenors=([i * 12 for i in custom_yearly_tenors] if custom_yearly_tenors else None),
                    custom_price_col=f"{bs_method}_spot_rate",
                    continuous_compounded_zero=continuous_compounded_zero,
                )
        else:
            curr_spot_df = _calc_spot_rates_on_tenors(
                yield_curve=curr_curve,
                on_rate=on_rate,
                custom_price_col=f"{bs_method}_spot_rate",
                continuous_compounded_zero=continuous_compounded_zero,
            )

        spot_dfs.append(curr_spot_df)

    if len(spot_dfs) == 1:
        zero_rates_df = spot_dfs[0]
    else:
        maturity_dates = spot_dfs[0]["maturity_date"].to_list()
        tenors = spot_dfs[0]["time_to_maturity"].to_list()
        merged_df = pd.concat(
            [df[[col for col in df.columns if "spot_rate" in col]] for df in spot_dfs],
            axis=1,
        )
        avg_spot_rate_col = merged_df.mean(axis=1).to_list()

        merged_df.insert(0, "maturity_date", maturity_dates)
        merged_df.insert(1, "time_to_maturity", tenors)
        merged_df["avg_spot_rate"] = avg_spot_rate_col
        zero_rates_df = merged_df

    to_return_dict = {
        "ql_zero_curve_obj": None,
        "spot_rate_df": None,
        "scipy_interp_funcs": None,
    }

    if return_ql_zero_curve:
        if len(ql_bootstrap_interp_methods) > 1:
            print("Get Spot Rates - multiple bs methods passed - returning ql zero curve based on first bs method")
        bs_method = ql_bootstrap_interp_methods[0]
        if bs_method.split("_")[1] == "f":
            ql_fit_method = ql_piecewise_methods[bs_method]
            zero_curve = ql.FittedBondDiscountCurve(bond_settlement_date, bond_helpers, day_count, ql_fit_method())
        else:
            zero_curve = ql_piecewise_methods[bs_method](bond_settlement_date, bond_helpers, day_count)
        zero_curve.enableExtrapolation()
        to_return_dict["ql_zero_curve_obj"] = zero_curve

    to_return_dict["spot_rate_df"] = zero_rates_df

    if return_scipy_interp_func:
        scipy_interp_funcs = {}
        for bs_method in ql_bootstrap_interp_methods:
            scipy_interp_funcs[bs_method] = scipy.interpolate.interp1d(
                to_return_dict["spot_rate_df"]["time_to_maturity"],
                to_return_dict["spot_rate_df"][f"{bs_method}_spot_rate"],
                axis=0,
                kind="linear",
                # bounds_error=False,
                # fill_value="extrapolate",
            )
        to_return_dict["scipy_interp_funcs"] = scipy_interp_funcs

    return to_return_dict


def get_par_rates(
    spot_rates: List[float],
    tenors: List[int],
    select_every_nth_spot_rate: Optional[int] = None,
) -> pd.DataFrame:
    if select_every_nth_spot_rate:
        spot_rates = spot_rates[::select_every_nth_spot_rate]
    par_rates = []
    for tenor in tenors:
        periods = np.arange(0, tenor + 0.5, 0.5)
        curr_spot_rates = spot_rates[: len(periods)].copy()
        discount_factors = [1 / (1 + (s / 100) / 2) ** (2 * t) for s, t in zip(curr_spot_rates, periods)]
        sum_of_dfs = sum(discount_factors[:-1])
        par_rate = (1 - discount_factors[-1]) / sum_of_dfs * 2
        par_rates.append(par_rate * 100)

    return pd.DataFrame(
        {
            "tenor": tenors,
            "par_rate": par_rates,
        }
    )


# TODO match ql interp method with scipy interp func
def get_spot_rates_fitter(
    curve_set_df: pd.DataFrame,
    as_of_date: datetime,
    on_rate: float,
    ql_fitting_methods: Optional[
        List[
            Literal[
                "ql_f_ns",
                "ql_f_nss",
                "ql_f_sp",
                "ql_f_es",
                "ql_f_cbs",
            ]
        ]
    ] = ["ql_f_nss"],
    ql_zero_curve_interp_method: Optional[
        Literal[
            "ql_z_interp_log_lin",
            "ql_z_interp_cubic",
            "ql_z_interp_nat_cubic",
            "ql_z_interp_log_cubic",
            "ql_z_interp_monot_cubic",
        ]
    ] = None,
    daily_interpolation: Optional[bool] = False,
    simple_poly: Optional[int] = None,
    knots: Optional[List[float]] = None,
):
    ql_fitting_methods_dict: Dict[str, ql.DiscountCurve | ql.ZeroCurve] = {
        "ql_f_ns": ql.NelsonSiegelFitting,
        "ql_f_nss": ql.SvenssonFitting,
        "ql_f_sp": ql.SimplePolynomialFitting,
        "ql_f_es": ql.ExponentialSplinesFitting,
        "ql_f_cbs": ql.CubicBSplinesFitting,
    }

    ql_zero_curve_interp_methods_dict: Dict[str, ql.ZeroCurve] = {
        "ql_z_interp_log_lin": ql.LogLinearZeroCurve,
        "ql_z_interp_cubic": ql.CubicZeroCurve,
        "ql_z_interp_nat_cubic": ql.NaturalCubicZeroCurve,
        "ql_z_interp_log_cubic": ql.LogCubicZeroCurve,
        "ql_z_interp_monot_cubic": ql.MonotonicCubicZeroCurve,
    }

    price_cols = ["bid_price", "offer_price", "mid_price", "eod_price"]
    required_cols = ["issue_date", "maturity_date", "int_rate"]
    price_col_exists = any(item in curve_set_df.columns for item in price_cols)
    missing_required_cols = [item for item in required_cols if item not in curve_set_df.columns]

    if not price_col_exists:
        raise ValueError(f"Build Spot Curve - Couldn't find a valid price col in your curve set df - one of {price_cols}")
    if missing_required_cols:
        raise ValueError(f"Build Spot Curve - Missing required curve set cols: {missing_required_cols}")

    price_col = next((item for item in price_cols if item in curve_set_df.columns), None)
    calendar = ql.UnitedStates(m=ql.UnitedStates.GovernmentBond)
    today = calendar.adjust(pydatetime_to_quantlib_date(py_datetime=as_of_date))
    ql.Settings.instance().evaluationDate = today

    t_plus = 1
    bond_settlement_date = calendar.advance(today, ql.Period(t_plus, ql.Days))
    frequency = ql.Semiannual
    day_count = ql.ActualActual(ql.ActualActual.ISDA)
    par = 100.0

    bond_helpers = []
    for _, row in curve_set_df.iterrows():
        maturity = pydatetime_to_quantlib_date(row["maturity_date"])
        if np.isnan(row["int_rate"]):
            quote = ql.QuoteHandle(ql.SimpleQuote(row[price_col]))
            tbill = ql.ZeroCouponBond(
                t_plus,
                calendar,
                par,
                maturity,
                ql.ModifiedFollowing,
                100.0,
                bond_settlement_date,
            )
            helper = ql.BondHelper(quote, tbill)
        else:
            schedule = ql.Schedule(
                bond_settlement_date,
                maturity,
                ql.Period(frequency),
                calendar,
                ql.ModifiedFollowing,
                ql.ModifiedFollowing,
                ql.DateGeneration.Backward,
                False,
            )
            helper = ql.FixedRateBondHelper(
                ql.QuoteHandle(ql.SimpleQuote(row[price_col])),
                t_plus,
                100.0,
                schedule,
                [row["int_rate"] / 100],
                day_count,
                ql.ModifiedFollowing,
                par,
            )

        bond_helpers.append(helper)

    ql_curves: Dict[str, Dict[str, ql.DiscountCurve | ql.ZeroCurve | pd.DataFrame]] = {}
    for fit_method in ql_fitting_methods:
        if fit_method == "ql_f_sp" and not simple_poly:
            continue
        if fit_method == "ql_f_cbs" and not knots:
            continue

        ql_fit_method = ql_fitting_methods_dict[fit_method]

        if fit_method == "ql_f_sp":
            called_ql_fit_method = ql_fit_method(simple_poly)
        elif fit_method == "ql_f_cbs":
            called_ql_fit_method = ql_fit_method(knots)
        else:
            called_ql_fit_method = ql_fit_method()

        curr_curve = ql.FittedBondDiscountCurve(bond_settlement_date, bond_helpers, day_count, called_ql_fit_method)
        curr_curve.enableExtrapolation()
        if fit_method not in ql_curves:
            ql_curves[fit_method] = {
                "ql_fitted_curve": None,
                "ql_zero_curve": None,
                "zero_interp_func": None,
                "df_interp_func": None,
                "comparison_df": None,
            }

        ql_curves[fit_method]["ql_curve"] = curr_curve

        if daily_interpolation:
            dates = [bond_settlement_date + ql.Period(i, ql.Days) for i in range(0, 12 * 30 * 30, 1)]
        else:
            dates = [bond_settlement_date + ql.Period(i, ql.Months) for i in range(0, 12 * 30, 1)]

        discount_factors = [curr_curve.discount(d) for d in dates]
        ttm = [(ql.Date.to_date(d) - ql.Date.to_date(bond_settlement_date)).days / 365 for d in dates]

        eq_zero_rates = []
        eq_zero_rates_dec = []
        for d in dates:
            yrs = (ql.Date.to_date(d) - ql.Date.to_date(bond_settlement_date)).days / 365.0
            zero_rate = curr_curve.zeroRate(yrs, ql.Continuous, True)
            eq_rate = zero_rate.equivalentRate(day_count, ql.Continuous, ql.NoFrequency, bond_settlement_date, d).rate()
            eq_zero_rates.append(eq_rate * 100)
            eq_zero_rates_dec.append(eq_rate)
        eq_zero_rates[0] = on_rate
        eq_zero_rates_dec[0] = on_rate / 100

        ql_curves[fit_method]["zero_interp_func"] = scipy.interpolate.interp1d(ttm, eq_zero_rates, axis=0, kind="linear", fill_value="extrapolate")
        ql_curves[fit_method]["df_interp_func"] = scipy.interpolate.interp1d(ttm, discount_factors, axis=0, kind="linear", fill_value="extrapolate")
        zero_curve = (
            ql.ZeroCurve(dates, eq_zero_rates_dec, day_count)
            if not ql_zero_curve_interp_method
            else ql_zero_curve_interp_methods_dict[ql_zero_curve_interp_method](dates, eq_zero_rates_dec, day_count)
        )
        zero_curve.enableExtrapolation()
        ql_curves[fit_method]["ql_zero_curve"] = zero_curve

    return ql_curves


def reprice_bonds_single_zero_curve(as_of_date: datetime, ql_zero_curve: ql.ZeroCurve, curve_set_df: pd.DataFrame) -> pd.DataFrame:
    yield_curve_handle = ql.YieldTermStructureHandle(ql_zero_curve)
    engine = ql.DiscountingBondEngine(yield_curve_handle)

    price_cols = ["bid_price", "offer_price", "mid_price", "eod_price"]
    required_cols = ["issue_date", "maturity_date", "int_rate"]
    price_col_exists = any(item in curve_set_df.columns for item in price_cols)
    missing_required_cols = [item for item in required_cols if item not in curve_set_df.columns]

    if not price_col_exists:
        raise ValueError(f"Build Spot Curve - Couldn't find a valid price col in your curve set df - one of {price_cols}")
    if missing_required_cols:
        raise ValueError(f"Build Spot Curve - Missing required curve set cols: {missing_required_cols}")

    price_col = next((item for item in price_cols if item in curve_set_df.columns), None)
    quote_type = price_col.split("_")[0]
    yield_col = f"{quote_type}_yield"

    if yield_col not in curve_set_df.columns:
        raise ValueError(f"Build Spot Curve - Missing required curve set cols: {yield_col}")

    calendar = ql.UnitedStates(m=ql.UnitedStates.GovernmentBond)
    today = calendar.adjust(pydatetime_to_quantlib_date(py_datetime=as_of_date))
    ql.Settings.instance().evaluationDate = today

    t_plus = 1
    bond_settlement_date = calendar.advance(today, ql.Period(t_plus, ql.Days))
    frequency = ql.Semiannual
    day_count = ql.ActualActual(ql.ActualActual.ISDA)
    par = 100.0

    bonds: List[Dict[str, ql.FixedRateBond | ql.ZeroCouponBond]] = []
    for _, row in curve_set_df.iterrows():
        try:
            maturity = pydatetime_to_quantlib_date(row["maturity_date"])
            if np.isnan(row["int_rate"]):
                bond_ql = ql.ZeroCouponBond(
                    t_plus,
                    calendar,
                    par,
                    maturity,
                    ql.ModifiedFollowing,
                    100.0,
                    bond_settlement_date,
                )
                bond_rl: rl.FixedRateBond = rl.Bill(
                    termination=row["maturity_date"],
                    effective=row["issue_date"],
                    calendar="nyc",
                    modifier="NONE",
                    currency="usd",
                    convention="Act360",
                    settle=1,
                    curves="bill_curve",
                    calc_mode="us_gbb",
                )
            else:
                schedule = ql.Schedule(
                    bond_settlement_date,
                    maturity,
                    ql.Period(frequency),
                    calendar,
                    ql.ModifiedFollowing,
                    ql.ModifiedFollowing,
                    ql.DateGeneration.Backward,
                    False,
                )
                bond_ql = ql.FixedRateBond(
                    t_plus,
                    100.0,
                    schedule,
                    [row["int_rate"] / 100],
                    day_count,
                    ql.ModifiedFollowing,
                )
                bond_rl: rl.FixedRateBond = rl.FixedRateBond(
                    effective=row["issue_date"],
                    termination=row["maturity_date"],
                    fixed_rate=row["int_rate"],
                    spec="ust",
                    calc_mode="ust_31bii",
                )

            curr_accrued_amount = bond_rl.accrued(quantlib_date_to_pydatetime(bond_settlement_date))
            bond_ql.setPricingEngine(engine)
            curr_calced_npv = bond_ql.NPV()
            curr_calced_ytm = (
                bond_ql.bondYield(
                    curr_calced_npv,
                    day_count,
                    ql.Compounded,
                    frequency,
                    bond_settlement_date,
                )
                * 100
            )
            bonds.append(
                {
                    "cusip": row["cusip"],
                    "label": row["label"],
                    "issue_date": row["issue_date"],
                    "maturity_date": row["maturity_date"],
                    "time_to_maturity": row["time_to_maturity"],
                    "high_investment_rate": row["high_investment_rate"],
                    "int_rate": row["int_rate"],
                    "rank": row["rank"] if "rank" in curve_set_df.columns else None,
                    "outstanding": (row["outstanding_amt"] if "outstanding_amt" in curve_set_df.columns else None),
                    "soma_holdings": (row["parValue"] if "parValue" in curve_set_df.columns else None),
                    "stripping_amount": (row["portion_stripped_amt"] if "portion_stripped_amt" in curve_set_df.columns else None),
                    "free_float": (row["free_float"] if "free_float" in curve_set_df.columns else None),
                    yield_col: row[yield_col],
                    price_col: row[price_col],
                    "accured": curr_accrued_amount,
                    "repriced_npv": curr_calced_npv,
                    "repriced_ytm": curr_calced_ytm,
                    "price_spread": row[price_col] + curr_accrued_amount - curr_calced_npv,
                    "ytm_spread": (row[yield_col] - curr_calced_ytm) * 100,
                }
            )
        except Exception as e:
            print(row["cusip"], e)

    return pd.DataFrame(bonds)


def reprice_bonds(
    as_of_date: datetime,
    ql_zero_curves: Dict[str, ql.ZeroCurve],
    curve_set_df: pd.DataFrame,
):
    price_cols = ["bid_price", "offer_price", "mid_price", "eod_price"]
    required_cols = ["issue_date", "maturity_date", "int_rate"]
    price_col_exists = any(item in curve_set_df.columns for item in price_cols)
    missing_required_cols = [item for item in required_cols if item not in curve_set_df.columns]

    if not price_col_exists:
        raise ValueError(f"Build Spot Curve - Couldn't find a valid price col in your curve set df - one of {price_cols}")
    if missing_required_cols:
        raise ValueError(f"Build Spot Curve - Missing required curve set cols: {missing_required_cols}")

    price_col = next((item for item in price_cols if item in curve_set_df.columns), None)
    quote_type = price_col.split("_")[0]
    yield_col = f"{quote_type}_yield"

    if yield_col not in curve_set_df.columns:
        raise ValueError(f"Build Spot Curve - Missing required curve set cols: {yield_col}")

    calendar = ql.UnitedStates(m=ql.UnitedStates.GovernmentBond)
    today = calendar.adjust(pydatetime_to_quantlib_date(py_datetime=as_of_date))
    ql.Settings.instance().evaluationDate = today

    t_plus = 1
    bond_settlement_date = calendar.advance(today, ql.Period(t_plus, ql.Days))
    frequency = ql.Semiannual
    day_count = ql.ActualActual(ql.ActualActual.ISDA)
    par = 100.0

    bonds: List[Dict[str, ql.FixedRateBond | ql.ZeroCouponBond]] = []
    for _, row in curve_set_df.iterrows():
        try:
            maturity = pydatetime_to_quantlib_date(row["maturity_date"])
            if np.isnan(row["int_rate"]):
                bond_ql = ql.ZeroCouponBond(
                    t_plus,
                    calendar,
                    par,
                    maturity,
                    ql.ModifiedFollowing,
                    100.0,
                    bond_settlement_date,
                )
                bond_rl: rl.FixedRateBond = rl.Bill(
                    termination=row["maturity_date"],
                    effective=row["issue_date"],
                    calendar="nyc",
                    modifier="NONE",
                    currency="usd",
                    convention="Act360",
                    settle=1,
                    curves="bill_curve",
                    calc_mode="us_gbb",
                )
            else:
                schedule = ql.Schedule(
                    bond_settlement_date,
                    maturity,
                    ql.Period(frequency),
                    calendar,
                    ql.ModifiedFollowing,
                    ql.ModifiedFollowing,
                    ql.DateGeneration.Backward,
                    False,
                )
                bond_ql = ql.FixedRateBond(
                    t_plus,
                    100.0,
                    schedule,
                    [row["int_rate"] / 100],
                    day_count,
                    ql.ModifiedFollowing,
                )
                bond_rl: rl.FixedRateBond = rl.FixedRateBond(
                    effective=row["issue_date"],
                    termination=row["maturity_date"],
                    fixed_rate=row["int_rate"],
                    spec="ust",
                    calc_mode="ust_31bii",
                )

            curr_accrued_amount = bond_rl.accrued(quantlib_date_to_pydatetime(bond_settlement_date))
            curr_row = {
                "cusip": row["cusip"],
                "label": row["label"],
                "issue_date": row["issue_date"],
                "maturity_date": row["maturity_date"],
                "time_to_maturity": row["time_to_maturity"],
                "high_investment_rate": row["high_investment_rate"],
                "int_rate": row["int_rate"],
                "rank": row["rank"] if "rank" in curve_set_df.columns else None,
                "outstanding": (row["outstanding_amt"] if "outstanding_amt" in curve_set_df.columns else None),
                "soma_holdings": (row["parValue"] if "parValue" in curve_set_df.columns else None),
                "stripping_amount": (row["portion_stripped_amt"] if "portion_stripped_amt" in curve_set_df.columns else None),
                "free_float": (row["free_float"] if "free_float" in curve_set_df.columns else None),
                yield_col: row[yield_col],
                price_col: row[price_col],
                "accured": curr_accrued_amount,
            }

            for label, ql_zero_curve in ql_zero_curves.items():
                yield_curve_handle = ql.YieldTermStructureHandle(ql_zero_curve)
                engine = ql.DiscountingBondEngine(yield_curve_handle)
                bond_ql.setPricingEngine(engine)
                curr_calced_npv = bond_ql.NPV()
                curr_calced_ytm = (
                    bond_ql.bondYield(
                        curr_calced_npv,
                        day_count,
                        ql.Compounded,
                        frequency,
                        bond_settlement_date,
                    )
                    * 100
                )

                curr_price_spread = row[price_col] + curr_accrued_amount - curr_calced_npv
                curr_ytm_spread = (row[yield_col] - curr_calced_ytm) * 100

                curr_row[f"{label}_repriced_npv"] = curr_calced_npv
                curr_row[f"{label}_repriced_ytm"] = curr_calced_ytm
                curr_row[f"{label}_price_spread"] = curr_price_spread
                curr_row[f"{label}_ytm_spread"] = curr_ytm_spread

            bonds.append(curr_row)

        except Exception as e:
            print(row["cusip"], e)

    return pd.DataFrame(bonds)
