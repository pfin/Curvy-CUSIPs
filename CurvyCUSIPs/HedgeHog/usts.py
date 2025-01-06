from datetime import datetime, timedelta
from typing import Dict, Optional

import numpy as np
import pandas as pd
import QuantLib as ql
import scipy
import ujson as json
from termcolor import colored

from CurvyCUSIPs.USTs import USTs


def pydatetime_to_quantlib_date(py_datetime: datetime) -> ql.Date:
    return ql.Date(py_datetime.day, py_datetime.month, py_datetime.year)


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


def calc_ust_implied_curve(n: float | int, scipy_interp_curve: scipy.interpolate.interpolate, return_scipy=False) -> Dict[float, float]:
    cfs = np.arange(0.5, 30 + 1, 0.5)
    implied_spot_rates = []
    first_n_cfs = 0
    for t in cfs:
        if t > n:
            Z_t_temp = scipy_interp_curve(t)
            Z_n = scipy_interp_curve(n)
            Z_n_t = (Z_t_temp * t - Z_n * n) / (t - n)
            implied_spot_rates.append(Z_n_t)
        else:
            if return_scipy:
                # implied_spot_rates.append(0)
                first_n_cfs += 1
            else:
                implied_spot_rates.append(np.nan)

    if return_scipy:
        return scipy.interpolate.CubicSpline(x=cfs[first_n_cfs:], y=implied_spot_rates)
    return dict(zip(cfs, implied_spot_rates))


def calc_ust_metrics(
    bond_info: str,
    curr_price: float,
    curr_ytm: float,
    on_rate: float,
    as_of_date: datetime,
    scipy_interp: scipy.interpolate.interpolate,
    print_bond_info=False,
):
    calendar = ql.UnitedStates(ql.UnitedStates.GovernmentBond)
    day_count = ql.ActualActual(ql.ActualActual.ISDA)
    times = np.arange(0, 30.5, 0.5)

    dates = []
    zero_rates = []
    today = calendar.adjust(pydatetime_to_quantlib_date(py_datetime=as_of_date))
    ql.Settings.instance().evaluationDate = today
    t_plus = 2
    bond_settlement_date = calendar.advance(today, ql.Period(t_plus, ql.Days))

    for t in times:
        if t == 0:
            dates.append(today)
            zero_rate = scipy_interp(0)
            zero_rates.append(float(zero_rate) / 100)
        else:
            maturity_date = calendar.advance(pydatetime_to_quantlib_date(as_of_date), ql.Period(int(round(t * 365)), ql.Days))
            dates.append(maturity_date)
            zero_rate = scipy_interp(t)
            zero_rates.append(float(zero_rate) / 100)

    zero_curve = ql.ZeroCurve(dates, zero_rates, day_count, calendar)
    yield_curve_handle = ql.YieldTermStructureHandle(zero_curve)
    engine = ql.DiscountingBondEngine(yield_curve_handle)

    ql_fixed_rate_bond_obj = to_quantlib_fixed_rate_bond_obj(bond_info=bond_info, as_of_date=as_of_date, print_bond_info=print_bond_info)
    ql_fixed_rate_bond_obj.setPricingEngine(engine)

    try:
        zspread = ql.BondFunctions.zSpread(
            ql_fixed_rate_bond_obj,
            curr_price,
            yield_curve_handle.currentLink(),
            day_count,
            ql.Compounded,
            ql.Semiannual,
            pydatetime_to_quantlib_date(as_of_date),
            1.0e-16,
            1000000,
            0.0,
        )
        spread1 = ql.SimpleQuote(zspread)
        spread_handle1 = ql.QuoteHandle(spread1)
        ts_spreaded1 = ql.ZeroSpreadedTermStructure(yield_curve_handle, spread_handle1, ql.Compounded, ql.Semiannual)
        ts_spreaded_handle1 = ql.YieldTermStructureHandle(ts_spreaded1)
        ycsin = ts_spreaded_handle1
        bond_engine = ql.DiscountingBondEngine(ycsin)
        ql_fixed_rate_bond_obj.setPricingEngine(bond_engine)
        zspread_impl_clean_price = ql_fixed_rate_bond_obj.cleanPrice()
        zspread = zspread * 10000
    except:
        zspread = None
        zspread_impl_clean_price = None

    rate = ql.InterestRate(curr_ytm / 100, day_count, ql.Compounded, ql.Semiannual)
    bps_value = ql.BondFunctions.basisPointValue(ql_fixed_rate_bond_obj, rate, bond_settlement_date)
    dv01_1mm = bps_value * 1_000_000 / 100
    impl_spot_3m_fwds = calc_ust_implied_curve(n=0.25, scipy_interp_curve=scipy_interp, return_scipy=True)
    impl_spot_6m_fwds = calc_ust_implied_curve(n=0.5, scipy_interp_curve=scipy_interp, return_scipy=True)
    impl_spot_12m_fwds = calc_ust_implied_curve(n=1, scipy_interp_curve=scipy_interp, return_scipy=True)
    bond_ttm: timedelta = bond_info["maturity_date"] - as_of_date

    metrics = {
        "Date": as_of_date,
        "zspread": zspread,
        "zspread_impl_clean_price": zspread_impl_clean_price,
        "clean_price": ql.BondFunctions.cleanPrice(ql_fixed_rate_bond_obj, rate),
        "dirty_price": ql.BondFunctions.dirtyPrice(ql_fixed_rate_bond_obj, yield_curve_handle.currentLink(), bond_settlement_date),
        "accrued_amount": ql.BondFunctions.accruedAmount(ql_fixed_rate_bond_obj, bond_settlement_date),
        "bps": bps_value,
        "dv01_1mm": dv01_1mm,
        "mac_duration": ql.BondFunctions.duration(ql_fixed_rate_bond_obj, rate, ql.Duration.Macaulay),
        "mod_duration": ql.BondFunctions.duration(ql_fixed_rate_bond_obj, rate, ql.Duration.Modified),
        "convexity": ql.BondFunctions.convexity(ql_fixed_rate_bond_obj, rate, bond_settlement_date),
        "basis_point_value": ql.BondFunctions.basisPointValue(ql_fixed_rate_bond_obj, rate, bond_settlement_date),
        "yield_value_basis_point": ql.BondFunctions.yieldValueBasisPoint(ql_fixed_rate_bond_obj, rate, bond_settlement_date),
        "rough_carry": curr_ytm - on_rate,
        "rough_3m_rolldown": (impl_spot_3m_fwds(float(bond_ttm.days / 365)) - curr_ytm) * 100,
        "rough_6m_rolldown": (impl_spot_6m_fwds(float(bond_ttm.days / 365)) - curr_ytm) * 100,
        "rough_12m_rolldown": (impl_spot_12m_fwds(float(bond_ttm.days / 365)) - curr_ytm) * 100,
    }

    return metrics


def dv01_neutral_curve_hedge_ratio(
    as_of_date: datetime,
    front_leg_bond_row: Dict | pd.Series,
    back_leg_bond_row: Dict | pd.Series,
    usts_obj: USTs,
    scipy_interp_curve: scipy.interpolate.interpolate,
    repo_rate: float,
    quote_type: Optional[str] = "eod",
    spread_dv01: Optional[int] = None,
    front_leg_par_amount: Optional[int] = None,
    back_leg_par_amount: Optional[int] = None,
    yvx_beta_adjustment: Optional[int] = None,
    total_trade_par_amount: Optional[int] = None,
    verbose: Optional[bool] = True,
    very_verbose: Optional[bool] = False,
    custom_beta_weighted_title: Optional[str] = None,
):
    if isinstance(front_leg_bond_row, pd.Series) or isinstance(front_leg_bond_row, pd.DataFrame):
        front_leg_bond_row = front_leg_bond_row.to_dict("records")[0]
    if isinstance(back_leg_bond_row, pd.Series) or isinstance(back_leg_bond_row, pd.DataFrame):
        back_leg_bond_row = back_leg_bond_row.to_dict("records")[0]

    front_leg_info = usts_obj.cusip_to_ust_label(cusip=front_leg_bond_row["cusip"])
    back_leg_info = usts_obj.cusip_to_ust_label(cusip=back_leg_bond_row["cusip"])

    front_leg_metrics = calc_ust_metrics(
        bond_info=front_leg_info,
        curr_price=front_leg_bond_row[f"{quote_type}_price"],
        curr_ytm=front_leg_bond_row[f"{quote_type}_yield"],
        as_of_date=as_of_date,
        scipy_interp=scipy_interp_curve,
        on_rate=repo_rate,
    )
    back_leg_metrics = calc_ust_metrics(
        bond_info=back_leg_info,
        curr_price=back_leg_bond_row[f"{quote_type}_price"],
        curr_ytm=back_leg_bond_row[f"{quote_type}_yield"],
        as_of_date=as_of_date,
        scipy_interp=scipy_interp_curve,
        on_rate=repo_rate,
    )

    front_leg_ttm: float = (front_leg_info["maturity_date"] - as_of_date).days / 365
    back_leg_ttm: float = (back_leg_info["maturity_date"] - as_of_date).days / 365
    impl_spot_3m_fwds = calc_ust_implied_curve(n=0.25, scipy_interp_curve=scipy_interp_curve, return_scipy=True)
    impl_spot_6m_fwds = calc_ust_implied_curve(n=0.5, scipy_interp_curve=scipy_interp_curve, return_scipy=True)
    impl_spot_12m_fwds = calc_ust_implied_curve(n=1, scipy_interp_curve=scipy_interp_curve, return_scipy=True)

    if very_verbose:
        print("Front Leg Info: ")
        print(front_leg_bond_row)
        print(front_leg_metrics)
        print("Back Leg Info: ")
        print(back_leg_bond_row)
        print(back_leg_metrics)

    if front_leg_bond_row["rank"] == 0 and back_leg_bond_row["rank"] == 0:
        print(f"{front_leg_bond_row["original_security_term"].split("-")[0]}s{back_leg_bond_row["original_security_term"].split("-")[0]}s")
    print(f"{front_leg_bond_row["ust_label"]} / {back_leg_bond_row["ust_label"]}") if verbose else None

    hr = back_leg_metrics["bps"] / front_leg_metrics["bps"]
    print(colored(f"BPV Neutral Hedge Ratio: {hr}", "light_blue")) if verbose else None
    if yvx_beta_adjustment:
        title = custom_beta_weighted_title or "Beta Weighted Hedge Ratio"
        (print(colored(f"{title}: {hr * yvx_beta_adjustment:3f}", "light_magenta")) if verbose else None)
        hr = hr * yvx_beta_adjustment

    if spread_dv01:
        back_leg_par_amount = np.abs(spread_dv01 / back_leg_metrics["bps"] * 100)

    if total_trade_par_amount:
        normalized_total = hr - 1
        back_leg_par_amount = total_trade_par_amount / normalized_total

    if front_leg_par_amount and back_leg_par_amount:
        raise ValueError("'front_leg_par_amount' and 'back_leg_par_amount' are both defined!")
    if not front_leg_par_amount and not back_leg_par_amount:
        back_leg_par_amount = 1_000_000
    if back_leg_par_amount:
        front_leg_par_amount = back_leg_par_amount * hr
    elif front_leg_par_amount:
        back_leg_par_amount = front_leg_par_amount / hr

    if verbose:
        print(
            f"Front Leg: {front_leg_bond_row["ust_label"]} (OST {front_leg_bond_row["original_security_term"]}, TTM = {front_leg_bond_row["time_to_maturity"]:3f}) Par Amount = {front_leg_par_amount :_}"
        )
        print(
            f"Back Leg: {back_leg_bond_row["ust_label"]} (OST {back_leg_bond_row["original_security_term"]}, TTM = {back_leg_bond_row["time_to_maturity"]:3f}) Par Amount = {round(back_leg_par_amount):_}"
        )
        print(f"Total Trade Par Amount: {front_leg_par_amount - back_leg_par_amount:_}")
        # risk_weight = (front_leg_par_amount * front_leg_metrics["bps"] / 100) / (back_leg_par_amount * back_leg_metrics["bps"] / 100)
        # print(f"Risk Weights: {risk_weight:3f} : 100")

    return {
        "current_spread": (back_leg_bond_row[f"{quote_type}_yield"] - front_leg_bond_row[f"{quote_type}_yield"]) * 100,
        "current_bpv_neutral_spread": (
            back_leg_bond_row[f"{quote_type}_yield"]
            - (front_leg_bond_row[f"{quote_type}_yield"] * (back_leg_metrics["bps"] / front_leg_metrics["bps"]))
        )
        * 100,
        "current_beta_weighted_spread": (
            (back_leg_bond_row[f"{quote_type}_yield"] - (front_leg_bond_row[f"{quote_type}_yield"] * hr)) * 100 if yvx_beta_adjustment else None
        ),
        "rough_3m_impl_fwd_spread": (impl_spot_3m_fwds(back_leg_ttm) - impl_spot_3m_fwds(front_leg_ttm)) * 100,
        "rough_6m_impl_fwd_spread": (impl_spot_6m_fwds(back_leg_ttm) - impl_spot_6m_fwds(front_leg_ttm)) * 100,
        "rough_12m_impl_fwd_spread": (impl_spot_12m_fwds(back_leg_ttm) - impl_spot_12m_fwds(front_leg_ttm)) * 100,
        "front_leg_metrics": front_leg_metrics,
        "back_leg_metrics": back_leg_metrics,
        "bpv_hedge_ratio": back_leg_metrics["bps"] / front_leg_metrics["bps"],
        "beta_weighted_hedge_ratio": ((back_leg_metrics["bps"] / front_leg_metrics["bps"]) * yvx_beta_adjustment if yvx_beta_adjustment else None),
        "front_leg_par_amount": front_leg_par_amount,
        "back_leg_par_amount": back_leg_par_amount,
        "spread_dv01": np.abs(back_leg_metrics["bps"] * back_leg_par_amount / 100),
        "rough_3m_carry_roll": (back_leg_metrics["rough_carry"] + back_leg_metrics["rough_3m_rolldown"])
        - hr * (front_leg_metrics["rough_carry"] + front_leg_metrics["rough_3m_rolldown"]),
        "rough_6m_carry_roll": (back_leg_metrics["rough_carry"] + back_leg_metrics["rough_6m_rolldown"])
        - hr * (front_leg_metrics["rough_carry"] + front_leg_metrics["rough_6m_rolldown"]),
        "rough_12m_carry_roll": (back_leg_metrics["rough_carry"] + back_leg_metrics["rough_12m_rolldown"])
        - hr * (front_leg_metrics["rough_carry"] + front_leg_metrics["rough_12m_rolldown"]),
    }


# reference point is buying the belly => fly spread down
def dv01_neutral_butterfly_hedge_ratio(
    as_of_date: datetime,
    front_wing_bond_row: Dict | pd.Series,
    belly_bond_row: Dict | pd.Series,
    back_wing_bond_row: Dict | pd.Series,
    usts_obj: USTs,
    scipy_interp_curve: scipy.interpolate.interpolate,
    quote_type: Optional[str] = "eod",
    front_wing_par_amount: Optional[int] = None,
    belly_par_amount: Optional[int] = None,
    back_wing_par_amount: Optional[int] = None,
    total_trade_par_amount: Optional[int] = None,
    yvx_front_wing_beta_adjustment: Optional[int] = None,
    yvx_back_wing_beta_adjustment: Optional[int] = None,
    verbose: Optional[bool] = True,
    very_verbose: Optional[bool] = False,
):
    sofr_df = usts_obj.curve_data_fetcher.nyfrb_data_fetcher.get_sofr_fixings_df(start_date=as_of_date, end_date=as_of_date)
    repo_rate = sofr_df.iloc[-1]["percentRate"]

    if isinstance(front_wing_bond_row, pd.Series) or isinstance(front_wing_bond_row, pd.DataFrame):
        front_wing_bond_row = front_wing_bond_row.to_dict("records")[0]
    if isinstance(belly_bond_row, pd.Series) or isinstance(belly_bond_row, pd.DataFrame):
        belly_bond_row = belly_bond_row.to_dict("records")[0]
    if isinstance(back_wing_bond_row, pd.Series) or isinstance(back_wing_bond_row, pd.DataFrame):
        back_wing_bond_row = back_wing_bond_row.to_dict("records")[0]

    front_wing_info = usts_obj.cusip_to_ust_label(cusip=front_wing_bond_row["cusip"])
    belly_info = usts_obj.cusip_to_ust_label(cusip=belly_bond_row["cusip"])
    back_wing_info = usts_obj.cusip_to_ust_label(cusip=back_wing_bond_row["cusip"])

    front_wing_metrics = calc_ust_metrics(
        bond_info=front_wing_info,
        curr_price=front_wing_bond_row[f"{quote_type}_price"],
        curr_ytm=front_wing_bond_row[f"{quote_type}_yield"],
        as_of_date=as_of_date,
        scipy_interp=scipy_interp_curve,
        on_rate=repo_rate,
    )
    belly_metrics = calc_ust_metrics(
        bond_info=belly_info,
        curr_price=belly_bond_row[f"{quote_type}_price"],
        curr_ytm=belly_bond_row[f"{quote_type}_yield"],
        as_of_date=as_of_date,
        scipy_interp=scipy_interp_curve,
        on_rate=repo_rate,
    )
    back_wing_metrics = calc_ust_metrics(
        bond_info=back_wing_info,
        curr_price=back_wing_bond_row[f"{quote_type}_price"],
        curr_ytm=back_wing_bond_row[f"{quote_type}_yield"],
        as_of_date=as_of_date,
        scipy_interp=scipy_interp_curve,
        on_rate=repo_rate,
    )

    front_wing_ttm: float = (front_wing_info["maturity_date"] - as_of_date).days / 365
    belly_ttm: float = (belly_info["maturity_date"] - as_of_date).days / 365
    back_wing_ttm: float = (back_wing_info["maturity_date"] - as_of_date).days / 365
    impl_spot_3m_fwds = calc_ust_implied_curve(n=0.25, scipy_interp_curve=scipy_interp_curve, return_scipy=True)
    impl_spot_6m_fwds = calc_ust_implied_curve(n=0.5, scipy_interp_curve=scipy_interp_curve, return_scipy=True)
    impl_spot_12m_fwds = calc_ust_implied_curve(n=1, scipy_interp_curve=scipy_interp_curve, return_scipy=True)

    if very_verbose:
        print("Front Wing Info: ")
        print(front_wing_info)
        print(front_wing_metrics)

        print("Belly Info: ")
        print(belly_info)
        print(belly_metrics)

        print("Back Wing Info: ")
        print(back_wing_info)
        print(back_wing_metrics)

    hedge_ratios = {
        "front_wing_hr": belly_metrics["bps"] / front_wing_metrics["bps"] * 0.5,
        "belly_hr": 1,
        "back_wing_hr": belly_metrics["bps"] / back_wing_metrics["bps"] * 0.5,
    }

    if verbose:
        if front_wing_bond_row["rank"] == 0 and belly_bond_row["rank"] == 0 and back_wing_bond_row["rank"] == 0:
            print(
                f"{front_wing_bond_row["original_security_term"].split("-")[0]}s{belly_bond_row["original_security_term"].split("-")[0]}s{back_wing_bond_row["original_security_term"].split("-")[0]}s"
            )

        (print(f"{front_wing_bond_row["ust_label"]} - {belly_bond_row["ust_label"]} - {back_wing_bond_row["ust_label"]} Fly") if verbose else None)
        print(colored(f"BPV Neutral Hedge Ratio:", "light_blue")) if verbose else None
        print(json.dumps(hedge_ratios, indent=4)) if verbose else None

        if yvx_front_wing_beta_adjustment and yvx_back_wing_beta_adjustment:
            print(colored(f"Beta Weighted Hedge Ratio:", "light_magenta")) if verbose else None
            hedge_ratios = {
                "front_wing_hr": belly_metrics["bps"] / front_wing_metrics["bps"] * yvx_front_wing_beta_adjustment,
                "belly_hr": 1,
                "back_wing_hr": belly_metrics["bps"] / back_wing_metrics["bps"] * yvx_back_wing_beta_adjustment,
            }
            print(json.dumps(hedge_ratios, indent=4)) if verbose else None

        if total_trade_par_amount is not None:
            if front_wing_par_amount is not None or belly_par_amount is not None or back_wing_par_amount is not None:
                raise ValueError("Cannot provide total_trade_par_amount along with individual leg par amounts.")

            total_hr_abs = abs(hedge_ratios["front_wing_hr"]) + abs(hedge_ratios["belly_hr"]) + abs(hedge_ratios["back_wing_hr"])
            belly_par_amount = total_trade_par_amount / total_hr_abs
            front_wing_par_amount = hedge_ratios["front_wing_hr"] * belly_par_amount
            back_wing_par_amount = hedge_ratios["back_wing_hr"] * belly_par_amount

        else:
            if belly_par_amount:
                front_wing_par_amount = hedge_ratios["front_wing_hr"] * belly_par_amount
                belly_par_amount = belly_par_amount
                back_wing_par_amount = hedge_ratios["back_wing_hr"] * belly_par_amount
            elif front_wing_par_amount:
                front_wing_par_amount = front_wing_par_amount
                belly_par_amount = front_wing_par_amount / hedge_ratios["front_wing_hr"]
                back_wing_par_amount = hedge_ratios["back_wing_hr"] * (front_wing_par_amount / hedge_ratios["front_wing_hr"])
            elif back_wing_par_amount:
                front_wing_par_amount = hedge_ratios["front_wing_hr"] * (back_wing_par_amount / hedge_ratios["back_wing_hr"])
                belly_par_amount = back_wing_par_amount / hedge_ratios["back_wing_hr"]
                back_wing_par_amount = back_wing_par_amount

        print(
            f"Front Wing: {front_wing_bond_row["ust_label"]} (OST {front_wing_bond_row["original_security_term"]}, TTM = {front_wing_bond_row["time_to_maturity"]:3f}) Par Amount = {front_wing_par_amount:_}"
        )
        print(
            f"Belly: {belly_bond_row["ust_label"]} (OST {belly_bond_row["original_security_term"]}, TTM = {belly_bond_row["time_to_maturity"]:3f}) Par Amount = {belly_par_amount:_}"
        )
        print(
            f"Back Wing: {back_wing_bond_row["ust_label"]} (OST {back_wing_bond_row["original_security_term"]}, TTM = {back_wing_bond_row["time_to_maturity"]:3f}) Par Amount = {back_wing_par_amount:_}"
        )
        print(f"Total Trade Par Amount: {front_wing_par_amount + belly_par_amount + back_wing_par_amount:_}")
        (
            print(
                f"Risk Weights - Front Wing: {yvx_front_wing_beta_adjustment:.3%}, Back Wing: {yvx_back_wing_beta_adjustment:.3%}, Sum: {yvx_front_wing_beta_adjustment + yvx_back_wing_beta_adjustment:.3%}"
            )
            if yvx_front_wing_beta_adjustment and yvx_back_wing_beta_adjustment
            else None
        )

    return {
        "curr_spread": (
            (belly_bond_row[f"{quote_type}_yield"] - front_wing_bond_row[f"{quote_type}_yield"])
            - (back_wing_bond_row[f"{quote_type}_yield"] - belly_bond_row[f"{quote_type}_yield"])
        )
        * 100,
        "rough_3m_impl_fwd_spread": (
            (impl_spot_3m_fwds(belly_ttm) - impl_spot_3m_fwds(front_wing_ttm)) - (impl_spot_3m_fwds(back_wing_ttm) - impl_spot_3m_fwds(belly_ttm))
        )
        * 100,
        "rough_6m_impl_fwd_spread": (
            (impl_spot_6m_fwds(belly_ttm) - impl_spot_6m_fwds(front_wing_ttm)) - (impl_spot_6m_fwds(back_wing_ttm) - impl_spot_6m_fwds(belly_ttm))
        )
        * 100,
        "rough_12m_impl_fwd_spread": (
            (impl_spot_12m_fwds(belly_ttm) - impl_spot_12m_fwds(front_wing_ttm)) - (impl_spot_12m_fwds(back_wing_ttm) - impl_spot_12m_fwds(belly_ttm))
        )
        * 100,
        "front_wing_metrics": front_wing_metrics,
        "belly_metrics": belly_metrics,
        "back_wing_metrics": back_wing_metrics,
        "bpv_neutral_hedge_ratio": {
            "front_wing_hr": belly_metrics["bps"] / front_wing_metrics["bps"] / 2,
            "belly_hr": 1,
            "back_wing_hr": belly_metrics["bps"] / back_wing_metrics["bps"] / 2,
        },
        "beta_weighted_hedge_ratio": (hedge_ratios if yvx_front_wing_beta_adjustment and yvx_back_wing_beta_adjustment else None),
        "front_wing_par_amount": front_wing_par_amount,
        "belly_par_amount": belly_par_amount,
        "back_leg_par_amount": back_wing_par_amount,
        "spread_dv01": np.abs(belly_metrics["bps"] * belly_par_amount / 100),
        "rough_3m_carry_roll": (belly_metrics["rough_carry"] + belly_metrics["rough_3m_rolldown"])
        - (hedge_ratios["front_wing_hr"] * (front_wing_metrics["rough_carry"] + front_wing_metrics["rough_3m_rolldown"]))
        - (hedge_ratios["back_wing_hr"] * (back_wing_metrics["rough_carry"] + back_wing_metrics["rough_3m_rolldown"])),
        "rough_6m_carry_roll": (belly_metrics["rough_carry"] + belly_metrics["rough_6m_rolldown"])
        - (hedge_ratios["front_wing_hr"] * (front_wing_metrics["rough_carry"] + front_wing_metrics["rough_6m_rolldown"]))
        - (hedge_ratios["back_wing_hr"] * (back_wing_metrics["rough_carry"] + back_wing_metrics["rough_6m_rolldown"])),
        "rough_12m_carry_roll": (belly_metrics["rough_carry"] + belly_metrics["rough_12m_rolldown"])
        - (hedge_ratios["front_wing_hr"] * (front_wing_metrics["rough_carry"] + front_wing_metrics["rough_12m_rolldown"]))
        - (hedge_ratios["back_wing_hr"] * (back_wing_metrics["rough_carry"] + back_wing_metrics["rough_12m_rolldown"])),
    }
