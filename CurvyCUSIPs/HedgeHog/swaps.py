from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Literal

import numpy as np
import pandas as pd
import rateslib as rl
import QuantLib as ql
import ujson as json
from termcolor import colored

from CurvyCUSIPs.utils.dtcc_swaps_utils import datetime_to_ql_date


@dataclass
class SwapLeg:
    trade_date: datetime
    original_tenor: str
    original_fixed_rate: float
    weighting: float = None
    key: str = None
    notional: str = None
    type: Literal["receiver", "payer"] = None


def ql_curve_to_rl_curve(ql_curve: ql.DiscountCurve | ql.ZeroCurve | ql.ForwardCurve):
    ql_dates: Tuple[ql.Date] = ql_curve.dates()
    rl_curve = rl.Curve(
        id="sofr",
        convention="Act360",
        calendar="nyc",
        modifier="MF",
        interpolation="log_linear",
        nodes=dict(zip([datetime(ql_dt.year(), ql_dt.month(), ql_dt.dayOfMonth()) for ql_dt in ql_dates], list(ql_curve.discounts()))),
    )
    return rl_curve


def calibrate_rl_curve(date: datetime, tenor_spot_df: pd.DataFrame):
    rl_curve = rl.Curve(
        id="sofr",
        convention="Act360",
        calendar="nyc",
        modifier="MF",
        interpolation="log_linear",
        nodes={**{date: 1.0}, **{rl.add_tenor(date, tenor, "MF", "stk"): 1.0 for tenor in tenor_spot_df["Tenor"]}},
    )

    sofr_args = dict(effective=date, spec="usd_irs", curves="sofr")
    rl_solver = rl.Solver(
        curves=[rl_curve],
        instruments=[rl.IRS(termination=_, **sofr_args) for _ in tenor_spot_df["Tenor"]],
        s=tenor_spot_df["Spot"],
        instrument_labels=tenor_spot_df["Tenor"],
        id="us_rates",
    )

    return rl_curve, rl_solver


def swap_leg_portfolio_to_rl_portfolio(swap_portfolio: List[SwapLeg]):
    return [rl.IRS(rl.dt(2022, 1, 1), "1m", "A", curves="sofr") for swap_leg in swap_portfolio]


def book_metrics(
    swap_portfolio: List[SwapLeg],
    ql_curve: ql.DiscountCurve | ql.ZeroCurve | ql.ForwardCurve,
    ql_yts: ql.RelinkableYieldTermStructureHandle,
    ql_sofr: ql.Sofr,
    agg_c_and_r_results: Optional[bool] = False,
):
    cal = ql.UnitedStates(ql.UnitedStates.GovernmentBond)
    ql_yts.linkTo(ql_curve)
    engine = ql.DiscountingSwapEngine(ql_yts)

    book: Dict[str, ql.OvernightIndexedSwap] = {}
    carry_roll_results = {}
    book_bps = 0
    skip_c_and_r_calc = False
    for swap_leg in swap_portfolio:
        if swap_leg.weighting == 0:
            continue

        if "Fwd" in swap_leg.original_tenor:
            fwd_tenor_str, swap_tenor_str = swap_leg.original_tenor.split(" Fwd ")
        else:
            swap_tenor_str = swap_leg.original_tenor
            fwd_tenor_str = "0D"

        effective_date = cal.advance(datetime_to_ql_date(swap_leg.trade_date), ql.Period("2D"))
        effective_date = cal.advance(effective_date, ql.Period(fwd_tenor_str))

        ql.Settings.instance().evaluationDate = datetime_to_ql_date(swap_leg.trade_date)
        swap: ql.OvernightIndexedSwap = ql.MakeOIS(
            ql.Period(swap_tenor_str),
            ql_sofr,
            swap_leg.original_fixed_rate,
            ql.Period(fwd_tenor_str),
            swapType=ql.OvernightIndexedSwap.Receiver if swap_leg.type == "receiver" else ql.OvernightIndexedSwap.Payer,
            effectiveDate=effective_date,
            terminationDate=cal.advance(effective_date, ql.Period(swap_tenor_str)),
            paymentAdjustmentConvention=ql.ModifiedFollowing,
            paymentLag=2,
            fixedLegDayCount=ql.Actual360(),
            nominal=swap_leg.notional if swap_leg.notional else -1 if swap_leg.weighting < 0 else 1,
        )
        swap.setPricingEngine(engine)

        curr_fwd_days = swap.startDate() - (ql.Date.todaysDate() - ql.Period("1D"))
        copy_swap_tenor_str = swap_tenor_str
        curr_tenor = (
            f"{curr_fwd_days}D Fwd {swap_tenor_str}" if curr_fwd_days > 0 else f"{np.round((swap.maturityDate() - ql.Date.todaysDate()) / 360, 3)}Y"
        )
        book[swap_leg.key or curr_tenor] = swap
        book_bps += swap.fixedLegBPS() * swap_leg.weighting

        if "Fwd" in curr_tenor:
            fwd_tenor_str, swap_tenor_str = curr_tenor.split(" Fwd ")
        else:
            swap_tenor_str = curr_tenor
            fwd_tenor_str = "0D"

        rolldown_dict = {}
        carry_dict = {}

        if "W" in copy_swap_tenor_str or "M" in copy_swap_tenor_str:
            skip_c_and_r_calc = True
            continue

        for horizon_days, horizon_months in dict(zip(["30D", "60D", "90D", "180D", "360D"], ["1M", "2M", "3M", "6M", "12M"])).items():
            rolled_maturity: ql.Period = (
                ql.Period(swap_tenor_str) - ql.Period(horizon_days)
                if "D" in swap_tenor_str
                else ql.Period(swap_tenor_str) - ql.Period(horizon_months)
            )
            if rolled_maturity.length() == 0:
                rolldown_dict[horizon_days] = None
                carry_dict[horizon_days] = None
                continue

            rolled_swap: ql.OvernightIndexedSwap = ql.MakeOIS(
                rolled_maturity,
                ql_sofr,
                0,
                ql.Period(fwd_tenor_str),
                effectiveDate=effective_date,
                terminationDate=cal.advance(effective_date, rolled_maturity),
                paymentAdjustmentConvention=ql.ModifiedFollowing,
                paymentLag=2,
                fixedLegDayCount=ql.Actual360(),
            )
            rolled_swap.setPricingEngine(engine)
            rolldown_dict[horizon_days] = (swap.fairRate() - rolled_swap.fairRate()) * 10_000

            fwd_rolled_effective_date = cal.advance(effective_date, ql.Period(fwd_tenor_str) + ql.Period(horizon_days))
            fwd_rolled_swap: ql.OvernightIndexedSwap = ql.MakeOIS(
                rolled_maturity,
                ql_sofr,
                0,
                ql.Period(fwd_tenor_str) + ql.Period(horizon_days),
                effectiveDate=fwd_rolled_effective_date,
                terminationDate=cal.advance(fwd_rolled_effective_date, rolled_maturity),
                paymentAdjustmentConvention=ql.ModifiedFollowing,
                paymentLag=2,
                fixedLegDayCount=ql.Actual360(),
            )
            carry_dict[horizon_days] = (fwd_rolled_swap.fairRate() - swap.fairRate()) * 10_000

        carry_roll_results[swap_leg.key or curr_tenor] = {"roll": rolldown_dict, "carry": carry_dict}

    if agg_c_and_r_results and not skip_c_and_r_calc:
        weights = np.array([s.weighting for s in swap_portfolio])
        total_c_and_r_df = pd.DataFrame(
            {
                "Total C+R (bps)": pd.DataFrame({key: value["carry"] for key, value in carry_roll_results.items()}).dot(weights)
                + pd.DataFrame({key: value["roll"] for key, value in carry_roll_results.items()}).dot(weights)
            }
        )

    return {
        "total_carry_and_roll": total_c_and_r_df if agg_c_and_r_results and not skip_c_and_r_calc else None,
        "bps_running": carry_roll_results,
        "book": book,
        "book_bps": book_bps,
    }


def dv01_neutral_curve_hedge_ratio(
    front_leg_swap: SwapLeg,
    back_leg_swap: SwapLeg,
    ql_curve: ql.DiscountCurve | ql.ZeroCurve | ql.ForwardCurve,
    ql_yts: ql.RelinkableYieldTermStructureHandle,
    ql_sofr: ql.Sofr,
    curve_pvbp: Optional[int] = None,
    front_leg_notional: Optional[int] = None,
    back_leg_notional: Optional[int] = None,
    total_trade_notional: Optional[int] = None,
    beta_adjustment_wrt_back_leg: Optional[int] = None,
    verbose: Optional[bool] = True,
    custom_beta_weighted_title: Optional[str] = None,
):
    front_leg_swap.key = "front_wing"
    back_leg_swap.key = "back_wing"

    if front_leg_notional and front_leg_notional * front_leg_swap.weighting < 0:
        raise ValueError(
            f"Front Notional and Front Weighting does not match - check 'front_leg_notional': {front_leg_notional} and 'front_leg_swap.weighting': {front_leg_swap.weighting}"
        )
    if back_leg_notional and back_leg_notional * back_leg_swap.weighting < 0:
        raise ValueError(
            f"Back Notional and Back Weighting does not match - check 'back_leg_notional': {back_leg_notional} and 'back_leg_swap.weighting': {back_leg_swap.weighting}"
        )

    front_leg_swap.key = "front_leg"
    back_leg_swap.key = "back_leg"
    book_metrics_dict = book_metrics(swap_portfolio=[front_leg_swap, back_leg_swap], ql_curve=ql_curve, ql_yts=ql_yts, ql_sofr=ql_sofr)
    book: Dict[str, ql.OvernightIndexedSwap] = book_metrics_dict["book"]
    cr_bps_running_dict = book_metrics_dict["bps_running"]

    bpv_hr = book["back_leg"].fixedLegBPS() / book["front_leg"].fixedLegBPS()
    print(colored(f"BPV Neutral Hedge Ratio: {bpv_hr}", "light_blue")) if verbose else None

    if beta_adjustment_wrt_back_leg:
        title = custom_beta_weighted_title or "Beta Weighted Hedge Ratio"
        hr = bpv_hr * beta_adjustment_wrt_back_leg
        (print(colored(f"{title}: {hr:3f}", "light_magenta")) if verbose else None)
    else:
        hr = bpv_hr

    is_long_or_short = front_leg_swap.weighting / np.abs(front_leg_swap.weighting)

    if curve_pvbp:
        back_leg_notional = np.abs(curve_pvbp / book["back_leg"].fixedLegBPS()) * is_long_or_short * -1

    if total_trade_notional:
        normalized_total = np.abs(hr) - 1
        back_leg_notional = total_trade_notional / normalized_total

    if front_leg_notional and back_leg_notional:
        raise ValueError("'front_leg_notional' and 'back_leg_par_amount' are both defined!")
    if not front_leg_notional and not back_leg_notional:
        back_leg_notional = 1_000_000 if back_leg_swap.weighting > 0 else -1_000_000
    if back_leg_notional:
        front_leg_notional = back_leg_notional * hr
    elif front_leg_notional:
        back_leg_notional = front_leg_notional / hr

    total_trade_notional_check = front_leg_notional + back_leg_notional
    if total_trade_notional and np.abs(total_trade_notional - total_trade_notional_check) > 1000:
        print(total_trade_notional, total_trade_notional_check)
        raise ValueError("Total Trade Notional Mismatch")
    else:
        total_trade_notional = total_trade_notional_check

    notional_scaled_front_leg_pvbp = book["front_leg"].fixedLegBPS() * np.abs(front_leg_notional)
    notional_scaled_back_leg_pvbp = book["back_leg"].fixedLegBPS() * np.abs(back_leg_notional)
    derived_curve_pvbp = (
        np.abs(notional_scaled_back_leg_pvbp) * is_long_or_short * beta_adjustment_wrt_back_leg
        if beta_adjustment_wrt_back_leg
        else np.abs(notional_scaled_back_leg_pvbp) * is_long_or_short
    )

    if curve_pvbp and np.abs((np.abs(derived_curve_pvbp) * is_long_or_short) - (np.abs(curve_pvbp) * is_long_or_short)) > 100:
        print((np.abs(derived_curve_pvbp) * is_long_or_short), (np.abs(curve_pvbp) * is_long_or_short))
        raise ValueError("Fly PVBP mismatch")
    else:
        curve_pvbp = derived_curve_pvbp

    print(f"{front_leg_swap.original_tenor}: Notional = {front_leg_notional:_.3f}, PVBP = {notional_scaled_front_leg_pvbp:_.3f}")
    print(f"{back_leg_swap.original_tenor}: Notional = {back_leg_notional:_.3f}, PVBP = {notional_scaled_back_leg_pvbp:_.3f}")
    print(f"Net Notional: {total_trade_notional:_.3f}")

    if beta_adjustment_wrt_back_leg:
        print(f"BVP Neutral PVBP: {np.abs(curve_pvbp) * is_long_or_short:_.3f}")
        print(
            f"Beta Weighted PVBP (bull steepening): {(2 * notional_scaled_front_leg_pvbp +  notional_scaled_back_leg_pvbp) * is_long_or_short:_.3f}"
        )
    else:
        print(f"BVP Neutral PVBP: {np.abs(curve_pvbp) * is_long_or_short:_.3f}")

    def _calc_curve_bps_running_carry_and_roll(
        tenor: Literal["30D", "60D", "90D", "180D"],
        cr_bps_running_results: Dict,
        front_leg_weighting: float,
        back_leg_weighting: float,
        front_leg_key="front_leg",
        back_leg_key="back_leg",
    ):
        front = (cr_bps_running_results[front_leg_key]["carry"][tenor] * front_leg_weighting) + (
            cr_bps_running_results[front_leg_key]["roll"][tenor] * front_leg_weighting
        )
        back = (cr_bps_running_results[back_leg_key]["carry"][tenor] * back_leg_weighting) + (
            cr_bps_running_results[back_leg_key]["roll"][tenor] * back_leg_weighting
        )
        return front + back

    return {
        "current_curve_bps": (
            (back_leg_swap.original_fixed_rate * np.abs(back_leg_swap.weighting))
            - (front_leg_swap.original_fixed_rate * np.abs(beta_adjustment_wrt_back_leg or front_leg_swap.weighting))
        )
        * 10_000,
        "current_curve_pvbp": (
            (2 * notional_scaled_front_leg_pvbp + notional_scaled_back_leg_pvbp) * is_long_or_short
            if beta_adjustment_wrt_back_leg
            else np.abs(curve_pvbp) * is_long_or_short
        ),
        "bpv_neutral_curve_bps": (
            (back_leg_swap.original_fixed_rate * np.abs(back_leg_swap.weighting))
            - (front_leg_swap.original_fixed_rate * np.abs(front_leg_swap.weighting))
        )
        * 10_000,
        "bpv_neutral_curve_pvbp": np.abs(curve_pvbp) * is_long_or_short,
        "1M_carry_and_roll_bps_running": _calc_curve_bps_running_carry_and_roll(
            tenor="30D",
            cr_bps_running_results=cr_bps_running_dict,
            front_leg_weighting=front_leg_swap.weighting,
            back_leg_weighting=back_leg_swap.weighting,
        ),
        "1M_carry_and_roll_bps_running_beta_weighted": (
            _calc_curve_bps_running_carry_and_roll(
                tenor="30D",
                cr_bps_running_results=cr_bps_running_dict,
                front_leg_weighting=beta_adjustment_wrt_back_leg,
                back_leg_weighting=back_leg_swap.weighting,
            )
            if beta_adjustment_wrt_back_leg
            else None
        ),
        "2M_carry_and_roll_bps_running": _calc_curve_bps_running_carry_and_roll(
            tenor="60D",
            cr_bps_running_results=cr_bps_running_dict,
            front_leg_weighting=front_leg_swap.weighting,
            back_leg_weighting=back_leg_swap.weighting,
        ),
        "2M_carry_and_roll_bps_running_beta_weighted": (
            _calc_curve_bps_running_carry_and_roll(
                tenor="60D",
                cr_bps_running_results=cr_bps_running_dict,
                front_leg_weighting=beta_adjustment_wrt_back_leg,
                back_leg_weighting=back_leg_swap.weighting,
            )
            if beta_adjustment_wrt_back_leg
            else None
        ),
        "3M_carry_and_roll_bps_running": _calc_curve_bps_running_carry_and_roll(
            tenor="90D",
            cr_bps_running_results=cr_bps_running_dict,
            front_leg_weighting=front_leg_swap.weighting,
            back_leg_weighting=back_leg_swap.weighting,
        ),
        "3M_carry_and_roll_bps_running_beta_weighted": (
            _calc_curve_bps_running_carry_and_roll(
                tenor="90D",
                cr_bps_running_results=cr_bps_running_dict,
                front_leg_weighting=beta_adjustment_wrt_back_leg,
                back_leg_weighting=back_leg_swap.weighting,
            )
            if beta_adjustment_wrt_back_leg
            else None
        ),
        "6M_carry_and_roll_bps_running": _calc_curve_bps_running_carry_and_roll(
            tenor="180D",
            cr_bps_running_results=cr_bps_running_dict,
            front_leg_weighting=front_leg_swap.weighting,
            back_leg_weighting=back_leg_swap.weighting,
        ),
        "6M_carry_and_roll_bps_running_beta_weighted": (
            _calc_curve_bps_running_carry_and_roll(
                tenor="180D",
                cr_bps_running_results=cr_bps_running_dict,
                front_leg_weighting=beta_adjustment_wrt_back_leg,
                back_leg_weighting=back_leg_swap.weighting,
            )
            if beta_adjustment_wrt_back_leg
            else None
        ),
        "total_trade_notional": total_trade_notional,
        "front_leg": {
            "current_yield": front_leg_swap.original_fixed_rate,
            "current_weighted_yield": front_leg_swap.original_fixed_rate * front_leg_swap.weighting,
            "hr": hr,
            "bpv": bpv_hr,
            "notional": front_leg_notional,
            "pvbp_per_mm": book["front_leg"].fixedLegBPS() * 1_000_000,
            "pvbp_leg": book["front_leg"].fixedLegBPS() * front_leg_notional,
            "bpv_bps_running": {
                "1M_carry": cr_bps_running_dict["front_leg"]["carry"]["30D"] * front_leg_swap.weighting,
                "1M_roll": cr_bps_running_dict["front_leg"]["roll"]["30D"] * front_leg_swap.weighting,
                "2M_carry": cr_bps_running_dict["front_leg"]["carry"]["60D"] * front_leg_swap.weighting,
                "2M_roll": cr_bps_running_dict["front_leg"]["roll"]["60D"] * front_leg_swap.weighting,
                "3M_carry": cr_bps_running_dict["front_leg"]["carry"]["90D"] * front_leg_swap.weighting,
                "3M_roll": cr_bps_running_dict["front_leg"]["roll"]["90D"] * front_leg_swap.weighting,
                "6M_carry": cr_bps_running_dict["front_leg"]["carry"]["180D"] * front_leg_swap.weighting,
                "6M_roll": cr_bps_running_dict["front_leg"]["roll"]["180D"] * front_leg_swap.weighting,
            },
            "beta_weighted_bps_runnung": (
                {
                    "1M_carry": cr_bps_running_dict["front_leg"]["carry"]["30D"] * np.abs(beta_adjustment_wrt_back_leg) * is_long_or_short,
                    "1M_roll": cr_bps_running_dict["front_leg"]["roll"]["30D"] * np.abs(beta_adjustment_wrt_back_leg) * is_long_or_short,
                    "2M_carry": cr_bps_running_dict["front_leg"]["carry"]["60D"] * np.abs(beta_adjustment_wrt_back_leg) * is_long_or_short,
                    "2M_roll": cr_bps_running_dict["front_leg"]["roll"]["60D"] * np.abs(beta_adjustment_wrt_back_leg) * is_long_or_short,
                    "3M_carry": cr_bps_running_dict["front_leg"]["carry"]["90D"] * np.abs(beta_adjustment_wrt_back_leg) * is_long_or_short,
                    "3M_roll": cr_bps_running_dict["front_leg"]["roll"]["90D"] * np.abs(beta_adjustment_wrt_back_leg) * is_long_or_short,
                    "6M_carry": cr_bps_running_dict["front_leg"]["carry"]["180D"] * np.abs(beta_adjustment_wrt_back_leg) * is_long_or_short,
                    "6M_roll": cr_bps_running_dict["front_leg"]["roll"]["180D"] * np.abs(beta_adjustment_wrt_back_leg) * is_long_or_short,
                }
                if beta_adjustment_wrt_back_leg
                else None
            ),
        },
        "back_leg": {
            "current_yield": back_leg_swap.original_fixed_rate,
            "current_weighted_yield": back_leg_swap.original_fixed_rate * back_leg_swap.weighting,
            "hr": 1,
            "bpv": 1,
            "notional": back_leg_notional,
            "pvbp_per_mm": book["back_leg"].fixedLegBPS() * 1_000_000,
            "pvbp_leg": book["back_leg"].fixedLegBPS() * back_leg_notional,
            "bpv_bps_running": {
                "1M_carry": cr_bps_running_dict["back_leg"]["carry"]["30D"] * back_leg_swap.weighting,
                "1M_roll": cr_bps_running_dict["back_leg"]["roll"]["30D"] * back_leg_swap.weighting,
                "2M_carry": cr_bps_running_dict["back_leg"]["carry"]["60D"] * back_leg_swap.weighting,
                "2M_roll": cr_bps_running_dict["back_leg"]["roll"]["60D"] * back_leg_swap.weighting,
                "3M_carry": cr_bps_running_dict["back_leg"]["carry"]["90D"] * back_leg_swap.weighting,
                "3M_roll": cr_bps_running_dict["back_leg"]["roll"]["90D"] * back_leg_swap.weighting,
                "6M_carry": cr_bps_running_dict["back_leg"]["carry"]["180D"] * back_leg_swap.weighting,
                "6M_roll": cr_bps_running_dict["back_leg"]["roll"]["180D"] * back_leg_swap.weighting,
            },
        },
        "ql_ois_obj": book,
    }


def dv01_neutral_butterfly_hedge_ratio(
    front_leg_swap: SwapLeg,
    belly_swap: SwapLeg,
    back_leg_swap: SwapLeg,
    ql_curve: ql.DiscountCurve | ql.ZeroCurve | ql.ForwardCurve,
    ql_yts: ql.RelinkableYieldTermStructureHandle,
    ql_sofr: ql.Sofr,
    fly_pvbp: Optional[int] = None,
    front_wing_notional: Optional[int] = None,
    belly_notional: Optional[int] = None,
    back_wing_notional: Optional[int] = None,
    total_trade_notional: Optional[int] = None,
    front_wing_beta_adjustment_wrt_belly: Optional[int] = None,
    back_wing_beta_adjustment_wrt_belly: Optional[int] = None,
    verbose: Optional[bool] = True,
):
    front_leg_swap.key = "front_wing"
    belly_swap.key = "belly"
    back_leg_swap.key = "back_wing"

    if belly_notional and belly_notional * belly_swap.weighting < 0:
        raise ValueError(
            f"Belly Notional and Belly Weighting does not match - check 'belly_notional': {belly_notional} and 'belly_swap.weighting': {belly_swap.weighting}"
        )
    if front_wing_notional and front_wing_notional * front_leg_swap.weighting < 0:
        raise ValueError(
            f"Front Notional and Front Weighting does not match - check 'front_wing_notional': {front_wing_notional} and 'front_wing.weighting': {front_leg_swap.weighting}"
        )
    if back_wing_notional and back_wing_notional * back_leg_swap.weighting < 0:
        raise ValueError(
            f"Back Notional and Back Weighting does not match - check 'back_wing_notional': {back_wing_notional} and 'back_wing.weighting': {back_leg_swap.weighting}"
        )
    if total_trade_notional and belly_swap.weighting * total_trade_notional > 0:
        raise ValueError(f"Total Trade Notional and Belly Weighting matches - if short the fly, you should be net negative notional")
    if fly_pvbp and belly_swap.weighting * fly_pvbp > 0:
        raise ValueError(f"Fly PVBP and Belly Weighting matches - if short the fly, you should be net negative pvbp/short delta risk")

    book_metrics_dict = book_metrics(swap_portfolio=[front_leg_swap, belly_swap, back_leg_swap], ql_curve=ql_curve, ql_yts=ql_yts, ql_sofr=ql_sofr)
    book: Dict[str, ql.OvernightIndexedSwap] = book_metrics_dict["book"]
    cr_bps_running_dict = book_metrics_dict["bps_running"]

    # explictly sign/check signs
    front_wring_bpv_hr = (book["belly"].fixedLegBPS() / book[front_leg_swap.key].fixedLegBPS()) * front_leg_swap.weighting
    front_wring_bpv_hr = np.abs(front_wring_bpv_hr) if front_leg_swap.weighting > 0 else np.abs(front_wring_bpv_hr) * -1

    back_wring_bpv_hr = (book["belly"].fixedLegBPS() / book[back_leg_swap.key].fixedLegBPS()) * np.abs(back_leg_swap.weighting)
    back_wring_bpv_hr = np.abs(back_wring_bpv_hr) if back_leg_swap.weighting > 0 else np.abs(back_wring_bpv_hr) * -1

    bpv_hedge_ratios = {
        "front_wing_hr": front_wring_bpv_hr,
        "belly_hr": belly_swap.weighting,
        "back_wing_hr": back_wring_bpv_hr,
    }
    print(colored(f"BPV Neutral Hedge Ratio:", "light_blue")) if verbose else None
    print(json.dumps(bpv_hedge_ratios, indent=4)) if verbose else None

    is_beta_weighted = front_wing_beta_adjustment_wrt_belly and back_wing_beta_adjustment_wrt_belly
    if is_beta_weighted:
        # explictly sign/check signs
        front_wring_beta_hr = (book["belly"].fixedLegBPS() / book[front_leg_swap.key].fixedLegBPS()) * front_wing_beta_adjustment_wrt_belly
        front_wring_beta_hr = np.abs(front_wring_beta_hr) if front_leg_swap.weighting > 0 else np.abs(front_wring_beta_hr) * -1
        back_wring_beta_hr = (book["belly"].fixedLegBPS() / book[back_leg_swap.key].fixedLegBPS()) * back_wing_beta_adjustment_wrt_belly
        back_wring_beta_hr = np.abs(back_wring_beta_hr) if back_leg_swap.weighting > 0 else np.abs(back_wring_beta_hr) * -1
        print(colored(f"Beta Weighted Hedge Ratio:", "light_magenta")) if verbose else None
        hedge_ratios = {
            "front_wing_hr": front_wring_beta_hr,
            "belly_hr": belly_swap.weighting,
            "back_wing_hr": back_wring_beta_hr,
        }
        print(json.dumps(hedge_ratios, indent=4)) if verbose else None
    else:
        hedge_ratios = bpv_hedge_ratios

    is_long_or_short = (belly_swap.weighting / np.abs(belly_swap.weighting)) * -1
    if fly_pvbp:
        abs_belly_notional = np.abs(fly_pvbp / book["belly"].fixedLegBPS() * 2)
        belly_notional = abs_belly_notional if hedge_ratios["belly_hr"] > 0 else abs_belly_notional * -1

    if total_trade_notional is not None:
        if front_wing_notional is not None or belly_notional is not None or back_wing_notional is not None:
            raise ValueError("Cannot provide total_trade_notional along with individual leg par amounts.")

        total_hr_abs = hedge_ratios["front_wing_hr"] + hedge_ratios["belly_hr"] + hedge_ratios["back_wing_hr"]
        abs_belly_notional = np.abs((total_trade_notional / total_hr_abs))
        belly_notional = abs_belly_notional if hedge_ratios["belly_hr"] > 0 else abs_belly_notional * -1
        front_wing_notional = hedge_ratios["front_wing_hr"] * np.abs(belly_notional)
        back_wing_notional = hedge_ratios["back_wing_hr"] * np.abs(belly_notional)

        test = front_wing_notional + belly_notional + back_wing_notional
        if np.abs(test - total_trade_notional) > 1000:
            raise ValueError(f"Total Notional is off - check your signs!")

    else:
        if belly_notional:
            front_wing_notional = hedge_ratios["front_wing_hr"] * np.abs(belly_notional)
            belly_notional = belly_notional
            back_wing_notional = hedge_ratios["back_wing_hr"] * np.abs(belly_notional)
        elif front_wing_notional:
            front_wing_notional = front_wing_notional
            abs_belly_notional = np.abs(np.abs(front_wing_notional) / hedge_ratios["front_wing_hr"])
            belly_notional = abs_belly_notional if hedge_ratios["belly_hr"] > 0 else abs_belly_notional * -1
            abs_back_wing_notional = np.abs(hedge_ratios["back_wing_hr"] * (np.abs(front_wing_notional) / hedge_ratios["front_wing_hr"]))
            back_wing_notional = abs_back_wing_notional if hedge_ratios["back_wing_hr"] > 0 else abs_back_wing_notional * -1
        elif back_wing_notional:
            abs_front_wing_notional = np.abs(hedge_ratios["front_wing_hr"] * (np.abs(back_wing_notional) / hedge_ratios["back_wing_hr"]))
            front_wing_notional = abs_front_wing_notional if hedge_ratios["front_wing_hr"] > 0 else abs_front_wing_notional * -1
            abs_belly_notional = np.abs(np.abs(back_wing_notional) / hedge_ratios["back_wing_hr"])
            belly_notional = abs_belly_notional if hedge_ratios["belly_hr"] > 0 else abs_belly_notional * -1
            back_wing_notional = back_wing_notional

        total_trade_notional = front_wing_notional + belly_notional + back_wing_notional

    notional_scaled_front_wing_pvbp = book["front_wing"].fixedLegBPS() * np.abs(front_wing_notional) * -1
    notional_scaled_belly_pvbp = book["belly"].fixedLegBPS() * np.abs(belly_notional) * -1
    notional_scaled_back_wing_pvbp = book["back_wing"].fixedLegBPS() * np.abs(back_wing_notional) * -1

    if is_beta_weighted:
        derived_fly_pvbp = (
            (notional_scaled_belly_pvbp * np.abs(belly_swap.weighting))
            + (notional_scaled_front_wing_pvbp * front_wing_beta_adjustment_wrt_belly)
            + (notional_scaled_back_wing_pvbp * back_wing_beta_adjustment_wrt_belly)
        )
    else:
        derived_fly_pvbp = (
            (notional_scaled_belly_pvbp * np.abs(belly_swap.weighting))
            + (notional_scaled_front_wing_pvbp * np.abs(front_leg_swap.weighting))
            + (notional_scaled_back_wing_pvbp * np.abs(back_leg_swap.weighting))
        )
        if fly_pvbp and np.abs((np.abs(derived_fly_pvbp) * is_long_or_short) - (np.abs(fly_pvbp) * is_long_or_short)) > 100:
            print((np.abs(derived_fly_pvbp) * is_long_or_short), (np.abs(fly_pvbp) * is_long_or_short))
            raise ValueError(f"Fly PVBP mismatch: ")
        else:
            fly_pvbp = derived_fly_pvbp

    (
        print(f"{front_leg_swap.original_tenor}: Fixed Notional = {front_wing_notional:_.3f}, PVBP = {notional_scaled_front_wing_pvbp:_.3f}")
        if verbose
        else None
    )
    print(f"{belly_swap.original_tenor}: Fixed Notional = {belly_notional:_.3f}, PVBP = {notional_scaled_belly_pvbp:_.3f}") if verbose else None
    (
        print(f"{back_leg_swap.original_tenor}: Fixed Notional = {back_wing_notional:_.3f}, PVBP = {notional_scaled_back_wing_pvbp:_.3f}")
        if verbose
        else None
    )
    print(f"Net Fixed Notional: {total_trade_notional:_.3f}") if verbose else None

    bpv_neutral_fly_pvbp = np.abs(notional_scaled_belly_pvbp) * 0.5 * is_long_or_short
    if is_beta_weighted and verbose:
        print(f"BPV Neutral Fly PVBP: {bpv_neutral_fly_pvbp:_.3f}")
        print(f"Beta Weighted Fly PVBP (naive): {np.abs(derived_fly_pvbp) * is_long_or_short:_.3f}")
        risk = front_wing_beta_adjustment_wrt_belly + back_wing_beta_adjustment_wrt_belly
        print(
            f"Risk Weights - Front Wing: {front_wing_beta_adjustment_wrt_belly:.3%}, Back Wing: {back_wing_beta_adjustment_wrt_belly:.3%}, Sum: {risk:.3%}"
        )
    else:
        print(f"BPV Neutral Fly PVBP: {np.abs(derived_fly_pvbp) * is_long_or_short:_.3f}") if verbose else None

    def _calc_fly_bps_running_carry_and_roll(
        tenor: Literal["30D", "60D", "90D", "180D"],
        cr_bps_running_results: Dict,
        front_wing_weighting: float,
        belly_wing_weighting: float,
        back_wing_weighting: float,
        front_wing_key="front_wing",
        belly_key="belly",
        back_wing_key="back_wing",
    ):
        front = (cr_bps_running_results[front_wing_key]["carry"][tenor] * front_wing_weighting) + (
            cr_bps_running_results[front_wing_key]["roll"][tenor] * front_wing_weighting
        )
        belly = (cr_bps_running_results[belly_key]["carry"][tenor] * belly_wing_weighting) + (
            cr_bps_running_results[belly_key]["roll"][tenor] * belly_wing_weighting
        )
        back = (cr_bps_running_results[back_wing_key]["carry"][tenor] * back_wing_weighting) + (
            cr_bps_running_results[back_wing_key]["roll"][tenor] * back_wing_weighting
        )
        return front + belly + back

    if verbose:
        cr_3m_bps_running = _calc_fly_bps_running_carry_and_roll(
            tenor="90D",
            cr_bps_running_results=cr_bps_running_dict,
            front_wing_weighting=front_leg_swap.weighting,
            belly_wing_weighting=belly_swap.weighting,
            back_wing_weighting=back_leg_swap.weighting,
        )
        print(f"3M bps Running Carry & Roll: {cr_3m_bps_running:.3f} bps")

        bw_cr_3m_bps_running = (
            _calc_fly_bps_running_carry_and_roll(
                tenor="90D",
                cr_bps_running_results=cr_bps_running_dict,
                front_wing_weighting=front_wing_beta_adjustment_wrt_belly * is_long_or_short,
                belly_wing_weighting=belly_swap.weighting,
                back_wing_weighting=back_wing_beta_adjustment_wrt_belly * is_long_or_short,
            )
            if is_beta_weighted
            else None
        )
        print(f"Beta Weigted 3M bps Running Carry & Roll: {bw_cr_3m_bps_running:.3f} bps") if is_beta_weighted else None

    return {
        "is_long": is_long_or_short,
        "risk_weights": [front_leg_swap.weighting, belly_swap.weighting, back_leg_swap.weighting],
        "current_fly_bps": (
            (belly_swap.original_fixed_rate * belly_swap.weighting)
            + (front_leg_swap.original_fixed_rate * front_leg_swap.weighting)
            + (back_leg_swap.original_fixed_rate * back_leg_swap.weighting)
        )
        * 10_000,
        "current_fly_pvbp": np.abs(derived_fly_pvbp) * is_long_or_short,
        "bpv_neutral_fly_bps": (
            (belly_swap.original_fixed_rate * 1) + (front_leg_swap.original_fixed_rate * -0.5) + (back_leg_swap.original_fixed_rate * -0.5)
        )
        * 10_000,
        "bpv_neutral_fly_pvbp": bpv_neutral_fly_pvbp,
        "1M_carry_and_roll_bps_running": _calc_fly_bps_running_carry_and_roll(
            tenor="30D",
            cr_bps_running_results=cr_bps_running_dict,
            front_wing_weighting=front_leg_swap.weighting,
            belly_wing_weighting=belly_swap.weighting,
            back_wing_weighting=back_leg_swap.weighting,
        ),
        "1M_carry_and_roll_bps_running_beta_weigted": (
            _calc_fly_bps_running_carry_and_roll(
                tenor="30D",
                cr_bps_running_results=cr_bps_running_dict,
                front_wing_weighting=front_wing_beta_adjustment_wrt_belly * is_long_or_short,
                belly_wing_weighting=belly_swap.weighting,
                back_wing_weighting=back_wing_beta_adjustment_wrt_belly * is_long_or_short,
            )
            if is_beta_weighted
            else None
        ),
        "2M_carry_and_roll_bps_running": _calc_fly_bps_running_carry_and_roll(
            tenor="60D",
            cr_bps_running_results=cr_bps_running_dict,
            front_wing_weighting=front_leg_swap.weighting,
            belly_wing_weighting=belly_swap.weighting,
            back_wing_weighting=back_leg_swap.weighting,
        ),
        "2M_carry_and_roll_bps_running_beta_weigted": (
            _calc_fly_bps_running_carry_and_roll(
                tenor="60D",
                cr_bps_running_results=cr_bps_running_dict,
                front_wing_weighting=front_wing_beta_adjustment_wrt_belly * is_long_or_short,
                belly_wing_weighting=belly_swap.weighting,
                back_wing_weighting=back_wing_beta_adjustment_wrt_belly * is_long_or_short,
            )
            if is_beta_weighted
            else None
        ),
        "3M_carry_and_roll_bps_running": _calc_fly_bps_running_carry_and_roll(
            tenor="90D",
            cr_bps_running_results=cr_bps_running_dict,
            front_wing_weighting=front_leg_swap.weighting,
            belly_wing_weighting=belly_swap.weighting,
            back_wing_weighting=back_leg_swap.weighting,
        ),
        "3M_carry_and_roll_bps_running_beta_weigted": (
            _calc_fly_bps_running_carry_and_roll(
                tenor="90D",
                cr_bps_running_results=cr_bps_running_dict,
                front_wing_weighting=front_wing_beta_adjustment_wrt_belly * is_long_or_short,
                belly_wing_weighting=belly_swap.weighting,
                back_wing_weighting=back_wing_beta_adjustment_wrt_belly * is_long_or_short,
            )
            if is_beta_weighted
            else None
        ),
        "6M_carry_and_roll_bps_running": _calc_fly_bps_running_carry_and_roll(
            tenor="180D",
            cr_bps_running_results=cr_bps_running_dict,
            front_wing_weighting=front_leg_swap.weighting,
            belly_wing_weighting=belly_swap.weighting,
            back_wing_weighting=back_leg_swap.weighting,
        ),
        "6M_carry_and_roll_bps_running_beta_weigted": (
            _calc_fly_bps_running_carry_and_roll(
                tenor="180D",
                cr_bps_running_results=cr_bps_running_dict,
                front_wing_weighting=front_wing_beta_adjustment_wrt_belly * is_long_or_short,
                belly_wing_weighting=belly_swap.weighting,
                back_wing_weighting=back_wing_beta_adjustment_wrt_belly * is_long_or_short,
            )
            if is_beta_weighted
            else None
        ),
        "total_trade_notional": total_trade_notional,
        "front_wing": {
            "current_yield": front_leg_swap.original_fixed_rate,
            "current_weighted_yield": front_leg_swap.original_fixed_rate * front_leg_swap.weighting,
            "hr": hedge_ratios["front_wing_hr"],
            "bpv_hr": bpv_hedge_ratios["front_wing_hr"],
            "notional": front_wing_notional,
            "pvbp_per_mm": book["front_wing"].fixedLegBPS() * 1_000_000,
            "pvbp_leg": book["front_wing"].fixedLegBPS() * front_wing_notional,
            "bpv_neutral_bps_running": {
                "1M_carry": cr_bps_running_dict["front_wing"]["carry"]["30D"] * front_leg_swap.weighting,
                "1M_roll": cr_bps_running_dict["front_wing"]["roll"]["30D"] * front_leg_swap.weighting,
                "2M_carry": cr_bps_running_dict["front_wing"]["carry"]["60D"] * front_leg_swap.weighting,
                "2M_roll": cr_bps_running_dict["front_wing"]["roll"]["60D"] * front_leg_swap.weighting,
                "3M_carry": cr_bps_running_dict["front_wing"]["carry"]["90D"] * front_leg_swap.weighting,
                "3M_roll": cr_bps_running_dict["front_wing"]["roll"]["90D"] * front_leg_swap.weighting,
                "6M_carry": cr_bps_running_dict["front_wing"]["carry"]["180D"] * front_leg_swap.weighting,
                "6M_roll": cr_bps_running_dict["front_wing"]["roll"]["180D"] * front_leg_swap.weighting,
            },
            "beta_weighted_bps_running": (
                {
                    "1M_carry": cr_bps_running_dict["front_wing"]["carry"]["30D"] * front_wing_beta_adjustment_wrt_belly * is_long_or_short,
                    "1M_roll": cr_bps_running_dict["front_wing"]["roll"]["30D"] * front_wing_beta_adjustment_wrt_belly * is_long_or_short,
                    "2M_carry": cr_bps_running_dict["front_wing"]["carry"]["60D"] * front_wing_beta_adjustment_wrt_belly * is_long_or_short,
                    "2M_roll": cr_bps_running_dict["front_wing"]["roll"]["60D"] * front_wing_beta_adjustment_wrt_belly * is_long_or_short,
                    "3M_carry": cr_bps_running_dict["front_wing"]["carry"]["90D"] * front_wing_beta_adjustment_wrt_belly * is_long_or_short,
                    "3M_roll": cr_bps_running_dict["front_wing"]["roll"]["90D"] * front_wing_beta_adjustment_wrt_belly * is_long_or_short,
                    "6M_carry": cr_bps_running_dict["front_wing"]["carry"]["180D"] * front_wing_beta_adjustment_wrt_belly * is_long_or_short,
                    "6M_roll": cr_bps_running_dict["front_wing"]["roll"]["180D"] * front_wing_beta_adjustment_wrt_belly * is_long_or_short,
                }
                if is_beta_weighted
                else None
            ),
        },
        "belly": {
            "current_yield": belly_swap.original_fixed_rate,
            "current_weighted_yield": belly_swap.original_fixed_rate * belly_swap.weighting,
            "hr": hedge_ratios["belly_hr"],
            "bpv_hr": bpv_hedge_ratios["belly_hr"],
            "notional": belly_notional,
            "pvbp_per_mm": book["belly"].fixedLegBPS() * 1_000_000,
            "pvbp_leg": book["belly"].fixedLegBPS() * belly_notional,
            "bpv_neutral_bps_running": {
                "1M_carry": cr_bps_running_dict["belly"]["carry"]["30D"] * belly_swap.weighting,
                "1M_roll": cr_bps_running_dict["belly"]["roll"]["30D"] * belly_swap.weighting,
                "2M_carry": cr_bps_running_dict["belly"]["carry"]["60D"] * belly_swap.weighting,
                "2M_roll": cr_bps_running_dict["belly"]["roll"]["60D"] * belly_swap.weighting,
                "3M_carry": cr_bps_running_dict["belly"]["carry"]["90D"] * belly_swap.weighting,
                "3M_roll": cr_bps_running_dict["belly"]["roll"]["90D"] * belly_swap.weighting,
                "6M_carry": cr_bps_running_dict["belly"]["carry"]["180D"] * belly_swap.weighting,
                "6M_roll": cr_bps_running_dict["belly"]["roll"]["180D"] * belly_swap.weighting,
            },
        },
        "back_wing": {
            "current_yield": back_leg_swap.original_fixed_rate,
            "current_weighted_yield": back_leg_swap.original_fixed_rate * back_leg_swap.weighting,
            "hr": hedge_ratios["back_wing_hr"],
            "bpv_hr": bpv_hedge_ratios["back_wing_hr"],
            "notional": back_wing_notional,
            "pvbp_per_mm": book["back_wing"].fixedLegBPS() * 1_000_000,
            "pvbp_leg": book["back_wing"].fixedLegBPS() * back_wing_notional,
            "bpv_neutral_bps_running": {
                "1M_carry": cr_bps_running_dict["back_wing"]["carry"]["30D"] * back_leg_swap.weighting,
                "1M_roll": cr_bps_running_dict["back_wing"]["roll"]["30D"] * back_leg_swap.weighting,
                "2M_carry": cr_bps_running_dict["back_wing"]["carry"]["60D"] * back_leg_swap.weighting,
                "2M_roll": cr_bps_running_dict["back_wing"]["roll"]["60D"] * back_leg_swap.weighting,
                "3M_carry": cr_bps_running_dict["back_wing"]["carry"]["90D"] * back_leg_swap.weighting,
                "3M_roll": cr_bps_running_dict["back_wing"]["roll"]["90D"] * back_leg_swap.weighting,
                "6M_carry": cr_bps_running_dict["back_wing"]["carry"]["180D"] * back_leg_swap.weighting,
                "6M_roll": cr_bps_running_dict["back_wing"]["roll"]["180D"] * back_leg_swap.weighting,
            },
            "beta_weighted_bps_running": (
                {
                    "1M_carry": cr_bps_running_dict["back_wing"]["carry"]["30D"] * back_wing_beta_adjustment_wrt_belly * is_long_or_short,
                    "1M_roll": cr_bps_running_dict["back_wing"]["roll"]["30D"] * back_wing_beta_adjustment_wrt_belly * is_long_or_short,
                    "2M_carry": cr_bps_running_dict["back_wing"]["carry"]["60D"] * back_wing_beta_adjustment_wrt_belly * is_long_or_short,
                    "2M_roll": cr_bps_running_dict["back_wing"]["roll"]["60D"] * back_wing_beta_adjustment_wrt_belly * is_long_or_short,
                    "3M_carry": cr_bps_running_dict["back_wing"]["carry"]["90D"] * back_wing_beta_adjustment_wrt_belly * is_long_or_short,
                    "3M_roll": cr_bps_running_dict["back_wing"]["roll"]["90D"] * back_wing_beta_adjustment_wrt_belly * is_long_or_short,
                    "6M_carry": cr_bps_running_dict["back_wing"]["carry"]["180D"] * back_wing_beta_adjustment_wrt_belly * is_long_or_short,
                    "6M_roll": cr_bps_running_dict["back_wing"]["roll"]["180D"] * back_wing_beta_adjustment_wrt_belly * is_long_or_short,
                }
                if is_beta_weighted
                else None
            ),
        },
        "ql_ois_obj": book,
    }
