from typing import Dict, List, Optional, Literal
from joblib import Parallel, delayed
from datetime import datetime

import pandas as pd
import numpy as np
import QuantLib as ql
import tqdm
from copy import copy

from CurvyCUSIPs.HedgeHog.swaps import SwapLeg, book_metrics
from CurvyCUSIPs.utils.dtcc_swaps_utils import datetime_to_ql_date, ql_date_to_datetime


def build_ql_vol_cube_handle(vol_cube_dict: Dict[Literal[-200, -100, -50, -25, -10, 0, 10, 25, 50, 100, 200], pd.DataFrame], ql_sofr: ql.Sofr):
    atm_vol_surface = vol_cube_dict[0]
    atm_vol_surface = atm_vol_surface.drop("9M")
    expiries = [ql.Period(e) for e in atm_vol_surface.index]
    tails = [ql.Period(t) for t in atm_vol_surface.columns]
    atm_swaption_vol_matrix = ql.SwaptionVolatilityMatrix(
        ql.UnitedStates(ql.UnitedStates.GovernmentBond),
        ql.ModifiedFollowing,
        expiries,
        tails,
        ql.Matrix([[vol / 10_000 for vol in row] for row in atm_vol_surface.values]),
        ql.Actual360(),
        False,
        ql.Normal,
    )
    vol_spreads = []
    strike_spreads = [float(k) / 10000 for k in vol_cube_dict.keys()]
    strike_offsets = sorted(vol_cube_dict.keys(), key=lambda x: int(x))
    for option_tenor in atm_vol_surface.index:
        for swap_tenor in atm_vol_surface.columns:
            vol_spread_row = [
                ql.QuoteHandle(
                    ql.SimpleQuote((vol_cube_dict[strike].loc[option_tenor, swap_tenor] - atm_vol_surface.loc[option_tenor, swap_tenor]) / 10_000)
                )
                for strike in strike_offsets
            ]
            vol_spreads.append(vol_spread_row)
    return ql.SwaptionVolatilityStructureHandle(
        ql.InterpolatedSwaptionVolatilityCube(
            ql.SwaptionVolatilityStructureHandle(atm_swaption_vol_matrix),
            expiries,
            tails,
            strike_spreads,
            vol_spreads,
            ql.OvernightIndexedSwapIndex("SOFR-OIS", ql.Period("1D"), 2, ql.USDCurrency(), ql_sofr),
            ql.OvernightIndexedSwapIndex("SOFR-OIS", ql.Period("1D"), 2, ql.USDCurrency(), ql_sofr),
            False,
        )
    )


def build_ql_swaption_with_vol_shift(
    ql_curve: ql.DiscountCurve | ql.ZeroCurve | ql.ForwardCurve,
    swaption: ql.Swaption,
    base_vol_handle: ql.RelinkableSwaptionVolatilityStructureHandle,
    shift: float,
) -> ql.Swaption:
    swaption_copy = ql.Swaption(swaption.underlying(), swaption.exercise())
    original_vol = base_vol_handle.currentLink()
    spread_quote = ql.SimpleQuote(shift)
    spreaded_vol = ql.SpreadedSwaptionVolatility(ql.SwaptionVolatilityStructureHandle(original_vol), ql.QuoteHandle(spread_quote))
    bumped_vol_handle = ql.RelinkableSwaptionVolatilityStructureHandle(spreaded_vol)
    bumped_engine = ql.BachelierSwaptionEngine(ql.RelinkableYieldTermStructureHandle(ql_curve), bumped_vol_handle)
    swaption_copy.setPricingEngine(bumped_engine)
    return swaption_copy


def calc_swaption_greeks(
    ql_date: ql.Date,
    original_swaption: ql.Swaption,
    swaption_strike_bumped_up: ql.Swaption,
    swaption_strike_bumped_down: ql.Swaption,
    swaption_vol_bumped_up: ql.Swaption,
    swaption_vol_bumped_down: ql.Swaption,
    swaption_strike_up_vol_up: ql.Swaption,
    swaption_strike_up_vol_down: ql.Swaption,
    swaption_strike_down_vol_up: ql.Swaption,
    swaption_strike_down_vol_down: ql.Swaption,
    dStrike: float,
    dVol: float,
    is_receiver: bool,
    is_long: bool,
):
    ql.Settings.instance().evaluationDate = ql_date

    dv01 = original_swaption.delta() / 10_000
    gamma = ((swaption_strike_bumped_up.delta() / 10_000) - (swaption_strike_bumped_down.delta() / 10_000)) / (2 * dStrike) / 10_000
    vega = original_swaption.vega() / 10_000
    volga = ((swaption_vol_bumped_up.vega() / 10_000) - (swaption_vol_bumped_down.vega() / 10_000)) / (2 * dVol)

    vanna = (
        (
            swaption_strike_up_vol_up.NPV()
            - swaption_strike_up_vol_down.NPV()
            - swaption_strike_down_vol_up.NPV()
            + swaption_strike_down_vol_down.NPV()
        )
        / (4.0 * dStrike * dVol)
    ) / 10_000

    price_today = original_swaption.NPV()
    ql.Settings.instance().evaluationDate = ql_date + 1
    theta = price_today - original_swaption.NPV()
    charm = dv01 - (original_swaption.delta() / 10_000)
    veta = vega - (original_swaption.vega() / 10_000)
    ql.Settings.instance().evaluationDate = ql_date

    return {
        "dv01": (np.abs(dv01) if is_receiver else np.abs(dv01) * -1 if is_long else (np.abs(dv01) if is_receiver else np.abs(dv01) * -1) * -1),
        "gamma_01": np.abs(gamma) if is_long else np.abs(gamma) * -1,
        "vega_01": np.abs(vega) if is_long else np.abs(vega) * -1,
        "volga_01": np.abs(volga) if is_long else np.abs(volga) * -1,
        "vanna_01": np.float64(vanna),
        "theta_1d": -1 * np.abs(theta) if is_long else np.abs(theta) * -1,
        "charm_1d": -1 * np.abs(charm) if is_long else np.abs(charm) * -1,
        "veta_1d": -1 * np.abs(veta) if is_long else np.abs(veta) * -1,
    }, price_today


def price_swaption_locally(
    swap_leg: SwapLeg,
    discount_curve_dict: Dict[datetime, float],
    vol_cube_dict: Dict[Literal[-200, -100, -50, -25, -10, 0, 10, 25, 50, 100, 200], pd.DataFrame],
    sofr_fixings_dates: List[datetime],
    sofr_fixings: List[float],
    bump_size_bps: Optional[float] = 1,
    vol_bump_normal: float = 1,
):
    ql_date = datetime_to_ql_date(swap_leg.trade_date)
    ql.Settings.instance().evaluationDate = ql_date
    cal = ql.UnitedStates(ql.UnitedStates.GovernmentBond)

    ql_yts = ql.RelinkableYieldTermStructureHandle()
    ql_sofr = ql.Sofr(ql_yts)
    ql_sofr.addFixings(
        [datetime_to_ql_date(ed) for ed in sofr_fixings_dates],
        [fixing / 100 for fixing in sofr_fixings],
        forceOverwrite=True,
    )

    # underlying ois discount curve
    ql_discount_curve = ql.DiscountCurve(
        [datetime_to_ql_date(d) for d in discount_curve_dict.keys()], [d for d in discount_curve_dict.values()], ql.Actual360(), cal
    )
    ql_yts.linkTo(ql_discount_curve)
    underlying_engine = ql.DiscountingSwapEngine(ql_yts)

    # ois swaption cube
    ql_vol_cube_handle = build_ql_vol_cube_handle(vol_cube_dict=vol_cube_dict, ql_sofr=ql_sofr)
    swaption_engine = ql.BachelierSwaptionEngine(ql.RelinkableYieldTermStructureHandle(ql_discount_curve), ql_vol_cube_handle)

    # underlying ois ql objects
    if "Fwd" in swap_leg.original_tenor:
        fwd_tenor_str, swap_tenor_str = swap_leg.original_tenor.split(" Fwd ")
    else:
        swap_tenor_str = swap_leg.original_tenor
        fwd_tenor_str = "0D"

    effective_date = cal.advance(ql_date, ql.Period("2D"))
    effective_date = cal.advance(effective_date, ql.Period(fwd_tenor_str))

    underlying_swap: ql.OvernightIndexedSwap = ql.MakeOIS(
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
    underlying_swap_strike_bumped_up: ql.OvernightIndexedSwap = ql.MakeOIS(
        ql.Period(swap_tenor_str),
        ql_sofr,
        swap_leg.original_fixed_rate + (bump_size_bps / 10_000),
        ql.Period(fwd_tenor_str),
        swapType=ql.OvernightIndexedSwap.Receiver if swap_leg.type == "receiver" else ql.OvernightIndexedSwap.Payer,
        effectiveDate=effective_date,
        terminationDate=cal.advance(effective_date, ql.Period(swap_tenor_str)),
        paymentAdjustmentConvention=ql.ModifiedFollowing,
        paymentLag=2,
        fixedLegDayCount=ql.Actual360(),
        nominal=swap_leg.notional if swap_leg.notional else -1 if swap_leg.weighting < 0 else 1,
    )
    underlying_swap_strike_bumped_down: ql.OvernightIndexedSwap = ql.MakeOIS(
        ql.Period(swap_tenor_str),
        ql_sofr,
        swap_leg.original_fixed_rate - (bump_size_bps / 10_000),
        ql.Period(fwd_tenor_str),
        swapType=ql.OvernightIndexedSwap.Receiver if swap_leg.type == "receiver" else ql.OvernightIndexedSwap.Payer,
        effectiveDate=effective_date,
        terminationDate=cal.advance(effective_date, ql.Period(swap_tenor_str)),
        paymentAdjustmentConvention=ql.ModifiedFollowing,
        paymentLag=2,
        fixedLegDayCount=ql.Actual360(),
        nominal=swap_leg.notional if swap_leg.notional else -1 if swap_leg.weighting < 0 else 1,
    )

    underlying_swap.setPricingEngine(underlying_engine)
    underlying_swap_strike_bumped_up.setPricingEngine(underlying_engine)
    underlying_swap_strike_bumped_down.setPricingEngine(underlying_engine)

    # swaption ql objects
    fwd_start_ql_date = cal.advance(ql_date, ql.Period(fwd_tenor_str))
    fwd_end_ql_date = cal.advance(fwd_start_ql_date, ql.Period(swap_tenor_str))

    dStrike = bump_size_bps / 10_000
    dVol = vol_bump_normal / 10_000

    original_swaption = ql.Swaption(underlying_swap, ql.EuropeanExercise(fwd_start_ql_date))
    swaption_strike_bumped_up = ql.Swaption(underlying_swap_strike_bumped_up, ql.EuropeanExercise(fwd_start_ql_date))
    swaption_strike_bumped_down = ql.Swaption(underlying_swap_strike_bumped_down, ql.EuropeanExercise(fwd_start_ql_date))

    original_swaption.setPricingEngine(swaption_engine)
    swaption_strike_bumped_up.setPricingEngine(swaption_engine)
    swaption_strike_bumped_down.setPricingEngine(swaption_engine)

    swaption_vol_bumped_up = build_ql_swaption_with_vol_shift(
        ql_curve=ql_discount_curve, swaption=original_swaption, base_vol_handle=ql_vol_cube_handle, shift=+dVol
    )
    swaption_vol_bumped_down = build_ql_swaption_with_vol_shift(
        ql_curve=ql_discount_curve, swaption=original_swaption, base_vol_handle=ql_vol_cube_handle, shift=-dVol
    )
    swaption_strike_up_vol_up = build_ql_swaption_with_vol_shift(
        ql_curve=ql_discount_curve, swaption=swaption_strike_bumped_up, base_vol_handle=ql_vol_cube_handle, shift=+dVol
    )
    swaption_strike_up_vol_down = build_ql_swaption_with_vol_shift(
        ql_curve=ql_discount_curve, swaption=swaption_strike_bumped_up, base_vol_handle=ql_vol_cube_handle, shift=-dVol
    )
    swaption_strike_down_vol_up = build_ql_swaption_with_vol_shift(
        ql_curve=ql_discount_curve, swaption=swaption_strike_bumped_down, base_vol_handle=ql_vol_cube_handle, shift=+dVol
    )
    swaption_strike_down_vol_down = build_ql_swaption_with_vol_shift(
        ql_curve=ql_discount_curve, swaption=swaption_strike_bumped_down, base_vol_handle=ql_vol_cube_handle, shift=-dVol
    )

    greeks, price_today = calc_swaption_greeks(
        ql_date=ql_date,
        original_swaption=original_swaption,
        swaption_strike_bumped_up=swaption_strike_bumped_up,
        swaption_strike_bumped_down=swaption_strike_bumped_down,
        swaption_vol_bumped_up=swaption_vol_bumped_up,
        swaption_vol_bumped_down=swaption_vol_bumped_down,
        swaption_strike_up_vol_up=swaption_strike_up_vol_up,
        swaption_strike_up_vol_down=swaption_strike_up_vol_down,
        swaption_strike_down_vol_up=swaption_strike_down_vol_up,
        swaption_strike_down_vol_down=swaption_strike_down_vol_down,
        dStrike=dStrike,
        dVol=dVol,
        is_receiver=swap_leg.type == "receiver",
        is_long=swap_leg.weighting > 0,
    )

    normal_vol = (
        original_swaption.impliedVolatility(
            price=price_today,
            discountCurve=ql.YieldTermStructureHandle(ql_discount_curve),
            guess=100 / 10_000,
            accuracy=1e-3,
            maxEvaluations=500,
            type=ql.Normal,
        )
        * 10_000
    )
    atmf = np.float64(
        ql.ImpliedTermStructure(ql.YieldTermStructureHandle(ql_discount_curve), fwd_start_ql_date)
        .forwardRate(fwd_start_ql_date, fwd_end_ql_date, ql.Actual360(), ql.Compounded, ql.Annual, True)
        .rate()
    )
    strike = np.float64(underlying_swap.fixedRate())

    return {
        "atm_strike": atmf,
        "strike": strike,
        "strike_offset_bps": (strike - atmf) * 10_000,
        "npv": np.float64(price_today),
        "normal_vol": np.float64(normal_vol),
        "bpvol": normal_vol / np.sqrt(252),
        "spot_prem_bps": (price_today / underlying_swap.nominal()) * 10_000,
        "fwd_prem_bps": (original_swaption.forwardPrice() / underlying_swap.nominal()) * 10_000,
        "ql": {
            "swaption": {
                "european_exercise_date": ql_date_to_datetime(fwd_start_ql_date),
            },
            "underlying": {
                "swap_tenor": swap_tenor_str,
                "original_fixed_rate": swap_leg.original_fixed_rate,
                "fwd_tenor": fwd_tenor_str,
                "swap_type": swap_leg.type,
                "effective_date": ql_date_to_datetime(effective_date),
                "termination_date": ql_date_to_datetime(cal.advance(effective_date, ql.Period(swap_tenor_str))),
                "nominal": swap_leg.notional if swap_leg.notional else -1 if swap_leg.weighting < 0 else 1,
            },
        },
        "greeks": greeks,
    }


def swaption_book_metrics_parallel(
    underlyings: List[SwapLeg],
    ql_curve: ql.DiscountCurve | ql.ZeroCurve | ql.ForwardCurve,
    sofr_fixings_dates: List[datetime],
    sofr_fixings: List[float],
    vol_cube_dict: Dict[Literal[-200, -100, -50, -25, -10, 0, 10, 25, 50, 100, 200], pd.DataFrame],
    bump_size_bps: Optional[float] = 1,
    vol_bump_normal: Optional[float] = 1,
    n_jobs: Optional[int] = 1,
):

    discount_curve_dict = dict(zip([ql_date_to_datetime(ql_date) for ql_date in ql_curve.dates()], [d for d in ql_curve.discounts()]))
    results = Parallel(n_jobs=n_jobs)(
        delayed(price_swaption_locally)(
            swap_leg=leg,
            discount_curve_dict=discount_curve_dict,
            vol_cube_dict=vol_cube_dict,
            sofr_fixings_dates=sofr_fixings_dates,
            sofr_fixings=sofr_fixings,
            bump_size_bps=bump_size_bps,
            vol_bump_normal=vol_bump_normal,
        )
        for leg in tqdm.tqdm(underlyings, desc="PRICING SWAPTIONS...")
    )

    ql_date = datetime_to_ql_date(underlyings[0].trade_date)
    ql.Settings.instance().evaluationDate = ql_date

    ql_yts = ql.RelinkableYieldTermStructureHandle()
    ql_sofr = ql.Sofr(ql_yts)
    ql_sofr.addFixings(
        [datetime_to_ql_date(ed) for ed in sofr_fixings_dates],
        [fixing / 100 for fixing in sofr_fixings],
        forceOverwrite=True,
    )

    ql_yts.linkTo(ql_curve)
    underlying_engine = ql.DiscountingSwapEngine(ql_yts)

    ql_vol_cube_handle = build_ql_vol_cube_handle(vol_cube_dict=vol_cube_dict, ql_sofr=ql_sofr)
    swaption_engine = ql.BachelierSwaptionEngine(ql.RelinkableYieldTermStructureHandle(ql_curve), ql_vol_cube_handle)

    pricer_results = {}
    for leg, result in tqdm.tqdm(zip(underlyings, results), desc="REBUILDING QL OBJECTS..."):
        underlying_swap_params = result["ql"]["underlying"]

        underlying_swap: ql.OvernightIndexedSwap = ql.MakeOIS(
            ql.Period(underlying_swap_params["swap_tenor"]),
            ql_sofr,
            underlying_swap_params["original_fixed_rate"],
            ql.Period(underlying_swap_params["fwd_tenor"]),
            swapType=ql.OvernightIndexedSwap.Receiver if underlying_swap_params["swap_type"] == "receiver" else ql.OvernightIndexedSwap.Payer,
            effectiveDate=datetime_to_ql_date(underlying_swap_params["effective_date"]),
            terminationDate=datetime_to_ql_date(underlying_swap_params["termination_date"]),
            paymentAdjustmentConvention=ql.ModifiedFollowing,
            paymentLag=2,
            fixedLegDayCount=ql.Actual360(),
            nominal=underlying_swap_params["nominal"],
        )
        underlying_swap.setPricingEngine(underlying_engine)

        original_swaption = ql.Swaption(underlying_swap, ql.EuropeanExercise(datetime_to_ql_date(result["ql"]["swaption"]["european_exercise_date"])))
        original_swaption.setPricingEngine(swaption_engine)

        del result["ql"]
        result["ql"] = {
            "swaption": original_swaption,
            "underlying": underlying_swap,
        }
        pricer_results[leg.key] = result

    return pricer_results


# TODO PARALLELIZATION
def swaption_book_metrics(
    underlyings: List[SwapLeg],
    ql_curve: ql.DiscountCurve | ql.ZeroCurve | ql.ForwardCurve,
    ql_yts: ql.RelinkableYieldTermStructureHandle,
    ql_sofr: ql.Sofr,
    swaption_vol_handle: ql.RelinkableSwaptionVolatilityStructureHandle,
    bump_size_bps: Optional[float] = 1,
    vol_bump_normal: float = 1,
):
    date = None
    includes_strike_bumped_swaptions: List[SwapLeg] = []
    long_or_short = {}
    for swap_leg in underlyings:
        if swap_leg.type == "straddle":
            receiver_swap_leg = SwapLeg(**swap_leg.__dict__)
            receiver_swap_leg.type = "receiver"
            receiver_swap_leg.key = f"{receiver_swap_leg.key}_rec"

            payer_swap_leg = SwapLeg(**swap_leg.__dict__)
            payer_swap_leg.type = "payer"
            payer_swap_leg.key = f"{payer_swap_leg.key}_pay"

            includes_strike_bumped_swaptions.append(receiver_swap_leg)
            includes_strike_bumped_swaptions.append(
                SwapLeg(
                    trade_date=receiver_swap_leg.trade_date,
                    original_tenor=receiver_swap_leg.original_tenor,
                    original_fixed_rate=receiver_swap_leg.original_fixed_rate + (bump_size_bps / 10_000),
                    weighting=receiver_swap_leg.weighting,
                    key=f"{receiver_swap_leg.key}_bumped_up",
                    notional=receiver_swap_leg.notional,
                    type=receiver_swap_leg.type,
                )
            )
            includes_strike_bumped_swaptions.append(
                SwapLeg(
                    trade_date=receiver_swap_leg.trade_date,
                    original_tenor=receiver_swap_leg.original_tenor,
                    original_fixed_rate=receiver_swap_leg.original_fixed_rate - (bump_size_bps / 10_000),
                    weighting=receiver_swap_leg.weighting,
                    key=f"{receiver_swap_leg.key}_bumped_down",
                    notional=receiver_swap_leg.notional,
                    type=receiver_swap_leg.type,
                )
            )

            includes_strike_bumped_swaptions.append(payer_swap_leg)
            includes_strike_bumped_swaptions.append(
                SwapLeg(
                    trade_date=payer_swap_leg.trade_date,
                    original_tenor=payer_swap_leg.original_tenor,
                    original_fixed_rate=payer_swap_leg.original_fixed_rate + (bump_size_bps / 10_000),
                    weighting=payer_swap_leg.weighting,
                    key=f"{payer_swap_leg.key}_bumped_up",
                    notional=payer_swap_leg.notional,
                    type=payer_swap_leg.type,
                )
            )
            includes_strike_bumped_swaptions.append(
                SwapLeg(
                    trade_date=payer_swap_leg.trade_date,
                    original_tenor=payer_swap_leg.original_tenor,
                    original_fixed_rate=payer_swap_leg.original_fixed_rate - (bump_size_bps / 10_000),
                    weighting=payer_swap_leg.weighting,
                    key=f"{payer_swap_leg.key}_bumped_down",
                    notional=payer_swap_leg.notional,
                    type=payer_swap_leg.type,
                )
            )

            long_or_short[receiver_swap_leg.key] = 1 if receiver_swap_leg.weighting > 0 else -1
            long_or_short[payer_swap_leg.key] = 1 if payer_swap_leg.weighting > 0 else -1

        else:
            includes_strike_bumped_swaptions.append(swap_leg)
            includes_strike_bumped_swaptions.append(
                SwapLeg(
                    trade_date=swap_leg.trade_date,
                    original_tenor=swap_leg.original_tenor,
                    original_fixed_rate=swap_leg.original_fixed_rate + (bump_size_bps / 10_000),
                    weighting=swap_leg.weighting,
                    key=f"{swap_leg.key}_bumped_up",
                    notional=swap_leg.notional,
                    type=swap_leg.type,
                )
            )
            includes_strike_bumped_swaptions.append(
                SwapLeg(
                    trade_date=swap_leg.trade_date,
                    original_tenor=swap_leg.original_tenor,
                    original_fixed_rate=swap_leg.original_fixed_rate - (bump_size_bps / 10_000),
                    weighting=swap_leg.weighting,
                    key=f"{swap_leg.key}_bumped_down",
                    notional=swap_leg.notional,
                    type=swap_leg.type,
                )
            )

            long_or_short[swap_leg.original_tenor] = 1 if swap_leg.weighting > 0 else -1

        date = swap_leg.trade_date

    ql_date = datetime_to_ql_date(date)
    underlying_book_metrics = book_metrics(swap_portfolio=includes_strike_bumped_swaptions, ql_curve=ql_curve, ql_yts=ql_yts, ql_sofr=ql_sofr)
    underlying_ql_book: Dict[str, ql.OvernightIndexedSwap] = underlying_book_metrics["book"]

    ql.Settings.instance().evaluationDate = ql_date
    engine = ql.BachelierSwaptionEngine(ql.RelinkableYieldTermStructureHandle(ql_curve), swaption_vol_handle)

    dStrike = bump_size_bps / 10_000
    dVol = vol_bump_normal / 10_000

    swaption_book_metrics = {}
    for tenor in tqdm.tqdm(underlying_ql_book.keys(), desc="PRICING SWAPTIONS..."):
        if "bumped" in tenor:
            continue

        expiry_tenor_str, tail_tenor_str = tenor.split(" Fwd ")
        if "_rec" in tail_tenor_str or "_pay" in tail_tenor_str:
            tail_tenor_str = tail_tenor_str.split("_")[0]

        fwd_start_ql_date = ql.UnitedStates(ql.UnitedStates.GovernmentBond).advance(ql_date, ql.Period(expiry_tenor_str))
        fwd_end_ql_date = ql.UnitedStates(ql.UnitedStates.GovernmentBond).advance(fwd_start_ql_date, ql.Period(tail_tenor_str))

        swaption = ql.Swaption(underlying_ql_book[tenor], ql.EuropeanExercise(fwd_start_ql_date))
        swaption.setPricingEngine(engine)

        swaption_strike_bumped_up = ql.Swaption(underlying_ql_book[f"{tenor}_bumped_up"], ql.EuropeanExercise(fwd_start_ql_date))
        swaption_strike_bumped_up.setPricingEngine(engine)

        swaption_strike_bumped_down = ql.Swaption(underlying_ql_book[f"{tenor}_bumped_down"], ql.EuropeanExercise(fwd_start_ql_date))
        swaption_strike_bumped_down.setPricingEngine(engine)

        ql.Settings.instance().evaluationDate = ql_date

        # basics: delta, gamma, vega, volga
        dv01 = swaption.delta() / 10_000
        gamma = (
            ((swaption_strike_bumped_up.delta() / 10_000) - (swaption_strike_bumped_down.delta() / 10_000)) / (2 * (bump_size_bps / 10_000)) / 10_000
        )
        vega = swaption.vega() / 10_000

        swaption_vol_bumped_up = build_ql_swaption_with_vol_shift(
            ql_curve=ql_curve, swaption=swaption, base_vol_handle=swaption_vol_handle, shift=+(vol_bump_normal / 10000)
        )
        swaption_vol_bumped_down = build_ql_swaption_with_vol_shift(
            ql_curve=ql_curve, swaption=swaption, base_vol_handle=swaption_vol_handle, shift=-(vol_bump_normal / 10000)
        )
        volga = ((swaption_vol_bumped_up.vega() / 10_000) - (swaption_vol_bumped_down.vega() / 10_000)) / (2 * (vol_bump_normal / 10_000))

        # vanna (DvegaDspot or DdeltaDvol; cross-partial derivative of the option price with respect to the underlying fwd and vol)
        swaption_strike_up_vol_up = build_ql_swaption_with_vol_shift(
            ql_curve=ql_curve, swaption=swaption_strike_bumped_up, base_vol_handle=swaption_vol_handle, shift=+dVol
        )
        swaption_strike_up_vol_down = build_ql_swaption_with_vol_shift(
            ql_curve=ql_curve, swaption=swaption_strike_bumped_up, base_vol_handle=swaption_vol_handle, shift=-dVol
        )
        swaption_strike_down_vol_up = build_ql_swaption_with_vol_shift(
            ql_curve=ql_curve, swaption=swaption_strike_bumped_down, base_vol_handle=swaption_vol_handle, shift=+dVol
        )
        swaption_strike_down_vol_down = build_ql_swaption_with_vol_shift(
            ql_curve=ql_curve, swaption=swaption_strike_bumped_down, base_vol_handle=swaption_vol_handle, shift=-dVol
        )
        vanna = (
            (
                swaption_strike_up_vol_up.NPV()
                - swaption_strike_up_vol_down.NPV()
                - swaption_strike_down_vol_up.NPV()
                + swaption_strike_down_vol_down.NPV()
            )
            / (4.0 * dStrike * dVol)
        ) / 10_000

        # time decay/bleed greeks
        price_today = swaption.NPV()
        ql.Settings.instance().evaluationDate = ql_date + 1
        theta = price_today - swaption.NPV()
        charm = dv01 - (swaption.delta() / 10_000)
        veta = vega - (swaption.vega() / 10_000)
        ql.Settings.instance().evaluationDate = ql_date

        is_receiver = underlying_ql_book[tenor].fixedLegBPS() > 0

        normal_vol = (
            swaption.impliedVolatility(
                price=price_today,
                discountCurve=ql.YieldTermStructureHandle(ql_curve),
                guess=100 / 10_000,
                accuracy=1e-3,
                maxEvaluations=500,
                type=ql.Normal,
            )
            * 10_000
        )
        atmf = np.float64(
            ql.ImpliedTermStructure(ql.YieldTermStructureHandle(ql_curve), fwd_start_ql_date)
            .forwardRate(fwd_start_ql_date, fwd_end_ql_date, ql.Actual360(), ql.Compounded, ql.Annual, True)
            .rate()
        )
        strike = np.float64(underlying_ql_book[tenor].fixedRate())

        swaption_book_metrics[tenor] = {
            "atm_strike": atmf,
            "strike": strike,
            "strike_offset_bps": (strike - atmf) * 10_000,
            "npv": np.float64(price_today),
            "normal_vol": np.float64(normal_vol),
            "bpvol": normal_vol / np.sqrt(252),
            "spot_prem_bps": (price_today / underlying_ql_book[tenor].nominal()) * 10_000,
            "fwd_prem_bps": (swaption.forwardPrice() / underlying_ql_book[tenor].nominal()) * 10_000,
            "ql": {
                "swaption": swaption,
                "underlying": underlying_ql_book[tenor],
            },
            "greeks": {
                "dv01": (
                    np.abs(dv01)
                    if is_receiver
                    else np.abs(dv01) * -1 if long_or_short[tenor] == 1 else (np.abs(dv01) if is_receiver else np.abs(dv01) * -1) * -1
                ),
                "gamma_01": np.abs(gamma) if long_or_short[tenor] == 1 else np.abs(gamma) * -1,
                "vega_01": np.abs(vega) if long_or_short[tenor] == 1 else np.abs(vega) * -1,
                "volga_01": np.abs(volga) if long_or_short[tenor] == 1 else np.abs(volga) * -1,
                "vanna_01": np.float64(vanna),
                "theta_1d": -1 * np.abs(theta) if long_or_short[tenor] == 1 else np.abs(theta) * -1,
                "charm_1d": -1 * np.abs(charm) if long_or_short[tenor] == 1 else np.abs(charm) * -1,
                "veta_1d": -1 * np.abs(veta) if long_or_short[tenor] == 1 else np.abs(veta) * -1,
            },
        }

    return swaption_book_metrics
