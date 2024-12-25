import logging
import traceback
from datetime import datetime
from typing import Dict, List, Literal, Optional, Tuple

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import QuantLib as ql
import scipy.interpolate
import tqdm
import ujson as json
from IPython.display import display
from numpy.typing import ArrayLike, NDArray
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import BDay, CustomBusinessDay
from pysabr import Hagan2002LognormalSABR, Hagan2002NormalSABR
from termcolor import colored

from CurvyCUSIPs.CurveDataFetcher import CurveDataFetcher
from CurvyCUSIPs.CurveInterpolator import GeneralCurveInterpolator
from CurvyCUSIPs.S490Swaps import S490Swaps
from CurvyCUSIPs.utils.dtcc_swaps_utils import datetime_to_ql_date, tenor_to_years
from CurvyCUSIPs.utils.ShelveDBWrapper import ShelveDBWrapper


class S490SwaptionsMarker:
    s490_swaps: S490Swaps
    s490_vol_cube_markings_db: ShelveDBWrapper = None
    s490_atm_vol_grid_markings_db: ShelveDBWrapper = None
    _current_sabr_state: Dict[
        str,  # Swaption tenor as a string
        Dict[str, float],  # "atmf", "sabr_beta", "sabr_alpha", "sabr_rho", "sabr_nu" and atm strike offsets
    ] = {}

    _logger = logging.getLogger(__name__)
    _debug_verbose: bool = False
    _error_verbose: bool = False
    _info_verbose: bool = False
    _no_logs_plz: bool = False

    def __init__(
        self,
        s490_swaps: S490Swaps,
        s490_vol_cube_markings_db_path: str,
        s490_atm_vol_grird_markings_db_path: str,
        init_markings_db: Optional[bool] = False,
        debug_verbose: Optional[bool] = False,
        info_verbose: Optional[bool] = False,
        error_verbose: Optional[bool] = False,
    ):
        self.s490_swaps = s490_swaps
        self.s490_vol_cube_markings_db = self.setup_db(s490_vol_cube_markings_db_path, init_markings_db)
        self.s490_atm_vol_grid_markings_db = self.setup_db(s490_atm_vol_grird_markings_db_path, init_markings_db)

        self._debug_verbose = debug_verbose
        self._error_verbose = error_verbose
        self._info_verbose = info_verbose
        self._no_logs_plz = not debug_verbose and not error_verbose and not info_verbose

        self._setup_logger()

    def setup_db(self, path: str, should_create):
        db = ShelveDBWrapper(path, should_create)
        db.open()

        if len(db.keys()) == 0:
            print(
                colored(
                    f"Warning: {path} is empty",
                    "yellow",
                )
            )
        else:
            most_recent_db_dt = datetime.strptime(max(db.keys()), "%Y-%m-%d")
            self._logger.info(f"Most recent date in db: {most_recent_db_dt}")
            if ((datetime.today() - BDay(1)) - most_recent_db_dt).days >= 1:
                print(
                    colored(
                        f"{path} is behind --- cd into 'scripts' and run 'update_sofr_ois_db.py' to update --- most recent date in db: {most_recent_db_dt}",
                        "yellow",
                    )
                )

        return db

    def _setup_logger(self):
        if not self._logger.hasHandlers():
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
            self._logger.addHandler(handler)

        if self._debug_verbose:
            self._logger.setLevel(logging.DEBUG)
        elif self._info_verbose:
            self._logger.setLevel(logging.INFO)
        elif self._error_verbose:
            self._logger.setLevel(logging.ERROR)
        else:
            self._logger.setLevel(logging.WARNING)

        if self._debug_verbose or self._info_verbose or self._error_verbose:
            self._logger.setLevel(logging.DEBUG)

        if self._no_logs_plz:
            self._logger.disabled = True
            self._logger.propagate = False

    @staticmethod
    def scipy_interp_func(xx, yy, kind="linear", logspace=False):
        if logspace:
            log_y = np.log(yy)
            interp_func = scipy.interpolate.interp1d(xx, log_y, kind=kind, fill_value="extrapolate", bounds_error=False)

            def log_linear_interp(x_new):
                return np.exp(interp_func(x_new))

            return log_linear_interp

        return scipy.interpolate.interp1d(xx, yy, kind=kind, fill_value="extrapolate", bounds_error=False)
    
    # Hagan, P. S., Kumar, D., Lesniewski, A. S., & Woodward, D. E. (2002). Managing Smile Risk. Risk Magazine.
    @staticmethod
    def sabr_implied_vol(F: float, K: float, T: float, alpha: float, beta: float, rho: float, nu: float):
        r"""
        Calculate the SABR implied volatility using the Hagan et al. (2002) approximation.

        Parameters:
        F (float): Forward price
        K (float): Strike price
        T (float): Time to maturity
        alpha (float): Volatility parameter
        beta (float): Elasticity parameter
        rho (float): Correlation parameter
        nu (float): Volatility of volatility

        Returns:
        float: Implied Black-Scholes volatility

        ATM Case (F == K)
        $$
        \sigma_{\text{BS}} = \frac{\alpha}{F^{1-\beta}} \left(1 + \left[\frac{(1-\beta)^2}{24} \frac{\alpha^2}{F^{2(1-\beta)}} + \frac{\rho \beta \nu \alpha}{4 F^{1-\beta}} + \frac{(2-3\rho^2)\nu^2}{24}\right] T \right)
        $$

        Non-ATM Case (F != K)
        $$
        \sigma_{\text{BS}} = \frac{\alpha}{(F K)^{\frac{1-\beta}{2}}} \cdot \frac{z}{x(z)} \left(1 + \frac{(1-\beta)^2}{24} \log^2\left(\frac{F}{K}\right) + \frac{(1-\beta)^4}{1920} \log^4\left(\frac{F}{K}\right)\right) \left(1 + \left[\frac{(1-\beta)^2}{24} \frac{\alpha^2}{(F K)^{1-\beta}} + \frac{\rho \beta \nu \alpha}{4 (F K)^{\frac{1-\beta}{2}}} + \frac{(2-3\rho^2)\nu^2}{24}\right] T \right)
        $$
        where,
        $$
        z = \frac{\nu}{\alpha} (F K)^{\frac{1-\beta}{2}} \log\left(\frac{F}{K}\right)
        $$
        $$
        x(z) = \log\left(\frac{\sqrt{1 - 2 \rho z + z^2} + z - \rho}{1 - \rho}\right)
        $$

        Invalid Strike Edge Case:
        $$
        K \leq 0 \quad \Rightarrow \quad \text{Implied Volatility} = \text{NaN}
        $$
        """

        # TODO handle negative strikes
        # Return NaN for invalid (negative) strikes
        if K <= 0:
            return np.nan

        if F == K:
            FK_beta = F ** (1 - beta)
            vol = (alpha / FK_beta) * (
                1
                + (((1 - beta) ** 2 / 24) * (alpha**2) / (FK_beta**2) + (rho * beta * nu * alpha) / (4 * FK_beta) + ((2 - 3 * rho**2) * nu**2 / 24))
                * T
            )
        else:
            logFK = np.log(F / K)
            FK_beta = (F * K) ** ((1 - beta) / 2)
            z = (nu / alpha) * FK_beta * logFK
            x_z = np.log((np.sqrt(1 - 2 * rho * z + z**2) + z - rho) / (1 - rho))

            # division by zero or very small z
            if abs(z) < 1e-12 or x_z == 0.0:
                z_over_x_z = 1.0
            else:
                z_over_x_z = z / x_z

            vol = (
                (alpha / FK_beta)
                * z_over_x_z
                * (1 + ((1 - beta) ** 2 / 24) * logFK**2 + ((1 - beta) ** 4 / 1920) * logFK**4)
                * (
                    1
                    + (
                        ((1 - beta) ** 2 / 24) * (alpha**2) / (FK_beta**2)
                        + (rho * beta * nu * alpha) / (4 * FK_beta)
                        + ((2 - 3 * rho**2) * nu**2 / 24)
                    )
                    * T
                )
            )

        return vol

    def detect_and_merge_straddles(self, df: pd.DataFrame):
        group_columns = [
            "Execution Timestamp",
            "Option Tenor",
            "Underlying Tenor",
            "Strike Price",
            "Option Premium Amount",
            "Notional Amount",
            "Option Premium per Notional",
            "IV",
            "IV bp/day",
        ]
        grouped = df.groupby(group_columns)
        new_rows = []

        def is_straddle(group_df):
            if len(group_df) == 2:
                styles = group_df["Style"].unique()
                if set(styles) == set(["payer", "receiver"]):
                    return True
            return False

        for _, group_df in grouped:
            if is_straddle(group_df):
                base_row = group_df.iloc[0].copy()
                base_row["Style"] = "straddle"
                new_rows.append(base_row)
            else:
                for _, row in group_df.iterrows():
                    new_rows.append(row)

        straddle_df = pd.DataFrame(new_rows)
        straddle_df.reset_index(drop=True, inplace=True)
        return straddle_df

    def split_straddles(self, df: pd.DataFrame):
        straddle_rows = df[df["Style"] == "straddle"]
        non_straddle_rows = df[df["Style"] != "straddle"]
        new_rows = []
        quantity_columns = ["Option Premium Amount", "Notional Amount", "Option Premium per Notional", "IV", "IV bp/day"]
        for _, row in straddle_rows.iterrows():
            row_payer = row.copy()
            row_receiver = row.copy()
            for col in quantity_columns:
                if pd.notnull(row[col]):
                    row_payer[col] = row[col] / 2
                    row_receiver[col] = row[col] / 2

            row_payer["Style"] = "payer"
            row_payer["UPI FISN"] = "NA/O Call Epn OIS USD"
            row_receiver["Style"] = "receiver"
            row_receiver["UPI FISN"] = "NA/O P Epn OIS USD"

            new_rows.append(row_payer)
            new_rows.append(row_receiver)

        result_df = pd.concat([non_straddle_rows, pd.DataFrame(new_rows)], ignore_index=True)
        return result_df

    def format_swaption_premium_close(self, df: pd.DataFrame, normalize_premium=True):
        df = df.copy()
        df["Swaption Tenor"] = df["Option Tenor"] + df["Underlying Tenor"]
        df = df.sort_values(by=["Swaption Tenor", "Strike Price", "IV bp/day", "ATMF", "Event timestamp"])
        if normalize_premium:
            df["Premium"] = df["Option Premium Amount"] / df["Notional Amount"]
        else:
            df["Premium"] = df["Option Premium Amount"]

        ohlc_df = (
            df.groupby(["Swaption Tenor", "Strike Price"])
            .agg(
                Premium=("Premium", "last"),
                IV_Daily_BPs=("IV bp/day", "last"),
                ATMF=("ATMF", "last"),
            )
            .reset_index()
        )
        return ohlc_df

    def create_s490_swaption_time_and_sales(
        self,
        data_fetcher: CurveDataFetcher,
        start_date: datetime,
        end_date: datetime,
        model: Optional[Literal["lognormal", "normal"]] = "normal",
        calc_greeks: Optional[bool] = False,
    ) -> Tuple[Dict[datetime, pd.DataFrame], Dict[datetime, pd.DataFrame]]:
        if model != "normal" and model != "lognormal":
            raise ValueError(f"Bad Model Param: {model}")

        calendar = ql.UnitedStates(ql.UnitedStates.GovernmentBond)
        day_count = ql.Actual360()
        swaption_dict: Dict[datetime, pd.DataFrame] = data_fetcher.dtcc_sdr_fetcher.fetch_historical_swaption_time_and_sales(
            start_date=start_date,
            end_date=end_date,
            underlying_swap_types=["Fixed_Float_OIS", "Fixed_Float"],
            underlying_reference_floating_rates=["USD-SOFR-COMPOUND", "USD-SOFR-OIS Compound", "USD-SOFR"],
            underlying_ccy="USD",
            underlying_reference_floating_rate_term_value=1,
            underlying_reference_floating_rate_term_unit="DAYS",
            underlying_notional_schedule="Constant",
            underlying_delivery_types=["CASH", "PHYS"],
        )

        fwd_dict, ql_curves_dict = self.s490_swaps.s490_nyclose_fwd_curve_matrices(
            start_date=start_date,
            end_date=end_date,
            swaption_time_and_sales_dict_for_fwds=swaption_dict,
            ql_piecewise_method="logLinearDiscount",
            ql_compounding=ql.Compounded,
            ql_compounding_freq=ql.Daily,
            use_ql_implied_ts=True,
        )

        if len(fwd_dict.keys()) == 0:
            raise ValueError("Forward rates calc error")

        modifed_swaption_time_and_sales = {}
        ohlc_premium = {}
        for bdate in tqdm.tqdm(
            pd.date_range(start=start_date, end=end_date, freq=CustomBusinessDay(calendar=USFederalHolidayCalendar())), "Fitting Swaption Trades..."
        ):
            ql.Settings.instance().evaluationDate = datetime_to_ql_date(bdate.to_pydatetime())
            curr_ql_curve = ql_curves_dict[bdate.to_pydatetime()]
            curr_ql_curve.enableExtrapolation()
            curr_curve_handle = ql.YieldTermStructureHandle(curr_ql_curve)
            swaption_pricing_engine = ql.BlackSwaptionEngine(
                curr_curve_handle, ql.QuoteHandle(ql.SimpleQuote(0.0)), ql.ActualActual(ql.ActualActual.ISMA)
            )

            curr_swaption_time_and_sales_with_iv = []
            curr_swaption_time_and_sales = swaption_dict[bdate.to_pydatetime()].to_dict("records")
            err_count = 0
            for i, swaption_trade_dict in enumerate(curr_swaption_time_and_sales):
                if swaption_trade_dict["Fwd"] != "0D":
                    continue

                try:
                    atmf = (
                        fwd_dict[bdate.to_pydatetime()]
                        .loc[
                            fwd_dict[bdate.to_pydatetime()]["Tenor"] == swaption_trade_dict["Underlying Tenor"],
                            f"{swaption_trade_dict["Option Tenor"]} Fwd",
                        ]
                        .values[0]
                        / 100
                    )
                    swaption_trade_dict["ATMF"] = atmf
                    swaption_trade_dict["OTM"] = (swaption_trade_dict["Strike Price"] - swaption_trade_dict["ATMF"]) * 100 * 100
                    option_premium = swaption_trade_dict["Option Premium Amount"] / swaption_trade_dict["Notional Amount"]

                    curr_sofr_ois_swap = ql.MakeOIS(
                        ql.Period(swaption_trade_dict["Underlying Tenor"]),
                        ql.OvernightIndex("SOFR", 1, ql.USDCurrency(), calendar, day_count, curr_curve_handle),
                        swaption_trade_dict["Strike Price"],
                        ql.Period(swaption_trade_dict["Option Tenor"]),
                        paymentLag=2,
                        settlementDays=2,
                        calendar=calendar,
                        convention=ql.ModifiedFollowing,
                    )
                    curr_sofr_ois_swaption = ql.Swaption(
                        curr_sofr_ois_swap,
                        ql.EuropeanExercise(
                            # calendar.advance(
                            #     datetime_to_ql_date(swaption_trade_dict["Effective Date"]),
                            #     ql.Period(swaption_trade_dict["Option Tenor"]),
                            #     ql.ModifiedFollowing,
                            # )
                            datetime_to_ql_date(swaption_trade_dict["Effective Date"])
                            + ql.Period(swaption_trade_dict["Option Tenor"])
                        ),
                    )
                    curr_sofr_ois_swaption.setPricingEngine(swaption_pricing_engine)

                    if model == "normal":
                        implied_vol = curr_sofr_ois_swaption.impliedVolatility(
                            price=option_premium,
                            discountCurve=curr_curve_handle,
                            guess=0.5,
                            accuracy=1e-1,
                            maxEvaluations=750,
                            minVol=0,
                            maxVol=5.0,
                            type=ql.Normal,
                        )
                        swaption_trade_dict["IV"] = implied_vol * 100 * 100
                        swaption_trade_dict["IV bp/day"] = swaption_trade_dict["IV"] / np.sqrt(252)
                    elif model == "lognormal":
                        implied_vol = curr_sofr_ois_swaption.impliedVolatility(
                            price=option_premium,
                            discountCurve=curr_curve_handle,
                            guess=0.01,
                            accuracy=1e-1,
                            maxEvaluations=750,
                            minVol=0,
                            maxVol=5.0,
                        )
                        swaption_trade_dict["IV"] = implied_vol * 100
                        # TODO: need to implement Le Floc’h's since approx only works for near ATM
                        swaption_trade_dict["IV bp/day"] = (swaption_trade_dict["IV"] * atmf / np.sqrt(252)) * 100
                    else:
                        raise ValueError("Not reachable")

                    if calc_greeks:
                        pass

                    curr_swaption_time_and_sales_with_iv.append(swaption_trade_dict)

                except Exception as e:
                    self.s490_swaps._logger.error(
                        colored(
                            f"Error at {i} {swaption_trade_dict["Effective Date"]} - {swaption_trade_dict["Option Tenor"]}x{swaption_trade_dict["Underlying Tenor"]}: {str(e)}",
                            "red",
                        )
                    )
                    err_count += 1

            curr_swaption_time_and_sales_with_iv_df = pd.DataFrame(curr_swaption_time_and_sales_with_iv)
            # modifed_swaption_time_and_sales[bdate.to_pydatetime()] = curr_swaption_time_and_sales_with_iv_df
            # ohlc_premium[bdate.to_pydatetime()] = self.format_swaption_premium_close(curr_swaption_time_and_sales_with_iv_df)

            # avoid double counting straddles after UPI migration (see below for info):
            # https://www.clarusft.com/swaption-volumes-by-strike-q1-2024/
            with_straddles_df = self.detect_and_merge_straddles(curr_swaption_time_and_sales_with_iv_df)
            split_straddles_df = self.split_straddles(with_straddles_df)
            modifed_swaption_time_and_sales[bdate.to_pydatetime()] = split_straddles_df
            ohlc_premium[bdate.to_pydatetime()] = self.format_swaption_premium_close(split_straddles_df)

        self.s490_swaps._logger.error(colored(f"Errors Count: {err_count}", "red"))
        return modifed_swaption_time_and_sales, ohlc_premium, ql_curves_dict

    def _mark_s490_vol_cube_markings_db(
        self,
        date: datetime,
        swaption_tenor: str,
        atmf: float,
        sabr_beta: float,
        sabr_alpha: float,
        sabr_rho: float,
        sabr_nu: float,
        smile_df: pd.DataFrame,
        strike_offsets: ArrayLike,
    ):
        try:
            date_str = date.strftime("%Y-%m-%d")
            if self.s490_vol_cube_markings_db.exists(date_str):
                curr_vol_cube_markings = self.s490_vol_cube_markings_db.get(date_str)
            else:
                curr_vol_cube_markings = {}

            curr_vol_cube_markings[swaption_tenor] = {
                "atmf": atmf,
                "sabr_beta": sabr_beta,
                "sabr_alpha": sabr_alpha,
                "sabr_rho": sabr_rho,
                "sabr_nu": sabr_nu,
            }
            for strike_offset in strike_offsets:
                if strike_offset == 0:
                    strike_offset_str = f"{abs(int(strike_offset))}"
                elif strike_offset > 0:
                    strike_offset_str = f"+{abs(int(strike_offset))}"
                else:
                    strike_offset_str = f"-{abs(int(strike_offset))}"

                curr_offset_df = smile_df[smile_df["Offset (bps)"] == strike_offset]
                if curr_offset_df.empty:
                    raise ValueError(f"curr_offset_df is empty for {swaption_tenor} at offset {strike_offset} on {date_str}")
                curr_vol_cube_markings[swaption_tenor][strike_offset_str] = {
                    "normal": curr_offset_df["Implied (%/annual)"].iloc[-1],
                    "bp_vol": curr_offset_df["Implied (bps/day)"].iloc[-1],
                }
                self.s490_vol_cube_markings_db.set(date_str, curr_vol_cube_markings)
            return True, "Sucess"

        except Exception:
            self._logger.error(f"Swaption Vol Cube Marking DB Write Failed for {swaption_tenor} on {date}: {traceback.format_exc()}")
            return False, traceback.format_exc()

    def _mark_s490_atm_vol_term_structure_markings_db(
        self,
        date: datetime,
        option_maturity: str,
        vol_type: Literal["bp_vol", "normal"] = "bp_vol",
        gci_interp_func_str: Optional[
            Literal[
                "linear_interpolation",
                "log_linear_interpolation",
                "cubic_spline_interpolation",
                "cubic_hermite_interpolation",
                "pchip_interpolation",
                "monotone_convex",
            ]
        ] = None,
        show_plot: Optional[bool] = False,
        hard_coded_vols: Optional[Dict[int, float]] = None,
    ):
        if not self.s490_vol_cube_markings_db.exists(date.strftime("%Y-%m-%d")):
            raise ValueError(f"s490_vol_cube_markings_db is empty for {date.date()}")

        curr_vol_cube: Dict[str, np.float] = self.s490_vol_cube_markings_db.get(date.strftime("%Y-%m-%d"))
        curr_atm_vol_term_structure = {}
        sabr_params = {}
        for tenor in curr_vol_cube.keys():
            if option_maturity in tenor.split("x")[0]:
                sabr_params[tenor] = {}
                curr_atm_vol_term_structure[tenor] = curr_vol_cube[tenor]["0"][vol_type]
                sabr_params[tenor]["sabr_beta"] = curr_vol_cube[tenor]["sabr_beta"]
                sabr_params[tenor]["sabr_alpha"] = curr_vol_cube[tenor]["sabr_alpha"]
                sabr_params[tenor]["sabr_rho"] = curr_vol_cube[tenor]["sabr_rho"]
                sabr_params[tenor]["sabr_nu"] = curr_vol_cube[tenor]["sabr_nu"]

        curr_atm_vol_term_structure = dict(
            sorted(curr_atm_vol_term_structure.items(), key=lambda x: ql.Period(x[0].split("x")[-1]).normalized().length())
        )
        sorted_data = dict(sorted(curr_atm_vol_term_structure.items(), key=lambda x: ql.Period(x[0].split("x")[-1]).normalized().length()))
        print(colored(f"Current {option_maturity} Vol Data in s490_vol_cube_markings_db: {sorted_data}", "blue"))
        print(colored(f"Current {option_maturity} SABR Params s490_vol_cube_markings_db: {json.dumps(sabr_params, indent=4)}", "blue"))
        tenors = np.array([ql.Period(key.split("x")[-1]).normalized().length() for key in sorted_data.keys()])
        values = np.array(list(sorted_data.values()))

        interp_underlying_tenors = np.array([1, 2, 3, 5, 7, 10, 30])
        if not gci_interp_func_str:
            interp_vol = self.scipy_interp_func(tenors, values, kind="linear", logspace=True)(interp_underlying_tenors)
        else:
            gci = GeneralCurveInterpolator(tenors, values)
            gci_function = getattr(gci, gci_interp_func_str)
            interp_vol = gci_function(return_func=True)(interp_underlying_tenors)

        if show_plot:
            plt.figure(figsize=(10, 6))
            plt.plot(tenors, values, "o", label="Original Data", markersize=8)
            plt.plot(interp_underlying_tenors, interp_vol, "-", label="Interpolated Curve")
            plt.title("Interpolated Term Structure", fontsize=16)
            plt.xlabel("Tenor (Years)", fontsize=12)
            plt.ylabel("Value", fontsize=12)
            plt.legend(fontsize=12)
            plt.grid(axis="both", linestyle="--", alpha=0.7)
            plt.show()

        curr_atm_grid = {}
        if self.s490_atm_vol_grid_markings_db.exists(date.strftime("%Y-%m-%d")):
            curr_atm_grid = self.s490_atm_vol_grid_markings_db.get(date.strftime("%Y-%m-%d"))

        curr_atm_grid[option_maturity] = {}
        interp_atm_row = dict(zip(interp_underlying_tenors, interp_vol))

        if hard_coded_vols:
            for hard_coded_tenor, hard_coded_vol in hard_coded_vols.items():
                interp_atm_row[hard_coded_tenor] = hard_coded_vol

        curr_atm_grid[option_maturity][vol_type] = dict(zip(interp_underlying_tenors, interp_vol))
        self.s490_atm_vol_grid_markings_db.set(date.strftime("%Y-%m-%d"), curr_atm_grid)

        print(colored(f"Interpolated ATM Row: {json.dumps(interp_atm_row, indent=4)}", "blue"))
        print(colored(f"Wrote ATM Term Structure for {option_maturity} on {date.date()}", "green"))

    def plot_s490_atm_vol_term_structure_markings_db(self, date: datetime, option_maturity: str, vol_type: Literal["bp_vol", "normal"] = "bp_vol"):
        if not self.s490_atm_vol_grid_markings_db.exists(date.strftime("%Y-%m-%d")):
            raise ValueError(f"{date.date()} not in s490_atm_vol_grid_markings_db!")

        curr_atm_grid_dict: Dict[str, Dict[str, Dict[int, float]]] = self.s490_atm_vol_grid_markings_db.get(date.strftime("%Y-%m-%d"))
        if option_maturity not in curr_atm_grid_dict.keys():
            raise ValueError(f"{option_maturity} on {date.date()} not in s490_atm_vol_grid_markings_db!")

        curr_atm_row_dict: Dict[str, Dict[int, float]] = curr_atm_grid_dict[option_maturity]
        if vol_type not in curr_atm_row_dict.keys():
            raise ValueError(f"{vol_type} for {option_maturity} on {date.date()} not in s490_atm_vol_grid_markings_db!")

        curr_atm_vol = curr_atm_row_dict[vol_type]
        print(curr_atm_vol)

    # TODO: Le Floc's RSS DF + GN refinement SABR calibration implementation
    def create_sabr_smile_interactive(
        self,
        swaption_time_and_sales: pd.DataFrame,
        option_tenor: str,
        underlying_tenor: str,
        model: Literal["normal", "lognormal"] = "lognormal",
        offsets_bps=np.array([-200, -150, -100, -50, -25, 0, 25, 50, 100, 150, 200]),
        payer_skew_anchor_bpvol: Optional[float] = None,
        receiver_skew_anchor_bpvol: Optional[float] = None,
        skew_offset_anchor_bps: Optional[int] = 100,
        anchor_weight: Optional[int] = 10,
        initial_beta=1,
        show_trades=False,
        scale_by_notional=False,
        drop_trades_idxs: Optional[List[int]] = None,
        ploty_height=750,
        max_alpha: Optional[float] = None,
        year_day_count=252,
        synthetic_swaption: Optional[Dict[str, str]] = None,
    ):
        selected_swaption_df = swaption_time_and_sales[
            (swaption_time_and_sales["Option Tenor"] == option_tenor) & (swaption_time_and_sales["Underlying Tenor"] == underlying_tenor)
        ]
        selected_swaption_df = selected_swaption_df.reset_index(drop=True)

        if drop_trades_idxs:
            selected_swaption_df = selected_swaption_df.drop(drop_trades_idxs)

        df_cols = [
            "Event timestamp",
            "Direction",
            "Style",
            "Notional Amount",
            "Strike Price",
            "OTM",
            "Option Premium per Notional",
            "IV",
            "IV bp/day",
            "ATMF",
            "UPI FISN",
            "UPI Underlier Name",
        ]
        if show_trades:
            display(selected_swaption_df[df_cols])

        if selected_swaption_df.empty and synthetic_swaption is None:
            raise ValueError(f"No data for option tenor {option_tenor} and underlying tenor {underlying_tenor}")

        if synthetic_swaption is not None:
            for col in df_cols:
                if col not in synthetic_swaption:
                    synthetic_swaption[col] = None

            synthetic_swaption_df = pd.DataFrame([synthetic_swaption])
            selected_swaption_df = pd.concat([selected_swaption_df, synthetic_swaption_df])

        grouped = selected_swaption_df.groupby("Strike Price")
        average_vols = grouped["IV"].mean().reset_index()
        unique_strikes: NDArray[np.float64] = average_vols["Strike Price"].values
        average_vols: NDArray[np.float64] = average_vols["IV"].values / 100

        F = selected_swaption_df["ATMF"].iloc[-1]
        T = tenor_to_years(option_tenor)
        offsets_decimal = offsets_bps / 10000
        strikes = F + offsets_decimal
        valid_indices = strikes > 0
        valid_strikes: NDArray[np.float64] = strikes[valid_indices]
        valid_offsets_bps: NDArray[np.int64] = offsets_bps[valid_indices]

        payer_skew_dict = {}
        receiver_skew_dict = {}

        if model == "lognormal":
            calibration_lognormal = Hagan2002LognormalSABR(f=F, shift=0, t=T, beta=initial_beta).fit(unique_strikes, average_vols)
            alpha_calibrated, rho_calibrated, nu_calibrated = calibration_lognormal
            atm_vol = self.sabr_implied_vol(F, F, T, alpha=alpha_calibrated, beta=initial_beta, rho=rho_calibrated, nu=nu_calibrated) * 100
            vols = [
                Hagan2002LognormalSABR(
                    f=F, shift=0, t=T, v_atm_n=atm_vol * F, beta=initial_beta, rho=rho_calibrated, volvol=nu_calibrated
                ).lognormal_vol(strike)
                * 100
                for strike in valid_strikes
            ]

        elif model == "normal":
            calibration_normal = Hagan2002NormalSABR(f=F, shift=0, t=T, beta=initial_beta).fit(unique_strikes, average_vols)
            alpha_calibrated, rho_calibrated, nu_calibrated = calibration_normal
            atm_vol = self.sabr_implied_vol(F, F, T, alpha=alpha_calibrated, beta=initial_beta, rho=rho_calibrated, nu=nu_calibrated) * 100
            if payer_skew_anchor_bpvol or receiver_skew_anchor_bpvol:
                unique_strikes: List[np.float64] = unique_strikes.tolist()
                average_vols: List[np.float64] = average_vols.tolist()

                if payer_skew_anchor_bpvol:
                    payer_skew_dict["offset"] = (F + skew_offset_anchor_bps / 10000,)
                    payer_skew_dict["vol"] = (atm_vol * F * 10000 + (payer_skew_anchor_bpvol * np.sqrt(year_day_count)),)

                    for _ in range(anchor_weight):
                        unique_strikes.append(payer_skew_dict["offset"][0])
                        average_vols.append(payer_skew_dict["vol"][0] / 100)

                if receiver_skew_anchor_bpvol:
                    receiver_skew_dict["offset"] = (F - skew_offset_anchor_bps / 10000,)
                    receiver_skew_dict["vol"] = (atm_vol * F * 10000 + (receiver_skew_anchor_bpvol * np.sqrt(year_day_count)),)

                    for _ in range(anchor_weight):
                        unique_strikes.append(receiver_skew_dict["offset"][0])
                        average_vols.append(receiver_skew_dict["vol"][0] / 100)

                unique_strikes = np.array(unique_strikes)
                average_vols = np.array(average_vols)
                calibration_normal = Hagan2002NormalSABR(f=F, shift=0, t=T, beta=initial_beta).fit(unique_strikes, average_vols)
                alpha_calibrated, rho_calibrated, nu_calibrated = calibration_normal
                atm_vol = ql.sabrVolatility(F, F, T, alpha_calibrated, initial_beta, max(nu_calibrated, 0), max(rho_calibrated, 0.99), ql.Normal) * 100

            vols = [
                Hagan2002NormalSABR(f=F, shift=0, t=T, v_atm_n=atm_vol * F, beta=initial_beta, rho=rho_calibrated, volvol=nu_calibrated).normal_vol(
                    strike
                )
                * 100
                * 100
                for strike in valid_strikes
            ]

        else:
            raise ValueError(f"Bad Model Param: {model}")

        atm_vol = atm_vol * 100
        smile_dict = dict(zip(valid_offsets_bps.astype(str), vols))

        fig = go.FigureWidget()
        fig.add_trace(go.Scatter(x=valid_offsets_bps, y=vols, mode="lines", name="SABR Smile"))

        if len(payer_skew_dict.keys()) == 2 and len(receiver_skew_dict.keys()) == 2:
            fig.add_trace(
                go.Scatter(
                    x=[skew_offset_anchor_bps],
                    y=[payer_skew_dict["vol"][0]],
                    mode="markers",
                    marker=dict(color="blue", size=10),
                    name="Payer Skew Anchor",
                    hovertemplate=("Payer Skew Anchor Offset: %{x:3f} bps<br>" + "Implied Volatility: %{y:.3f}<br>" "<extra></extra>"),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=[skew_offset_anchor_bps * -1],
                    y=[receiver_skew_dict["vol"][0]],
                    mode="markers",
                    marker=dict(color="blue", size=10),
                    name="Receiver Skew Anchor",
                    hovertemplate=("Receiver Skew Anchor Offset: %{x:3f} bps<br>" + "Implied Volatility: %{y:.3f}<br>" "<extra></extra>"),
                )
            )

        direction_color_map = {
            None: "blue",
            "buyer": "green",
            "underwritter": "red",
        }

        if scale_by_notional:
            min_size = 5
            max_size = 20
            notional_values = selected_swaption_df["Notional Amount"]
            notional_values = notional_values.replace(0, np.nan)  # Avoid log(0) issues
            log_scaled_sizes = np.log(notional_values)
            log_scaled_sizes = min_size + (log_scaled_sizes - log_scaled_sizes.min()) * (max_size - min_size) / (
                log_scaled_sizes.max() - log_scaled_sizes.min()
            )

        for idx, row in selected_swaption_df.iterrows():
            fig.add_trace(
                go.Scatter(
                    x=[(row["Strike Price"] - F) * 10000],
                    y=[row["IV"]],
                    name=f"{row['Event timestamp'].time() if row['Event timestamp'] is not None else row['Event timestamp']}-{row['Style']} {row['Direction']}",
                    mode="markers",
                    marker=dict(size=log_scaled_sizes[idx] if scale_by_notional else 8, color=direction_color_map[row["Direction"]]),
                    hovertemplate=(
                        "Strike Offset: %{x:.3f} bps<br>"
                        "IV bps/annual: " + str(round(row["IV"] if row["IV"] is not None else 0, 3)) + "<br>"
                        "IV bps/day: " + str(round(row["IV bp/day"] if row["IV bp/day"] is not None else 0, 3)) + "<br>"
                        "Strike Price: " + str(round(row["Strike Price"] if row["Strike Price"] is not None else 0 * 100, 3)) + "<br>"
                        "ATMF: " + str(round(row["ATMF"] if row["ATMF"] is not None else 0 * 100, 3)) + "<br>"
                        "Notional Amount ($): " + str(row["Notional Amount"]) + "<br>"
                        "Option Premium Amount ($): " + str(row["Option Premium Amount"]) + "<br>"
                        "Option Premium per Notional ($): "
                        + str(round(row["Option Premium per Notional"] if row["Option Premium per Notional"] is not None else 0, 3))
                        + "<br>"
                        "Execution Timestamp: " + str(row["Event timestamp"]) + "<br>"
                        "Direction: " + str(row["Direction"]) + "<br>"
                        "Style: " + str(row["Style"]) + "<br>"
                        "<extra></extra>"
                    ),
                )
            )

        fig.add_trace(
            go.Scatter(
                x=[0],
                y=[smile_dict["0"]],
                mode="markers",
                marker=dict(color="blue", size=10),
                name="ATM Vol",
                hovertemplate=("ATM Strike Offset: 0 bps<br>" + "ATM Implied Volatility: %{y:.3f}<br>" "<extra></extra>"),
            )
        )
        fig.update_layout(
            title="",
            xaxis_title="Strike Offset from ATMF (bps)",
            yaxis_title="Implied Volatility (%)",
            legend=dict(font=dict(size=12)),
            hovermode="closest",
            template="plotly_dark",
            height=ploty_height,
            title_x=0.5,
        )
        fig.update_xaxes(
            showgrid=True,
            showspikes=True,
            spikecolor="white",
            spikesnap="cursor",
            spikemode="across",
        )
        fig.update_yaxes(showgrid=True, showspikes=True, spikecolor="white", spikesnap="cursor", spikethickness=0.5)

        slider_layout = widgets.Layout(width="1250px")
        beta_slider = widgets.FloatSlider(
            value=initial_beta, min=0.0, max=1.0, step=0.000001, description="β", continuous_update=True, readout_format=".6f", layout=slider_layout
        )
        alpha_slider = widgets.FloatSlider(
            value=alpha_calibrated,
            min=0.00001,
            max=max_alpha or 0.1,
            step=0.000001,
            description="α",
            continuous_update=True,
            readout_format=".6f",
            layout=slider_layout,
        )
        rho_slider = widgets.FloatSlider(
            value=rho_calibrated,
            min=-0.99999,
            max=0.99999,
            step=0.000001,
            description="Ρ",
            continuous_update=True,
            readout_format=".6f",
            layout=slider_layout,
        )
        nu_slider = widgets.FloatSlider(
            value=nu_calibrated,
            min=0.0001,
            max=2.0,
            step=0.000001,
            description="ν",
            continuous_update=True,
            readout_format=".6f",
            layout=slider_layout,
        )

        smile_output = widgets.Output()

        self._current_sabr_state = {}

        def update_sabr_smile(beta, alpha, rho, nu):
            atm_vol = self.sabr_implied_vol(F, F, T, alpha, beta, rho, nu) * 100
            if model == "lognormal":
                vols = [
                    Hagan2002LognormalSABR(f=F, shift=0, t=T, v_atm_n=atm_vol * F, beta=beta, rho=rho, volvol=nu).lognormal_vol(strike) * 100
                    for strike in valid_strikes
                ]
            elif model == "normal":
                vols = [
                    Hagan2002NormalSABR(f=F, shift=0, t=T, v_atm_n=atm_vol * F, beta=beta, rho=rho, volvol=nu).normal_vol(strike) * 100 * 100
                    for strike in valid_strikes
                ]
            else:
                raise ValueError("Unreachable")

            smile_dict = dict(zip(valid_offsets_bps.astype(str), vols))
            smile_dict_bps_vol = dict(
                zip(
                    valid_offsets_bps.astype(str),
                    (
                        [round(vol / np.sqrt(year_day_count), 3) for vol in smile_dict.values()]
                        if model == "normal"
                        else [None for _ in smile_dict.values()]
                    ),
                )
            )
            daily_atm_iv = smile_dict["0"] * F / np.sqrt(year_day_count) if model == "lognormal" else smile_dict["0"] / np.sqrt(year_day_count)
            smile_df = pd.DataFrame(
                {
                    "Offset (bps)": [int(offset) for offset in smile_dict.keys()],
                    "Implied (%/annual)": [round(vol, 3) for vol in smile_dict.values()],
                    "Implied (bps/day)": (
                        [round(vol / np.sqrt(year_day_count), 3) for vol in smile_dict.values()]
                        if model == "normal"
                        else [None for _ in smile_dict.values()]
                    ),
                }
            )
            smile_df = smile_df.sort_values("Offset (bps)").reset_index(drop=True)
            self._current_sabr_state = {
                "Date": selected_swaption_df["Event timestamp"].iloc[-1].to_pydatetime(),
                "swaption_tenor": f"{option_tenor}x{underlying_tenor}",
                "smile_df": smile_df,
                "atmf": F,
                "beta": beta,
                "alpha": alpha,
                "rho": rho,
                "nu": nu,
            }

            with fig.batch_update():
                fig.data[0].y = vols
                if payer_skew_anchor_bpvol and receiver_skew_anchor_bpvol:
                    fig.data[1].y = [smile_dict["0"] + (payer_skew_anchor_bpvol * np.sqrt(year_day_count))]
                    fig.data[2].y = [smile_dict["0"] + (receiver_skew_anchor_bpvol * np.sqrt(year_day_count))]

                fig.data[-1].y = [smile_dict["0"]]
                if model == "normal":
                    title_lines = [
                        f"{option_tenor} x {underlying_tenor} SABR {model.capitalize()} Vol Smile - {"pysabr"} Implementation --- DTCC Reported Trades From {selected_swaption_df['Event timestamp'].iloc[-1].to_pydatetime().date()}",
                        "-" * 75,
                        f"ATMF Strike: {F * 100:.3f}%, SABR ATM {model.capitalize()} Vol: {smile_dict["0"]:.3f} bps, {str(round(daily_atm_iv, 3))} bps/day",
                        "-" * 75,
                        f"Rec. Skew: {round(smile_dict_bps_vol[f"-{skew_offset_anchor_bps}"] - smile_dict_bps_vol["0"], 3)}, Pay. Skew: {round(smile_dict_bps_vol[f"{skew_offset_anchor_bps}"] - smile_dict_bps_vol["0"], 3)}",
                    ]
                else:
                    title_lines = [
                        f"{option_tenor} x {underlying_tenor} SABR {model.capitalize()} Vol Smile - {"pysabr"} Implementation --- DTCC Reported Trades From {selected_swaption_df['Event timestamp'].iloc[-1].to_pydatetime().date()}",
                        "-" * 75,
                        f"ATMF Strike: {F * 100:.3f}%, SABR ATM {model.capitalize()} Vol: {smile_dict["0"]:.3f} %, {str(round(daily_atm_iv * 100, 3))} bps/day",
                    ]
                fig.layout.title.text = "<br>".join(title_lines)

            with smile_output:
                smile_output.clear_output(wait=True)
                display(smile_df.style)

        write_db_button = widgets.Button(
            description="Write markings to DB", button_style="success", icon="check", layout=widgets.Layout(width="500px", height="40px")
        )

        def on_write_db_button_clicked(b):
            params = self._current_sabr_state
            if not params:
                print("No parameters to write yet. Adjust sliders first.")
                return
            success, db_write_message = self._mark_s490_vol_cube_markings_db(
                date=params["Date"],
                swaption_tenor=params["swaption_tenor"],
                smile_df=params["smile_df"],
                atmf=params["atmf"],
                sabr_beta=params["beta"],
                sabr_alpha=params["alpha"],
                sabr_rho=params["rho"],
                sabr_nu=params["nu"],
                strike_offsets=offsets_bps,
            )
            if success:
                print(f"Successfully wrote SABR markings to Vol Cube DB: {db_write_message}")
            else:
                print(f"Failed to write SABR markings to Vol Cube DB: {db_write_message}")

        write_db_button.on_click(on_write_db_button_clicked)

        ui = widgets.VBox([beta_slider, alpha_slider, rho_slider, nu_slider, write_db_button])
        out = widgets.interactive_output(update_sabr_smile, {"beta": beta_slider, "alpha": alpha_slider, "rho": rho_slider, "nu": nu_slider})
        display(smile_output, ui, fig, out)
