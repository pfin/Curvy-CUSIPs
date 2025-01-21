import logging
from datetime import datetime
from typing import Dict, List, Literal, Optional, Tuple, TypeAlias

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import QuantLib as ql
import tqdm
import tqdm.asyncio
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import BDay, CustomBusinessDay
from termcolor import colored
from functools import reduce

from CurvyCUSIPs.S490Swaps import S490Swaps
from CurvyCUSIPs.utils.ShelveDBWrapper import ShelveDBWrapper
from CurvyCUSIPs.CurveDataFetcher import CurveDataFetcher

JSON: TypeAlias = dict[str, "JSON"] | list["JSON"] | str | int | float | bool | None


# TODO
# - prerelease numba for 3.13
class S490Swaptions:
    s490_swaps: S490Swaps
    s490_atm_vol_timeseries_db: ShelveDBWrapper
    s490_vol_cube_timeseries_db: ShelveDBWrapper

    _logger = logging.getLogger(__name__)
    _debug_verbose: bool = False
    _error_verbose: bool = False
    _info_verbose: bool = False
    _no_logs_plz: bool = False

    def __init__(
        self,
        s490_swaps: S490Swaps,
        atm_vol_timeseries_db_path=None,
        s490_vol_cube_timeseries_db_path=None,
        debug_verbose: Optional[bool] = False,
        info_verbose: Optional[bool] = False,
        error_verbose: Optional[bool] = False,
    ):
        self.s490_swaps = s490_swaps
        
        if atm_vol_timeseries_db_path:
            self.s490_atm_vol_timeseries_db = self.setup_db(atm_vol_timeseries_db_path)
        else:
            self.s490_atm_vol_timeseries_db = None
            
        if s490_vol_cube_timeseries_db_path:
            self.s490_vol_cube_timeseries_db = self.setup_db(s490_vol_cube_timeseries_db_path)
        else:
            self.s490_vol_cube_timeseries_db = None

        self._debug_verbose = debug_verbose
        self._error_verbose = error_verbose
        self._info_verbose = info_verbose
        self._no_logs_plz = not debug_verbose and not error_verbose and not info_verbose
        self._setup_logger()

    def setup_db(self, db_path: str, create=False):
        """Setup shelve database with proper path handling."""
        try:
            db = ShelveDBWrapper(db_path, create=create)
            db.open()
            
            if len(db.keys()) == 0:
                print(colored(f"Warning: {db_path} is empty", "yellow"))
            else:
                most_recent_db_dt = datetime.strptime(max(db.keys()), "%Y-%m-%d")
                self._logger.info(f"Most recent date in db: {most_recent_db_dt}")
                if ((datetime.today() - BDay(1)) - most_recent_db_dt).days >= 1:
                    print(
                        colored(
                            f"{db_path} is behind --- cd into 'scripts' and run update script to update --- most recent date in db: {most_recent_db_dt}",
                            "yellow",
                        )
                    )
            return db
        except Exception as e:
            self._logger.error(f"Failed to setup database at {db_path}: {str(e)}")
            raise

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

    def _github_headers(self, path: str):
        return {
            "authority": "raw.githubusercontent.com",
            "method": "GET",
            "path": path,
            "scheme": "https",
            "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
            "accept-encoding": "gzip, deflate, br, zstd",
            "accept-language": "en-US,en;q=0.9",
            "cache-control": "no-cache",
            "dnt": "1",
            "pragma": "no-cache",
            "priority": "u=0, i",
            "sec-ch-ua": '"Not)A;Brand";v="99", "Google Chrome";v="127", "Chromium";v="127"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"Windows"',
            "sec-fetch-dest": "document",
            "sec-fetch-mode": "navigate",
            "sec-fetch-site": "none",
            "sec-fetch-user": "?1",
            "upgrade-insecure-requests": "1",
            "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36",
        }

    def get_vol_surfaces(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        bdates: Optional[List[datetime]] = None,
        date: Optional[datetime] = None,
        strike_offset: Optional[Literal[-200, -100, -50, -25, -10, 0, 10, 25, 50, 100, 200]] = 0,
        tail: Optional[Literal["1Y", "2Y", "3Y", "4Y", "5Y", "6Y", "7Y", "8Y", "9Y", "10Y", "15Y", "20Y", "25Y", "30Y"]] = None,
        expiry: Optional[Literal["1M", "3M", "6M", "1Y", "2Y", "3Y", "4Y", "5Y", "6Y", "7Y", "8Y", "9Y", "10Y", "15Y", "20Y", "25Y", "30Y"]] = None,
        plot_surfaces: Optional[bool] = False,
        use_ploty: Optional[bool] = False,
    ) -> Dict[datetime, pd.DataFrame]:
        vol_surface_dict: Dict[datetime, pd.DataFrame] = {}
        if start_date and end_date:
            bdates = [
                ts.to_pydatetime()
                for ts in pd.date_range(start=start_date, end=end_date, freq=CustomBusinessDay(calendar=USFederalHolidayCalendar()))
            ]
        elif date:
            bdates = [date]

        def matplotlib_vol_surface_plotter(vol_surface_df, title: str, xlabel: str, ylabel: str, zlabel: str):
            X, Y = np.meshgrid(range(len(vol_surface_df.columns)), range(len(vol_surface_df.index)))
            Z = vol_surface_df.values
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection="3d")
            surf = ax.plot_surface(X, Y, Z, cmap="viridis", edgecolor="k", alpha=0.9)
            ax.set_title(title, fontsize=16)
            ax.set_xlabel(xlabel, fontsize=12)
            ax.set_ylabel(ylabel, fontsize=12)
            ax.set_zlabel(zlabel, fontsize=12)
            ax.set_xticks(range(len(vol_surface_df.columns)))
            ax.set_xticklabels(vol_surface_df.columns, rotation=45, fontsize=10)
            ax.set_yticks(range(len(vol_surface_df.index)))
            ax.set_yticklabels(vol_surface_df.index, fontsize=10)
            fig.colorbar(surf, shrink=0.5, aspect=10, label="Nornmal Vol")
            plt.show()

        def plotly_vol_surface_plotter(vol_surface_df, title: str, xlabel: str, ylabel: str, zlabel: str):
            X, Y = np.meshgrid(range(len(vol_surface_df.columns)), range(len(vol_surface_df.index)))
            Z = vol_surface_df.values

            fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale="RdYlGn_r", showscale=True)])
            fig.update_traces(contours_z=dict(show=True, usecolormap=True, highlightcolor="limegreen", project_z=True))
            fig.update_layout(
                title=title,
                scene=dict(
                    xaxis=dict(title=xlabel, tickvals=list(range(len(vol_surface_df.columns))), ticktext=vol_surface_df.columns),
                    yaxis=dict(title=ylabel, tickvals=list(range(len(vol_surface_df.index))), ticktext=vol_surface_df.index),
                    zaxis=dict(title=zlabel),
                    aspectratio={"x": 1, "y": 1, "z": 0.6},
                    # camera_eye={"x": 0, "y": -1, "z": 0.5},
                ),
                template="plotly_dark",
                height=750,
                width=1250,
            )
            fig.update_xaxes(showspikes=True, spikecolor="white", spikesnap="cursor", spikemode="across")
            fig.update_yaxes(
                showspikes=True,
                spikecolor="white",
                spikesnap="cursor",
                spikethickness=0.5,
            )
            fig.show()

        for bdate in bdates:
            try:
                # Expiry-Tail surface
                if not tail and not expiry:
                    if strike_offset == 0:
                        vol_surface_df = pd.DataFrame(self.s490_atm_vol_timeseries_db.get(bdate.strftime("%Y-%m-%d")))
                    else:
                        strike_offset = str(strike_offset)
                        vol_surface_df = pd.DataFrame(self.s490_vol_cube_timeseries_db.get(bdate.strftime("%Y-%m-%d"))[strike_offset])

                    vol_surface_df = vol_surface_df.rename(columns={"Option Tenor": "Expiry"})
                    vol_surface_df = vol_surface_df.set_index("Expiry")
                    vol_surface_dict[bdate] = vol_surface_df

                    if plot_surfaces:
                        vol_surface_df = vol_surface_df.iloc[::-1].T
                        if use_ploty:
                            plotly_vol_surface_plotter(
                                vol_surface_df,
                                f"{bdate.date()} ATM {"+" + str(strike_offset) if strike_offset > 0 else strike_offset} Expiry-Tail Vol Surface",
                                "Tail",
                                "Expiry",
                                "Normal Vol",
                            )
                        else:
                            matplotlib_vol_surface_plotter(
                                vol_surface_df,
                                f"{bdate.date()} ATM {"+" + str(strike_offset) if strike_offset > 0 else strike_offset} Expiry-Tail Vol Surface",
                                "Tail",
                                "Expiry",
                                "Normal Vol",
                            )

                # Expiry-Strike Surface
                elif tail and not expiry:
                    vol_cube: Dict[str, List[Dict[str, float]]] = self.s490_vol_cube_timeseries_db.get(bdate.strftime("%Y-%m-%d"))
                    expiry_strike_surface = []
                    for curr_strike_offset, expiry_tail_surface in vol_cube.items():
                        for expiry_vol_dict in expiry_tail_surface:
                            expiry_strike_surface.append(
                                {
                                    "Strike": curr_strike_offset,
                                    "Expiry": expiry_vol_dict["Option Tenor"],
                                    "Normal Vol": expiry_vol_dict[tail],
                                }
                            )

                    vol_surface_df = pd.DataFrame(expiry_strike_surface)
                    vol_surface_df["Strike"] = pd.to_numeric(vol_surface_df["Strike"])
                    vol_surface_df = vol_surface_df.sort_values(by=["Strike"])
                    vol_surface_df = vol_surface_df.pivot(index="Strike", columns="Expiry", values="Normal Vol").dropna(axis="columns")
                    vol_surface_df = vol_surface_df[sorted(vol_surface_df.columns, key=lambda x: ql.Period(x))]
                    vol_surface_dict[bdate] = vol_surface_df

                    if plot_surfaces:
                        vol_surface_df = vol_surface_df.T
                        if use_ploty:
                            plotly_vol_surface_plotter(
                                vol_surface_df,
                                f"{bdate.date()} {tail} Tail, Expiry-Strike Vol Surface",
                                "ATM Strike Offsets",
                                "Expiry",
                                "Normal Vol",
                            )
                        else:
                            matplotlib_vol_surface_plotter(
                                vol_surface_df,
                                f"{bdate.date()} {tail} Tail, Expiry-Strike Vol Surface",
                                "ATM Strike Offsets",
                                "Expiry",
                                "Normal Vol",
                            )

                # Tail-Strike Surface
                elif expiry and not tail:
                    vol_cube: Dict[str, List[Dict[str, float]]] = self.s490_vol_cube_timeseries_db.get(bdate.strftime("%Y-%m-%d"))
                    expiry_strike_surface = []
                    for curr_strike_offset, expiry_tail_surface in vol_cube.items():
                        for expiry_vol_dict in expiry_tail_surface:
                            if expiry == expiry_vol_dict["Option Tenor"]:
                                expiry_vol_dict_copy = expiry_vol_dict.copy()
                                del expiry_vol_dict_copy["Option Tenor"]
                                expiry_strike_surface.append({"Strike": curr_strike_offset} | expiry_vol_dict_copy)

                    vol_surface_df = pd.DataFrame(expiry_strike_surface)
                    vol_surface_dict[bdate] = vol_surface_df

                    if plot_surfaces:
                        vol_surface_df = vol_surface_df.set_index("Strike")
                        vol_surface_df = vol_surface_df.T
                        if use_ploty:
                            plotly_vol_surface_plotter(
                                vol_surface_df,
                                f"{bdate.date()} {expiry} Expiry, Tail-Strike Vol Surface",
                                "ATM Strike Offsets",
                                "Tail",
                                "Normal Vol",
                            )
                        else:
                            matplotlib_vol_surface_plotter(
                                vol_surface_df,
                                f"{bdate.date()} {expiry} Expiry, Tail-Strike Vol Surface",
                                "ATM Strike Offsets",
                                "Tail",
                                "Normal Vol",
                            )
                else:
                    raise ValueError(f"Bad Params for {bdate}: {tail}, {expiry}")
                
            except Exception as e:
                self._logger.error(f"'get_vol_surfaces' Error at {bdate}: {e}")

        return vol_surface_dict

    def _format_swaption_tenor_key(self, tenor, strike_offset):
        strike_offset = int(strike_offset)
        if strike_offset > 0:
            key = f"{tenor}_+{strike_offset}"
        elif strike_offset < 0:
            key = f"{tenor}_-{strike_offset}"
        else:
            key = f"{tenor}_ATM"
        return key

    def get_vol_timeseries(
        self,
        tenor_strike_pairs: List[Tuple[str, str | int]],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        bdates: Optional[List[datetime]] = None,
        plot_timeseries: Optional[bool] = False,
        default_vol: Optional[Literal["Normal Vol", "Bpvol"]] = "Bpvol",
        one_df: Optional[bool] = False,
    ) -> Dict[datetime, pd.DataFrame]:
        vol_timeseries: Dict[str, List[Dict[datetime, float, float]]] = {}
        for tenor, strike_offset in tenor_strike_pairs:
            key = self._format_swaption_tenor_key(tenor, strike_offset)
            vol_timeseries[key] = []

        if start_date and end_date:
            bdates = [
                ts.to_pydatetime()
                for ts in pd.date_range(start=start_date, end=end_date, freq=CustomBusinessDay(calendar=USFederalHolidayCalendar()))
            ]

        errors = []
        for bdate in tqdm.tqdm(bdates, desc="FETCHING VOL TIMESERIES"):
            try:
                for tenor, strike_offset in tenor_strike_pairs:
                    key = self._format_swaption_tenor_key(tenor, strike_offset)
                    if int(strike_offset) != 0:
                        strike_offset = str(strike_offset)
                        otm_vol_df = pd.DataFrame(self.s490_vol_cube_timeseries_db.get(bdate.strftime("%Y-%m-%d"))[strike_offset])
                        otm_vol_df = otm_vol_df.rename(columns={"Option Tenor": "Expiry"})
                        otm_vol_df = otm_vol_df.set_index("Expiry")
                        normal_vol = otm_vol_df.loc[tenor.split("x")[0]][tenor.split("x")[1]]
                        vol_timeseries[key].append({"Date": bdate, "Normal Vol": normal_vol, "Bpvol": normal_vol / np.sqrt(252)})
                    else:
                        atm_vol_df = pd.DataFrame(self.s490_atm_vol_timeseries_db.get(bdate.strftime("%Y-%m-%d")))
                        atm_vol_df = atm_vol_df.rename(columns={"Option Tenor": "Expiry"})
                        atm_vol_df = atm_vol_df.set_index("Expiry")
                        normal_vol = atm_vol_df.loc[tenor.split("x")[0]][tenor.split("x")[1]]
                        vol_timeseries[key].append({"Date": bdate, "Normal Vol": normal_vol, "Bpvol": normal_vol / np.sqrt(252)})
            except Exception as e:
                errors.append({"Date": bdate, "Error": str(e)})

        self._logger.warning("ATM Vol Timeseries Errors Report: ")
        self._logger.warning(pd.DataFrame(errors))

        dfs = []
        if plot_timeseries:
            plt.figure()
            default_title = []
            for tenor, strike_offset in tenor_strike_pairs:
                key = self._format_swaption_tenor_key(tenor, strike_offset)
                default_title.append(key)
                vol_timeseries[key] = pd.DataFrame(vol_timeseries[key])
                (line,) = plt.plot(vol_timeseries[key]["Date"], vol_timeseries[key][default_vol], label=key)
                avg_vol = vol_timeseries[key][default_vol].mean()
                plt.axhline(y=avg_vol, color=line.get_color(), linestyle="--", alpha=0.6)
                most_recent_date = vol_timeseries[key]["Date"].iloc[-1].strftime("%Y-%m-%d")
                most_recent_vol = vol_timeseries[key][default_vol].iloc[-1]
                line.set_label(f"{key} - Latest: {most_recent_date}, {most_recent_vol:.2f} {default_vol}")
                if one_df:
                    curr_df = pd.DataFrame(vol_timeseries[key])[["Date", default_vol]]
                    curr_df = curr_df.rename(columns={default_vol: key})
                    dfs.append(curr_df)

            plt.xlabel("Date")
            plt.ylabel(default_vol)
            plt.title(f"{", ".join(str(x) for x in default_title) if len(default_title) > 1 else default_title[0]} {default_vol}")
            plt.legend(fontsize="medium")
            plt.show()
        else:
            for tenor, strike_offset in tenor_strike_pairs:
                key = self._format_swaption_tenor_key(tenor, strike_offset)
                if one_df:
                    curr_df = pd.DataFrame(vol_timeseries[key])[["Date", default_vol]]
                    curr_df = curr_df.rename(columns={default_vol: key})
                    dfs.append(curr_df)
                else:
                    vol_timeseries[key] = pd.DataFrame(vol_timeseries[key])

        if one_df:
            return reduce(lambda left, right: left.merge(right, on="Date", how="outer"), dfs)

        return vol_timeseries

    def get_ql_atm_surface_handle(self, date: datetime):
        atm_vol_grid_df = self.get_vol_surfaces(date=date, strike_offset=0)[date]
        return ql.SwaptionVolatilityStructureHandle(
            ql.SwaptionVolatilityMatrix(
                ql.UnitedStates(ql.UnitedStates.GovernmentBond),
                ql.ModifiedFollowing,
                [ql.Period(e) for e in atm_vol_grid_df.index],
                [ql.Period(t) for t in atm_vol_grid_df.columns],
                ql.Matrix([[vol / 10_000 for vol in row] for row in atm_vol_grid_df.values]),
                ql.Actual360(),
                False,
                ql.Normal,
            )
        )

    def get_vol_cube(self, date: datetime) -> Dict[Literal[-200, -100, -50, -25, -10, 0, 10, 25, 50, 100, 200], pd.DataFrame]:
        return {
            -200: self.get_vol_surfaces(date=date, strike_offset=-200)[date],
            -100: self.get_vol_surfaces(date=date, strike_offset=-100)[date],
            -50: self.get_vol_surfaces(date=date, strike_offset=-50)[date],
            -25: self.get_vol_surfaces(date=date, strike_offset=-25)[date],
            -10: self.get_vol_surfaces(date=date, strike_offset=-10)[date],
            0: self.get_vol_surfaces(date=date, strike_offset=0)[date],
            10: self.get_vol_surfaces(date=date, strike_offset=10)[date],
            25: self.get_vol_surfaces(date=date, strike_offset=25)[date],
            50: self.get_vol_surfaces(date=date, strike_offset=50)[date],
            100: self.get_vol_surfaces(date=date, strike_offset=100)[date],
            200: self.get_vol_surfaces(date=date, strike_offset=200)[date],
        }
    
    def get_ql_vol_cube_handle(
        self,
        date: Optional[datetime] = None,
        vol_cube_dict: Optional[Dict[Literal[-200, -100, -50, -25, -10, 0, 10, 25, 50, 100, 200], pd.DataFrame]] = None,
    ) -> ql.SwaptionVolatilityStructureHandle:
        if not vol_cube_dict:
            assert date
            vol_cube_dict = self.get_vol_cube(date=date)

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

        ql_vol_cube_handle = ql.SwaptionVolatilityStructureHandle(
            ql.InterpolatedSwaptionVolatilityCube(
                ql.SwaptionVolatilityStructureHandle(atm_swaption_vol_matrix),
                expiries,
                tails,
                strike_spreads,
                vol_spreads,
                ql.OvernightIndexedSwapIndex("SOFR-OIS", ql.Period("1D"), 2, ql.USDCurrency(), self.s490_swaps._ql_sofr),
                ql.OvernightIndexedSwapIndex("SOFR-OIS", ql.Period("1D"), 2, ql.USDCurrency(), self.s490_swaps._ql_sofr),
                False,
            )
        )

        return ql_vol_cube_handle

    def create_s490_swaption_time_and_sales(
        self,
        start_date: datetime,
        end_date: datetime,
        data_fetcher: CurveDataFetcher,
        model: Literal["normal", "lognormal"] = "normal",
    ) -> Tuple[Dict[datetime, pd.DataFrame], Dict[datetime, Dict[str, float]], Dict[datetime, ql.YieldTermStructure]]:
        """Fetch and process swaption time and sales data."""
        try:
            # Fetch swaption trades from DTCC
            swaption_time_and_sales_dict = data_fetcher.dtcc_sdr_fetcher.fetch_historical_swaption_time_and_sales(
                start_date=start_date,
                end_date=end_date,
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

            # Process results
            close_premium_dict = {}
            ql_curves_dict = {}

            # Calculate implied vols for each date
            for date, df in swaption_time_and_sales_dict.items():
                # Get QuantLib curve for this date
                ql_curves = self.s490_swaps.build_sofr_ois_curve(
                    as_of_date=date,
                    data_fetcher=data_fetcher
                )
                ql_curves_dict[date] = ql_curves

                # Calculate IVs using the curve
                df['IV'] = df.apply(lambda row: self._calculate_implied_vol(
                    premium=row['Option Premium per Notional'],
                    strike=row['Strike Price'],
                    option_tenor=row['Option Tenor'],
                    underlying_tenor=row['Underlying Tenor'],
                    curve=ql_curves['discount_curve'],
                    model=model
                ), axis=1)

            return swaption_time_and_sales_dict, close_premium_dict, ql_curves_dict
            
        except Exception as e:
            self._logger.error(f"Failed to fetch swaption data: {str(e)}")
            raise

    def _calculate_closing_premium(
        self,
        trades_df: pd.DataFrame,
        ql_curve: ql.YieldTermStructure,
        model: str
    ) -> Dict[str, float]:
        """Calculate closing premium values from trades.
        
        Helper method for create_s490_swaption_time_and_sales.
        """
        # Group trades by option/underlying tenor pair
        grouped = trades_df.groupby(['Option Tenor', 'Underlying Tenor'])
        
        premiums = {}
        for (opt_tenor, und_tenor), group in grouped:
            key = f"{opt_tenor}x{und_tenor}"
            
            # Use last trade of the day as closing premium
            if not group.empty:
                last_trade = group.iloc[-1]
                premiums[key] = float(last_trade['Option Premium per Notional'])
                
        return premiums

    def _calculate_implied_vol(
        self,
        premium: float,
        strike: float,
        option_tenor: str,
        underlying_tenor: str,
        curve: ql.YieldTermStructure,
        model: str
    ) -> float:
        """Calculate implied volatility using Bachelier formula for normal model."""
        try:
            # Convert tenors to QuantLib periods
            opt_period = ql.Period(option_tenor)
            und_period = ql.Period(underlying_tenor)
            
            # Get forward rate
            forward = curve.forwardRate(
                opt_period,
                und_period,
                ql.Actual365Fixed(),
                ql.Simple
            ).rate()
            
            # Get time to expiry in years
            expiry = float(opt_period.length()) / 12.0  # Convert months to years
            
            if model == "normal":
                # Use QuantLib's Bachelier calculator
                vol = ql.bachelierBlackFormulaImpliedVol(
                    ql.Option.Call,  # Type doesn't matter for ATM
                    strike,
                    forward,
                    expiry,
                    premium
                )
                return vol
            elif model == "lognormal":
                raise NotImplementedError("Lognormal model not yet implemented")
            else:
                raise ValueError(f"Unknown model: {model}")
            
        except Exception as e:
            self._logger.warning(f"Failed to calculate IV: {str(e)}")
            return float('nan')
