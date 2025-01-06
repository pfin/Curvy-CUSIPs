import itertools
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Literal, Optional, Tuple, Annotated

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import random
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import QuantLib as ql
from joblib import Parallel, delayed
import tqdm
from tqdm_joblib import tqdm_joblib
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import BDay, CustomBusinessDay
from scipy.stats import zscore, tstd, norm
from sklearn.decomposition import PCA
from termcolor import colored

from CurvyCUSIPs.CurveDataFetcher import CurveDataFetcher
from CurvyCUSIPs.utils.dtcc_swaps_utils import DEFAULT_SWAP_TENORS, datetime_to_ql_date, tenor_to_years
from CurvyCUSIPs.utils.ShelveDBWrapper import ShelveDBWrapper
from CurvyCUSIPs.utils.mean_reversion import simulate_mean_reversion_ou


class S490Swaps:
    s490_nyclose_db: ShelveDBWrapper = None
    curve_data_fetcher: CurveDataFetcher = None
    _ql_yts: ql.RelinkableYieldTermStructureHandle = None
    _ql_sofr: ql.Sofr = None

    _logger = logging.getLogger(__name__)
    _debug_verbose: bool = False
    _error_verbose: bool = False
    _info_verbose: bool = False
    _no_logs_plz: bool = False

    def __init__(
        self,
        s490_curve_db_path: str,
        curve_data_fetcher: CurveDataFetcher,
        ql_yts: Optional[ql.RelinkableYieldTermStructureHandle] = None,
        debug_verbose: Optional[bool] = False,
        info_verbose: Optional[bool] = False,
        error_verbose: Optional[bool] = False,
    ):
        self._debug_verbose = debug_verbose
        self._error_verbose = error_verbose
        self._info_verbose = info_verbose
        self._no_logs_plz = not debug_verbose and not error_verbose and not info_verbose
        self._setup_logger()

        self.setup_s490_nyclose_db(s490_curve_db_path=s490_curve_db_path)
        self.curve_data_fetcher = curve_data_fetcher

        self._ql_yts = ql_yts if ql_yts else ql.RelinkableYieldTermStructureHandle()
        self._ql_sofr = ql.Sofr(self._ql_yts)
        try:
            sofr_fixings_df = self.curve_data_fetcher.nyfrb_data_fetcher.get_sofr_fixings_df(
                start_date=datetime(2018, 4, 2),
                end_date=datetime.today(),
            )
            sofr_fixings_df_date_col = "effectiveDate"
            sofr_fixings_df_fixings_col = "percentRate"
        except Exception as e:
            self._logger.error(colored(f"S490Swaos init: Failed to fetch SOFR Fixing from NYFRB: {e}", "red"))
            sofr_fixings_df = self.curve_data_fetcher.fred_data_fetcher.fred.get_multiple_series(
                series_ids=["SOFR"], observation_start=datetime(2018, 4, 2), observation_end=datetime.today(), one_df=True
            )
            sofr_fixings_df = sofr_fixings_df.reset_index(names="Date")
            sofr_fixings_df_date_col = "Date"
            sofr_fixings_df_fixings_col = "SOFR"

        sofr_fixings_df = sofr_fixings_df.dropna()
        self._ql_sofr.addFixings(
            [datetime_to_ql_date(ed) for ed in sofr_fixings_df[sofr_fixings_df_date_col]],
            [fixing / 100 for fixing in sofr_fixings_df[sofr_fixings_df_fixings_col]],
        )

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

    def setup_s490_nyclose_db(self, s490_curve_db_path: str):
        self.s490_nyclose_db = ShelveDBWrapper(s490_curve_db_path) if s490_curve_db_path else None
        self.s490_nyclose_db.open()

        most_recent_db_dt = datetime.fromtimestamp(int(max(self.s490_nyclose_db.keys())))
        self._logger.info(f"Most recent date in db: {most_recent_db_dt}")
        if ((datetime.today() - BDay(1)) - most_recent_db_dt).days >= 1:
            print(
                colored(
                    f"{s490_curve_db_path} is behind --- cd into 'scripts' and run 'update_sofr_ois_db.py' to update --- most recent date in db: {most_recent_db_dt}",
                    "yellow",
                )
            )

    def s490_nyclose_term_structure_ts(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        bdates = pd.date_range(start=start_date, end=end_date, freq=CustomBusinessDay(calendar=USFederalHolidayCalendar()))
        ts_term_structures = []
        for curr_date in bdates:
            try:
                str_ts = str(int(curr_date.to_pydatetime().timestamp()))
                ohlc_df = pd.DataFrame(self.s490_nyclose_db.get(str_ts)["ohlc"])
                curr_term_structure = {"Date": curr_date}
                curr_term_structure = curr_term_structure | dict(zip(DEFAULT_SWAP_TENORS, ohlc_df["Close"] * 100))
                ts_term_structures.append(curr_term_structure)
            except Exception as e:
                self._logger.error(colored(f'"s490_nyclose_term_structure_ts" Something went wrong at {curr_date}: {e}'), "red")

        df = pd.DataFrame(ts_term_structures)
        df = df.set_index("Date")
        return df

    @staticmethod
    def swap_spreads_term_structure(swaps_term_structure_ts_df: pd.DataFrame, cash_term_structure_ts_df: pd.DataFrame, is_cmt=False):
        CT_TENORS = ["CT3M", "CT6M", "CT1", "CT2", "CT3", "CT5", "CT7", "CT10", "CT20", "CT30"]
        CMT_TENORS = ["CMT3M", "CMT6M", "CMT1", "CMT2", "CMT3", "CMT5", "CMT7", "CMT10", "CMT20", "CMT30"]
        SWAP_TENORS = ["3M", "6M", "12M", "2Y", "3Y", "5Y", "7Y", "10Y", "20Y", "30Y"]
        aligned_index = swaps_term_structure_ts_df.index.intersection(cash_term_structure_ts_df.index)
        swaps_aligned = swaps_term_structure_ts_df.loc[aligned_index, SWAP_TENORS]
        cash_aligned = cash_term_structure_ts_df.loc[aligned_index, CMT_TENORS if is_cmt else CT_TENORS]
        swap_spreads = (swaps_aligned.subtract(cash_aligned.values, axis=0)) * 100
        return swap_spreads

    def s490_nyclose_fwd_curve_matrices(
        self,
        start_date: datetime,
        end_date: datetime,
        fwd_tenors: Optional[List[str]] = ["1D", "1W", "1M", "2M", "3M", "6M", "12M", "18M", "2Y", "3Y", "5Y", "10Y", "15Y"],
        rm_swap_tenors: Optional[List[str]] = None,
        swaption_time_and_sales_dict_for_fwds: Optional[Dict[datetime, pd.DataFrame]] = None,
        ql_piecewise_method: Literal[
            "logLinearDiscount", "logCubicDiscount", "linearZero", "cubicZero", "linearForward", "splineCubicDiscount"
        ] = "logLinearDiscount",
        ql_zero_curve_method: Optional[ql.ZeroCurve] = ql.ZeroCurve,
        ql_compounding=ql.Compounded,
        ql_compounding_freq=ql.Annual,
        use_ql_implied_ts: Optional[bool] = True,
        level_bps_adj: Optional[int] = 0,
    ) -> Tuple[Dict[datetime, pd.DataFrame], Dict[datetime, ql.DiscountCurve | ql.ZeroCurve | ql.ForwardCurve]]:
        bdates = pd.date_range(start=start_date, end=end_date, freq=CustomBusinessDay(calendar=USFederalHolidayCalendar()))
        fwd_term_structure_grids: Dict[datetime, pd.DataFrame] = {}
        ql_curves: Dict[datetime, ql.DiscountCurve | ql.ZeroCurve | ql.ForwardCurve] = {}

        for curr_date in tqdm.tqdm(bdates, "Building Implied Fwd Curves..."):
            try:
                str_ts = str(int(curr_date.to_pydatetime().timestamp()))
                if not self.s490_nyclose_db.exists(str_ts):
                    continue
                ql_piecewise_term_struct_nodes: Dict = self.s490_nyclose_db.get(str_ts)[ql_piecewise_method]
                if "Discount" in ql_piecewise_method:
                    ql_curve = ql.DiscountCurve(
                        [datetime_to_ql_date(mat) for mat in ql_piecewise_term_struct_nodes.keys()],
                        ql_piecewise_term_struct_nodes.values(),
                        ql.Actual360(),
                        ql.UnitedStates(ql.UnitedStates.GovernmentBond),
                    )
                elif "Zero" in ql_piecewise_method:
                    ql_curve = ql_zero_curve_method(
                        [datetime_to_ql_date(mat) for mat in ql_piecewise_term_struct_nodes.keys()],
                        ql_piecewise_term_struct_nodes.values(),
                        ql.Actual360(),
                        ql.UnitedStates(ql.UnitedStates.GovernmentBond),
                    )
                elif "Forward" in ql_piecewise_method:
                    ql_curve = ql.ForwardCurve(
                        [datetime_to_ql_date(mat) for mat in ql_piecewise_term_struct_nodes.keys()],
                        ql_piecewise_term_struct_nodes.values(),
                        ql.Actual360(),
                        ql.UnitedStates(ql.UnitedStates.GovernmentBond),
                    )
                else:
                    raise ValueError("Bad ql_piecewise_method method passed in")

                ohlc_df = pd.DataFrame(self.s490_nyclose_db.get(str_ts)["ohlc"])
                ohlc_df["Expiry"] = pd.to_numeric(ohlc_df["Expiry"], errors="coerce")
                ohlc_df["Expiry"] = pd.to_datetime(ohlc_df["Expiry"], errors="coerce", unit="s")
                if rm_swap_tenors:
                    tenors = list(set(DEFAULT_SWAP_TENORS) - set(rm_swap_tenors))
                    ohlc_df = ohlc_df[ohlc_df["Tenor"].isin(tenors)]
                    dict_for_df = {"Tenor": tenors, "Spot": ohlc_df["Close"] * 100}
                else:
                    dict_for_df = {"Tenor": DEFAULT_SWAP_TENORS, "Spot": ohlc_df["Close"] * 100}

                ql_curve.enableExtrapolation()
                ql.Settings.instance().evaluationDate = datetime_to_ql_date(curr_date)
                ql_curves[curr_date] = ql_curve

                if swaption_time_and_sales_dict_for_fwds is not None:
                    swaption_ts_df = swaption_time_and_sales_dict_for_fwds[curr_date.to_pydatetime()]
                    fwd_tenors = swaption_ts_df["Option Tenor"].unique()

                for fwd_tenor in fwd_tenors:
                    fwd_start_date = ql.UnitedStates(ql.UnitedStates.GovernmentBond).advance(datetime_to_ql_date(curr_date), ql.Period(fwd_tenor))

                    ql_curve_to_use = ql_curve
                    if use_ql_implied_ts:
                        implied_ts = ql.ImpliedTermStructure(ql.YieldTermStructureHandle(ql_curve), fwd_start_date)
                        implied_ts.enableExtrapolation()
                        ql_curve_to_use = implied_ts

                    fwds_list = []
                    for underlying_tenor in ohlc_df["Tenor"]:
                        try:
                            fwd_end_date = ql.UnitedStates(ql.UnitedStates.GovernmentBond).advance(fwd_start_date, ql.Period(underlying_tenor))
                            forward_rate = ql_curve_to_use.forwardRate(
                                fwd_start_date, fwd_end_date, ql.Actual360(), ql_compounding, ql_compounding_freq, True
                            ).rate()
                            fwds_list.append(forward_rate * 100 + (level_bps_adj / 100))
                        except Exception as e:
                            self._logger.error(f"Error computing forward rate on {curr_date.date()} for {fwd_tenor}x{underlying_tenor}: {e}")
                            fwds_list.append(float("nan"))

                    dict_for_df[f"{fwd_tenor} Fwd"] = fwds_list

                df = pd.DataFrame(dict_for_df)
                df["period"] = df["Tenor"].apply(lambda x: ql.Period(x))
                df = df.sort_values(by="period").drop(columns=["period"]).reset_index(drop=True)
                fwd_term_structure_grids[curr_date] = df

            except Exception as e:
                self._logger.error(colored(f'"s490_nyclose_fwd_grid_term_structures" Something went wrong at {curr_date}: {e}'), "red")
                print(e)

        return fwd_term_structure_grids, ql_curves

    @staticmethod
    def fwd_grid_dict_curve_plotter(
        tenor_date_pairs: List[Tuple[str, datetime]], fwd_grid_dict: Dict[datetime, pd.DataFrame], use_plotly: Optional[bool] = False
    ):
        if use_plotly:
            fig = go.Figure()
            for tenor, dt in tenor_date_pairs:
                fig.add_trace(
                    go.Scatter(
                        x=[tenor_to_years(tenor) for tenor in fwd_grid_dict[dt]["Tenor"].to_list()],
                        y=fwd_grid_dict[dt][tenor],
                        mode="lines",
                        name=f"{dt.date()} - {tenor}",
                    )
                )
            fig.update_layout(
                title="SOFR OIS Fwd Curve",
                xaxis_title="Tenor",
                yaxis_title="Yield",
                font=dict(size=11),
                template="plotly_dark",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                height=750,
            )
            fig.update_xaxes(showspikes=True, spikecolor="white", spikesnap="cursor", spikemode="across", showgrid=True)
            fig.update_yaxes(showspikes=True, spikecolor="white", spikesnap="cursor", spikethickness=0.5, showgrid=True)
            fig.show()
        else:
            for tenor, dt in tenor_date_pairs:
                plt.plot(
                    [tenor_to_years(tenor) for tenor in fwd_grid_dict[dt]["Tenor"].to_list()],
                    fwd_grid_dict[dt][tenor],
                    label=f"{dt.date()} - {tenor}",
                )

            plt.xlabel("Tenor")
            plt.ylabel("Yield")
            plt.title("SOFR OIS Fwd Curve")
            plt.legend(fontsize="large")
            plt.grid(True)
            plt.show()

    @staticmethod
    def fwd_grid_dict_timeseries_plotter(
        fwd_grid_dict: Dict[datetime, pd.DataFrame],
        tenors_to_plot: List[str],
        bdates: List[pd.Timestamp],
        tenors_to_plot_on_secondary_axis: Optional[List[str]] = None,
        custom_title: Optional[str] = None,
        use_plotly: Optional[bool] = False,
        custom_fly_weights: Optional[Annotated[List[float], 3]] = None,
    ):
        S490Swaps._general_fwd_dict_df_timeseries_plotter(
            fwd_dict_df=fwd_grid_dict,
            tenors_to_plot=tenors_to_plot,
            tenors_to_plot_on_secondary_axis=tenors_to_plot_on_secondary_axis,
            bdates=bdates,
            tqdm_desc="PLOTTING SWAPS",
            custom_title=custom_title,
            use_plotly=use_plotly,
            custom_fly_weights=custom_fly_weights,
        )

    @staticmethod
    def _calculate_cross_df_zscores(dict_df: Dict[datetime, pd.DataFrame]) -> pd.DataFrame:
        dict_df = dict(sorted(dict_df.items()))
        df_list = list(dict_df.values())
        array_3d = np.stack([df.values for df in df_list])
        zscores = zscore(array_3d, axis=0, nan_policy="omit")
        zscore_dfs = []
        for i in range(len(df_list)):
            zscore_df = pd.DataFrame(zscores[i], index=df_list[0].index, columns=df_list[0].columns)
            zscore_dfs.append(zscore_df)
        return zscore_dfs

    @staticmethod
    def pca_on_fwd_grids(
        fwd_grid_dict: Dict[datetime, pd.DataFrame],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        indy_fwd_strips: Optional[bool] = False,
        run_on_level_changes: Optional[bool] = False,
        n_components: Optional[int] = 3,
        rm_swap_tenors: Optional[List[str]] = None,
        rolling_window: Optional[int] = None,
        n_jobs: Optional[int] = 1,  # note: only for rolling pca, parallelization can reduce wall-clock time but not overall CPU time
        n_jobs_parent: Optional[int] = None,
        n_jobs_child: Optional[int] = None,
        verbose: Optional[bool] = False,
    ):
        df_list = []
        for obs_date, df_forward in fwd_grid_dict.items():
            tmp = df_forward.copy()
            tmp.insert(0, "Date", obs_date)
            df_list.append(tmp)

        big_df = pd.concat(df_list, ignore_index=True)
        if rm_swap_tenors:
            big_df = big_df[~big_df["Tenor"].isin(rm_swap_tenors)]
        if start_date:
            big_df = big_df[big_df["Date"] >= start_date]
        if end_date:
            big_df = big_df[big_df["Date"] <= end_date]

        melted_df = big_df.melt(id_vars=["Date", "Tenor"], value_vars=list(big_df.columns[2:]), var_name="FwdType", value_name="Rate")
        long_df = melted_df.pivot(index="Date", columns=["Tenor", "FwdType"], values="Rate")

        def _ql_period_wrapper(x):
            if x == "Tenor":
                return ql.Period("-0D")
            elif x == "Spot":
                return ql.Period("0D")
            elif "Fwd" in x:
                return ql.Period(str(x).split(" Fwd")[0])

        def _merge_residuals_by_date(fwd_residual_dict):
            date_dict = {}
            for fwd_type, date_map in fwd_residual_dict.items():
                for dt, sub_df in date_map.items():
                    if dt not in date_dict:
                        date_dict[dt] = pd.DataFrame(index=sub_df.index)
                    date_dict[dt][fwd_type] = sub_df["Residual"]

            for dt, df in date_dict.items():
                df = df[sorted(df.columns, key=lambda x: _ql_period_wrapper(x))]
                date_dict[dt] = df

            return date_dict

        if rolling_window:
            sorted_dates = sorted(long_df.index)
            if len(sorted_dates) < rolling_window:
                raise ValueError(f"Not enough observations to do rolling PCA of window size={rolling_window}.")

            if indy_fwd_strips:
                # two levels of parallelism for rolling pca on indy fwd strips
                if n_jobs_parent and n_jobs_child:
                    fwd_types = long_df.columns.levels[1]
                    rolling_pca_results_dict = {}
                    master_residuals_for_anchor = pd.DataFrame(index=long_df.index, columns=long_df.columns, dtype=float)

                    # inner worker for anchor_idx
                    def _rolling_pca_worker(sub_df: pd.DataFrame, sorted_dates_, anchor_idx: int, run_on_level_changes: bool):
                        anchor_date = sorted_dates_[anchor_idx]
                        window_dates = sorted_dates_[anchor_idx - rolling_window + 1 : anchor_idx + 1]
                        window_data = sub_df.loc[window_dates]

                        pca = PCA(n_components=n_components)

                        if run_on_level_changes:
                            window_diff = window_data.diff().fillna(0.0)
                            pca.fit(window_diff)
                            reconst_changes = pca.inverse_transform(pca.transform(window_diff))
                            reconst_changes_df = pd.DataFrame(
                                reconst_changes,
                                index=window_diff.index,
                                columns=window_diff.columns,
                            )
                            reconst_levels_window = reconst_changes_df.cumsum() + window_data.iloc[0]
                            residuals_window = window_data - reconst_levels_window
                        else:
                            pca.fit(window_data)
                            reconst_levels_window = pd.DataFrame(
                                pca.inverse_transform(pca.transform(window_data)),
                                index=window_data.index,
                                columns=window_data.columns,
                            )
                            residuals_window = window_data - reconst_levels_window

                        anchor_resid = residuals_window.loc[[anchor_date]]

                        if run_on_level_changes:
                            loadings_df = pd.DataFrame(
                                pca.components_,
                                columns=window_diff.columns,
                                index=[f"PC{i+1}" for i in range(n_components)],
                            )
                            scores_df = pd.DataFrame(
                                pca.transform(window_diff),
                                index=window_diff.index,
                                columns=[f"PC{i+1}" for i in range(n_components)],
                            )
                        else:
                            loadings_df = pd.DataFrame(
                                pca.components_,
                                columns=window_data.columns,
                                index=[f"PC{i+1}" for i in range(n_components)],
                            )
                            scores_df = pd.DataFrame(
                                pca.transform(window_data),
                                index=window_data.index,
                                columns=[f"PC{i+1}" for i in range(n_components)],
                            )

                        residuals_long = residuals_window.stack().reset_index()
                        residuals_long.columns = ["ObsDate", "Tenor", "Residual"]
                        residuals_timeseries_dict = {}
                        for dt_, sub_sub_df in residuals_long.groupby("ObsDate"):
                            pivoted = sub_sub_df.pivot_table(index="Tenor", values="Residual", aggfunc="first").reset_index()
                            pivoted["period"] = pivoted["Tenor"].apply(lambda x: ql.Period(x))
                            pivoted = pivoted.sort_values(by="period").drop(columns=["period"]).reset_index(drop=True).set_index("Tenor")
                            residuals_timeseries_dict[dt_] = pivoted * 100

                        residual_zscores = S490Swaps._calculate_cross_df_zscores(residuals_timeseries_dict)
                        return (
                            anchor_date,
                            {
                                "explained_variance_ratio_": pca.explained_variance_ratio_,
                                "loadings_df": loadings_df,
                                "scores_df": scores_df,
                                "reconstructed_window": reconst_levels_window,
                                "residuals_window": residuals_window,
                                "residual_timeseries_dict": residuals_timeseries_dict,
                                "residual_timeseries_zscore_dict": dict(zip(residuals_timeseries_dict.keys(), residual_zscores)),
                                "rich_cheap_zscore_anchor": residual_zscores[-1] if len(residual_zscores) > 0 else None,
                                "anchor_resid": anchor_resid,
                            },
                        )

                    # outer worker for a single fwd_type
                    def _pca_for_one_fwd_type(fwd_type: str):
                        sub_df = long_df.xs(fwd_type, axis=1, level="FwdType").loc[sorted_dates]
                        anchor_idx_values = range(rolling_window - 1, len(sorted_dates))

                        # parallelize across anchor_idx here, but limit to n_jobs_child.
                        par_results = Parallel(n_jobs=n_jobs_child)(
                            delayed(_rolling_pca_worker)(sub_df, sorted_dates, aidx, run_on_level_changes) for aidx in anchor_idx_values
                        )
                        return fwd_type, dict(par_results)

                    # outer parallel over fwd_types, using n_jobs_parent for the outer level
                    # fwd_types_results = Parallel(n_jobs=n_jobs_parent)(
                    #     delayed(_pca_for_one_fwd_type)(fwd_type)
                    #     for fwd_type in tqdm.tqdm(fwd_types, desc="ROLLING PCA ON INDY FWDs (parent loop)...")
                    # )
                    fwd_types_results = [
                        result
                        for result in tqdm.tqdm(
                            Parallel(return_as="generator", n_jobs=n_jobs_parent)(delayed(_pca_for_one_fwd_type)(fwd_type) for fwd_type in fwd_types),
                            desc="ROLLING PCA ON INDY FWDS..." if not run_on_level_changes else "ROLLING PCA ON INDY FWDS TIME STEP LEVEL CHANGES...",
                            total=len(fwd_types),
                        )
                    ]

                    for fwd_type, fwd_type_dict in tqdm.tqdm(fwd_types_results, desc="CLEANING UP ROLLING PCA RESULTS..."):
                        rolling_pca_results_dict[fwd_type] = fwd_type_dict
                        for anchor_date, res_dict in fwd_type_dict.items():
                            anchor_resid_df = res_dict["anchor_resid"]
                            for tenor_col in anchor_resid_df.columns:
                                master_residuals_for_anchor.loc[anchor_date, (tenor_col, fwd_type)] = anchor_resid_df.loc[anchor_date, tenor_col]

                    rich_cheap_residual_zscore_timeseries_dict: Dict[datetime, pd.DataFrame] = {}
                    for anchor_date in long_df.index:
                        curr_dict = {}
                        for fwd_type in fwd_types:
                            if anchor_date in rolling_pca_results_dict[fwd_type]:
                                rc_anchor = rolling_pca_results_dict[fwd_type][anchor_date]["rich_cheap_zscore_anchor"]
                                if rc_anchor is not None:
                                    curr_dict[fwd_type] = rc_anchor["Residual"]

                        curr_df = pd.DataFrame(curr_dict)
                        if not curr_df.empty:
                            cols_sorted = sorted(curr_df.columns, key=lambda x: _ql_period_wrapper(x))
                            curr_df = curr_df[cols_sorted]
                        rich_cheap_residual_zscore_timeseries_dict[anchor_date] = curr_df

                    return {
                        "rolling_windows": rolling_window,
                        "rolling_pca_results_per_fwd": rolling_pca_results_dict,
                        "master_anchor_date_residuals": master_residuals_for_anchor,
                        "rich_cheap_residual_zscore_timeseries_dict": rich_cheap_residual_zscore_timeseries_dict,
                    }

                # single level of parallelism for rolling PCA on indy fwd strips
                else:
                    fwd_types = long_df.columns.levels[1]
                    rolling_pca_results_dict = {}
                    master_residuals_for_anchor = pd.DataFrame(index=long_df.index, columns=long_df.columns, dtype=float)

                    for fwd_type in tqdm.tqdm(
                        fwd_types,
                        "ROLLING PCA ON INDY FWDS..." if not run_on_level_changes else "ROLLING PCA ON INDY FWDS TIME STEP LEVEL CHANGES...",
                        total=len(fwd_types),
                    ):
                        sub_df = long_df.xs(fwd_type, axis=1, level="FwdType").loc[sorted_dates]

                        # anchor_idx is an integer pointing to a position in sorted_dates.
                        def _rolling_pca_worker(anchor_idx: int, run_on_level_changes: bool):
                            anchor_date = sorted_dates[anchor_idx]
                            window_dates = sorted_dates[anchor_idx - rolling_window + 1 : anchor_idx + 1]
                            window_data = sub_df.loc[window_dates]

                            pca = PCA(n_components=n_components)

                            if run_on_level_changes:
                                window_diff = window_data.diff().fillna(0.0)
                                pca.fit(window_diff)
                                reconst_changes = pca.inverse_transform(pca.transform(window_diff))
                                reconst_changes_df = pd.DataFrame(
                                    reconst_changes,
                                    index=window_diff.index,
                                    columns=window_diff.columns,
                                )
                                reconst_levels_window = reconst_changes_df.cumsum() + window_data.iloc[0]
                                residuals_window = window_data - reconst_levels_window
                            else:
                                pca.fit(window_data)
                                reconst_levels_window = pd.DataFrame(
                                    pca.inverse_transform(pca.transform(window_data)),
                                    index=window_data.index,
                                    columns=window_data.columns,
                                )
                                residuals_window = window_data - reconst_levels_window

                            anchor_resid = residuals_window.loc[[anchor_date]]

                            if run_on_level_changes:
                                loadings_df = pd.DataFrame(
                                    pca.components_,
                                    columns=window_diff.columns,
                                    index=[f"PC{i+1}" for i in range(n_components)],
                                )
                                scores_df = pd.DataFrame(
                                    pca.transform(window_diff),
                                    index=window_diff.index,
                                    columns=[f"PC{i+1}" for i in range(n_components)],
                                )
                            else:
                                loadings_df = pd.DataFrame(
                                    pca.components_,
                                    columns=window_data.columns,
                                    index=[f"PC{i+1}" for i in range(n_components)],
                                )
                                scores_df = pd.DataFrame(
                                    pca.transform(window_data),
                                    index=window_data.index,
                                    columns=[f"PC{i+1}" for i in range(n_components)],
                                )

                            residuals_long = residuals_window.stack().reset_index()
                            residuals_long.columns = ["ObsDate", "Tenor", "Residual"]
                            residuals_timeseries_dict = {}
                            for dt_, sub_sub_df in residuals_long.groupby("ObsDate"):
                                pivoted = sub_sub_df.pivot_table(index="Tenor", values="Residual", aggfunc="first").reset_index()
                                pivoted["period"] = pivoted["Tenor"].apply(lambda x: ql.Period(x))
                                pivoted = pivoted.sort_values(by="period").drop(columns=["period"]).reset_index(drop=True).set_index("Tenor")
                                residuals_timeseries_dict[dt_] = pivoted * 100

                            residual_zscores = S490Swaps._calculate_cross_df_zscores(residuals_timeseries_dict)
                            return (
                                anchor_date,
                                {
                                    "explained_variance_ratio_": pca.explained_variance_ratio_,
                                    "loadings_df": loadings_df,
                                    "scores_df": scores_df,
                                    "reconstructed_window": reconst_levels_window,
                                    "residuals_window": residuals_window,
                                    "residual_timeseries_dict": residuals_timeseries_dict,
                                    "residual_timeseries_zscore_dict": dict(zip(residuals_timeseries_dict.keys(), residual_zscores)),
                                    "rich_cheap_zscore_anchor": residual_zscores[-1] if len(residual_zscores) > 0 else None,
                                    "anchor_resid": anchor_resid,
                                },
                            )

                        anchor_idx_values = range(rolling_window - 1, len(sorted_dates))
                        if verbose:
                            # par_results = [
                            #     result
                            #     for result in tqdm.tqdm(
                            #         Parallel(return_as="generator", n_jobs=n_jobs)(delayed(_rolling_pca_worker)(aidx) for aidx in anchor_idx_values),
                            #         desc=f"{fwd_type} STRIP ROLLING PCA CALC...",
                            #         total=len(anchor_idx_values),
                            #     )
                            # ]
                            par_results = Parallel(n_jobs=n_jobs)(
                                delayed(_rolling_pca_worker)(aidx, run_on_level_changes)
                                for aidx in tqdm.tqdm(anchor_idx_values, desc=f"{fwd_type} STRIP ROLLING PCA CALC...")
                            )
                        else:
                            par_results = Parallel(n_jobs=n_jobs)(
                                delayed(_rolling_pca_worker)(aidx, run_on_level_changes) for aidx in anchor_idx_values
                            )

                        fwd_type_dict = dict(par_results)
                        for anchor_date, res_dict in fwd_type_dict.items():
                            for tenor_col in res_dict["anchor_resid"].columns:
                                master_residuals_for_anchor.loc[anchor_date, (tenor_col, fwd_type)] = res_dict["anchor_resid"].loc[
                                    anchor_date, tenor_col
                                ]

                        rolling_pca_results_dict[fwd_type] = fwd_type_dict

                    rich_cheap_residual_zscore_timeseries_dict: Dict[datetime, pd.DataFrame] = {}
                    tenors = rolling_pca_results_dict.keys()
                    for anchor_date in long_df.index:
                        curr_dict = {}
                        for tenor in tenors:
                            if anchor_date in rolling_pca_results_dict[tenor]:
                                curr_dict[tenor] = rolling_pca_results_dict[tenor][anchor_date]["rich_cheap_zscore_anchor"]["Residual"]

                        curr_df = pd.DataFrame(curr_dict)
                        if not curr_df.empty:
                            cols_sorted = sorted(curr_df.columns, key=lambda x: _ql_period_wrapper(x))
                            curr_df = curr_df[cols_sorted]
                        rich_cheap_residual_zscore_timeseries_dict[anchor_date] = curr_df

                    return {
                        "rolling_windows": rolling_window,
                        "rolling_pca_results_per_fwd": rolling_pca_results_dict,
                        "master_anchor_date_residuals": master_residuals_for_anchor,
                        "rich_cheap_residual_zscore_timeseries_dict": rich_cheap_residual_zscore_timeseries_dict,
                    }

            # rolling pca across fwd grid
            else:
                long_df = long_df.loc[sorted_dates]

                def _rolling_pca_worker(anchor_idx):
                    anchor_date = sorted_dates[anchor_idx]
                    window_dates = sorted_dates[anchor_idx - rolling_window + 1 : anchor_idx + 1]
                    window_data = long_df.loc[window_dates]

                    pca = PCA(n_components=n_components)
                    pca.fit(window_data)

                    loadings_df = pd.DataFrame(
                        pca.components_,
                        columns=window_data.columns,
                        index=[f"PC{i+1}" for i in range(n_components)],
                    )
                    scores_df = pd.DataFrame(
                        pca.transform(window_data),
                        index=window_data.index,
                        columns=[f"PC{i+1}" for i in range(n_components)],
                    )
                    reconstructed_window = pd.DataFrame(
                        pca.inverse_transform(pca.transform(window_data)),
                        index=window_data.index,
                        columns=window_data.columns,
                    )
                    residuals_window = window_data - reconstructed_window
                    anchor_resid = residuals_window.loc[[anchor_date]]
                    residuals_long = residuals_window.stack(["Tenor", "FwdType"]).reset_index()
                    residuals_long.columns = ["ObsDate", "Tenor", "FwdType", "Residual"]
                    residuals_timeseries_dict = {}
                    for dt_, sub_df in residuals_long.groupby("ObsDate"):
                        pivoted = sub_df.pivot(index="Tenor", columns="FwdType", values="Residual")
                        pivoted = pivoted.reset_index()
                        pivoted["period"] = pivoted["Tenor"].apply(lambda x: ql.Period(x))
                        pivoted = pivoted.sort_values(by="period").drop(columns=["period"]).reset_index(drop=True).set_index("Tenor")
                        pivoted = pivoted[sorted(pivoted.columns, key=lambda x: _ql_period_wrapper(x))]
                        residuals_timeseries_dict[dt_] = pivoted * 100

                    residual_zscores = S490Swaps._calculate_cross_df_zscores(residuals_timeseries_dict)
                    return (
                        anchor_date,
                        {
                            "evr": pca.explained_variance_ratio_,
                            "loading_df": loadings_df,
                            "scores_df": scores_df,
                            "reconstructed_window": reconstructed_window,
                            "residuals_window": residuals_window,
                            "residual_timeseries_dict": residuals_timeseries_dict,
                            "residual_timeseries_zscore_dict": dict(zip(residuals_timeseries_dict.keys(), residual_zscores)),
                            "rich_cheap_zscore_anchor": residual_zscores[-1] if len(residual_zscores) > 0 else None,
                            "anchor_resid": anchor_resid,
                        },
                    )

                anchor_idx_values = range(rolling_window - 1, len(sorted_dates))
                # with tqdm_joblib(tqdm.tqdm(total=len(anchor_idx_values))):
                #     par_results = Parallel(n_jobs=n_jobs)(delayed(_rolling_pca_worker)(aidx) for aidx in anchor_idx_values)
                par_results = Parallel(n_jobs=n_jobs)(
                    delayed(_rolling_pca_worker)(aidx) for aidx in tqdm.tqdm(anchor_idx_values, desc="ROLLING PCA ON FWD GRID...")
                )
                rolling_pca_results_dict = dict(par_results)

                master_residuals_for_anchor = pd.DataFrame(index=long_df.index, columns=long_df.columns, dtype=float)
                for anchor_date, res_dict in tqdm.tqdm(rolling_pca_results_dict.items(), desc="CLEANING UP ROLLING PCA RESULTS..."):
                    anchor_resid = res_dict["anchor_resid"]
                    for col_ in anchor_resid.columns:
                        master_residuals_for_anchor.loc[anchor_date, col_] = anchor_resid.loc[anchor_date, col_]

                rich_cheap_residual_zscore_timeseries_dict: Dict[datetime, pd.DataFrame] = {}
                for anchor_date in long_df.index:
                    if anchor_date in rolling_pca_results_dict:
                        rich_cheap_residual_zscore_timeseries_dict[anchor_date] = rolling_pca_results_dict[anchor_date]["rich_cheap_zscore_anchor"]
                    else:
                        rich_cheap_residual_zscore_timeseries_dict[anchor_date] = None

                return {
                    "rolling_windows": rolling_window,
                    "rolling_pca_results": rolling_pca_results_dict,
                    "master_anchor_date_residuals": master_residuals_for_anchor,
                    "rich_cheap_residual_zscore_timeseries_dict": rich_cheap_residual_zscore_timeseries_dict,
                }

        # non-rolling pca case
        else:

            # non-rolling pca on indy fwd strips
            if indy_fwd_strips:

                def _pca_for_one_fwd_type(fwd_type: str):
                    sub_df = long_df.xs(fwd_type, axis=1, level="FwdType")

                    if run_on_level_changes:
                        sub_df_diff = sub_df.diff().fillna(0.0)
                        pca = PCA(n_components=n_components)
                        pca.fit(sub_df_diff)

                        reconst_changes = pca.inverse_transform(pca.transform(sub_df_diff))
                        reconst_changes_df = pd.DataFrame(reconst_changes, index=sub_df_diff.index, columns=sub_df_diff.columns)

                        reconst_levels_df = reconst_changes_df.cumsum() + sub_df.iloc[0]
                        reconstructed_sub = reconst_levels_df
                        residuals_sub = sub_df - reconstructed_sub

                        loadings_df = pd.DataFrame(
                            pca.components_,
                            columns=sub_df_diff.columns,
                            index=[f"PC{i+1}" for i in range(n_components)],
                        )
                        scores_df = pd.DataFrame(
                            pca.transform(sub_df_diff),
                            index=sub_df_diff.index,
                            columns=[f"PC{i+1}" for i in range(n_components)],
                        )

                    else:
                        pca = PCA(n_components=n_components)
                        pca.fit(sub_df)
                        loadings_df = pd.DataFrame(
                            pca.components_,
                            columns=sub_df.columns,
                            index=[f"PC{i+1}" for i in range(n_components)],
                        )
                        scores_df = pd.DataFrame(
                            pca.transform(sub_df),
                            index=sub_df.index,
                            columns=[f"PC{i+1}" for i in range(n_components)],
                        )
                        reconstructed_sub = pd.DataFrame(
                            pca.inverse_transform(pca.transform(sub_df)),
                            index=sub_df.index,
                            columns=sub_df.columns,
                        )
                        residuals_sub = sub_df - reconstructed_sub

                    residuals_long = residuals_sub.stack().reset_index()
                    residuals_long.columns = ["ObsDate", "Tenor", "Residual"]
                    residuals_timeseries_dict = {}
                    for obs_date, sub_sub_df in residuals_long.groupby("ObsDate"):
                        pivoted = sub_sub_df.pivot_table(index="Tenor", values="Residual", aggfunc="first").reset_index()
                        pivoted["period"] = pivoted["Tenor"].apply(lambda x: ql.Period(x))
                        pivoted = pivoted.sort_values(by="period").drop(columns=["period"]).reset_index(drop=True).set_index("Tenor")
                        residuals_timeseries_dict[obs_date] = pivoted * 100

                    residual_zscores = S490Swaps._calculate_cross_df_zscores(residuals_timeseries_dict)
                    pca_dict = {
                        "explained_variance_ratio_": pca.explained_variance_ratio_,
                        "loadings_df": loadings_df,
                        "scores_df": scores_df,
                        "reconstructed_df_sub": reconstructed_sub,
                        "residuals_df_sub": residuals_sub,
                        "residual_timeseries_dict": residuals_timeseries_dict,
                        "residual_timeseries_zscore_dict": dict(zip(residuals_timeseries_dict.keys(), residual_zscores)),
                        "rich_cheap_zscore_heatmap": residual_zscores[-1] if residual_zscores else None,
                    }
                    return (fwd_type, pca_dict, reconstructed_sub, residuals_sub)

                pca_results_dict = {}
                reconstructed_all = pd.DataFrame(index=long_df.index, columns=long_df.columns, dtype=float)
                residuals_all = pd.DataFrame(index=long_df.index, columns=long_df.columns, dtype=float)
                fwd_types = long_df.columns.levels[1]
                par_results = Parallel(n_jobs=n_jobs)(
                    delayed(_pca_for_one_fwd_type)(fwd_type) for fwd_type in tqdm.tqdm(fwd_types, desc="PCA ON INDY FWD STRIPS...")
                )

                for fwd_type, pca_dict, reconstructed_sub, residuals_sub in tqdm.tqdm(
                    par_results, desc="CLEANING UP PCA ON INDY FWD STRIPS RESULTS..."
                ):
                    pca_results_dict[fwd_type] = pca_dict
                    for tenor_col in reconstructed_sub.columns:
                        reconstructed_all[(tenor_col, fwd_type)] = reconstructed_sub[tenor_col]
                        residuals_all[(tenor_col, fwd_type)] = residuals_sub[tenor_col]

                residuals_long_all = residuals_all.stack(["Tenor", "FwdType"]).reset_index()
                residuals_long_all.columns = ["ObsDate", "Tenor", "FwdType", "Residual"]

                heatmap_dict = {}
                for fwd_type in pca_results_dict.keys():
                    heat = pca_results_dict[fwd_type]["rich_cheap_zscore_heatmap"]
                    # Only add if not None
                    if heat is not None:
                        heatmap_dict[fwd_type] = heat
                rich_cheap_map_df = pd.concat(heatmap_dict, axis=1)
                rich_cheap_map_df.columns = rich_cheap_map_df.columns.get_level_values(0)
                rich_cheap_map_df = rich_cheap_map_df[sorted(rich_cheap_map_df.columns, key=lambda x: _ql_period_wrapper(x))]

                return {
                    "multi_pca_results_per_fwd": pca_results_dict,
                    "reconstructed_df": reconstructed_all,
                    "residuals_df": residuals_all,
                    "residual_timeseries_dict": _merge_residuals_by_date(
                        {fwd_type: pca_results_dict[fwd_type]["residual_timeseries_dict"] for fwd_type in pca_results_dict.keys()}
                    ),
                    "rich_cheap_residual_zscore_timeseries_dict": _merge_residuals_by_date(
                        {fwd_type: pca_results_dict[fwd_type]["residual_timeseries_zscore_dict"] for fwd_type in pca_results_dict.keys()}
                    ),
                    "rich_cheap_zscore_heatmap": rich_cheap_map_df,
                }

            else:
                if run_on_level_changes:
                    long_df_diff = long_df.diff().fillna(0.0)
                    pca = PCA(n_components=n_components)
                    pca.fit(long_df_diff)
                    loadings_df = pd.DataFrame(
                        data=pca.components_,
                        columns=long_df_diff.columns,
                        index=[f"PC{i+1}" for i in range(n_components)],
                    )
                    scores_df = pd.DataFrame(
                        data=pca.transform(long_df_diff),
                        index=long_df_diff.index,
                        columns=[f"PC{i+1}" for i in range(n_components)],
                    )
                    reconstructed_diff = pca.inverse_transform(pca.transform(long_df_diff))
                    reconstructed_diff_df = pd.DataFrame(
                        reconstructed_diff,
                        index=long_df_diff.index,
                        columns=long_df_diff.columns,
                    )
                    reconstructed_df = reconstructed_diff_df.cumsum() + long_df.iloc[0]
                    residuals_df = long_df - reconstructed_df

                else:
                    pca = PCA(n_components=n_components)
                    pca.fit(long_df)
                    loadings_df = pd.DataFrame(
                        data=pca.components_,
                        columns=long_df.columns,
                        index=[f"PC{i+1}" for i in range(n_components)],
                    )
                    scores_df = pd.DataFrame(
                        data=pca.transform(long_df),
                        index=long_df.index,
                        columns=[f"PC{i+1}" for i in range(n_components)],
                    )
                    reconstructed_df = pd.DataFrame(
                        data=pca.inverse_transform(pca.transform(long_df)),
                        index=long_df.index,
                        columns=long_df.columns,
                    )
                    residuals_df = long_df - reconstructed_df

                residuals_df = long_df - reconstructed_df
                residuals_long = residuals_df.stack(["Tenor", "FwdType"]).reset_index()
                residuals_long.columns = ["ObsDate", "Tenor", "FwdType", "Residual"]
                residuals_timeseries_dict = {}
                for obs_date, sub_df in residuals_long.groupby("ObsDate"):
                    pivoted = sub_df.pivot(index="Tenor", columns="FwdType", values="Residual").reset_index()
                    pivoted.columns.name = None
                    pivoted["period"] = pivoted["Tenor"].apply(lambda x: ql.Period(x))
                    pivoted = pivoted.sort_values(by="period").drop(columns=["period"]).reset_index(drop=True).set_index("Tenor")
                    pivoted = pivoted[sorted(pivoted.columns, key=lambda x: _ql_period_wrapper(x))]
                    residuals_timeseries_dict[obs_date] = pivoted * 100

                residual_zscores = S490Swaps._calculate_cross_df_zscores(residuals_timeseries_dict)
                return {
                    "evr": pca.explained_variance_ratio_,
                    "loading_df": loadings_df,
                    "scores_df": scores_df,
                    "reconstructed_df": reconstructed_df,
                    "residuals_df": residuals_df,
                    "residual_timeseries_dict": residuals_timeseries_dict,
                    "rich_cheap_residual_zscore_timeseries_dict": dict(zip(residuals_timeseries_dict.keys(), residual_zscores)),
                    "rich_cheap_zscore_heatmap": residual_zscores[-1],
                }

    @staticmethod
    def pca_residual_timeseries_plotter(
        pca_results: Dict,
        tenors_to_plot: List[str],
        use_plotly: Optional[bool] = False,
        key: Optional[str] = "residual_timeseries_dict",
        custom_fly_weights: Optional[Annotated[List[float], 3]] = None,
    ):
        tenors_to_plot = list(set(tenors_to_plot))
        residuals_df_dict: Dict[datetime, pd.DataFrame] = pca_results[key]
        S490Swaps._general_fwd_dict_df_timeseries_plotter(
            fwd_dict_df=residuals_df_dict,
            tenors_to_plot=tenors_to_plot,
            bdates=list([key for key in residuals_df_dict.keys() if residuals_df_dict[key] is not None and not residuals_df_dict[key].empty]),
            tqdm_desc="PLOTTING PCA RESIDUAL ZSCORES",
            custom_title=f"PCA Residuals: {tenors_to_plot[0] if len(tenors_to_plot) == 0 else ", ".join(tenors_to_plot)}",
            yaxis_title="Residuals (bps)",
            tenor_is_df_index=True,
            use_plotly=use_plotly,
            custom_fly_weights=custom_fly_weights,
            should_scale=False,
        )

    @staticmethod
    def pca_residual_credit_suisse_BBar_plot(
        pca_results: Dict,
        tenors_to_plot: Optional[List[str]] = [  # TODO assign tenors_to_plot based on tenors in pca_results
            "12M Fwd 12M",
            "12M Fwd 18M",
            "12M Fwd 2Y",
            "12M Fwd 3Y",
            "12M Fwd 4Y",
            "12M Fwd 5Y",
            "12M Fwd 6Y",
            "12M Fwd 7Y",
            "12M Fwd 8Y",
            "12M Fwd 9Y",
            "12M Fwd 10Y",
            "12M Fwd 12Y",
            "12M Fwd 15Y",
            "12M Fwd 20Y",
            "12M Fwd 25Y",
            "12M Fwd 30Y",
            "12M Fwd 40Y",
            "12M Fwd 50Y",
        ],
        bday_offsets: Optional[Annotated[List[int], 3]] = None,
        title: Optional[str] = "Latest PCA Residuals",
    ):
        if bday_offsets:
            if len(bday_offsets) > 3:
                raise ValueError(f"'bday_offsets' must be length 3")
            bday_offsets = sorted(bday_offsets)

        residuals_df_dict: Dict[datetime, pd.DataFrame] = pca_results["residual_timeseries_dict"]
        bdates = residuals_df_dict.keys()

        latest_date = max(bdates)
        latest_df = residuals_df_dict[latest_date]
        latest_df = latest_df.copy()
        latest_df = latest_df.reset_index()

        fig = go.Figure()
        # tenor_colors = {
        #     tenor: f"rgb({np.random.randint(0, 50)}, {np.random.randint(0, 50)}, {np.random.randint(100, 255)})" for tenor in tenors_to_plot
        # }
        tenor_colors = {tenor: "rgb(120, 154, 255)" for tenor in tenors_to_plot}

        for tenor in tenors_to_plot:
            groups = tenor.split(" ")
            if len(groups) == 3:
                fig.add_trace(
                    go.Bar(
                        x=[tenor],
                        y=[latest_df.loc[latest_df["Tenor"] == " ".join(groups[2:]), " ".join(groups[:2])].values[0]],
                        marker_color=tenor_colors[tenor],
                        name=f"{latest_date.strftime('%Y-%m-%d')}",
                        showlegend=False,
                    )
                )
            else:
                fig.add_trace(
                    go.Bar(
                        x=[tenor],
                        y=[latest_df.loc[latest_df["Tenor"] == tenor, "Spot"].values[0]],
                        marker_color=tenor_colors[tenor],
                        name=f"{latest_date.strftime('%Y-%m-%d')}",
                        showlegend=False,
                    )
                )

        if bday_offsets:
            bdate_offsets = []
            for offset in bday_offsets:
                bdate_offsets.append(latest_date - BDay(offset))
            alpha_values = [0.5, 0.35, 0.2]
            bar_widths = [0.6, 0.45, 0.3]
            for i, date in enumerate(bdate_offsets):
                df = residuals_df_dict[date]
                df = df.copy()
                df = df.reset_index()
                for tenor in tenors_to_plot:
                    groups = tenor.split(" ")
                    if len(groups) == 3:
                        fig.add_trace(
                            go.Bar(
                                x=[tenor],
                                y=[df.loc[df["Tenor"] == " ".join(groups[2:]), " ".join(groups[:2])].values[0]],
                                marker_color=tenor_colors[tenor],
                                marker_opacity=alpha_values[i],
                                name=f"{bdate_offsets[i].strftime('%Y-%m-%d')}",
                                width=bar_widths[i],
                                showlegend=False,
                            )
                        )
                    else:
                        fig.add_trace(
                            go.Bar(
                                x=[tenor],
                                y=[df.loc[latest_df["Tenor"] == tenor, "Spot"].values[0]],
                                marker_color=tenor_colors[tenor],
                                marker_opacity=alpha_values[i],
                                name=f"{bdate_offsets[i].strftime('%Y-%m-%d')}",
                                width=bar_widths[i],
                                showlegend=False,
                            )
                        )

        fig.update_layout(
            title=title,
            xaxis_title="Tenor",
            yaxis_title="Residuals (bps)",
            barmode="overlay",  # Bars will overlap
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, title="Dates"),
            height=750,
            font=dict(size=11),
            template="plotly_dark",
        )
        fig.update_xaxes(showspikes=True, spikecolor="white", spikesnap="cursor", spikemode="across", showgrid=True)
        fig.update_yaxes(showspikes=True, spikecolor="white", spikesnap="cursor", spikethickness=0.5, showgrid=True)
        fig.show(
            config={
                "modeBarButtonsToAdd": [
                    "drawline",
                    "drawopenpath",
                    "drawclosedpath",
                    "drawcircle",
                    "drawrect",
                    "eraseshape",
                ]
            }
        )

    @staticmethod
    def most_mispriced_pca_resid_zscores(
        df: pd.DataFrame,
        top_n: Optional[int] = 5,
        across_grid: Optional[bool] = False,
        curve_weights: Optional[Annotated[List[int], 2]] = [1, 1],
        fly_weights: Optional[Annotated[List[int], 3]] = [1, 2, 1],
        exclusive_tenors: Optional[Dict[str, List[str]]] = None,
        exclude_tenors: Optional[Dict[str, List[str]]] = None,
    ) -> Dict[Annotated[Literal["curve", "fly"], 2], pd.DataFrame]:
        if across_grid:
            points = [(tenor, fwd, df.loc[tenor, fwd]) for tenor in df.index for fwd in df.columns]

            curve_records = []
            for p1, p2 in itertools.combinations(points, 2):
                tenor1, fwd1, z1 = p1
                tenor2, fwd2, z2 = p2
                spread = (curve_weights[0] * z1) - (curve_weights[1] * z2)
                curve_records.append(
                    {
                        "Tenor1": f"{fwd1}x{tenor1}",
                        "Tenor2": f"{fwd2}x{tenor2}",
                        "ZScore-Spread": spread,
                        "Trade": "steepener" if spread > 0 else "flattener",
                        "Full Tenor": f"{fwd1} {tenor1}-{fwd2} {tenor2}",
                    }
                )

            curve_df = pd.DataFrame(curve_records)
            if exclude_tenors:
                for col, tenors_to_exclude in exclude_tenors.items():
                    if col in curve_df.columns:
                        curve_df = curve_df[~curve_df[col].isin(tenors_to_exclude)]
            top_curve_df = curve_df.sort_values("ZScore-Spread", ascending=False, key=lambda x: np.abs(x)).head(top_n).reset_index(drop=True)

            bfly_records = []
            for p1, p2, p3 in itertools.combinations(points, 3):
                tenor1, fwd1, z1 = p1
                tenor2, fwd2, z2 = p2
                tenor3, fwd3, z3 = p3
                bfly = (fly_weights[1] * z2) - (fly_weights[0] * z1) - (fly_weights[2] * z3)
                bfly_records.append(
                    {
                        "ShortWing": f"{fwd1}x{tenor1}",
                        "Belly": f"{fwd2}x{tenor2}",
                        "LongWing": f"{fwd3}x{tenor3}",
                        "Full Tenor": f"{fwd1} {tenor1}-{fwd2} {tenor2}-{fwd3} {tenor3}",
                        "ZScore-Spread": bfly,
                        "Trade": "rec belly" if bfly > 0 else "pay belly",
                    }
                )

            bfly_df = pd.DataFrame(bfly_records)
            if exclude_tenors:
                for col, tenors_to_exclude in exclude_tenors.items():
                    if col in bfly_df.columns:
                        bfly_df = bfly_df[~bfly_df[col].isin(tenors_to_exclude)]

            if exclusive_tenors:
                for col, exclusive_tenors in exclusive_tenors.items():
                    if col in bfly_df.columns:
                        bfly_df = bfly_df[bfly_df[col].isin(exclusive_tenors)]

            top_bfly_df = bfly_df.sort_values("ZScore-Spread", ascending=False, key=lambda x: np.abs(x)).head(top_n).reset_index(drop=True)

            return {"curve": top_curve_df, "fly": top_bfly_df}

        else:
            curve_records = []
            tenors = df.index.tolist()
            for col in df.columns:
                for t1, t2 in itertools.combinations(tenors, 2):
                    z1 = df.loc[t1, col]
                    z2 = df.loc[t2, col]
                    spread = (curve_weights[0] * z1) - (curve_weights[1] * z2)
                    curve_records.append(
                        {
                            "Forward": col,
                            "Tenor1": t1,
                            "Tenor2": t2,
                            "ZScore-Spread": spread,
                            "Trade": "steepener" if spread > 0 else "flattener",
                            "Full Tenor": f"{col} {t1}-{col} {t2}",
                        }
                    )

            curve_df = pd.DataFrame(curve_records)
            if exclude_tenors:
                for col, tenors_to_exclude in exclude_tenors.items():
                    if col in curve_df.columns:
                        curve_df = curve_df[~curve_df[col].isin(tenors_to_exclude)]
            top_curve_df = curve_df.sort_values("ZScore-Spread", ascending=False, key=lambda x: np.abs(x)).head(top_n).reset_index(drop=True)

            bfly_records = []
            for col in df.columns:
                for t_s, t_b, t_l in itertools.combinations(tenors, 3):
                    z_s = df.loc[t_s, col]
                    z_b = df.loc[t_b, col]
                    z_l = df.loc[t_l, col]
                    bfly = (fly_weights[1] * z_b) - (fly_weights[0] * z_s) - (fly_weights[2] * z_l)
                    bfly_records.append(
                        {
                            "Forward": col,
                            "ShortWing": t_s,
                            "Belly": t_b,
                            "LongWing": t_l,
                            "ZScore-Spread": bfly,
                            "Trade": "rec belly" if bfly > 0 else "pay belly",
                            "Full Tenor": f"{col} {t_s}-{col} {t_b}-{col} {t_l}",
                        }
                    )

            bfly_df = pd.DataFrame(bfly_records)
            if exclude_tenors:
                for col, tenors_to_exclude in exclude_tenors.items():
                    if col in bfly_df.columns:
                        bfly_df = bfly_df[~bfly_df[col].isin(tenors_to_exclude)]

            if exclusive_tenors:
                for col, exclusive_tenors in exclusive_tenors.items():
                    if col in bfly_df.columns:
                        bfly_df = bfly_df[bfly_df[col].isin(exclusive_tenors)]

            top_bfly_df = bfly_df.sort_values("ZScore-Spread", ascending=False, key=lambda x: np.abs(x)).head(top_n).reset_index(drop=True)

            return {"curve": top_curve_df, "fly": top_bfly_df}

    @staticmethod
    def backtester():
        pass

    @staticmethod
    def timeseries_builder(
        fwd_dict_df: Dict[datetime, pd.DataFrame],
        cols: List[str],
        bdates: Optional[List[pd.Timestamp] | List[datetime]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        tqdm_desc: Optional[str] = "BUILDING OIS SWAP TIMESERIES DF...",
        tenor_is_df_index: Optional[bool] = False,
        scale_curve: Optional[bool] = False,
        scale_fly: Optional[bool] = False,
        fifty_fifty_fly: Optional[bool] = False,
        custom_fly_weights: Optional[Annotated[List[float], 3]] = None,
    ) -> pd.DataFrame:
        if bdates is None and not start_date and not end_date:
            raise ValueError("Must specifiy 'bdates' or 'start_date' and 'end_date'")

        if bdates is None:
            bdates = pd.date_range(start=start_date, end=end_date, freq=CustomBusinessDay(calendar=USFederalHolidayCalendar()))

        dates = []
        to_plot: Dict[str, List[float]] = {}
        for bdate in tqdm.tqdm(bdates, desc=tqdm_desc):
            if not type(bdate) is datetime:
                bdate: pd.Timestamp = bdate
                bdate = bdate.to_pydatetime()

            try:
                curr_df = fwd_dict_df[bdate]
                if tenor_is_df_index:
                    curr_df = curr_df.copy()
                    curr_df = curr_df.reset_index()
                dates.append(bdate)

                for tenor in cols:
                    # spread case
                    if "-" in tenor:
                        legs = tenor.split("-")

                        # curve case
                        if len(legs) == 2:
                            leg1, leg2 = legs
                            if tenor not in to_plot:
                                to_plot[tenor] = []

                            group1 = leg1.split(" ")
                            group2 = leg2.split(" ")
                            if len(group1) == 3 and len(group2) == 3:
                                implied_fwd1 = " ".join(group1[:2])
                                leg1 = " ".join(group1[2:])
                                implied_fwd2 = " ".join(group2[:2])
                                leg2 = " ".join(group2[2:])
                            else:
                                implied_fwd1 = "Spot"
                                implied_fwd2 = "Spot"

                            spread = (
                                curr_df.loc[curr_df["Tenor"] == leg2, implied_fwd2].values[0]
                                - curr_df.loc[curr_df["Tenor"] == leg1, implied_fwd1].values[0]
                            )
                            if scale_curve:
                                spread = spread * 100

                            to_plot[tenor].append(spread)

                        # butterfly case
                        elif len(legs) == 3:
                            leg1, leg2, leg3 = legs
                            if tenor not in to_plot:
                                to_plot[tenor] = []

                            group1 = leg1.split(" ")
                            group2 = leg2.split(" ")
                            group3 = leg3.split(" ")
                            if len(group1) == 3 and len(group2) == 3:
                                implied_fwd1 = " ".join(group1[:2])
                                leg1 = " ".join(group1[2:])
                                implied_fwd2 = " ".join(group2[:2])
                                leg2 = " ".join(group2[2:])
                                implied_fwd3 = " ".join(group3[:2])
                                leg3 = " ".join(group3[2:])
                            else:
                                implied_fwd1 = "Spot"
                                implied_fwd2 = "Spot"
                                implied_fwd3 = "Spot"

                            if fifty_fifty_fly:
                                spread = (
                                    (1 * curr_df.loc[curr_df["Tenor"] == leg2, implied_fwd2].values[0])
                                    - (0.5 * curr_df.loc[curr_df["Tenor"] == leg1, implied_fwd1].values[0])
                                    - (0.5 * curr_df.loc[curr_df["Tenor"] == leg3, implied_fwd3].values[0])
                                )
                            elif custom_fly_weights:
                                spread = (
                                    (custom_fly_weights[1] * curr_df.loc[curr_df["Tenor"] == leg2, implied_fwd2].values[0])
                                    - (custom_fly_weights[0] * curr_df.loc[curr_df["Tenor"] == leg1, implied_fwd1].values[0])
                                    - (custom_fly_weights[2] * curr_df.loc[curr_df["Tenor"] == leg3, implied_fwd3].values[0])
                                )
                            else:
                                spread = (
                                    curr_df.loc[curr_df["Tenor"] == leg2, implied_fwd2].values[0]
                                    - curr_df.loc[curr_df["Tenor"] == leg1, implied_fwd1].values[0]
                                ) - (
                                    curr_df.loc[curr_df["Tenor"] == leg3, implied_fwd3].values[0]
                                    - curr_df.loc[curr_df["Tenor"] == leg2, implied_fwd2].values[0]
                                )
                            if scale_fly:
                                spread = spread * 100

                            to_plot[tenor].append(spread)

                        # bad format
                        else:
                            raise ValueError("Bad format passed in")
                    else:
                        groups = tenor.split(" ")
                        if len(groups) == 3:
                            fwd_tenor = tenor
                            if fwd_tenor not in to_plot:
                                to_plot[fwd_tenor] = []
                            to_plot[fwd_tenor].append(curr_df.loc[curr_df["Tenor"] == " ".join(groups[2:]), " ".join(groups[:2])].values[0])
                        else:
                            if tenor not in to_plot:
                                to_plot[tenor] = []
                            to_plot[tenor].append(curr_df.loc[curr_df["Tenor"] == tenor, "Spot"].values[0])

            except Exception as e:
                S490Swaps._logger.error(f"{tqdm_desc} Timeseries builder had an error at {bdate}: {e}")

        timeseries_dict = {"Date": dates}
        timeseries_dict = timeseries_dict | to_plot
        return pd.DataFrame(timeseries_dict)

    @staticmethod
    def _general_fwd_dict_df_timeseries_plotter(
        fwd_dict_df: Dict[datetime, pd.DataFrame],
        tenors_to_plot: List[str],
        bdates: List[pd.Timestamp] | List[datetime],
        tqdm_desc: str,
        tenors_to_plot_on_secondary_axis: Optional[List[str]] = None,
        custom_title: Optional[str] = None,
        yaxis_title: Optional[str] = None,
        use_plotly: Optional[bool] = False,
        tenor_is_df_index: Optional[bool] = False,
        custom_fly_weights: Optional[Annotated[List[float], 3]] = None,
        should_scale: Optional[bool] = True,
    ):
        df = S490Swaps.timeseries_builder(
            fwd_dict_df=fwd_dict_df,
            cols=tenors_to_plot + tenors_to_plot_on_secondary_axis if tenors_to_plot_on_secondary_axis else tenors_to_plot,
            bdates=bdates,
            tqdm_desc=tqdm_desc,
            tenor_is_df_index=tenor_is_df_index,
            scale_curve=should_scale,
            scale_fly=should_scale,
            custom_fly_weights=custom_fly_weights,
        )

        S490Swaps._general_df_plotter(
            df=df,
            cols_to_plot=tenors_to_plot,
            cols_to_plot_raxis=tenors_to_plot_on_secondary_axis,
            use_plotly=use_plotly,
            custom_title=custom_title,
            yaxis_title=yaxis_title,
        )

    @staticmethod
    def _general_df_plotter(
        df: pd.DataFrame,
        cols_to_plot: List[str],
        cols_to_plot_raxis: Optional[List[str]] = None,
        use_plotly: Optional[bool] = False,
        custom_title: Optional[str] = None,
        yaxis_title: Optional[str] = None,
        yaxis_title_r: Optional[str] = None,
        stds: Optional[Dict[str, List[int]]] = {},
        run_ou_mr_col_steps: Optional[Tuple[str, int, datetime | None]] = None,
        fill_ou_bands: Optional[bool] = False,
        show_1_sigma_ou_band: Optional[bool] = False,
        show_2_sigma_ou_band: Optional[bool] = False,
    ):
        if use_plotly:
            fig = go.Figure()
            for tenor in cols_to_plot:
                fig.add_trace(go.Scatter(x=df["Date"], y=df[tenor], mode="lines", name=f"{tenor} (lhs)", yaxis="y1"))
            if cols_to_plot_raxis:
                for tenor in cols_to_plot_raxis:
                    fig.add_trace(go.Scatter(x=df["Date"], y=df[tenor], mode="lines", name=f"{tenor} (rhs)", yaxis="y2"))

            fig.update_layout(
                title=(
                    custom_title or f"{" ".join(cols_to_plot)} (lhs) & {" ".join(cols_to_plot_raxis)} (rhs) Timeseries"
                    if cols_to_plot_raxis
                    else f"{" ".join(cols_to_plot)} Timeseries"
                ),
                xaxis_title="Date",
                yaxis=dict(title=yaxis_title if yaxis_title else ", ".join(cols_to_plot), side="left"),
                yaxis2=dict(
                    title=yaxis_title_r if yaxis_title_r else ", ".join(cols_to_plot_raxis) if cols_to_plot_raxis else None,
                    overlaying="y",
                    side="right",
                ),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, title="Tenors"),
                template="plotly_dark",
                font=dict(size=11),
                height=750,
            )

            fig.update_xaxes(showspikes=True, spikecolor="white", spikesnap="cursor", spikemode="across", showgrid=True)
            fig.update_yaxes(showspikes=True, spikecolor="white", spikesnap="cursor", spikethickness=0.5, showgrid=True)
            fig.show(
                config={
                    "modeBarButtonsToAdd": [
                        "drawline",
                        "drawopenpath",
                        "drawclosedpath",
                        "drawcircle",
                        "drawrect",
                        "eraseshape",
                    ]
                }
            )
        else:
            unique_colors = plt.cm.tab10.colors
            i = 0
            fig, ax_left = plt.subplots()

            lns = []
            for tenor in cols_to_plot:
                lns += ax_left.plot(
                    df["Date"],
                    df[tenor],
                    label=f"{tenor} (lhs)\nMost Recent: {df["Date"].iloc[-1].date()}, {np.round(df[tenor].iloc[-1], 3)}",
                    color=unique_colors[i],
                )
                i += 1

                if tenor in stds.keys():
                    level_std = tstd(df[tenor])
                    level_mean = np.mean(df[tenor])
                    lns += ax_left.plot(df["Date"], [level_mean] * len(df["Date"]), linestyle="--", color="red", label=f"Mean: {level_mean}")

                    for std in stds[tenor]:
                        curr_std_level = level_mean + (level_std * std)
                        curr_std_level_opp = level_mean + (level_std * std * -1)
                        curr = ax_left.plot(
                            df["Date"],
                            [curr_std_level] * len(df["Date"]),
                            linestyle="--",
                            label=f"± {std} STD: {np.round(curr_std_level, 3)}, {np.round(curr_std_level_opp, 3)}",
                            color="red",
                            alpha=0.75,
                        )
                        lns += curr
                        ax_left.plot(df["Date"], [curr_std_level_opp] * len(df["Date"]), linestyle="--", color=curr[0].get_color(), alpha=0.75)

            ax_left.set_ylabel(yaxis_title if yaxis_title else ", ".join(cols_to_plot))
            ax_left.tick_params(axis="y")

            if cols_to_plot_raxis:
                ax_right = ax_left.twinx()
                for tenor in cols_to_plot_raxis:
                    lns += ax_right.plot(
                        df["Date"],
                        df[tenor],
                        label=f"{tenor} (lhs)\nMost Recent: {df["Date"].iloc[-1].date()}, {np.round(df[tenor].iloc[-1], 3)}",
                        color=unique_colors[i],
                    )
                    i += 1

                    if tenor in stds.keys():
                        level_std = tstd(df[tenor])
                        level_mean = np.mean(df[tenor])
                        lns += ax_right.plot(df["Date"], [level_mean] * len(df["Date"]), linestyle="--", color="red", label=f"Mean: {level_mean}")

                        for std in stds[tenor]:
                            curr_std_level = level_mean + (level_std * std)
                            curr_std_level_opp = level_mean + (level_std * std * -1)
                            curr = ax_right.plot(
                                df["Date"],
                                [curr_std_level] * len(df["Date"]),
                                linestyle="--",
                                label=f"±{std} std: {np.round(curr_std_level, 3)}, {np.round(curr_std_level_opp, 3)}",
                                color="red",
                                alpha=0.75,
                            )
                            lns += curr
                            ax_right.plot(
                                df["Date"],
                                [curr_std_level_opp] * len(df["Date"]),
                                linestyle="--",
                                color=curr[0].get_color(),
                                alpha=0.75,
                            )

                ax_right.set_ylabel(yaxis_title_r if yaxis_title_r else ", ".join(cols_to_plot_raxis))
                ax_right.tick_params(axis="y")

            if run_ou_mr_col_steps:
                copy_df = df.copy()
                if len(run_ou_mr_col_steps) == 3:
                    tenor_for_ou, time_steps, dt_offset = run_ou_mr_col_steps
                    copy_df = copy_df[copy_df["Date"] <= dt_offset]
                else:
                    tenor_for_ou, time_steps = run_ou_mr_col_steps
                if not tenor_for_ou in cols_to_plot:
                    raise ValueError("OU Mean Reversion only supported for left axis plotting")

                ts_ou = copy_df.set_index("Date")[[tenor_for_ou]].dropna()
                ou_forecast, optimal_holding_period = simulate_mean_reversion_ou(ts_ou, steps=time_steps)

                base_case = np.round(ou_forecast["mean_reversion"].iloc[-1], 3)
                bull_case = np.round(ou_forecast["+2_sigma"].iloc[-1], 3)
                stop_loss = np.round(ou_forecast["-2_sigma"].iloc[-1], 3)

                lns += ax_left.plot(
                    ou_forecast.index,
                    ou_forecast["mean_reversion"],
                    label=f"OU Mean Reversion (t+1)\nTarget: {base_case}, Bull Case: {bull_case}, Stop-loss: {stop_loss}\nOptimal Holding Period: {np.ceil(optimal_holding_period)}d",
                    color="green",
                    linestyle="dashdot",
                )
                if fill_ou_bands:
                    ax_left.fill_between(
                        ou_forecast.index,
                        ou_forecast["+2_sigma"],
                        ou_forecast["-2_sigma"],
                        color="tab:green",
                        alpha=0.25,
                        label=f"±2 sigma corridor ({tenor_for_ou})",
                    )
                else:
                    if show_1_sigma_ou_band:
                        curr = ax_left.plot(ou_forecast.index, ou_forecast["+1_sigma"], color="green", alpha=0.7, linestyle="dashdot")
                        lns += curr
                        ax_left.plot(
                            ou_forecast.index,
                            ou_forecast["-1_sigma"],
                            label=f"OU Mean Reversion -1 Sigma",
                            color=curr[0].get_color(),
                            alpha=0.7,
                            linestyle="dashdot",
                        )

                    if show_2_sigma_ou_band:
                        curr = ax_left.plot(ou_forecast.index, ou_forecast["+2_sigma"], color="green", alpha=0.7, linestyle="dashdot")
                        lns += curr
                        ax_left.plot(
                            ou_forecast.index,
                            ou_forecast["-2_sigma"],
                            label=f"OU Mean Reversion -2 Sigma",
                            color=curr[0].get_color(),
                            alpha=0.7,
                            linestyle="dashdot",
                        )

            locator = mdates.AutoDateLocator(minticks=10, maxticks=20)
            formatter = mdates.DateFormatter("%Y-%m-%d")
            ax_left.xaxis.set_major_locator(locator)
            ax_left.xaxis.set_major_formatter(formatter)
            ax_left.set_xlabel("Date")

            labs = [l.get_label() for l in lns]
            ax_left.legend(lns, labs, loc=(0, 0))
            plt.title(
                custom_title or f"{' '.join(cols_to_plot)} (lhs) & {' '.join(cols_to_plot_raxis)} (rhs) Timeseries"
                if cols_to_plot_raxis
                else f"{' '.join(cols_to_plot)} Timeseries"
            )
            ax_left.grid(True)
            plt.xticks(rotation=15)
            fig.tight_layout()
            plt.show()
