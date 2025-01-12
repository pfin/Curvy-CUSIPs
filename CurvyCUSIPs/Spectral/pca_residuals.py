import logging
from datetime import datetime
from typing import Dict, List, Optional

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import plotly.graph_objects as go

import numpy as np
import pandas as pd
import QuantLib as ql
import tqdm
from joblib import Parallel, delayed
from scipy.stats import zscore
from sklearn.decomposition import PCA
from termcolor import colored

from CurvyCUSIPs.Spectral.R2PCA import R2PCA


class PCAGridResiduals:
    _timeseries_grids: Dict[datetime, pd.DataFrame]
    _master_date_col = "Date"
    _master_residual_col = "Residual"
    _master_tenor_col = "Tenor"
    _fwd_start_delimitter = "Fwd"
    _melted_val_name = "Rate"
    _melted_var_name = "FwdType"

    _logger = logging.getLogger(__name__)
    _debug_verbose: bool = False
    _error_verbose: bool = False
    _info_verbose: bool = False
    _no_logs_plz: bool = False

    def __init__(
        self,
        timeseries_grids: Dict[datetime, pd.DataFrame],
        tenor_col: Optional[str] = "Tenor",
        residual_col: Optional[str] = "Residual",
        date_col: Optional[str] = "Date",
        fwd_start_delimitter: Optional[str] = "Fwd",
        tenor_col_is_index: Optional[bool] = False,
        debug_verbose: Optional[bool] = False,
        info_verbose: Optional[bool] = False,
        error_verbose: Optional[bool] = False,
    ):
        self._debug_verbose = debug_verbose
        self._error_verbose = error_verbose
        self._info_verbose = info_verbose
        self._no_logs_plz = not debug_verbose and not error_verbose and not info_verbose
        self._setup_logger()

        self._timeseries_grids = timeseries_grids.copy()
        self._master_tenor_col = tenor_col
        self._master_date_col = date_col
        self._master_residual_col = residual_col
        self._fwd_start_delimitter = fwd_start_delimitter

        first_df = next(iter(self._timeseries_grids.values()))
        if first_df.index.name:
            tenor_col_is_index = True

        if tenor_col_is_index:
            for dt, df in self._timeseries_grids.items():
                self._timeseries_grids[dt] = df.reset_index()

        df_tenor_col = next(iter(self._timeseries_grids.values())).columns[0]
        if df_tenor_col != self._master_tenor_col:
            self._logger.warning(colored(f"WARNING: _master_tenor_col mismatch: {self._master_tenor_col} vs {df_tenor_col}", "yellow"))
            self._master_tenor_col = df_tenor_col

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

    def _ql_period_wrapper(self, x) -> ql.Period:
        if self._fwd_start_delimitter in x:
            return ql.Period(str(x).split(f" {self._fwd_start_delimitter}")[0])
        elif "Spot" in x:
            return ql.Period("0D")
        return ql.Period(x)

    def _merge_residuals_by_date(self, fwd_residual_dict: Dict[str, Dict[datetime, pd.DataFrame]]) -> Dict[datetime, pd.DataFrame]:
        date_dict = {}
        for fwd_type, date_map in fwd_residual_dict.items():
            for dt, sub_df in date_map.items():
                if dt not in date_dict:
                    date_dict[dt] = pd.DataFrame(index=sub_df.index)
                date_dict[dt][fwd_type] = sub_df["Residual"]

        for dt, df in date_dict.items():
            df = df[sorted(df.columns, key=lambda x: self._ql_period_wrapper(x))]
            date_dict[dt] = df

        return date_dict

    def _melt_and_pivot_timeseries_grids(
        self,
        timeseries_grids: Dict[datetime, pd.DataFrame],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        remove_underlying_tenors: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        df_list = []
        for dt, df in timeseries_grids.items():
            tmp = df.copy()
            tmp.insert(0, self._master_date_col, dt)
            df_list.append(tmp)

        big_df = pd.concat(df_list, ignore_index=True)
        if remove_underlying_tenors:
            big_df = big_df[~big_df[self._master_tenor_col].isin(remove_underlying_tenors)]
        if start_date:
            big_df = big_df[big_df[self._master_date_col] >= start_date]
        if end_date:
            big_df = big_df[big_df[self._master_date_col] <= end_date]

        melted_df = big_df.melt(
            id_vars=[self._master_date_col, self._master_tenor_col],
            value_vars=list(big_df.columns[2:]),
            var_name=self._melted_var_name,
            value_name=self._melted_val_name,
        )
        long_df = melted_df.pivot(index=self._master_date_col, columns=[self._master_tenor_col, self._melted_var_name], values=self._melted_val_name)

        return long_df

    def _calculate_cross_df_zscores(self, dict_df: Dict[datetime, pd.DataFrame]) -> List[pd.DataFrame]:
        dict_df = dict(sorted(dict_df.items()))
        df_list = list(dict_df.values())
        array_3d = np.stack([df.values for df in df_list])
        zscores = zscore(array_3d, axis=0, nan_policy="omit")
        zscore_dfs = []
        for i in range(len(df_list)):
            zscore_df = pd.DataFrame(zscores[i], index=df_list[0].index, columns=df_list[0].columns)
            zscore_dfs.append(zscore_df)
        return zscore_dfs

    # Two-levels of parallelism (PCA calc worker, across timeseries dict)
    def __double_parallel_individual_cols_rolling_window(
        self,
        long_df: pd.DataFrame,
        rolling_window: int,
        n_components: int,
        n_jobs_parent: int,
        n_jobs_child: int,
        run_on_level_changes: Optional[bool] = False,
    ):
        sorted_dates = sorted(long_df.index)
        if len(sorted_dates) < rolling_window:
            raise ValueError(f"Not enough observations to do rolling PCA of window size={rolling_window}.")

        fwd_types = long_df.columns.levels[1]
        rolling_pca_results_dict = {}
        master_residuals_for_anchor = pd.DataFrame(index=long_df.index, columns=long_df.columns, dtype=float)

        # inner worker for anchor_idx
        def ___rolling_pca_worker(sub_df: pd.DataFrame, sorted_dates_, anchor_idx: int, run_on_level_changes: bool):
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
            residuals_long.columns = [self._master_date_col, self._master_tenor_col, self._master_residual_col]
            residuals_timeseries_dict = {}
            for dt_, sub_sub_df in residuals_long.groupby(self._master_date_col):
                pivoted = sub_sub_df.pivot_table(index=self._master_tenor_col, values=self._master_residual_col, aggfunc="first").reset_index()
                pivoted["period_temp"] = pivoted[self._master_tenor_col].apply(lambda x: ql.Period(x))
                pivoted = pivoted.sort_values(by="period_temp").drop(columns=["period_temp"]).reset_index(drop=True).set_index(self._master_tenor_col)
                residuals_timeseries_dict[dt_] = pivoted * 100

            residual_zscores = self._calculate_cross_df_zscores(residuals_timeseries_dict)
            return (
                anchor_date,
                {
                    "explained_variance_ratio": pca.explained_variance_ratio_,
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

        # Outer Worker
        def ___outer_worker_pca_for_indy_col(fwd_type: str):
            sub_df = long_df.xs(fwd_type, axis=1, level=self._melted_var_name).loc[sorted_dates]
            anchor_idx_values = range(rolling_window - 1, len(sorted_dates))

            # parallelize across anchor_idx here, but limit to n_jobs_child.
            par_results = Parallel(n_jobs=n_jobs_child)(
                delayed(___rolling_pca_worker)(sub_df, sorted_dates, aidx, run_on_level_changes) for aidx in anchor_idx_values
            )
            return fwd_type, dict(par_results)

        fwd_types_results = [
            result
            for result in tqdm.tqdm(
                Parallel(return_as="generator", n_jobs=n_jobs_parent)(delayed(___outer_worker_pca_for_indy_col)(fwd_type) for fwd_type in fwd_types),
                desc="ROLLING PCA ON INDY COLS..." if not run_on_level_changes else "ROLLING PCA ON INDY COLS TIME STEP LEVEL CHANGES...",
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
                        curr_dict[fwd_type] = rc_anchor[self._master_residual_col]

            curr_df = pd.DataFrame(curr_dict)
            if not curr_df.empty:
                cols_sorted = sorted(curr_df.columns, key=lambda x: self._ql_period_wrapper(x))
                curr_df = curr_df[cols_sorted]
            rich_cheap_residual_zscore_timeseries_dict[anchor_date] = curr_df

        return {
            "rolling_windows": rolling_window,
            "rolling_pca_results_per_fwd": rolling_pca_results_dict,
            "master_anchor_date_residuals": master_residuals_for_anchor,
            "rich_cheap_residual_zscore_timeseries_dict": rich_cheap_residual_zscore_timeseries_dict,
        }

    def __parallel_individual_cols_rolling_window(
        self,
        long_df: pd.DataFrame,
        rolling_window: int,
        n_components: int,
        n_jobs: int,
        run_on_level_changes: Optional[bool] = False,
    ):
        sorted_dates = sorted(long_df.index)
        if len(sorted_dates) < rolling_window:
            raise ValueError(f"Not enough observations to do rolling PCA of window size={rolling_window}.")

        fwd_types = long_df.columns.levels[1]
        rolling_pca_results_dict = {}
        master_residuals_for_anchor = pd.DataFrame(index=long_df.index, columns=long_df.columns, dtype=float)

        for fwd_type in tqdm.tqdm(
            fwd_types,
            "ROLLING PCA ON INDY COLS..." if not run_on_level_changes else "ROLLING PCA ON INDY COLS TIME STEP LEVEL CHANGES...",
            total=len(fwd_types),
        ):
            sub_df = long_df.xs(fwd_type, axis=1, level=self._melted_var_name).loc[sorted_dates]

            def ___rolling_pca_worker(anchor_idx: int, run_on_level_changes: bool):
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
                residuals_long.columns = [self._master_date_col, self._master_tenor_col, self._master_residual_col]
                residuals_timeseries_dict = {}
                for dt_, sub_sub_df in residuals_long.groupby(self._master_date_col):
                    pivoted = sub_sub_df.pivot_table(index=self._master_tenor_col, values=self._master_residual_col, aggfunc="first").reset_index()
                    pivoted["period_temp"] = pivoted[self._master_tenor_col].apply(lambda x: ql.Period(x))
                    pivoted = (
                        pivoted.sort_values(by="period_temp").drop(columns=["period_temp"]).reset_index(drop=True).set_index(self._master_tenor_col)
                    )
                    residuals_timeseries_dict[dt_] = pivoted * 100

                residual_zscores = self._calculate_cross_df_zscores(residuals_timeseries_dict)
                return (
                    anchor_date,
                    {
                        "explained_variance_ratio": pca.explained_variance_ratio_,
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
            if not self._no_logs_plz:
                par_results = Parallel(n_jobs=n_jobs)(
                    delayed(___rolling_pca_worker)(aidx, run_on_level_changes)
                    for aidx in tqdm.tqdm(anchor_idx_values, desc=f"{fwd_type} STRIP ROLLING PCA CALC...")
                )
            else:
                par_results = Parallel(n_jobs=n_jobs)(delayed(___rolling_pca_worker)(aidx, run_on_level_changes) for aidx in anchor_idx_values)

            fwd_type_dict = dict(par_results)
            for anchor_date, res_dict in fwd_type_dict.items():
                for tenor_col in res_dict["anchor_resid"].columns:
                    master_residuals_for_anchor.loc[anchor_date, (tenor_col, fwd_type)] = res_dict["anchor_resid"].loc[anchor_date, tenor_col]

            rolling_pca_results_dict[fwd_type] = fwd_type_dict

        rich_cheap_residual_zscore_timeseries_dict: Dict[datetime, pd.DataFrame] = {}
        tenors = rolling_pca_results_dict.keys()
        for anchor_date in long_df.index:
            curr_dict = {}
            for tenor in tenors:
                if anchor_date in rolling_pca_results_dict[tenor]:
                    curr_dict[tenor] = rolling_pca_results_dict[tenor][anchor_date]["rich_cheap_zscore_anchor"][self._master_residual_col]

            curr_df = pd.DataFrame(curr_dict)
            if not curr_df.empty:
                cols_sorted = sorted(curr_df.columns, key=lambda x: self._ql_period_wrapper(x))
                curr_df = curr_df[cols_sorted]
            rich_cheap_residual_zscore_timeseries_dict[anchor_date] = curr_df

        return {
            "rolling_windows": rolling_window,
            "rolling_pca_results_per_fwd": rolling_pca_results_dict,
            "master_anchor_date_residuals": master_residuals_for_anchor,
            "rich_cheap_residual_zscore_timeseries_dict": rich_cheap_residual_zscore_timeseries_dict,
        }

    def __parallel_across_grid_rolling_window(
        self,
        long_df: pd.DataFrame,
        rolling_window: int,
        n_components: int,
        n_jobs: int,
        run_on_level_changes: Optional[bool] = False,
    ):
        sorted_dates = sorted(long_df.index)
        if len(sorted_dates) < rolling_window:
            raise ValueError(f"Not enough observations to do rolling PCA of window size={rolling_window}.")
        long_df = long_df.loc[sorted_dates]

        def ___rolling_pca_worker(anchor_idx: int, run_on_level_changes: bool):
            anchor_date = sorted_dates[anchor_idx]
            window_dates = sorted_dates[anchor_idx - rolling_window + 1 : anchor_idx + 1]
            window_data = long_df.loc[window_dates]

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
                reconstructed_window = reconst_changes_df.cumsum() + window_data.iloc[0]
            else:
                pca.fit(window_data)
                reconstructed_window = pd.DataFrame(
                    pca.inverse_transform(pca.transform(window_data)),
                    index=window_data.index,
                    columns=window_data.columns,
                )

            residuals_window = window_data - reconstructed_window
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

            residuals_window = window_data - reconstructed_window
            anchor_resid = residuals_window.loc[[anchor_date]]
            residuals_long = residuals_window.stack([self._master_tenor_col, self._melted_var_name]).reset_index()
            residuals_long.columns = [self._master_date_col, self._master_tenor_col, self._melted_var_name, self._master_residual_col]
            residuals_timeseries_dict = {}
            for dt_, sub_df in residuals_long.groupby(self._master_date_col):
                pivoted = sub_df.pivot(index=self._master_tenor_col, columns=self._melted_var_name, values=self._master_residual_col)
                pivoted = pivoted.reset_index()
                pivoted["period_temp"] = pivoted[self._master_tenor_col].apply(lambda x: ql.Period(x))
                pivoted = pivoted.sort_values(by="period_temp").drop(columns=["period_temp"]).reset_index(drop=True).set_index(self._master_tenor_col)
                pivoted = pivoted[sorted(pivoted.columns, key=lambda x: self._ql_period_wrapper(x))]
                residuals_timeseries_dict[dt_] = pivoted * 100

            residual_zscores = self._calculate_cross_df_zscores(residuals_timeseries_dict)
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
        par_results = Parallel(n_jobs=n_jobs)(
            delayed(___rolling_pca_worker)(aidx, run_on_level_changes) for aidx in tqdm.tqdm(anchor_idx_values, desc="ROLLING PCA ACROSS GRIDS...")
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

    def __parallel_individual_cols_all(
        self,
        long_df: pd.DataFrame,
        n_components: int,
        n_jobs: int,
        run_on_level_changes: Optional[bool] = False,
    ):
        def ___pca_worker_indy_col(fwd_type: str, run_on_level_changes: bool):
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
            residuals_long.columns = [self._master_date_col, self._master_tenor_col, self._master_residual_col]
            residuals_timeseries_dict = {}
            for obs_date, sub_sub_df in residuals_long.groupby(self._master_date_col):
                pivoted = sub_sub_df.pivot_table(index=self._master_tenor_col, values=self._master_residual_col, aggfunc="first").reset_index()
                pivoted["period_temp"] = pivoted[self._master_tenor_col].apply(lambda x: ql.Period(x))
                pivoted = pivoted.sort_values(by="period_temp").drop(columns=["period_temp"]).reset_index(drop=True).set_index(self._master_tenor_col)
                residuals_timeseries_dict[obs_date] = pivoted * 100

            residual_zscores = self._calculate_cross_df_zscores(residuals_timeseries_dict)
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
            delayed(___pca_worker_indy_col)(fwd_type, run_on_level_changes) for fwd_type in tqdm.tqdm(fwd_types, desc="PCA ON INDY COLS...")
        )

        for fwd_type, pca_dict, reconstructed_sub, residuals_sub in tqdm.tqdm(par_results, desc="CLEANING UP PCA ON INDY COLS RESULTS..."):
            pca_results_dict[fwd_type] = pca_dict
            for tenor_col in reconstructed_sub.columns:
                reconstructed_all[(tenor_col, fwd_type)] = reconstructed_sub[tenor_col]
                residuals_all[(tenor_col, fwd_type)] = residuals_sub[tenor_col]

        residuals_long_all = residuals_all.stack([self._master_tenor_col, self._melted_var_name]).reset_index()
        residuals_long_all.columns = [self._master_date_col, self._master_tenor_col, self._melted_var_name, self._master_residual_col]

        heatmap_dict = {}
        for fwd_type in pca_results_dict.keys():
            heat = pca_results_dict[fwd_type]["rich_cheap_zscore_heatmap"]
            if heat is not None:
                heatmap_dict[fwd_type] = heat
        rich_cheap_map_df = pd.concat(heatmap_dict, axis=1)
        rich_cheap_map_df.columns = rich_cheap_map_df.columns.get_level_values(0)
        rich_cheap_map_df = rich_cheap_map_df[sorted(rich_cheap_map_df.columns, key=lambda x: self._ql_period_wrapper(x))]

        return {
            "multi_pca_results_per_fwd": pca_results_dict,
            "reconstructed_df": reconstructed_all,
            "residuals_df": residuals_all,
            "residual_timeseries_dict": self._merge_residuals_by_date(
                {fwd_type: pca_results_dict[fwd_type]["residual_timeseries_dict"] for fwd_type in pca_results_dict.keys()}
            ),
            "rich_cheap_residual_zscore_timeseries_dict": self._merge_residuals_by_date(
                {fwd_type: pca_results_dict[fwd_type]["residual_timeseries_zscore_dict"] for fwd_type in pca_results_dict.keys()}
            ),
            "rich_cheap_zscore_heatmap": rich_cheap_map_df,
        }

    def __single_across_grid_all(self, long_df: pd.DataFrame, n_components: int, run_on_level_changes: Optional[bool] = False):
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
        residuals_long = residuals_df.stack([self._master_tenor_col, self._melted_var_name]).reset_index()
        residuals_long.columns = [self._master_date_col, self._master_tenor_col, self._melted_var_name, self._master_residual_col]
        residuals_timeseries_dict = {}
        for obs_date, sub_df in residuals_long.groupby(self._master_date_col):
            pivoted = sub_df.pivot(index=self._master_tenor_col, columns=self._melted_var_name, values=self._master_residual_col).reset_index()
            pivoted.columns.name = None
            pivoted["period_temp"] = pivoted[self._master_tenor_col].apply(lambda x: ql.Period(x))
            pivoted = pivoted.sort_values(by="period_temp").drop(columns=["period_temp"]).reset_index(drop=True).set_index(self._master_tenor_col)
            pivoted = pivoted[sorted(pivoted.columns, key=lambda x: self._ql_period_wrapper(x))]
            residuals_timeseries_dict[obs_date] = pivoted * 100

        residual_zscores = self._calculate_cross_df_zscores(residuals_timeseries_dict)
        return {
            "pca": pca,
            "loading_df": loadings_df,
            "scores_df": scores_df,
            "reconstructed_df": reconstructed_df,
            "residuals_df": residuals_df,
            "residual_timeseries_dict": residuals_timeseries_dict,
            "rich_cheap_residual_zscore_timeseries_dict": dict(zip(residuals_timeseries_dict.keys(), residual_zscores)),
        }
    
    def __single_r2pca_across_grid_all(
        self,
        long_df: pd.DataFrame,
        n_components: int,
        run_on_level_changes: bool = False,
    ):
        sorted_dates = sorted(long_df.index)
        data_df = long_df.loc[sorted_dates]  

        X = data_df.values  
        T_, D_ = X.shape
        X = X.reshape((1, T_, D_))  

        if run_on_level_changes:
            X_diff = np.diff(X, axis=1)
            r2pca = R2PCA(n_components=n_components, window_size=X_diff.shape[1])
            r2pca.fit(X_diff)
            changes_transformed = r2pca.transform(X_diff)   
            changes_recon = r2pca.inverse_transform(changes_transformed)  

            X_recon = np.zeros_like(X)  
            X_recon[:, 0] = X[:, 0]

            for i in range(1, T_):
                X_recon[:, i] = X_recon[:, i-1] + changes_recon[:, i-1]

            residuals = X - X_recon

            final_loadings = r2pca.components_[-1]  
            loading_df = pd.DataFrame(
                final_loadings,
                index=[f"PC{i+1}" for i in range(n_components)],
                columns=data_df.columns,
            )

            all_scores = changes_transformed[0]  
            score_idx = sorted_dates[1:]  
            scores_df = pd.DataFrame(
                all_scores,
                index=score_idx,
                columns=[f"PC{i+1}" for i in range(n_components)],
            )

        else:
            r2pca = R2PCA(n_components=n_components, window_size=T_)  
            r2pca.fit(X)
            X_transformed = r2pca.transform(X)      
            X_recon = r2pca.inverse_transform(X_transformed)  
            residuals = X - X_recon

            final_loadings = r2pca.components_[-1]  
            loading_df = pd.DataFrame(
                final_loadings,
                index=[f"PC{i+1}" for i in range(n_components)],
                columns=data_df.columns,
            )

            all_scores = X_transformed[0]  
            scores_df = pd.DataFrame(
                all_scores,
                index=sorted_dates,
                columns=[f"PC{i+1}" for i in range(n_components)],
            )

        residuals_df = pd.DataFrame(
            residuals[0],
            index=sorted_dates,
            columns=data_df.columns,
        )

        residuals_long = residuals_df.stack([self._master_tenor_col, self._melted_var_name]).reset_index()
        residuals_long.columns = [self._master_date_col, self._master_tenor_col, self._melted_var_name, self._master_residual_col]
        residuals_timeseries_dict = {}
        for obs_date, sub_df in residuals_long.groupby(self._master_date_col):
            pivoted = sub_df.pivot(index=self._master_tenor_col, columns=self._melted_var_name, values=self._master_residual_col).reset_index()
            pivoted.columns.name = None
            pivoted["period_temp"] = pivoted[self._master_tenor_col].apply(lambda x: ql.Period(x))
            pivoted = pivoted.sort_values(by="period_temp").drop(columns=["period_temp"]).reset_index(drop=True).set_index(self._master_tenor_col)
            pivoted = pivoted[sorted(pivoted.columns, key=lambda x: self._ql_period_wrapper(x))]
            residuals_timeseries_dict[obs_date] = pivoted * 100

        residual_zscores = self._calculate_cross_df_zscores(residuals_timeseries_dict)

        return {
            "r2pca_model": r2pca,
            "loading_df": loading_df,     
            "scores_df": scores_df,       
            "reconstructed_df": pd.DataFrame(X_recon[0], index=sorted_dates, columns=data_df.columns),
            "residuals_df": residuals_df,
            "residual_timeseries_dict": residuals_timeseries_dict,
            "rich_cheap_residual_zscore_timeseries_dict": dict(zip(residuals_timeseries_dict.keys(), residual_zscores)),
        }

    def runner(
        self,
        rolling_window: Optional[int] = None,
        run_on_indy_cols: Optional[bool] = False,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        remove_underlying_tenors: Optional[List[str]] = None,
        n_components: Optional[int] = 3,
        run_on_level_changes: Optional[bool] = False,
        use_r2_algo: Optional[bool] = False,  # TODO
        n_jobs: Optional[int] = None,
        n_jobs_parent: Optional[int] = None,
        n_jobs_child: Optional[int] = None,
    ):
        long_df = self._melt_and_pivot_timeseries_grids(
            timeseries_grids=self._timeseries_grids, start_date=start_date, end_date=end_date, remove_underlying_tenors=remove_underlying_tenors
        )

        if rolling_window:
            if run_on_indy_cols:
                if n_jobs_parent and n_jobs_child:
                    return self.__double_parallel_individual_cols_rolling_window(
                        long_df=long_df,
                        rolling_window=rolling_window,
                        n_components=n_components,
                        run_on_level_changes=run_on_level_changes,
                        n_jobs_parent=n_jobs_parent,
                        n_jobs_child=n_jobs_child,
                    )

                return self.__parallel_individual_cols_rolling_window(
                    long_df=long_df,
                    rolling_window=rolling_window,
                    n_components=n_components,
                    run_on_level_changes=run_on_level_changes,
                    n_jobs=n_jobs or 1,
                )

            return self.__parallel_across_grid_rolling_window(
                long_df=long_df,
                rolling_window=rolling_window,
                n_components=n_components,
                n_jobs=n_jobs or 1,
                run_on_level_changes=run_on_level_changes,
            )
        
        if use_r2_algo:
            return self.__single_r2pca_across_grid_all(long_df=long_df, n_components=n_components, run_on_level_changes=run_on_level_changes)

        if run_on_indy_cols:
            return self.__parallel_individual_cols_all(
                long_df=long_df, n_components=n_components, n_jobs=n_jobs or 1, run_on_level_changes=run_on_level_changes
            )

        return self.__single_across_grid_all(long_df=long_df, n_components=n_components, run_on_level_changes=run_on_level_changes)


    # @staticmethod
    # def pca_residual_timeseries_plotter(
    #     pca_results: Dict,
    #     tenors_to_plot: List[str],
    #     use_plotly: Optional[bool] = False,
    #     key: Optional[str] = "residual_timeseries_dict",
    #     custom_fly_weights: Optional[Annotated[List[float], 3]] = None,
    # ):
    #     tenors_to_plot = list(set(tenors_to_plot))
    #     residuals_df_dict: Dict[datetime, pd.DataFrame] = pca_results[key]
    #     S490Swaps._general_fwd_dict_df_timeseries_plotter(
    #         fwd_dict_df=residuals_df_dict,
    #         tenors_to_plot=tenors_to_plot,
    #         bdates=list([key for key in residuals_df_dict.keys() if residuals_df_dict[key] is not None and not residuals_df_dict[key].empty]),
    #         tqdm_desc="PLOTTING PCA RESIDUAL ZSCORES",
    #         custom_title=f"PCA Residuals: {tenors_to_plot[0] if len(tenors_to_plot) == 0 else ", ".join(tenors_to_plot)}",
    #         yaxis_title="Residuals (bps)",
    #         tenor_is_df_index=True,
    #         use_plotly=use_plotly,
    #         custom_fly_weights=custom_fly_weights,
    #         should_scale=False,
    #     )


    def _timeseries_dict_df_plotter(
        self,
        dict_df: Dict[datetime, pd.DataFrame],
        tenors_to_plot: List[str],
        bdates: List[pd.Timestamp] | List[datetime],
        tqdm_desc: str,
        custom_title: Optional[str] = None,
        yaxis_title: Optional[str] = None,
        use_plotly: Optional[bool] = False,
        should_scale_spreads: Optional[bool] = False,
    ):
        dates = []
        to_plot: Dict[str, List[float]] = {}
        for bdate in tqdm.tqdm(bdates, desc=tqdm_desc):
            if not type(bdate) is datetime:
                bdate: pd.Timestamp = bdate
                bdate = bdate.to_pydatetime()

            try:
                curr_df = dict_df[bdate]
                dates.append(bdate)

                for tenor in tenors_to_plot:
                    if "-" in tenor:
                        pass 
                        # legs = tenor.split("-")
                        # if len(legs) == 2:
                        #     leg1, leg2 = legs
                        #     if tenor not in to_plot:
                        #         to_plot[tenor] = []

                        #     group1 = leg1.split(" ")
                        #     group2 = leg2.split(" ")
                        #     if len(group1) == 3 and len(group2) == 3:
                        #         implied_fwd1 = " ".join(group1[:2])
                        #         leg1 = " ".join(group1[2:])
                        #         implied_fwd2 = " ".join(group2[:2])
                        #         leg2 = " ".join(group2[2:])
                        #     else:
                        #         implied_fwd1 = "Spot"
                        #         implied_fwd2 = "Spot"

                        #     spread = (
                        #         curr_df.loc[curr_df["Tenor"] == leg2, implied_fwd2].values[0]
                        #         - curr_df.loc[curr_df["Tenor"] == leg1, implied_fwd1].values[0]
                        #     )
                        #     if should_scale_spreads:
                        #         spread = spread * 100
                        #     to_plot[tenor].append(spread)
                        # elif len(legs) == 3:
                        #     leg1, leg2, leg3 = legs
                        #     if tenor not in to_plot:
                        #         to_plot[tenor] = []

                        #     group1 = leg1.split(" ")
                        #     group2 = leg2.split(" ")
                        #     group3 = leg3.split(" ")
                        #     if len(group1) == 3 and len(group2) == 3:
                        #         implied_fwd1 = " ".join(group1[:2])
                        #         leg1 = " ".join(group1[2:])
                        #         implied_fwd2 = " ".join(group2[:2])
                        #         leg2 = " ".join(group2[2:])
                        #         implied_fwd3 = " ".join(group3[:2])
                        #         leg3 = " ".join(group3[2:])
                        #     else:
                        #         implied_fwd1 = "Spot"
                        #         implied_fwd2 = "Spot"
                        #         implied_fwd3 = "Spot"

                        #     spread = (
                        #         curr_df.loc[curr_df["Tenor"] == leg2, implied_fwd2].values[0]
                        #         - curr_df.loc[curr_df["Tenor"] == leg1, implied_fwd1].values[0]
                        #         - curr_df.loc[curr_df["Tenor"] == leg3, implied_fwd3].values[0]
                        #         - curr_df.loc[curr_df["Tenor"] == leg2, implied_fwd2].values[0]
                        #     )
                        #     to_plot[tenor].append(spread)
                        # else:
                        #     raise ValueError("Bad tenor passed in")
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
                self._logger.error(f"{tqdm_desc} Timeseries plotter had an error at {bdate}: {e}")

        if use_plotly:
            fig = go.Figure()
            for tenor, rates in to_plot.items():
                fig.add_trace(go.Scatter(x=dates, y=rates, mode="lines", name=tenor))
            fig.update_layout(
                title=custom_title or "SOFR OIS Plot",
                xaxis_title="Date",
                yaxis_title=yaxis_title or "bps",
                legend_title="Tenors",
                font=dict(size=11),
                template="plotly_dark",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
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
            for tenor, rates in to_plot.items():
                plt.plot(
                    dates,
                    rates,
                    label=tenor,
                )

            ax = plt.gca()
            locator = mdates.AutoDateLocator(minticks=10, maxticks=20)
            formatter = mdates.DateFormatter("%Y-%m-%d")
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(formatter)

            plt.xlabel("Date")
            plt.ylabel(yaxis_title or "bps")
            plt.title(custom_title or "SOFR OIS Plot")
            plt.legend(fontsize="large")
            plt.grid(True)
            plt.xticks(rotation=15)
            plt.show()


# @staticmethod
# def pca_residual_credit_suisse_BBar_plot(
#     pca_results: Dict,
#     tenors_to_plot: Optional[List[str]] = [  # TODO assign tenors_to_plot based on tenors in pca_results
#         "12M Fwd 12M",
#         "12M Fwd 18M",
#         "12M Fwd 2Y",
#         "12M Fwd 3Y",
#         "12M Fwd 4Y",
#         "12M Fwd 5Y",
#         "12M Fwd 6Y",
#         "12M Fwd 7Y",
#         "12M Fwd 8Y",
#         "12M Fwd 9Y",
#         "12M Fwd 10Y",
#         "12M Fwd 12Y",
#         "12M Fwd 15Y",
#         "12M Fwd 20Y",
#         "12M Fwd 25Y",
#         "12M Fwd 30Y",
#         "12M Fwd 40Y",
#         "12M Fwd 50Y",
#     ],
#     bday_offsets: Optional[Annotated[List[int], 3]] = None,
#     title: Optional[str] = "Latest PCA Residuals",
# ):
#     if bday_offsets:
#         if len(bday_offsets) > 3:
#             raise ValueError(f"'bday_offsets' must be length 3")
#         bday_offsets = sorted(bday_offsets)

#     residuals_df_dict: Dict[datetime, pd.DataFrame] = pca_results["residual_timeseries_dict"]
#     bdates = residuals_df_dict.keys()

#     latest_date = max(bdates)
#     latest_df = residuals_df_dict[latest_date]
#     latest_df = latest_df.copy()
#     latest_df = latest_df.reset_index()

#     fig = go.Figure()
#     # tenor_colors = {
#     #     tenor: f"rgb({np.random.randint(0, 50)}, {np.random.randint(0, 50)}, {np.random.randint(100, 255)})" for tenor in tenors_to_plot
#     # }
#     tenor_colors = {tenor: "rgb(120, 154, 255)" for tenor in tenors_to_plot}

#     for tenor in tenors_to_plot:
#         groups = tenor.split(" ")
#         if len(groups) == 3:
#             fig.add_trace(
#                 go.Bar(
#                     x=[tenor],
#                     y=[latest_df.loc[latest_df["Tenor"] == " ".join(groups[2:]), " ".join(groups[:2])].values[0]],
#                     marker_color=tenor_colors[tenor],
#                     name=f"{latest_date.strftime('%Y-%m-%d')}",
#                     showlegend=False,
#                 )
#             )
#         else:
#             fig.add_trace(
#                 go.Bar(
#                     x=[tenor],
#                     y=[latest_df.loc[latest_df["Tenor"] == tenor, "Spot"].values[0]],
#                     marker_color=tenor_colors[tenor],
#                     name=f"{latest_date.strftime('%Y-%m-%d')}",
#                     showlegend=False,
#                 )
#             )

#     if bday_offsets:
#         bdate_offsets = []
#         for offset in bday_offsets:
#             bdate_offsets.append(latest_date - BDay(offset))
#         alpha_values = [0.5, 0.35, 0.2]
#         bar_widths = [0.6, 0.45, 0.3]
#         for i, date in enumerate(bdate_offsets):
#             df = residuals_df_dict[date]
#             df = df.copy()
#             df = df.reset_index()
#             for tenor in tenors_to_plot:
#                 groups = tenor.split(" ")
#                 if len(groups) == 3:
#                     fig.add_trace(
#                         go.Bar(
#                             x=[tenor],
#                             y=[df.loc[df["Tenor"] == " ".join(groups[2:]), " ".join(groups[:2])].values[0]],
#                             marker_color=tenor_colors[tenor],
#                             marker_opacity=alpha_values[i],
#                             name=f"{bdate_offsets[i].strftime('%Y-%m-%d')}",
#                             width=bar_widths[i],
#                             showlegend=False,
#                         )
#                     )
#                 else:
#                     fig.add_trace(
#                         go.Bar(
#                             x=[tenor],
#                             y=[df.loc[latest_df["Tenor"] == tenor, "Spot"].values[0]],
#                             marker_color=tenor_colors[tenor],
#                             marker_opacity=alpha_values[i],
#                             name=f"{bdate_offsets[i].strftime('%Y-%m-%d')}",
#                             width=bar_widths[i],
#                             showlegend=False,
#                         )
#                     )

#     fig.update_layout(
#         title=title,
#         xaxis_title="Tenor",
#         yaxis_title="Residuals (bps)",
#         barmode="overlay",  # Bars will overlap
#         legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, title="Dates"),
#         height=750,
#         font=dict(size=11),
#         template="plotly_dark",
#     )
#     fig.update_xaxes(showspikes=True, spikecolor="white", spikesnap="cursor", spikemode="across", showgrid=True)
#     fig.update_yaxes(showspikes=True, spikecolor="white", spikesnap="cursor", spikethickness=0.5, showgrid=True)
#     fig.show(
#         config={
#             "modeBarButtonsToAdd": [
#                 "drawline",
#                 "drawopenpath",
#                 "drawclosedpath",
#                 "drawcircle",
#                 "drawrect",
#                 "eraseshape",
#             ]
#         }
#     )


# @staticmethod
# def most_mispriced_pca_resid_zscores(
#     df: pd.DataFrame,
#     top_n: Optional[int] = 5,
#     across_grid: Optional[bool] = False,
#     curve_weights: Optional[Annotated[List[int], 2]] = [1, 1],
#     fly_weights: Optional[Annotated[List[int], 3]] = [1, 2, 1],
#     exclusive_tenors: Optional[Dict[str, List[str]]] = None,
#     exclude_tenors: Optional[Dict[str, List[str]]] = None,
# ) -> Dict[Annotated[Literal["curve", "fly"], 2], pd.DataFrame]:
#     if across_grid:
#         points = [(tenor, fwd, df.loc[tenor, fwd]) for tenor in df.index for fwd in df.columns]

#         curve_records = []
#         for p1, p2 in itertools.combinations(points, 2):
#             tenor1, fwd1, z1 = p1
#             tenor2, fwd2, z2 = p2
#             spread = (curve_weights[0] * z1) - (curve_weights[1] * z2)
#             curve_records.append(
#                 {
#                     "Tenor1": f"{fwd1}x{tenor1}",
#                     "Tenor2": f"{fwd2}x{tenor2}",
#                     "ZScore-Spread": spread,
#                     "Trade": "steepener" if spread > 0 else "flattener",
#                     "Full Tenor": f"{fwd1} {tenor1}-{fwd2} {tenor2}",
#                 }
#             )

#         curve_df = pd.DataFrame(curve_records)
#         if exclude_tenors:
#             for col, tenors_to_exclude in exclude_tenors.items():
#                 if col in curve_df.columns:
#                     curve_df = curve_df[~curve_df[col].isin(tenors_to_exclude)]
#         top_curve_df = curve_df.sort_values("ZScore-Spread", ascending=False, key=lambda x: np.abs(x)).head(top_n).reset_index(drop=True)

#         bfly_records = []
#         for p1, p2, p3 in itertools.combinations(points, 3):
#             tenor1, fwd1, z1 = p1
#             tenor2, fwd2, z2 = p2
#             tenor3, fwd3, z3 = p3
#             bfly = (fly_weights[1] * z2) - (fly_weights[0] * z1) - (fly_weights[2] * z3)
#             bfly_records.append(
#                 {
#                     "ShortWing": f"{fwd1}x{tenor1}",
#                     "Belly": f"{fwd2}x{tenor2}",
#                     "LongWing": f"{fwd3}x{tenor3}",
#                     "Full Tenor": f"{fwd1} {tenor1}-{fwd2} {tenor2}-{fwd3} {tenor3}",
#                     "ZScore-Spread": bfly,
#                     "Trade": "rec belly" if bfly > 0 else "pay belly",
#                 }
#             )

#         bfly_df = pd.DataFrame(bfly_records)
#         if exclude_tenors:
#             for col, tenors_to_exclude in exclude_tenors.items():
#                 if col in bfly_df.columns:
#                     bfly_df = bfly_df[~bfly_df[col].isin(tenors_to_exclude)]

#         if exclusive_tenors:
#             for col, exclusive_tenors in exclusive_tenors.items():
#                 if col in bfly_df.columns:
#                     bfly_df = bfly_df[bfly_df[col].isin(exclusive_tenors)]

#         top_bfly_df = bfly_df.sort_values("ZScore-Spread", ascending=False, key=lambda x: np.abs(x)).head(top_n).reset_index(drop=True)

#         return {"curve": top_curve_df, "fly": top_bfly_df}

#     else:
#         curve_records = []
#         tenors = df.index.tolist()
#         for col in df.columns:
#             for t1, t2 in itertools.combinations(tenors, 2):
#                 z1 = df.loc[t1, col]
#                 z2 = df.loc[t2, col]
#                 spread = (curve_weights[0] * z1) - (curve_weights[1] * z2)
#                 curve_records.append(
#                     {
#                         "Forward": col,
#                         "Tenor1": t1,
#                         "Tenor2": t2,
#                         "ZScore-Spread": spread,
#                         "Trade": "steepener" if spread > 0 else "flattener",
#                         "Full Tenor": f"{col} {t1}-{col} {t2}",
#                     }
#                 )

#         curve_df = pd.DataFrame(curve_records)
#         if exclude_tenors:
#             for col, tenors_to_exclude in exclude_tenors.items():
#                 if col in curve_df.columns:
#                     curve_df = curve_df[~curve_df[col].isin(tenors_to_exclude)]
#         top_curve_df = curve_df.sort_values("ZScore-Spread", ascending=False, key=lambda x: np.abs(x)).head(top_n).reset_index(drop=True)

#         bfly_records = []
#         for col in df.columns:
#             for t_s, t_b, t_l in itertools.combinations(tenors, 3):
#                 z_s = df.loc[t_s, col]
#                 z_b = df.loc[t_b, col]
#                 z_l = df.loc[t_l, col]
#                 bfly = (fly_weights[1] * z_b) - (fly_weights[0] * z_s) - (fly_weights[2] * z_l)
#                 bfly_records.append(
#                     {
#                         "Forward": col,
#                         "ShortWing": t_s,
#                         "Belly": t_b,
#                         "LongWing": t_l,
#                         "ZScore-Spread": bfly,
#                         "Trade": "rec belly" if bfly > 0 else "pay belly",
#                         "Full Tenor": f"{col} {t_s}-{col} {t_b}-{col} {t_l}",
#                     }
#                 )

#         bfly_df = pd.DataFrame(bfly_records)
#         if exclude_tenors:
#             for col, tenors_to_exclude in exclude_tenors.items():
#                 if col in bfly_df.columns:
#                     bfly_df = bfly_df[~bfly_df[col].isin(tenors_to_exclude)]

#         if exclusive_tenors:
#             for col, exclusive_tenors in exclusive_tenors.items():
#                 if col in bfly_df.columns:
#                     bfly_df = bfly_df[bfly_df[col].isin(exclusive_tenors)]

#         top_bfly_df = bfly_df.sort_values("ZScore-Spread", ascending=False, key=lambda x: np.abs(x)).head(top_n).reset_index(drop=True)

#         return {"curve": top_curve_df, "fly": top_bfly_df}
