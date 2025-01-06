import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd
import QuantLib as ql
import statsmodels.api as sm
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression

from CurvyCUSIPs.utils.arbitragelab import EngleGrangerPortfolio, JohansenPortfolio, construct_spread
from CurvyCUSIPs.utils.regression_utils import run_odr


# https://github.com/hudson-and-thames/arbitragelab/blob/master/arbitragelab/hedge_ratios/box_tiao.py
def get_box_tiao_hedge_ratio(price_data: pd.DataFrame, dependent_variable: str) -> Tuple[dict, pd.DataFrame, None, pd.Series]:
    """
    Perform Box-Tiao canonical decomposition on the assets dataframe.
    The resulting ratios are the weightings of each asset in the portfolio. There are N decompositions for N assets,
    where each column vector corresponds to one portfolio. The order of the weightings corresponds to the
    descending order of the eigenvalues.
    """

    def _least_square_VAR_fit(demeaned_price_data: pd.DataFrame) -> np.array:
        var_model = sm.tsa.VAR(demeaned_price_data)
        least_sq_est = np.squeeze(var_model.fit(1).coefs, axis=0)
        return least_sq_est, var_model

    X = price_data.copy()
    X = X[[dependent_variable] + [x for x in X.columns if x != dependent_variable]]

    demeaned = X - X.mean()
    least_sq_est, var_model = _least_square_VAR_fit(demeaned)
    covar = demeaned.cov()
    box_tiao_matrix = np.linalg.inv(covar) @ least_sq_est @ covar @ least_sq_est.T
    eigvals, eigvecs = np.linalg.eig(box_tiao_matrix)
    bt_eigvecs = eigvecs[:, np.argsort(eigvals)[::-1]]
    hedge_ratios = dict(zip(X.columns, bt_eigvecs[:, -1]))

    beta_weights = []
    for ticker, h in hedge_ratios.items():
        if ticker != dependent_variable:
            beta = -h / hedge_ratios[dependent_variable]
            hedge_ratios[ticker] = beta
            beta_weights.append(beta)
    hedge_ratios[dependent_variable] = 1.0
    residuals = construct_spread(price_data, hedge_ratios=hedge_ratios, dependent_variable=dependent_variable)

    return {
        "beta_weights": beta_weights,
        "hedge_ratios_dict": hedge_ratios,
        "X": X,
        "residuals": residuals,
        "results": var_model,
    }


# https://github.com/hudson-and-thames/arbitragelab/blob/master/arbitragelab/hedge_ratios/johansen.py
def get_johansen_hedge_ratio(price_data: pd.DataFrame, dependent_variable: str) -> Tuple[dict, pd.DataFrame, pd.Series, pd.Series]:
    """
    Get hedge ratio from Johansen test eigenvector
    https://en.wikipedia.org/wiki/Johansen_test
    """

    port = JohansenPortfolio()
    port.fit(price_data, dependent_variable)

    X = price_data.copy()
    X.drop(columns=dependent_variable, axis=1, inplace=True)

    y = price_data[dependent_variable].copy()

    hedge_ratios = port.hedge_ratios.iloc[0].to_dict()
    residuals = construct_spread(price_data, hedge_ratios=hedge_ratios, dependent_variable=dependent_variable)

    hedge_ratios_copy = hedge_ratios.copy()
    del hedge_ratios_copy[dependent_variable]

    return {
        "beta_weights": list(hedge_ratios_copy.values()),
        "hedge_ratios_dict": hedge_ratios,
        "X": X,
        "y": y,
        "residuals": residuals,
        "results": port,
    }


# https://github.com/hudson-and-thames/arbitragelab/blob/master/arbitragelab/hedge_ratios/half_life.py
def get_minimum_hl_hedge_ratio(price_data: pd.DataFrame, dependent_variable: str) -> Tuple[dict, pd.DataFrame, pd.Series, pd.Series, object]:
    """
    Get hedge ratio by minimizing spread half-life of mean reversion.
    https://quant.stackexchange.com/questions/77953/interpretation-and-intuition-behind-half-life-of-a-mean-reverting-process
    """

    def get_half_life_of_mean_reversion_ou_process(data: pd.Series) -> float:
        reg = LinearRegression(fit_intercept=True)
        training_data = data.shift(1).dropna().values.reshape(-1, 1)
        target_values = data.diff().dropna()
        reg.fit(X=training_data, y=target_values)
        half_life = -np.log(2) / reg.coef_[0]
        return half_life

    def _min_hl_function(beta: np.array, X: pd.DataFrame, y: pd.Series) -> float:
        spread = y - (beta * X).sum(axis=1)
        return abs(get_half_life_of_mean_reversion_ou_process(spread))

    X = price_data.copy()
    X.drop(columns=dependent_variable, axis=1, inplace=True)

    y = price_data[dependent_variable].copy()
    initial_guess = (y[0] / X).mean().values
    result = minimize(_min_hl_function, x0=initial_guess, method="BFGS", tol=1e-5, args=(X, y))

    if result.status != 0:
        warnings.warn("Minimum Half Life Optimization failed to converge. Please check output hedge ratio! The result can be unstable!")

    residuals = y - (result.x * X).sum(axis=1)

    hedge_ratios = result.x
    hedge_ratios_dict = dict(zip([dependent_variable] + X.columns.tolist(), np.insert(hedge_ratios, 0, 1.0)))

    return {
        "beta_weights": list(hedge_ratios),
        "hedge_ratios_dict": hedge_ratios_dict,
        "X": X,
        "y": y,
        "residuals": residuals,
        "results": result,
    }


# https://github.com/hudson-and-thames/arbitragelab/blob/master/arbitragelab/hedge_ratios/adf_optimal.py
def get_adf_optimal_hedge_ratio(price_data: pd.DataFrame, dependent_variable: str) -> Tuple[dict, pd.DataFrame, pd.Series, pd.Series, object]:
    """
    Get hedge ratio by minimizing ADF test statistic.
    https://www.statisticshowto.com/adf-augmented-dickey-fuller-test/
    """

    def _min_adf_stat(beta: np.array, X: pd.DataFrame, y: pd.Series) -> float:
        portfolio = EngleGrangerPortfolio()
        spread = y - (beta * X).sum(axis=1)
        portfolio.perform_eg_test(spread)
        return portfolio.adf_statistics.loc["statistic_value"].iloc[0]

    X = price_data.copy()
    X.drop(columns=dependent_variable, axis=1, inplace=True)
    y = price_data[dependent_variable].copy()
    initial_guess = (y[0] / X).mean().values
    result = minimize(_min_adf_stat, x0=initial_guess, method="BFGS", tol=1e-5, args=(X, y))

    if result.status != 0:
        warnings.warn("ADF Optimization failed to converge. Please check output hedge ratio! The result can be unstable!")

    residuals = y - (result.x * X).sum(axis=1)

    hedge_ratios = result.x
    hedge_ratios_dict = dict(zip([dependent_variable] + X.columns.tolist(), np.insert(hedge_ratios, 0, 1.0)))

    return {
        "beta_weights": list(hedge_ratios),
        "hedge_ratios_dict": hedge_ratios_dict,
        "X": X,
        "y": y,
        "residuals": residuals,
        "results": result,
    }


def beta_estimates(
    df: pd.DataFrame,
    x_cols: List[str],
    y_col: str,
    run_on_level_changes: Optional[bool] = False,
    x_errs: Optional[npt.ArrayLike] = None,
    y_errs: Optional[npt.ArrayLike] = None,
    pc_scores_df: Optional[pd.DataFrame] = None,
    loadings_df: Optional[pd.DataFrame] = None,
) -> Dict:
    df = df[["Date"] + x_cols + [y_col]].copy()
    df_level = df.copy()

    if len(x_cols) == 1:
        df["spread"] = df[y_col] - df[x_cols[0]]
    elif len(x_cols) == 2:
        df["spread"] = (df[y_col] - df[x_cols[0]]) - (df[x_cols[1]] - df[y_col])
    else:
        raise ValueError("Too many x_cols")

    if run_on_level_changes:
        date_col = df["Date"]
        df = df[x_cols + [y_col] + ["spread"]].diff()
        df["Date"] = date_col
    df = df.dropna()

    pc1_beta = None
    if loadings_df is not None:
        ep_x0_pc1 = loadings_df.loc[x_cols[0], "PC1"]
        ep_x0_pc2 = loadings_df.loc[x_cols[0], "PC2"]
        ep_x0_pc3 = loadings_df.loc[x_cols[0], "PC3"] if len(x_cols) > 1 else None

        ep_y_pc1 = loadings_df.loc[y_col, "PC1"]
        ep_y_pc2 = loadings_df.loc[y_col, "PC2"]
        ep_y_pc3 = loadings_df.loc[y_col, "PC3"] if len(x_cols) > 1 else None

        if len(x_cols) > 1:
            ep_x1_pc1 = loadings_df.loc[x_cols[1], "PC1"]
            ep_x1_pc2 = loadings_df.loc[x_cols[1], "PC2"]
            ep_x1_pc3 = loadings_df.loc[x_cols[1], "PC3"] if len(x_cols) > 1 else None

            # see  Doug Huggins, Christian Schaller Fixed Income Relative Value Analysis ed2 page 76 APPROPRIATE HEDGING
            r"""
                Hedge ratios against more factors are best calculated via matrix inversion. 
                For example, the hedge ratio for a 2Y-5Y-10Y butterfly which is neutral to changes in the first and 
                second factor can be calculated for a given notional $n_5$ for 5Y by:
                    $$
                        \begin{pmatrix}
                            n_2 \\
                            n_{10}
                        \end{pmatrix}
                        = 
                        \begin{pmatrix}
                            BPV_2 \cdot e_{12} & BPV_2 \cdot e_{22} \\
                            BPV_{10} \cdot e_{12} & BPV_{10} \cdot e_{22}
                            \end{pmatrix}^{-1} 
                            \begin{pmatrix}
                            - n_5 \cdot BPV_5 \cdot e_{15} \\
                            - n_5 \cdot BPV_5 \cdot e_{25}
                        \end{pmatrix}
                    $$
            """
            pc1_beta = list(
                np.dot(
                    np.linalg.inv(np.array([[ep_x0_pc1, ep_x1_pc1], [ep_x0_pc2, ep_x1_pc2]])),
                    np.array([ep_y_pc1, ep_y_pc2]),
                )
            )
        else:
            pc1_beta = ep_x0_pc1 / ep_y_pc1

    # avoiding divide by zero errors
    small_value = 1e-8
    if x_errs is not None:
        x_errs[x_errs == 0] = small_value
    if y_errs is not None:
        y_errs[y_errs == 0] = small_value

    regression_results = {
        "ols": sm.OLS(df[y_col], sm.add_constant(df[x_cols])).fit(),
        "tls": run_odr(df=df, x_cols=x_cols, y_col=y_col, x_errs=None, y_errs=None),
        # ODR becomes TLS if errors not specified
        "odr": (run_odr(df=df, x_cols=x_cols, y_col=y_col, x_errs=x_errs, y_errs=y_errs) if x_errs is not None or y_errs is not None else None),
        "box_tiao": get_box_tiao_hedge_ratio(df_level[["Date"] + x_cols + [y_col]].set_index("Date"), dependent_variable=y_col),
        "johansen": get_johansen_hedge_ratio(df_level[["Date"] + x_cols + [y_col]].set_index("Date"), dependent_variable=y_col),
        "minimum_half_life": get_minimum_hl_hedge_ratio(df_level[["Date"] + x_cols + [y_col]].set_index("Date"), dependent_variable=y_col),
        "adf_optimal": get_adf_optimal_hedge_ratio(df_level[["Date"] + x_cols + [y_col]].set_index("Date"), dependent_variable=y_col),
        "pcr_pc1": (sm.OLS(df["spread"].to_numpy(), sm.add_constant(pc_scores_df["PC1"].to_numpy())).fit() if pc_scores_df is not None else None),
        "pcr_pc2": (sm.OLS(df["spread"].to_numpy(), sm.add_constant(pc_scores_df["PC2"].to_numpy())).fit() if pc_scores_df is not None else None),
        "pcr_pc3": (sm.OLS(df["spread"].to_numpy(), sm.add_constant(pc_scores_df["PC3"].to_numpy())).fit() if pc_scores_df is not None else None),
    }

    beta_estimates = {
        "ols": (
            regression_results["ols"].params[1] if len(x_cols) == 1 else [regression_results["ols"].params[1], regression_results["ols"].params[2]]
        ),
        "tls": (regression_results["tls"].beta[1] if len(x_cols) == 1 else [regression_results["tls"].beta[1], regression_results["tls"].beta[2]]),
        "odr": (
            regression_results["odr"].beta[1]
            if (x_errs is not None or y_errs is not None) and len(x_cols) == 1
            else [regression_results["odr"].beta[1], regression_results["odr"].beta[2]] if x_errs or y_errs else None
        ),
        "pc1": pc1_beta,
        "box_tiao": regression_results["box_tiao"]["beta_weights"],
        "johansen": regression_results["johansen"]["beta_weights"],
        "minimum_half_life": regression_results["minimum_half_life"]["beta_weights"],
        "adf_optimal": regression_results["adf_optimal"]["beta_weights"],
    }

    pcs_exposures = {
        "pcr_pc1_exposure": regression_results["pcr_pc1"].params[1] if pc_scores_df is not None else None,
        "pcr_pc2_exposure": regression_results["pcr_pc2"].params[1] if pc_scores_df is not None else None,
        "pcr_pc3_exposure": (regression_results["pcr_pc3"].params[1] if pc_scores_df is not None and len(x_cols) > 1 else None),
        # checking 50-50 duration - if non-zero => exposures exists
        "epsilon_pc1_loadings_exposure": (
            ep_y_pc1 - (ep_x0_pc1 + ep_x1_pc1) / 2.0 if len(x_cols) > 1 else ep_x0_pc1 - ep_y_pc1 if loadings_df is not None else None
        ),
        "epsilon_pc2_loadings_exposure": (
            ep_y_pc2 - (ep_x0_pc2 + ep_x1_pc2) / 2.0 if len(x_cols) > 1 else ep_x0_pc2 - ep_y_pc2 if loadings_df is not None else None
        ),
        "epsilon_pc3_loadings_exposure": (
            (ep_y_pc3 - (ep_x0_pc3 + ep_x1_pc3) / 2.0 if len(x_cols) > 1 else ep_x0_pc3 - ep_y_pc3 if loadings_df is not None else None)
            if len(x_cols) > 1
            else None
        ),
    }

    return {"betas": beta_estimates, "regression_results": regression_results, "pc_exposures": pcs_exposures}


# TODO
def rolling_beta_estimates():
    pass
