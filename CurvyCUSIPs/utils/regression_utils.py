from types import SimpleNamespace
from typing import List, Optional, Tuple, Annotated

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import seaborn as sns
import statsmodels.api as sm
from scipy import stats
from scipy.odr import ODR, Model, RealData, Data
from scipy.stats import tstd, zscore, linregress

sns.set_style("whitegrid", {"grid.linestyle": "--"})

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)


def run_basic_linear_regression(
    x_series: pd.Series,
    y_series: pd.Series,
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    title: Optional[str] = None,
):
    if not x_label:
        x_label = x_series.name
    if not y_label:
        y_label = y_series.name

    Y = y_series
    X = x_series
    X = sm.add_constant(X)
    model = sm.OLS(Y, X)
    results = model.fit()
    print(results.summary())

    intercept = results.params[0]
    slope = results.params[1]
    r_squared = results.rsquared

    plt.figure()
    plt.scatter(x_series, y_series)

    regression_line = intercept + slope * x_series
    plt.plot(x_series, regression_line, color="red")

    plt.ylabel(x_label)
    plt.xlabel(y_label)
    plt.title(title or f"{y_label} Regressed on {x_label}")
    equation_text = f"y = {intercept:.3f} + {slope:.3f}x\nR² = {r_squared:.3f}\nSE = {results.bse["const"]:.3f}"
    plt.plot([], [], " ", label=f"{equation_text}")

    plt.legend(fontsize="x-large")
    plt.grid(True)
    plt.show()

    return results


def run_basic_linear_regression_df(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: Optional[str] = None,
    plot_most_recent: Optional[bool] = False,
    date_color_bar: Optional[bool] = False,
    on_diff: Optional[bool] = False,
    run_tls: Optional[bool] = False,
    run_gls: Optional[bool] = False,
    run_wls: Optional[bool | npt.ArrayLike] = None,
    no_plot: Optional[bool] = False
):
    if x_col not in df.columns or y_col not in df.columns:
        raise Exception(f"{x_col} or {y_col} not in df cols")

    df = df[["Date"] + [x_col, y_col]].copy()
    if on_diff:
        date_col = df["Date"]
        df = df[[x_col, y_col]].diff()
        df["Date"] = date_col
    df = df.dropna()

    y = df[y_col]
    X = df[x_col]
    X = sm.add_constant(X)

    regression_type = "OLS"

    if run_tls:
        regression_type = "TLS"

        def _linear_f_no_constant(beta: np.array, x_variable: np.array) -> np.array:
            _, b = beta[0], beta[1:]
            b.shape = (b.shape[0], 1)
            return (x_variable * b).sum(axis=0)

        linear = Model(_linear_f_no_constant)
        mydata = RealData(X.T, y)
        myodr = ODR(mydata, linear, beta0=np.ones(X.shape[1] + 1))
        out = myodr.run()
        if not no_plot:
            print(out.pprint())

        hedge_ratios = out.beta[1:]
        residuals = y - (X * hedge_ratios).sum(axis=1)

        x = df[x_col].to_numpy().flatten()
        y = df[y_col].to_numpy().flatten()

        intercept = out.beta[1]
        slope = out.beta[-1]
        sd_intercept = out.sd_beta[1]
        sd_slope = out.sd_beta[0]
        df_resid = len(x) - 2
        results = SimpleNamespace(
            params=pd.Series([intercept, slope], index=["const", x_col]),
            bse=pd.Series([sd_intercept, sd_slope], index=["const", x_col]),
            pvalues=pd.Series(
                [2 * (1 - stats.t.cdf(np.abs(intercept / sd_intercept), df=df_resid)), 2 * (1 - stats.t.cdf(np.abs(slope / sd_slope), df=df_resid))],
                index=["const", x_col],
            ),
            rsquared=1 - np.sum((y - (slope * x + intercept)) ** 2) / np.sum((y - np.mean(y)) ** 2),
            resid=residuals,
            model=SimpleNamespace(**{"endog_names": y_col, "exog_names": ["", x_col]}),
        )

        intercept = results.params[0]
        slope = results.params[1]
        r_squared = results.rsquared
        p_value = results.pvalues[1] if len(results.pvalues) > 1 else None
        slope_name = results.params.drop("const").index[0]
        se_intercept = results.bse[0]
        se_slope = results.bse[1]

    else:
        if run_gls:
            regression_type = "GLS"
            model = sm.GLS(y, X)
        elif run_wls:
            regression_type = "WLS"
            model = sm.WLS(y, X, weights=run_wls)
        else:
            model = sm.OLS(y, X)

        results = model.fit()
        if not no_plot:
            print(results.summary())

        intercept = results.params[0]
        slope = results.params[1]
        r_squared = results.rsquared
        p_value = results.pvalues[1] if len(results.pvalues) > 1 else None
        slope_name = results.params.drop("const").index[0]
        se_intercept = results.bse[0]
        se_slope = results.bse[1]

    if not no_plot:
        plt.figure()

        if date_color_bar:
            df["date_numeric"] = (df["Date"] - df["Date"].min()).dt.total_seconds()
            scatter = plt.scatter(df[x_col], df[y_col], c=df["date_numeric"], cmap="viridis")
            cbar = plt.colorbar(scatter)
            cbar.set_label("Date")
            cbar_ticks = np.linspace(df["date_numeric"].min(), df["date_numeric"].max(), num=10)
            cbar.set_ticks(cbar_ticks)
            cbar.set_ticklabels(pd.to_datetime(cbar_ticks, unit="s", origin=df["Date"].min()).strftime("%Y-%m-%d"))
        else:
            plt.scatter(df[x_col], df[y_col])

        if plot_most_recent:
            most_recent = df["Date"].iloc[-1]
            plt.scatter(
                df[x_col].iloc[-1],
                df[y_col].iloc[-1],
                color="orange",
                s=100,
                label=f"Most Recent: {most_recent}",
            )

        regression_line = intercept + slope * df[x_col]
        plt.plot(df[x_col], regression_line, color="blue")
        plt.xlabel(x_col if not on_diff else f"Δ{x_col}")
        plt.ylabel(y_col if not on_diff else f"Δ{y_col}")
        plt.title(
            title or f"{regression_type} - {y_col} Regressed on {x_col}" if not on_diff else f"{regression_type} - Δ{y_col} Regressed on Δ{x_col}",
            fontdict={"fontsize": "large"},
        )
        equation_text = (
            f"y = {intercept:.3f} + {slope:.3f}*{slope_name}\n"
            f"R² = {r_squared:.3f}\n"
            f"SE (Intercept) = {se_intercept:.3f}, SE ({slope_name}) = {se_slope:.3f}\n"
            f"p-value ({slope_name}) = {p_value:.3e}"
        )
        plt.plot([], [], " ", label=f"{equation_text}")
        plt.legend(fontsize="large")
        plt.grid(True)

        plt.show()

    return results


def run_multiple_linear_regression_df(
    df: pd.DataFrame,
    x_cols: List[str],
    y_col: str,
    title: Optional[str] = None,
    verbose: Optional[bool] = False,
    on_diff: Optional[bool] = False,
    use_adjusted_residuals: bool = False,
    show_plot: Optional[bool] = False,
):
    df = df[["Date"] + x_cols + [y_col]].copy()
    if on_diff:
        date_col = df["Date"]
        df = df[x_cols + [y_col]].diff()
        df["Date"] = date_col
    df = df.dropna()

    for col in x_cols + [y_col]:
        if col not in df.columns:
            raise Exception(f"{col} not in df columns")

    Y = df[y_col]
    X = df[x_cols]
    X = sm.add_constant(X)

    model = sm.OLS(Y, X)
    results = model.fit()
    if verbose:
        print(results.summary())

    intercept = results.params["const"]
    slopes = results.params.drop("const")
    r_squared = results.rsquared
    adj_r_squared = results.rsquared_adj

    Y_pred = results.fittedvalues
    residuals = results.resid

    if use_adjusted_residuals:
        df["adjusted_residuals"] = residuals
        for col in x_cols:
            X_adjusted = X.drop(columns=col)
            model_adjusted = sm.OLS(Y, X_adjusted).fit()
            residuals_adjusted = Y - model_adjusted.fittedvalues
            df[f"adjusted_residuals_{col}"] = residuals_adjusted

        return df[["Date"] + [f"adjusted_residuals_{col}" for col in x_cols]], results

    if show_plot:
        plt.figure()
        plt.scatter(Y_pred, Y)
        plt.xlabel("Predicted Values")
        plt.ylabel("Actual Values")
        plt.title("Actual vs Predicted")
        plt.grid(True)

        equation_text = f"y = {intercept:.3f}"
        for col in slopes.index:
            coef = slopes[col]
            equation_text += f" + {coef:.3f}*{col}"
        equation_text += f"\nR² = {r_squared:.3f}\nAdjusted R² = {adj_r_squared:.3f}"
        plt.plot([], [], " ", label=equation_text)
        plt.legend(fontsize="large")
        plt.show()

        plt.figure()
        plt.scatter(Y_pred, residuals)
        plt.axhline(y=0, color="red", linestyle="--")
        plt.xlabel("Predicted Values")
        plt.ylabel("Residuals")
        plt.title(title or "Residuals vs Predicted")
        plt.grid(True)
        plt.show()

    return df, results


def plot_residuals_timeseries(
    df: pd.DataFrame,
    results: sm.regression.linear_model.RegressionResultsWrapper,
    date_col: str = "Date",
    plot_zscores: Optional[bool] = False,
    stds: Optional[List[int]] = None,
    is_on_diff: Optional[bool] = False,
):
    residuals = results.resid
    if plot_zscores:
        residuals = zscore(residuals)

    if date_col not in df.columns:
        raise Exception(f"{date_col} not in df columns")

    if is_on_diff:
        df = df.copy()
        df = df.iloc[1:]

    r_squared = results.rsquared
    intercept = results.params[0]
    slope = results.params[1]
    p_value = results.pvalues[1] if len(results.pvalues) > 1 else None
    dependent_variable = results.model.endog_names
    independent_variables = results.model.exog_names[1:]
    slope_name = results.params.drop("const").index[0]

    if p_value is not None:
        title = f"Residuals of {dependent_variable} Regressed on {", ".join(independent_variables)} Over Time\n"
    else:
        title = f"Residuals of {dependent_variable} Regressed on {", ".join(independent_variables)} Over Time\n"

    plt.figure()
    plt.plot(df[date_col], residuals, linestyle="-", color="blue")
    plt.axhline(y=0, color="red", linestyle="--")
    equation_text = f"y = {intercept:.3f} + {slope:.3f}*{slope_name}\nR² = {r_squared:.3f}\nSE = {results.bse["const"]:.3f}"
    # for var in results.params.drop("const").index:
    #     p_val = results.pvalues[var]
    #     equation_text += f"\np-value ({var}) = {p_val:.3e}"
    plt.plot([], [], " ", label=f"{equation_text}")
    plt.plot(
        [],
        [],
        " ",
        label=f"Most Recent Residual ({df['Date'].iloc[-1].date()}): {residuals.iloc[-1]:.2f}",
    )

    if stds:
        resid_std = tstd(residuals)
        resid_mean = np.mean(residuals)
        plt.axhline(resid_mean, linestyle="--", color="red", label=f"Resid Mean: {resid_mean}")
        for std in stds:
            curr_std_level = resid_mean + resid_std * std
            curr = plt.axhline(curr_std_level, linestyle="--", label=f"+/- {std} STD Resid Level: {np.round(curr_std_level, 3)}")
            plt.axhline(curr_std_level * -1, linestyle="--", color=curr.get_color())

    plt.legend(loc=(0, 0), fontsize="large")
    plt.xlabel("Date")
    plt.ylabel("Residuals (bps)" if not plot_zscores else "Z-Scores")
    plt.title(title + ", Z-Scroes" if plot_zscores else title, fontdict={"fontsize": "large"})
    plt.grid(True)
    plt.show()


def modified_partial_regression_fly_plot(
    df: pd.DataFrame,
    x_cols: Annotated[List[str], 2],
    y_col: str,
    show_regression: Optional[bool] = False,
    show_residuals: Optional[bool] = False,
    show_residual_timeseries: Optional[bool] = False,
    plot_zscores: Optional[bool] = False,
    stds: Optional[List[int]] = None,
    verbose: Optional[bool] = False,
):
    X1 = sm.add_constant(df[x_cols[1]])
    model1 = sm.OLS(df[y_col], X1).fit()
    df[f"{y_col}_resid"] = model1.resid

    X2 = sm.add_constant(df[x_cols[0]])
    model2 = sm.OLS(df[f"{y_col}_resid"], X2).fit()
    df["partial_residuals"] = model2.fittedvalues

    if verbose:
        print(model2.summary())

    intercept = model2.params["const"]
    r_squared = model2.rsquared
    intercept = model2.params[0]
    slope = model2.params[1]
    p_value = model2.pvalues[1] if len(model2.pvalues) > 1 else None
    dependent_variable = model2.model.endog_names
    independent_variables = model2.model.exog_names[1]
    slope_name = model2.params.drop("const").index[0]

    if p_value is not None:
        title = f"Residuals of {dependent_variable} Regressed on {independent_variables} Over Time\n"
    else:
        title = f"Residuals of {dependent_variable} Regressed on {independent_variables} Over Time\n"

    # equation_text = f"y_adj_{y_col} = {coeff_xcol_0:.2f} x {x_cols[0]} {intercept:+.2f}"
    equation_text = f"y = {intercept:.3f} + {slope:.3f}*{slope_name}\nR² = {r_squared:.3f}\nSE = {model2.bse["const"]:.3f}"
    # for var in model2.params.drop("const").index:
    #     p_val = model2.pvalues[var]
    #     equation_text += f"\np-value ({var}) = {p_val:.3e}"

    if show_regression:
        plt.figure()
        plt.scatter(df[x_cols[0]], df[f"{y_col}_resid"], alpha=0.7)
        plt.plot([], [], " ", label=equation_text)
        plt.scatter(
            df[f"{x_cols[0]}"].iloc[-1],
            df[f"{y_col}_resid"].iloc[-1],
            label=f"Most Recent Observation: {df["Date"].iloc[-1].date()}",
            color="orange",
            s=100,
        )
        plt.plot(df[x_cols[0]], df["partial_residuals"], color="blue", linestyle="-")

        plt.title(f"Partial Regression: {title} (After Adjusting for {x_cols[1]})")
        plt.xlabel(f"{x_cols[0]}")
        plt.ylabel(f"Adjusted {y_col} Fly")
        plt.legend(fontsize="large")
        plt.grid(True)
        plt.show()

    if show_residuals:
        plt.figure()
        residuals = df[f"{y_col}_resid"] - df["partial_residuals"]
        plt.plot([], [], " ", label=equation_text)
        plt.scatter(df[x_cols[0]], residuals, alpha=0.7, color="red")
        plt.scatter(
            df[x_cols[0]].iloc[-1],
            residuals.iloc[-1],
            color="green",
            label=f"Most Recent Resid ({df['Date'].iloc[-1].date()}): {residuals.iloc[-1]:.2f}",
            s=100,
        )
        plt.axhline(0, color="black", linestyle="--")
        plt.title(f"Partial Regression: Residuals of {y_col} Fly vs {x_cols[0]} (After Adjusting for {x_cols[1]})")
        plt.xlabel(f"{x_cols[0]}")
        plt.ylabel("Residuals (bps)")
        plt.legend(fontsize="large")
        plt.grid(True)
        plt.show()

    if show_residual_timeseries:
        plt.figure()
        residuals = df[f"{y_col}_resid"] - df["partial_residuals"]
        if plot_zscores:
            residuals = zscore(residuals)

        plt.plot(df["Date"], residuals, linestyle="-", color="blue")
        plt.axhline(0, color="red", linestyle="--")
        plt.plot([], [], " ", label=equation_text)
        plt.plot(
            [],
            [],
            " ",
            label=f"Most Recent Resid ({df['Date'].iloc[-1].date()}): {residuals.iloc[-1]:.2f}",
        )

        if stds:
            resid_std = tstd(residuals)
            resid_mean = np.mean(residuals)
            plt.axhline(resid_mean, linestyle="--", color="red", label=f"Resid Mean: {resid_mean}")
            for std in stds:
                curr_std_level = resid_mean + resid_std * std
                curr = plt.axhline(curr_std_level, linestyle="--", label=f"+/- {std} STD Resid Level: {np.round(curr_std_level, 3)}")
                plt.axhline(curr_std_level * -1, linestyle="--", color=curr.get_color())

        plt.title(f"Partial Regression: Residuals of {y_col} Fly vs {x_cols[0]} (After Adjusting for {x_cols[1]})")
        plt.xlabel("Date")
        plt.ylabel("Residuals (bps)" if not plot_zscores else "Z-Scores")
        plt.legend(loc=(0, 0), fontsize="large")
        plt.grid(True)
        plt.show()


def run_rolling_regression_df(df: pd.DataFrame, x_col: str, y_col: str, window: int, title: Optional[str] = None):
    df = df.copy()

    if x_col not in df.columns or y_col not in df.columns:
        raise Exception(f"{x_col} or {y_col} not in df cols")

    def calculate_r_squared(x, y):
        Y = y
        X = x
        X = sm.add_constant(X)
        model = sm.OLS(Y, X)
        results = model.fit()
        r_squared = results.rsquared
        return r_squared

    rolling_r_squared = []
    for rolling_df in df[[x_col, y_col]].rolling(window=window):
        if len(rolling_df.index) < window:
            rolling_r_squared.append(np.nan)
        else:
            r_squared = calculate_r_squared(rolling_df[x_col], rolling_df[y_col])
            rolling_r_squared.append(r_squared)

    plt.figure()
    plt.plot(df["Date"], rolling_r_squared, label=f"Rolling R-squared (window={window})")

    most_recent = df["Date"].iloc[-1]
    plt.scatter(most_recent, rolling_r_squared[-1], color="purple", s=100, label=f"Most Recent: {most_recent}, R² = {rolling_r_squared[-1]}")

    plt.xlabel("Date")
    plt.ylabel("R-squared")
    plt.title(title or f"Rolling R-squared: {y_col} Regressed on {x_col}", fontdict={"fontsize": "x-large"})

    mean_r_squared = np.nanmean(rolling_r_squared)
    std_r_squared = np.nanstd(rolling_r_squared)
    stats_text = f"Mean R² = {mean_r_squared:.3f}\nStd R² = {std_r_squared:.3f}"
    plt.plot([], [], " ", label=stats_text)

    plt.legend(fontsize="x-large")
    plt.grid(True)
    plt.show()

    return rolling_r_squared


# Orthogonal Distance Regression
"""
https://docs.scipy.org/doc/scipy/reference/odr.html
https://www.mechanicalkern.com/static/odr_ams.pdf
https://www.reddit.com/r/mathematics/comments/12qkes4/confused_btw_orthogonal_linear_regressionolr/
"""


def run_odr(df: pd.DataFrame, x_cols: List[str], y_col: str, x_errs: Optional[npt.ArrayLike] = None, y_errs: Optional[npt.ArrayLike] = None):
    def orthoregress(
        x: pd.Series | npt.ArrayLike, y: pd.Series | npt.ArrayLike, x_errs: Optional[npt.ArrayLike] = None, y_errs: Optional[npt.ArrayLike] = None
    ):
        # calc weights (inverse variances)
        wd = None
        we = None
        if x_errs is not None:
            wd = 1.0 / np.square(x_errs)
        if y_errs is not None:
            we = 1.0 / np.square(y_errs)

        def f(p, x):
            return (p[0] * x) + p[1]

        od = ODR(Data(x, y, wd=wd, we=we), Model(f), beta0=linregress(x, y)[0:2])
        out = od.run()
        return out

    def orthoregress_multilinear(
        X: pd.DataFrame | pd.Series | npt.ArrayLike,
        y: pd.Series | npt.ArrayLike,
        x_errs: Optional[npt.ArrayLike] = None,
        y_errs: Optional[npt.ArrayLike] = None,
    ):
        # calc weights (inverse variances)
        wd = None
        we = None
        if x_errs is not None:
            x_errs = np.asarray(x_errs)
            wd = 1.0 / np.square(x_errs.T)  # transpose to match ODR shape
        if y_errs is not None:
            we = 1.0 / np.square(y_errs)

        def multilinear_f(p, x):
            return np.dot(p[:-1], x) + p[-1]

        X = np.asarray(X)
        y = np.asarray(y)
        X_odr = X.T
        y_flat = y.flatten()
        X_with_intercept = np.column_stack((X, np.ones(X.shape[0])))
        beta_init, _, _, _ = np.linalg.lstsq(X_with_intercept, y_flat, rcond=None)
        beta0 = np.append(beta_init[:-1], beta_init[-1])
        model = Model(multilinear_f)
        data = Data(X_odr, y, wd=wd, we=we)
        odr_instance = ODR(data, model, beta0=beta0)
        output = odr_instance.run()
        return output

    if len(x_cols) > 1:
        out = orthoregress_multilinear(df[x_cols], df[y_col], x_errs, y_errs)
    else:
        out = orthoregress(df[x_cols[0]], df[y_col], x_errs, y_errs)
    out.beta = np.roll(out.beta, 1)
    return out
