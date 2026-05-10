"""Small shared utilities for return summaries and portfolio calculations."""

import numpy as np
import pandas as pd
import statsmodels.api as sm


def safe_mean(series):
    """Return the numeric mean of a series, or NaN if no values are valid."""
    values = pd.to_numeric(pd.Series(series), errors="coerce").dropna()
    if values.empty:
        return np.nan
    return values.mean()


def compute_weighted_return(values, weights):
    """Compute a weighted average after dropping missing and nonpositive weights."""
    values = pd.to_numeric(pd.Series(values), errors="coerce")
    weights = pd.to_numeric(pd.Series(weights), errors="coerce")

    valid = values.notna() & weights.notna() & (weights > 0)
    if not valid.any():
        return np.nan

    values = values.loc[valid]
    weights = weights.loc[valid]
    total_weight = weights.sum()
    if total_weight <= 0:
        return np.nan

    return (values * weights / total_weight).sum()


def simple_t_stat(return_series):
    """Compute the usual t-statistic for the mean of a return series."""
    returns = pd.to_numeric(pd.Series(return_series), errors="coerce").dropna()
    if len(returns) < 2:
        return np.nan

    std = returns.std(ddof=1)
    if pd.isna(std) or std == 0:
        return np.nan

    return returns.mean() / (std / np.sqrt(len(returns)))


def newey_west_t_stat(return_series, maxlags=4):
    """Estimate the mean return and Newey-West/HAC t-statistic."""
    returns = pd.to_numeric(pd.Series(return_series), errors="coerce").dropna()
    if len(returns) < 3:
        return {
            "alpha": np.nan,
            "t_stat": np.nan,
            "p_value": np.nan,
            "n_obs": len(returns),
        }

    y = returns.to_numpy(dtype=float)
    x = np.ones((len(y), 1))
    try:
        model = sm.OLS(y, x).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
    except Exception:
        return {
            "alpha": np.nan,
            "t_stat": np.nan,
            "p_value": np.nan,
            "n_obs": len(returns),
        }

    return {
        "alpha": model.params[0],
        "t_stat": model.tvalues[0],
        "p_value": model.pvalues[0],
        "n_obs": int(model.nobs),
    }


def annualize_return(monthly_mean):
    """Annualize an arithmetic monthly mean return."""
    return monthly_mean * 12


def annualize_vol(monthly_vol):
    """Annualize a monthly volatility."""
    return monthly_vol * np.sqrt(12)


def summarize_return_series(return_series, annualization=12, nw_lags=4):
    """Summarize a monthly return series with raw and Newey-West t-stats."""
    returns = pd.to_numeric(pd.Series(return_series), errors="coerce").dropna()
    n_months = len(returns)

    if n_months == 0:
        return {
            "mean_monthly_return": np.nan,
            "annualized_return": np.nan,
            "monthly_volatility": np.nan,
            "annualized_volatility": np.nan,
            "sharpe_ratio": np.nan,
            "raw_t_stat": np.nan,
            "nw_t_stat": np.nan,
            "nw_p_value": np.nan,
            "positive_month_pct": np.nan,
            "n_months": 0,
        }

    mean_monthly = returns.mean()
    monthly_vol = returns.std(ddof=1) if n_months > 1 else np.nan
    annualized_return = mean_monthly * annualization
    annualized_volatility = monthly_vol * np.sqrt(annualization) if pd.notna(monthly_vol) else np.nan
    nw = newey_west_t_stat(returns, maxlags=nw_lags)

    return {
        "mean_monthly_return": mean_monthly,
        "annualized_return": annualized_return,
        "monthly_volatility": monthly_vol,
        "annualized_volatility": annualized_volatility,
        "sharpe_ratio": annualized_return / annualized_volatility
        if pd.notna(annualized_volatility) and annualized_volatility != 0
        else np.nan,
        "raw_t_stat": simple_t_stat(returns),
        "nw_t_stat": nw["t_stat"],
        "nw_p_value": nw["p_value"],
        "positive_month_pct": (returns > 0).mean(),
        "n_months": n_months,
    }
