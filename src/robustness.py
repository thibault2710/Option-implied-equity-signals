"""Additional IV spread robustness checks.

This module creates new diagnostic outputs only. It does not modify the
baseline monthly panel or previously saved backtest results.
"""

import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


MPLCONFIGDIR = Path(tempfile.gettempdir()) / "options_implied_signals_matplotlib"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


QUINTILE_RETURN_COLUMNS = [
    "signal_month",
    "return_month",
    "Q1",
    "Q2",
    "Q3",
    "Q4",
    "Q5",
    "LS",
    "n_stocks",
    "n_Q1",
    "n_Q2",
    "n_Q3",
    "n_Q4",
    "n_Q5",
    "weighting",
    "method",
]


def load_additional_robustness_inputs(processed_data_dir, raw_data_dir, tables_dir):
    """Load inputs for the additional IV spread robustness checks."""
    processed_data_dir = Path(processed_data_dir)
    raw_data_dir = Path(raw_data_dir)
    tables_dir = Path(tables_dir)

    monthly_panel = pd.read_parquet(processed_data_dir / "monthly_signal_panel.parquet")

    daily_columns = [
        "secid",
        "permno",
        "date",
        "iv_atm_call",
        "iv_atm_put",
        "iv_otm_put",
        "realized_var",
        "mktcap",
        "exchcd",
        "shrcd",
        "score",
    ]
    daily_signals_with_vrp = pd.read_parquet(
        processed_data_dir / "daily_signals_with_vrp.parquet",
        columns=daily_columns,
    )

    crsp_monthly = pd.read_parquet(raw_data_dir / "crsp_monthly_2018_2024.parquet")

    factor_summary_path = tables_dir / "factor_regression_summary.csv"
    factor_summary = (
        pd.read_csv(factor_summary_path) if factor_summary_path.exists() else pd.DataFrame()
    )

    baseline_all_path = tables_dir / "robustness_quintile_returns_iv_spread_adj_raw_all_ew.csv"
    baseline_100m_path = (
        tables_dir / "robustness_quintile_returns_iv_spread_adj_raw_mktcap_100m_ew.csv"
    )
    baseline_all_ew = pd.read_csv(baseline_all_path) if baseline_all_path.exists() else pd.DataFrame()
    baseline_100m_ew = (
        pd.read_csv(baseline_100m_path) if baseline_100m_path.exists() else pd.DataFrame()
    )

    print(f"Loaded monthly signal panel: {monthly_panel.shape}")
    print(f"Loaded daily signals with VRP: {daily_signals_with_vrp.shape}")
    print(f"Loaded CRSP monthly returns: {crsp_monthly.shape}")
    print(f"Loaded factor regression summary: {factor_summary.shape}")
    print(f"Loaded baseline IV spread EW all returns: {baseline_all_ew.shape}")
    print(f"Loaded baseline IV spread EW mktcap >= 100M returns: {baseline_100m_ew.shape}")

    return (
        monthly_panel,
        daily_signals_with_vrp,
        crsp_monthly,
        factor_summary,
        baseline_all_ew,
        baseline_100m_ew,
    )


def compute_weighted_return(
    df,
    return_col="ret_fwd_1m",
    weight_col="mktcap",
    value_weighted=True,
):
    """Compute equal-weighted or value-weighted average returns."""
    if df.empty:
        return np.nan

    returns = pd.to_numeric(df[return_col], errors="coerce")

    if not value_weighted:
        return returns.mean()

    weights = pd.to_numeric(df[weight_col], errors="coerce")
    valid = returns.notna() & weights.notna() & (weights > 0)
    if not valid.any():
        return np.nan

    returns = returns.loc[valid]
    weights = weights.loc[valid]
    weight_sum = weights.sum()
    if weight_sum <= 0:
        return np.nan

    return (returns * weights / weight_sum).sum()


def _t_stat(series):
    """Compute the t-statistic of a sample mean."""
    series = pd.to_numeric(series, errors="coerce").dropna()
    if len(series) <= 1:
        return np.nan
    volatility = series.std()
    if pd.isna(volatility) or volatility == 0:
        return np.nan
    return series.mean() / (volatility / np.sqrt(len(series)))


def summarize_ls_returns(results_df, method, weighting, label=None):
    """Summarize one monthly long-short return series."""
    ls = pd.to_numeric(results_df["LS"], errors="coerce").dropna()
    n_months = len(ls)
    mean_monthly_ls = ls.mean() if n_months else np.nan
    monthly_volatility = ls.std() if n_months else np.nan
    annualized_ls = mean_monthly_ls * 12
    annualized_volatility = monthly_volatility * np.sqrt(12)
    sharpe = (
        annualized_ls / annualized_volatility
        if pd.notna(annualized_volatility) and annualized_volatility != 0
        else np.nan
    )

    return pd.DataFrame(
        [
            {
                "label": label if label is not None else f"{method}_{weighting}",
                "method": method,
                "weighting": weighting,
                "mean_monthly_ls": mean_monthly_ls,
                "annualized_ls": annualized_ls,
                "monthly_volatility": monthly_volatility,
                "annualized_volatility": annualized_volatility,
                "sharpe_ratio": sharpe,
                "t_stat": _t_stat(ls),
                "positive_month_pct": (ls > 0).mean() if n_months else np.nan,
                "n_months": n_months,
                "avg_n_stocks": results_df["n_stocks"].mean()
                if "n_stocks" in results_df.columns
                else np.nan,
                "avg_n_Q1": results_df["n_Q1"].mean()
                if "n_Q1" in results_df.columns
                else np.nan,
                "avg_n_Q5": results_df["n_Q5"].mean()
                if "n_Q5" in results_df.columns
                else np.nan,
            }
        ]
    )


def _prepare_monthly_panel(panel):
    """Keep needed columns and normalize dtypes for monthly sorts."""
    needed_columns = [
        "permno",
        "signal_month",
        "return_month",
        "iv_spread_adj",
        "ret_fwd_1m",
        "mktcap",
    ]
    data = panel[[column for column in needed_columns if column in panel.columns]].copy()
    data = data.assign(
        permno=pd.to_numeric(data["permno"], errors="coerce").astype("Int64"),
        signal_month=pd.PeriodIndex(data["signal_month"].astype(str), freq="M"),
        return_month=pd.PeriodIndex(data["return_month"].astype(str), freq="M"),
        iv_spread_adj=pd.to_numeric(data["iv_spread_adj"], errors="coerce"),
        ret_fwd_1m=pd.to_numeric(data["ret_fwd_1m"], errors="coerce"),
        mktcap=pd.to_numeric(data["mktcap"], errors="coerce"),
    )
    return data.dropna(subset=["permno", "signal_month", "iv_spread_adj", "ret_fwd_1m"])


def _compute_monthly_quintile_returns(
    assigned_panel,
    quintile_col,
    weighting,
    method,
    return_col="ret_fwd_1m",
    weight_col="mktcap",
):
    """Compute Q1-Q5 and Q5-Q1 monthly returns from an assigned panel."""
    value_weighted = weighting == "vw"
    rows = []

    assigned = assigned_panel.dropna(subset=[quintile_col, return_col]).copy()
    assigned.loc[:, quintile_col] = assigned[quintile_col].astype("Int64")

    for signal_month, month_data in assigned.groupby("signal_month", sort=True):
        row = {
            "signal_month": signal_month,
            "return_month": month_data["return_month"].iloc[0],
            "n_stocks": len(month_data),
            "weighting": weighting,
            "method": method,
        }

        for quintile in range(1, 6):
            quintile_data = month_data.loc[month_data[quintile_col] == quintile]
            row[f"Q{quintile}"] = compute_weighted_return(
                quintile_data,
                return_col=return_col,
                weight_col=weight_col,
                value_weighted=value_weighted,
            )
            row[f"n_Q{quintile}"] = len(quintile_data)

        row["LS"] = row["Q5"] - row["Q1"]
        rows.append(row)

    results = pd.DataFrame(rows)
    if results.empty:
        return pd.DataFrame(columns=QUINTILE_RETURN_COLUMNS)
    return results[QUINTILE_RETURN_COLUMNS]


def _assign_signal_quintiles(panel, signal_col, quintile_col, n_quantiles=5):
    """Assign simple signal quintiles within each signal month."""
    assigned = panel.copy()
    assigned.loc[:, quintile_col] = pd.NA

    for _, month_data in assigned.groupby("signal_month", sort=True):
        valid = month_data.dropna(subset=[signal_col])
        if valid[signal_col].nunique() < n_quantiles:
            continue

        try:
            codes = pd.qcut(
                valid[signal_col],
                q=n_quantiles,
                labels=False,
                duplicates="drop",
            )
        except ValueError:
            continue

        if codes.nunique(dropna=True) < n_quantiles:
            continue

        assigned.loc[valid.index, quintile_col] = (codes + 1).astype("Int64")

    assigned = assigned.dropna(subset=[quintile_col]).copy()
    assigned.loc[:, quintile_col] = assigned[quintile_col].astype("Int64")
    return assigned


def assign_size_neutral_iv_spread_quintiles(
    panel,
    n_size_buckets=5,
    n_signal_quantiles=5,
):
    """Run size-bucket-neutral IV spread quintile sorts."""
    print("\n" + "=" * 80)
    print("Size-Bucket-Neutral IV Spread Sorts")
    print("=" * 80)

    data = _prepare_monthly_panel(panel)
    data = data.dropna(subset=["mktcap"])
    data = data.loc[data["mktcap"] > 0].copy()
    data.loc[:, "size_bucket"] = pd.NA
    data.loc[:, "iv_spread_quintile"] = pd.NA

    for _, month_data in data.groupby("signal_month", sort=True):
        if month_data["mktcap"].nunique() < n_size_buckets:
            continue

        try:
            size_codes = pd.qcut(
                month_data["mktcap"],
                q=n_size_buckets,
                labels=False,
                duplicates="drop",
            )
        except ValueError:
            continue

        if size_codes.nunique(dropna=True) < n_size_buckets:
            continue

        data.loc[month_data.index, "size_bucket"] = (size_codes + 1).astype("Int64")

        for size_code in sorted(size_codes.dropna().unique()):
            bucket_index = size_codes.loc[size_codes == size_code].index
            bucket = data.loc[bucket_index]

            if bucket["iv_spread_adj"].nunique() < n_signal_quantiles:
                continue

            try:
                signal_codes = pd.qcut(
                    bucket["iv_spread_adj"],
                    q=n_signal_quantiles,
                    labels=False,
                    duplicates="drop",
                )
            except ValueError:
                continue

            if signal_codes.nunique(dropna=True) < n_signal_quantiles:
                continue

            data.loc[bucket.index, "iv_spread_quintile"] = (
                signal_codes + 1
            ).astype("Int64")

    assigned = data.dropna(subset=["iv_spread_quintile"]).copy()
    assigned.loc[:, "iv_spread_quintile"] = assigned["iv_spread_quintile"].astype("Int64")
    assigned.loc[:, "size_bucket"] = assigned["size_bucket"].astype("Int64")

    print(f"Rows after size-neutral assignment: {len(assigned):,}")
    print(f"Months assigned: {assigned['signal_month'].nunique():,}")
    print(
        "Average assigned stocks per month: "
        f"{assigned.groupby('signal_month')['permno'].nunique().mean():,.1f}"
    )

    returns = []
    for weighting in ["ew", "vw"]:
        returns.append(
            _compute_monthly_quintile_returns(
                assigned,
                quintile_col="iv_spread_quintile",
                weighting=weighting,
                method="size_bucket_neutral",
            )
        )

    return pd.concat(returns, ignore_index=True)


def compute_size_residualized_iv_spread(panel):
    """Residualize IV spread on log market cap within each signal month."""
    print("\n" + "=" * 80)
    print("Residualizing IV Spread on Log Market Cap")
    print("=" * 80)

    data = _prepare_monthly_panel(panel)
    data = data.loc[data["mktcap"] > 0].copy()
    data.loc[:, "log_mktcap"] = np.log(data["mktcap"])
    data.loc[:, "iv_spread_size_resid"] = np.nan

    for _, month_data in data.groupby("signal_month", sort=True):
        valid = month_data.dropna(subset=["iv_spread_adj", "log_mktcap"])
        if len(valid) < 20 or valid["log_mktcap"].nunique() < 2:
            continue

        y = valid["iv_spread_adj"].to_numpy(dtype=float)
        x = np.column_stack(
            [
                np.ones(len(valid)),
                valid["log_mktcap"].to_numpy(dtype=float),
            ]
        )
        beta, *_ = np.linalg.lstsq(x, y, rcond=None)
        residual = y - x @ beta
        data.loc[valid.index, "iv_spread_size_resid"] = residual

    print(f"Rows with residualized IV spread: {data['iv_spread_size_resid'].notna().sum():,}")
    print(
        "Months with residualized IV spread: "
        f"{data.loc[data['iv_spread_size_resid'].notna(), 'signal_month'].nunique():,}"
    )

    return data


def run_size_neutral_iv_spread_sorts(panel, output_tables_dir=None):
    """Run both size-neutral IV spread robustness approaches."""
    output_tables_dir = Path(output_tables_dir) if output_tables_dir is not None else None
    if output_tables_dir is not None:
        output_tables_dir.mkdir(parents=True, exist_ok=True)

    bucket_returns = assign_size_neutral_iv_spread_quintiles(panel)

    residualized_panel = compute_size_residualized_iv_spread(panel)
    residualized_panel = residualized_panel.dropna(subset=["iv_spread_size_resid"]).copy()
    residualized_panel = _assign_signal_quintiles(
        residualized_panel,
        signal_col="iv_spread_size_resid",
        quintile_col="iv_spread_quintile",
    )

    residualized_returns = []
    for weighting in ["ew", "vw"]:
        residualized_returns.append(
            _compute_monthly_quintile_returns(
                residualized_panel,
                quintile_col="iv_spread_quintile",
                weighting=weighting,
                method="size_residualized",
            )
        )
    residualized_returns = pd.concat(residualized_returns, ignore_index=True)

    returns = pd.concat([bucket_returns, residualized_returns], ignore_index=True)

    summary_rows = []
    label_map = {
        ("size_bucket_neutral", "ew"): "Size Bucket Neutral EW",
        ("size_bucket_neutral", "vw"): "Size Bucket Neutral VW",
        ("size_residualized", "ew"): "Size Residualized EW",
        ("size_residualized", "vw"): "Size Residualized VW",
    }
    for (method, weighting), group in returns.groupby(["method", "weighting"]):
        summary_rows.append(
            summarize_ls_returns(
                group,
                method=method,
                weighting=weighting,
                label=label_map.get((method, weighting)),
            )
        )
    summary = pd.concat(summary_rows, ignore_index=True)

    if output_tables_dir is not None:
        returns_path = output_tables_dir / "iv_spread_size_neutral_returns.csv"
        summary_path = output_tables_dir / "iv_spread_size_neutral_summary.csv"
        returns.to_csv(returns_path, index=False)
        summary.to_csv(summary_path, index=False)
        print(f"Saved size-neutral returns: {returns_path}")
        print(f"Saved size-neutral summary: {summary_path}")

    return returns, summary


def _prepare_crsp_monthly_returns(crsp_monthly):
    """Prepare CRSP monthly returns with one row per permno-return_month."""
    needed_columns = ["permno", "date", "ret", "retx", "exchcd", "shrcd"]
    returns = crsp_monthly[[column for column in needed_columns if column in crsp_monthly.columns]].copy()

    returns = returns.assign(
        permno=pd.to_numeric(returns["permno"], errors="coerce").astype("Int64"),
        date_return=pd.to_datetime(returns["date"]),
        ret=pd.to_numeric(returns["ret"], errors="coerce"),
    )
    if "retx" in returns.columns:
        returns.loc[:, "retx"] = pd.to_numeric(returns["retx"], errors="coerce")

    returns.loc[:, "return_month"] = returns["date_return"].dt.to_period("M")
    returns = returns.dropna(subset=["permno", "return_month", "ret"])
    returns = returns.sort_values(["permno", "return_month", "date_return"])
    returns = returns.drop_duplicates(subset=["permno", "return_month"], keep="last")

    rename_columns = {"ret": "ret_fwd_1m"}
    if "retx" in returns.columns:
        rename_columns["retx"] = "retx_fwd_1m"
    returns = returns.rename(columns=rename_columns)
    returns = returns.drop(columns=["date"], errors="ignore")
    return returns.reset_index(drop=True)


def build_last5_iv_spread_monthly_panel(
    daily_signals_with_vrp,
    crsp_monthly,
    output_path=None,
):
    """Build a monthly panel using the last 5 daily IV observations each month."""
    print("\n" + "=" * 80)
    print("Building Last-5-Trading-Day IV Spread Monthly Panel")
    print("=" * 80)

    base_columns = [
        "secid",
        "permno",
        "date",
        "iv_atm_call",
        "iv_atm_put",
        "iv_otm_put",
        "realized_var",
        "mktcap",
        "exchcd",
        "shrcd",
        "score",
    ]
    available_columns = [
        column for column in base_columns if column in daily_signals_with_vrp.columns
    ]
    daily = daily_signals_with_vrp[available_columns].copy()

    daily = daily.assign(
        date=pd.to_datetime(daily["date"]),
        permno=pd.to_numeric(daily["permno"], errors="coerce").astype("Int64"),
    )
    for column in ["iv_atm_call", "iv_atm_put", "iv_otm_put", "realized_var", "mktcap"]:
        if column in daily.columns:
            daily.loc[:, column] = pd.to_numeric(daily[column], errors="coerce")

    daily = daily.dropna(
        subset=["permno", "date", "iv_atm_call", "iv_atm_put", "iv_otm_put", "realized_var"]
    )
    daily.loc[:, "signal_month"] = daily["date"].dt.to_period("M")
    daily = daily.sort_values(["permno", "signal_month", "date"])
    last5 = daily.groupby(["permno", "signal_month"], group_keys=False).tail(5)

    aggregations = {
        "signal_date": ("date", "max"),
        "n_daily_obs_used": ("date", "count"),
        "iv_atm_call": ("iv_atm_call", "mean"),
        "iv_atm_put": ("iv_atm_put", "mean"),
        "iv_otm_put": ("iv_otm_put", "mean"),
        "realized_var": ("realized_var", "mean"),
        "mktcap": ("mktcap", "mean"),
    }
    if "secid" in last5.columns:
        aggregations["secid"] = ("secid", "last")
    if "exchcd" in last5.columns:
        aggregations["exchcd"] = ("exchcd", "last")
    if "shrcd" in last5.columns:
        aggregations["shrcd"] = ("shrcd", "last")
    if "score" in last5.columns:
        aggregations["score"] = ("score", "min")

    monthly = (
        last5.groupby(["permno", "signal_month"], as_index=False)
        .agg(**aggregations)
        .reset_index(drop=True)
    )

    monthly.loc[:, "return_month"] = monthly["signal_month"] + 1
    monthly.loc[:, "iv_spread"] = monthly["iv_atm_call"] - monthly["iv_atm_put"]
    monthly.loc[:, "iv_skew"] = monthly["iv_otm_put"] - monthly["iv_atm_call"]
    monthly.loc[:, "implied_var"] = monthly["iv_atm_call"] ** 2
    monthly.loc[:, "vrp"] = monthly["implied_var"] - monthly["realized_var"]
    monthly.loc[:, "iv_spread_adj"] = monthly["iv_spread"]

    monthly_returns = _prepare_crsp_monthly_returns(crsp_monthly)
    panel = monthly.merge(
        monthly_returns,
        on=["permno", "return_month"],
        how="inner",
        suffixes=("", "_return"),
    )

    ordered_columns = [
        "secid",
        "permno",
        "signal_month",
        "signal_date",
        "return_month",
        "date_return",
        "n_daily_obs_used",
        "iv_atm_call",
        "iv_atm_put",
        "iv_otm_put",
        "iv_spread",
        "iv_skew",
        "implied_var",
        "realized_var",
        "vrp",
        "iv_spread_adj",
        "mktcap",
        "ret_fwd_1m",
        "retx_fwd_1m",
        "exchcd",
        "shrcd",
        "score",
    ]
    panel = panel[[column for column in ordered_columns if column in panel.columns]]

    print(f"Last-5 monthly signals before return merge: {len(monthly):,}")
    print(f"Last-5 monthly panel after return merge: {len(panel):,}")
    print(f"signal_month range: {panel['signal_month'].min()} to {panel['signal_month'].max()}")
    print(f"return_month range: {panel['return_month'].min()} to {panel['return_month'].max()}")
    print(f"Average daily observations used: {panel['n_daily_obs_used'].mean():.2f}")

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        panel.to_csv(output_path, index=False)
        print(f"Saved last-5 monthly panel: {output_path}")

    return panel


def run_last5_iv_spread_sort(last5_panel, output_tables_dir=None):
    """Run IV spread quintile sorts on the last-5-day monthly panel."""
    print("\n" + "=" * 80)
    print("Running Last-5-Day IV Spread Sorts")
    print("=" * 80)

    output_tables_dir = Path(output_tables_dir) if output_tables_dir is not None else None
    if output_tables_dir is not None:
        output_tables_dir.mkdir(parents=True, exist_ok=True)

    data = _prepare_monthly_panel(last5_panel)
    assigned = _assign_signal_quintiles(
        data,
        signal_col="iv_spread_adj",
        quintile_col="iv_spread_quintile",
    )

    returns = []
    for weighting in ["ew", "vw"]:
        returns.append(
            _compute_monthly_quintile_returns(
                assigned,
                quintile_col="iv_spread_quintile",
                weighting=weighting,
                method="last5",
            )
        )
    returns = pd.concat(returns, ignore_index=True)

    summary_rows = []
    for weighting, group in returns.groupby("weighting"):
        summary_rows.append(
            summarize_ls_returns(
                group,
                method="last5",
                weighting=weighting,
                label=f"Last5 {weighting.upper()}",
            )
        )
    summary = pd.concat(summary_rows, ignore_index=True)

    if output_tables_dir is not None:
        returns_path = output_tables_dir / "iv_spread_last5_returns.csv"
        summary_path = output_tables_dir / "iv_spread_last5_summary.csv"
        returns.to_csv(returns_path, index=False)
        summary.to_csv(summary_path, index=False)
        print(f"Saved last-5 returns: {returns_path}")
        print(f"Saved last-5 summary: {summary_path}")

    return returns, summary


def _clean_return_series(results_df):
    """Keep only timing and long-short return columns for factor regressions."""
    columns_to_keep = ["signal_month", "LS"]
    if "return_month" in results_df.columns:
        columns_to_keep.append("return_month")

    returns = results_df[columns_to_keep].copy()
    returns = returns.assign(
        signal_month=pd.PeriodIndex(returns["signal_month"].astype(str), freq="M"),
        LS=pd.to_numeric(returns["LS"], errors="coerce"),
    )
    if "return_month" in returns.columns:
        returns.loc[:, "return_month"] = pd.PeriodIndex(
            returns["return_month"].astype(str),
            freq="M",
        )
    else:
        returns.loc[:, "return_month"] = returns["signal_month"] + 1

    return returns.dropna(subset=["signal_month", "return_month", "LS"]).reset_index(drop=True)


def run_ff5_mom_alpha_if_possible(raw_data_dir, return_specs):
    """Run FF5 + Momentum regressions for selected robustness return series."""
    print("\n" + "=" * 80)
    print("Running FF5 + Momentum Alpha Checks")
    print("=" * 80)

    try:
        from src.regressions import load_factor_data, run_factor_regression

        factor_df = load_factor_data(raw_data_dir)
    except Exception as exc:
        print(f"Skipping factor alpha checks because factor data could not be loaded: {exc}")
        return pd.DataFrame()

    factor_cols = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "MOM"]
    rows = []

    for spec in return_specs:
        label = spec["label"]
        returns = _clean_return_series(spec["returns"])
        if returns.empty:
            print(f"Skipping {label}: no valid LS returns.")
            continue

        try:
            result = run_factor_regression(
                returns,
                factor_df,
                portfolio_label=label,
                model_name="FF5_MOM",
                factor_cols=factor_cols,
                nw_lags=4,
            )
            result["label"] = label
            result["method"] = spec.get("method")
            result["weighting"] = spec.get("weighting")
            rows.append(result)
            print(
                f"{label}: annualized alpha={result['alpha_annualized']:.2%}, "
                f"t-stat={result['alpha_tstat']:.2f}"
            )
        except Exception as exc:
            print(f"Skipping {label}: regression failed with error: {exc}")

    return pd.DataFrame(rows)


def _prepare_baseline_returns(returns_df, label, method):
    """Prepare a baseline return dataframe for summaries and charts."""
    if returns_df is None or returns_df.empty:
        return pd.DataFrame()

    data = returns_df.copy()
    data.loc[:, "signal_month"] = pd.PeriodIndex(data["signal_month"].astype(str), freq="M")
    if "return_month" in data.columns:
        data.loc[:, "return_month"] = pd.PeriodIndex(data["return_month"].astype(str), freq="M")
    else:
        data.loc[:, "return_month"] = data["signal_month"] + 1
    data.loc[:, "LS"] = pd.to_numeric(data["LS"], errors="coerce")
    data.loc[:, "method"] = method
    data.loc[:, "weighting"] = "ew"
    data.loc[:, "label"] = label
    return data


def build_additional_robustness_summary(
    baseline_all_ew,
    baseline_100m_ew,
    size_summary,
    last5_summary,
    new_alpha_df,
    factor_summary,
):
    """Combine baseline, return-summary, and alpha information into one table."""
    summaries = []

    baseline_all = _prepare_baseline_returns(
        baseline_all_ew,
        label="Baseline EW All",
        method="baseline_all",
    )
    if not baseline_all.empty:
        summaries.append(
            summarize_ls_returns(
                baseline_all,
                method="baseline_all",
                weighting="ew",
                label="Baseline EW All",
            )
        )

    baseline_100m = _prepare_baseline_returns(
        baseline_100m_ew,
        label="Baseline EW MktCap100M",
        method="baseline_mktcap_100m",
    )
    if not baseline_100m.empty:
        summaries.append(
            summarize_ls_returns(
                baseline_100m,
                method="baseline_mktcap_100m",
                weighting="ew",
                label="Baseline EW MktCap100M",
            )
        )

    summaries.extend([size_summary, last5_summary])
    combined = pd.concat(summaries, ignore_index=True)

    alpha_rows = []
    if factor_summary is not None and not factor_summary.empty:
        baseline_map = {
            "IV Spread EW All": ("Baseline EW All", "baseline_all", "ew"),
            "IV Spread EW MktCap100M": (
                "Baseline EW MktCap100M",
                "baseline_mktcap_100m",
                "ew",
            ),
        }
        baseline_alpha = factor_summary.loc[
            factor_summary["portfolio"].isin(baseline_map.keys())
            & (factor_summary["model"] == "FF5_MOM")
        ].copy()
        for _, row in baseline_alpha.iterrows():
            label, method, weighting = baseline_map[row["portfolio"]]
            alpha_rows.append(
                {
                    "label": label,
                    "method": method,
                    "weighting": weighting,
                    "alpha_annualized": row.get("alpha_annualized", np.nan),
                    "alpha_tstat": row.get("alpha_tstat", np.nan),
                    "alpha_pvalue": row.get("alpha_pvalue", np.nan),
                    "alpha_r_squared": row.get("r_squared", np.nan),
                    "alpha_n_months": row.get("n_months", np.nan),
                }
            )

    if new_alpha_df is not None and not new_alpha_df.empty:
        for _, row in new_alpha_df.iterrows():
            alpha_rows.append(
                {
                    "label": row.get("label", row.get("portfolio")),
                    "method": row.get("method"),
                    "weighting": row.get("weighting"),
                    "alpha_annualized": row.get("alpha_annualized", np.nan),
                    "alpha_tstat": row.get("alpha_tstat", np.nan),
                    "alpha_pvalue": row.get("alpha_pvalue", np.nan),
                    "alpha_r_squared": row.get("r_squared", np.nan),
                    "alpha_n_months": row.get("n_months", np.nan),
                }
            )

    if alpha_rows:
        alpha = pd.DataFrame(alpha_rows)
        combined = combined.merge(alpha, on=["label", "method", "weighting"], how="left")
    else:
        for column in [
            "alpha_annualized",
            "alpha_tstat",
            "alpha_pvalue",
            "alpha_r_squared",
            "alpha_n_months",
        ]:
            combined.loc[:, column] = np.nan

    ordered_columns = [
        "label",
        "method",
        "weighting",
        "mean_monthly_ls",
        "annualized_ls",
        "monthly_volatility",
        "annualized_volatility",
        "sharpe_ratio",
        "t_stat",
        "positive_month_pct",
        "n_months",
        "avg_n_stocks",
        "avg_n_Q1",
        "avg_n_Q5",
        "alpha_annualized",
        "alpha_tstat",
        "alpha_pvalue",
        "alpha_r_squared",
        "alpha_n_months",
    ]
    return combined[[column for column in ordered_columns if column in combined.columns]]


def _plot_cumulative_line(returns_df, label):
    """Plot one cumulative LS return line on the active axes."""
    if returns_df is None or returns_df.empty:
        return
    data = _clean_return_series(returns_df)
    if data.empty:
        return
    data = data.sort_values("return_month")
    data.loc[:, "cumulative_growth"] = (1 + data["LS"]).cumprod()
    plt.plot(data["return_month"].astype(str), data["cumulative_growth"], label=label)


def plot_size_neutral_cumulative_ls(size_neutral_returns, charts_dir, baseline_all_ew=None):
    """Plot cumulative LS returns for size-neutral IV spread sorts."""
    charts_dir = Path(charts_dir)
    charts_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    _plot_cumulative_line(baseline_all_ew, "Baseline EW All")

    for method, label in [
        ("size_bucket_neutral", "Size Bucket Neutral EW"),
        ("size_residualized", "Size Residualized EW"),
    ]:
        data = size_neutral_returns.loc[
            (size_neutral_returns["method"] == method)
            & (size_neutral_returns["weighting"] == "ew")
        ]
        _plot_cumulative_line(data, label)

    plt.title("IV Spread Size-Neutral Cumulative Long-Short Returns")
    plt.xlabel("Return month")
    plt.ylabel("Growth of $1")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    output_path = charts_dir / "iv_spread_size_neutral_cumulative_ls.png"
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved chart: {output_path}")


def plot_last5_cumulative_ls(last5_returns, charts_dir, baseline_all_ew=None):
    """Plot cumulative LS returns for baseline and last-5-day IV spread sorts."""
    charts_dir = Path(charts_dir)
    charts_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    _plot_cumulative_line(baseline_all_ew, "Baseline EW All")

    for weighting, label in [("ew", "Last5 EW"), ("vw", "Last5 VW")]:
        data = last5_returns.loc[last5_returns["weighting"] == weighting]
        _plot_cumulative_line(data, label)

    plt.title("IV Spread Last-5-Day Cumulative Long-Short Returns")
    plt.xlabel("Return month")
    plt.ylabel("Growth of $1")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    output_path = charts_dir / "iv_spread_last5_cumulative_ls.png"
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved chart: {output_path}")


def plot_additional_robustness_bars(summary_df, charts_dir):
    """Plot annualized LS returns and FF5 + Momentum alphas for key checks."""
    charts_dir = Path(charts_dir)
    charts_dir.mkdir(parents=True, exist_ok=True)

    labels_to_plot = [
        "Baseline EW All",
        "Baseline EW MktCap100M",
        "Size Bucket Neutral EW",
        "Size Residualized EW",
        "Last5 EW",
    ]
    plot_data = summary_df.loc[summary_df["label"].isin(labels_to_plot)].copy()
    plot_data.loc[:, "label"] = pd.Categorical(
        plot_data["label"],
        categories=labels_to_plot,
        ordered=True,
    )
    plot_data = plot_data.sort_values("label")

    has_alpha = "alpha_annualized" in plot_data.columns and plot_data["alpha_annualized"].notna().any()
    if has_alpha:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        plot_data.set_index("label")["annualized_ls"].plot(kind="bar", ax=axes[0])
        axes[0].set_title("Annualized LS Return")
        axes[0].set_xlabel("")
        axes[0].set_ylabel("Annualized return")
        axes[0].tick_params(axis="x", rotation=45)

        plot_data.set_index("label")["alpha_annualized"].plot(kind="bar", ax=axes[1])
        axes[1].set_title("FF5 + Momentum Alpha")
        axes[1].set_xlabel("")
        axes[1].set_ylabel("Annualized alpha")
        axes[1].tick_params(axis="x", rotation=45)
    else:
        fig, ax = plt.subplots(figsize=(10, 5))
        plot_data.set_index("label")["annualized_ls"].plot(kind="bar", ax=ax)
        ax.set_title("IV Spread Additional Robustness")
        ax.set_xlabel("")
        ax.set_ylabel("Annualized LS return")
        ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    output_path = charts_dir / "iv_spread_additional_robustness_bars.png"
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved chart: {output_path}")
