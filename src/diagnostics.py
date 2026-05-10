"""Focused diagnostics for the IV spread portfolio result."""

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


def load_diagnostics_inputs(processed_data_dir, tables_dir):
    """Load the monthly panel and selected IV spread result files."""
    processed_data_dir = Path(processed_data_dir)
    tables_dir = Path(tables_dir)

    monthly_panel = pd.read_parquet(processed_data_dir / "monthly_signal_panel.parquet")
    iv_spread_all_ew_returns = pd.read_csv(
        tables_dir / "robustness_quintile_returns_iv_spread_adj_raw_all_ew.csv"
    )
    iv_spread_100m_ew_returns = pd.read_csv(
        tables_dir / "robustness_quintile_returns_iv_spread_adj_raw_mktcap_100m_ew.csv"
    )
    factor_summary = pd.read_csv(tables_dir / "factor_regression_summary.csv")

    print(f"Loaded monthly panel: {monthly_panel.shape}")
    print(f"Loaded IV spread all EW returns: {iv_spread_all_ew_returns.shape}")
    print(f"Loaded IV spread mktcap >= 100M EW returns: {iv_spread_100m_ew_returns.shape}")
    print(f"Loaded factor regression summary: {factor_summary.shape}")

    return monthly_panel, iv_spread_all_ew_returns, iv_spread_100m_ew_returns, factor_summary


def assign_iv_spread_quintiles(panel, universe="all"):
    """Assign IV spread quintiles within each signal month."""
    if universe not in ["all", "mktcap_100m"]:
        raise ValueError(f"Unknown universe: {universe}")

    panel_q = panel.copy()
    panel_q = panel_q.assign(
        signal_month=pd.PeriodIndex(panel_q["signal_month"].astype(str), freq="M"),
        return_month=pd.PeriodIndex(panel_q["return_month"].astype(str), freq="M"),
        iv_spread_adj=pd.to_numeric(panel_q["iv_spread_adj"], errors="coerce"),
        mktcap=pd.to_numeric(panel_q["mktcap"], errors="coerce"),
    )

    if universe == "mktcap_100m":
        panel_q = panel_q.loc[panel_q["mktcap"] >= 100].copy()

    panel_q = panel_q.dropna(subset=["signal_month", "iv_spread_adj", "ret_fwd_1m"])
    panel_q.loc[:, "iv_spread_quintile"] = pd.NA

    def assign_one_month(group):
        group = group.copy()
        if group["iv_spread_adj"].nunique() < 5:
            return group
        try:
            quintile = pd.qcut(
                group["iv_spread_adj"],
                q=5,
                labels=False,
                duplicates="drop",
            )
        except ValueError:
            return group
        if quintile.nunique(dropna=True) < 5:
            return group
        group.loc[:, "iv_spread_quintile"] = (quintile + 1).astype("Int64")
        return group

    panel_q = (
        panel_q.groupby("signal_month", group_keys=False)
        .apply(assign_one_month)
        .dropna(subset=["iv_spread_quintile"])
        .copy()
    )
    panel_q.loc[:, "iv_spread_quintile"] = panel_q["iv_spread_quintile"].astype("Int64")

    avg_stocks = panel_q.groupby("signal_month")["permno"].nunique().mean()
    print(f"\nAssigned IV spread quintiles for universe={universe}")
    print(f"Rows: {len(panel_q):,}")
    print(f"Months: {panel_q['signal_month'].nunique():,}")
    print(f"Average stocks per month: {avg_stocks:,.1f}")

    return panel_q


def value_weighted_return(group, return_col="ret_fwd_1m", weight_col="mktcap"):
    """Compute a value-weighted return with positive market-cap weights."""
    returns = pd.to_numeric(group[return_col], errors="coerce")
    weights = pd.to_numeric(group[weight_col], errors="coerce")
    valid = returns.notna() & weights.notna() & (weights > 0)
    if not valid.any():
        return np.nan
    weights = weights.loc[valid]
    returns = returns.loc[valid]
    return (returns * weights / weights.sum()).sum()


def t_stat(series):
    """Compute the t-stat of a sample mean."""
    series = pd.to_numeric(series, errors="coerce").dropna()
    if len(series) <= 1 or series.std() == 0:
        return np.nan
    return series.mean() / (series.std() / np.sqrt(len(series)))


def compute_quintile_average_returns(panel_q, output_path=None):
    """Compute average next-month returns by IV spread quintile."""
    monthly_rows = []
    for (signal_month, quintile), group in panel_q.groupby(["signal_month", "iv_spread_quintile"]):
        monthly_rows.append(
            {
                "signal_month": signal_month,
                "quintile": int(quintile),
                "ew_return": group["ret_fwd_1m"].mean(),
                "vw_return": value_weighted_return(group),
                "n_stocks": group["permno"].nunique(),
            }
        )

    monthly_returns = pd.DataFrame(monthly_rows)

    rows = []
    for quintile in range(1, 6):
        group = monthly_returns.loc[monthly_returns["quintile"] == quintile]
        ew_monthly = group["ew_return"].mean()
        vw_monthly = group["vw_return"].mean()
        avg_n = group["n_stocks"].mean()
        rows.append(
            {
                "quintile": f"Q{quintile}",
                "ew_mean_monthly_return": ew_monthly,
                "vw_mean_monthly_return": vw_monthly,
                "ew_annualized_return": ew_monthly * 12,
                "vw_annualized_return": vw_monthly * 12,
                "avg_n_stocks": avg_n,
            }
        )

    result = pd.DataFrame(rows)
    q1 = result.loc[result["quintile"] == "Q1"].iloc[0]
    q5 = result.loc[result["quintile"] == "Q5"].iloc[0]
    spread_row = {
        "quintile": "Q5-Q1",
        "ew_mean_monthly_return": q5["ew_mean_monthly_return"] - q1["ew_mean_monthly_return"],
        "vw_mean_monthly_return": q5["vw_mean_monthly_return"] - q1["vw_mean_monthly_return"],
        "ew_annualized_return": q5["ew_annualized_return"] - q1["ew_annualized_return"],
        "vw_annualized_return": q5["vw_annualized_return"] - q1["vw_annualized_return"],
        "avg_n_stocks": np.nan,
    }
    result = pd.concat([result, pd.DataFrame([spread_row])], ignore_index=True)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output_path, index=False)
        print(f"Saved quintile average returns: {output_path}")

    return result


def compute_long_short_decomposition(panel_q, output_path=None):
    """Decompose IV spread long-short returns into long and short legs."""
    rows = []
    for signal_month, month in panel_q.groupby("signal_month"):
        q1 = month.loc[month["iv_spread_quintile"] == 1]
        q5 = month.loc[month["iv_spread_quintile"] == 5]

        row = {
            "signal_month": signal_month,
            "return_month": month["return_month"].iloc[0],
            "q1_ew_return": q1["ret_fwd_1m"].mean(),
            "q5_ew_return": q5["ret_fwd_1m"].mean(),
            "q1_vw_return": value_weighted_return(q1),
            "q5_vw_return": value_weighted_return(q5),
            "q1_avg_mktcap": q1["mktcap"].mean(),
            "q5_avg_mktcap": q5["mktcap"].mean(),
            "q1_avg_iv_spread": q1["iv_spread_adj"].mean(),
            "q5_avg_iv_spread": q5["iv_spread_adj"].mean(),
        }
        row["ls_ew_return"] = row["q5_ew_return"] - row["q1_ew_return"]
        row["ls_vw_return"] = row["q5_vw_return"] - row["q1_vw_return"]
        rows.append(row)

    monthly = pd.DataFrame(rows)

    summary_rows = []
    for weighting in ["ew", "vw"]:
        q1_col = f"q1_{weighting}_return"
        q5_col = f"q5_{weighting}_return"
        ls_col = f"ls_{weighting}_return"
        summary_rows.append(
            {
                "diagnostic": f"long_short_decomposition_{weighting}",
                "weighting": weighting,
                "mean_monthly_q1": monthly[q1_col].mean(),
                "mean_monthly_q5": monthly[q5_col].mean(),
                "mean_monthly_ls": monthly[ls_col].mean(),
                "annualized_q1": monthly[q1_col].mean() * 12,
                "annualized_q5": monthly[q5_col].mean() * 12,
                "annualized_ls": monthly[ls_col].mean() * 12,
                "tstat_q1": t_stat(monthly[q1_col]),
                "tstat_q5": t_stat(monthly[q5_col]),
                "tstat_ls": t_stat(monthly[ls_col]),
                "vol_q1": monthly[q1_col].std(),
                "vol_q5": monthly[q5_col].std(),
                "vol_ls": monthly[ls_col].std(),
                "positive_month_share_ls": (monthly[ls_col] > 0).mean(),
                "avg_q1_mktcap": monthly["q1_avg_mktcap"].mean(),
                "avg_q5_mktcap": monthly["q5_avg_mktcap"].mean(),
                "avg_q1_iv_spread": monthly["q1_avg_iv_spread"].mean(),
                "avg_q5_iv_spread": monthly["q5_avg_iv_spread"].mean(),
            }
        )

    summary = pd.DataFrame(summary_rows)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        monthly.to_csv(output_path, index=False)
        print(f"Saved long-short decomposition: {output_path}")

    return monthly, summary


def _prepare_returns_with_return_month(iv_spread_returns_df):
    """Ensure a return dataframe has signal_month and return_month period columns."""
    returns = iv_spread_returns_df.copy()
    returns.loc[:, "signal_month"] = pd.PeriodIndex(returns["signal_month"].astype(str), freq="M")
    if "return_month" in returns.columns:
        returns.loc[:, "return_month"] = pd.PeriodIndex(returns["return_month"].astype(str), freq="M")
    else:
        returns.loc[:, "return_month"] = returns["signal_month"] + 1
    returns.loc[:, "LS"] = pd.to_numeric(returns["LS"], errors="coerce")
    return returns


def compute_by_year_performance(iv_spread_returns_df, output_path=None):
    """Compute IV spread long-short performance by return year."""
    returns = _prepare_returns_with_return_month(iv_spread_returns_df)
    returns.loc[:, "return_year"] = pd.PeriodIndex(returns["return_month"], freq="M").year

    rows = []
    for year, group in returns.groupby("return_year"):
        ls = group["LS"].dropna()
        monthly_vol = ls.std()
        annual_vol = monthly_vol * np.sqrt(12)
        annual_return = ls.mean() * 12
        rows.append(
            {
                "return_year": year,
                "n_months": len(ls),
                "mean_monthly_ls": ls.mean(),
                "annualized_ls": annual_return,
                "monthly_volatility": monthly_vol,
                "annualized_volatility": annual_vol,
                "sharpe_ratio": annual_return / annual_vol if annual_vol != 0 else np.nan,
                "t_stat": t_stat(ls),
                "positive_month_pct": (ls > 0).mean(),
                "min_monthly_ls": ls.min(),
                "max_monthly_ls": ls.max(),
            }
        )

    result = pd.DataFrame(rows)
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output_path, index=False)
        print(f"Saved by-year performance: {output_path}")
    return result


def run_covid_exclusion_tests(iv_spread_returns_df, output_path=None):
    """Compute IV spread long-short performance excluding COVID windows."""
    returns = _prepare_returns_with_return_month(iv_spread_returns_df)

    samples = {
        "full_sample": pd.Series(True, index=returns.index),
        "exclude_mar_apr_2020": ~returns["return_month"].isin(
            [pd.Period("2020-03", freq="M"), pd.Period("2020-04", freq="M")]
        ),
        "exclude_mar_to_jun_2020": ~returns["return_month"].isin(
            pd.period_range("2020-03", "2020-06", freq="M")
        ),
        "exclude_2020_full_year": pd.PeriodIndex(returns["return_month"], freq="M").year != 2020,
    }

    rows = []
    for sample_name, mask in samples.items():
        ls = returns.loc[mask, "LS"].dropna()
        monthly_vol = ls.std()
        annual_vol = monthly_vol * np.sqrt(12)
        annual_return = ls.mean() * 12
        rows.append(
            {
                "sample": sample_name,
                "n_months": len(ls),
                "mean_monthly_ls": ls.mean(),
                "annualized_ls": annual_return,
                "monthly_volatility": monthly_vol,
                "annualized_volatility": annual_vol,
                "sharpe_ratio": annual_return / annual_vol if annual_vol != 0 else np.nan,
                "t_stat": t_stat(ls),
                "positive_month_pct": (ls > 0).mean(),
            }
        )

    result = pd.DataFrame(rows)
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output_path, index=False)
        print(f"Saved COVID exclusion tests: {output_path}")
    return result


def compute_quintile_characteristics(panel_q, output_path=None):
    """Compute average characteristics by IV spread quintile."""
    rows = []
    for quintile in range(1, 6):
        group = panel_q.loc[panel_q["iv_spread_quintile"] == quintile]
        rows.append(
            {
                "quintile": f"Q{quintile}",
                "avg_mktcap": group["mktcap"].mean(),
                "median_mktcap": group["mktcap"].median(),
                "avg_realized_var": group["realized_var"].mean(),
                "avg_implied_var": group["implied_var"].mean(),
                "avg_iv_atm_call": group["iv_atm_call"].mean(),
                "avg_iv_atm_put": group["iv_atm_put"].mean(),
                "avg_iv_otm_put": group["iv_otm_put"].mean(),
                "avg_iv_spread": group["iv_spread"].mean(),
                "avg_iv_skew": group["iv_skew"].mean(),
                "avg_vrp": group["vrp"].mean(),
                "avg_ret_fwd_1m": group["ret_fwd_1m"].mean(),
                "annualized_avg_ret_fwd_1m": group["ret_fwd_1m"].mean() * 12,
                "avg_n_stocks_per_month": group.groupby("signal_month")["permno"].nunique().mean(),
            }
        )

    result = pd.DataFrame(rows)
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output_path, index=False)
        print(f"Saved quintile characteristics: {output_path}")
    return result


def extract_factor_alpha_for_iv_spread(factor_summary, output_path=None):
    """Extract clean factor alpha table for IV spread portfolios."""
    portfolios = ["IV Spread EW All", "IV Spread EW MktCap100M"]
    columns = [
        "portfolio",
        "model",
        "alpha_annualized",
        "alpha_tstat",
        "alpha_pvalue",
        "r_squared",
        "n_months",
    ]
    alpha = factor_summary.loc[factor_summary["portfolio"].isin(portfolios), columns].copy()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        alpha.to_csv(output_path, index=False)
        print(f"Saved IV spread factor alpha table: {output_path}")
    return alpha


def plot_iv_spread_quintile_returns(quintile_returns_df, charts_dir):
    """Plot annualized Q1-Q5 returns for EW and VW portfolios."""
    charts_dir = Path(charts_dir)
    charts_dir.mkdir(parents=True, exist_ok=True)

    plot_data = quintile_returns_df.loc[quintile_returns_df["quintile"].isin([f"Q{i}" for i in range(1, 6)])]
    ax = plot_data.set_index("quintile")[["ew_annualized_return", "vw_annualized_return"]].plot(
        kind="bar",
        figsize=(9, 5),
    )
    ax.set_title("IV Spread Quintile Annualized Returns")
    ax.set_xlabel("IV spread quintile")
    ax.set_ylabel("Annualized return")
    ax.legend(["Equal-weighted", "Value-weighted"])
    plt.xticks(rotation=0)
    plt.tight_layout()
    output_path = charts_dir / "iv_spread_quintile_returns.png"
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved chart: {output_path}")


def plot_iv_spread_cumulative_ls(iv_spread_returns_df, charts_dir, iv_spread_100m_returns_df=None):
    """Plot cumulative IV spread long-short returns."""
    charts_dir = Path(charts_dir)
    charts_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    for label, returns_df in [
        ("All stocks EW", iv_spread_returns_df),
        ("MktCap >= 100M EW", iv_spread_100m_returns_df),
    ]:
        if returns_df is None:
            continue
        returns = _prepare_returns_with_return_month(returns_df).dropna(subset=["LS"]).copy()
        returns.loc[:, "cumulative_growth"] = (1 + returns["LS"]).cumprod()
        plt.plot(returns["return_month"].astype(str), returns["cumulative_growth"], label=label)

    plt.title("IV Spread Cumulative Long-Short Returns")
    plt.xlabel("Return month")
    plt.ylabel("Growth of $1")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    output_path = charts_dir / "iv_spread_cumulative_ls.png"
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved chart: {output_path}")


def plot_iv_spread_annual_ls_returns(by_year_df, charts_dir):
    """Plot annualized IV spread LS returns by return year."""
    charts_dir = Path(charts_dir)
    charts_dir.mkdir(parents=True, exist_ok=True)
    ax = by_year_df.set_index("return_year")["annualized_ls"].plot(kind="bar", figsize=(9, 5))
    ax.set_title("IV Spread Annualized Long-Short Return by Year")
    ax.set_xlabel("Return year")
    ax.set_ylabel("Annualized LS return")
    plt.xticks(rotation=0)
    plt.tight_layout()
    output_path = charts_dir / "iv_spread_annual_ls_returns.png"
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved chart: {output_path}")


def plot_iv_spread_quintile_characteristics(characteristics_df, charts_dir):
    """Plot selected characteristics by IV spread quintile."""
    charts_dir = Path(charts_dir)
    charts_dir.mkdir(parents=True, exist_ok=True)

    plot_data = characteristics_df.set_index("quintile")[["avg_mktcap", "avg_realized_var", "avg_implied_var"]]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, column in zip(axes, plot_data.columns):
        plot_data[column].plot(kind="bar", ax=ax)
        ax.set_title(column)
        ax.set_xlabel("Quintile")
        ax.tick_params(axis="x", rotation=0)
    plt.tight_layout()
    output_path = charts_dir / "iv_spread_quintile_characteristics.png"
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved chart: {output_path}")


def plot_iv_spread_factor_alpha(alpha_df, charts_dir):
    """Plot annualized factor alphas for IV spread portfolios."""
    charts_dir = Path(charts_dir)
    charts_dir.mkdir(parents=True, exist_ok=True)

    pivot = alpha_df.pivot(index="model", columns="portfolio", values="alpha_annualized")
    ax = pivot.plot(kind="bar", figsize=(10, 5))
    ax.set_title("IV Spread Factor Alpha")
    ax.set_xlabel("Factor model")
    ax.set_ylabel("Annualized alpha")
    plt.xticks(rotation=0)
    plt.tight_layout()
    output_path = charts_dir / "iv_spread_factor_alpha.png"
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved chart: {output_path}")


def build_diagnostics_summary(
    quintile_returns,
    decomposition_summary,
    by_year,
    covid_summary,
    characteristics,
    alpha_df,
):
    """Build a compact combined diagnostics summary table."""
    rows = []

    spread_row = quintile_returns.loc[quintile_returns["quintile"] == "Q5-Q1"].iloc[0]
    rows.append(
        {
            "diagnostic": "quintile_return_spread",
            "metric": "Q5-Q1 EW annualized return",
            "value": spread_row["ew_annualized_return"],
            "note": "Positive means Q5 outperforms Q1.",
        }
    )

    best_year = by_year.sort_values("annualized_ls", ascending=False).iloc[0]
    rows.append(
        {
            "diagnostic": "by_year",
            "metric": "best return year",
            "value": best_year["annualized_ls"],
            "note": f"Best year is {int(best_year['return_year'])} based on {int(best_year['n_months'])} return months.",
        }
    )

    full = covid_summary.loc[covid_summary["sample"] == "full_sample"].iloc[0]
    exclude_2020 = covid_summary.loc[covid_summary["sample"] == "exclude_2020_full_year"].iloc[0]
    rows.append(
        {
            "diagnostic": "covid_exclusion",
            "metric": "annualized LS excluding 2020",
            "value": exclude_2020["annualized_ls"],
            "note": f"Full-sample annualized LS is {full['annualized_ls']:.4f}.",
        }
    )

    q1_mktcap = characteristics.loc[characteristics["quintile"] == "Q1", "avg_mktcap"].iloc[0]
    q5_mktcap = characteristics.loc[characteristics["quintile"] == "Q5", "avg_mktcap"].iloc[0]
    rows.append(
        {
            "diagnostic": "characteristics",
            "metric": "Q5 average market cap / Q1 average market cap",
            "value": q5_mktcap / q1_mktcap if q1_mktcap != 0 else np.nan,
            "note": "Less than 1 means Q5 is smaller on average.",
        }
    )

    ff5_mom = alpha_df.loc[
        (alpha_df["portfolio"] == "IV Spread EW All") & (alpha_df["model"] == "FF5_MOM")
    ].iloc[0]
    rows.append(
        {
            "diagnostic": "factor_alpha",
            "metric": "IV Spread EW All FF5_MOM alpha t-stat",
            "value": ff5_mom["alpha_tstat"],
            "note": f"Annualized alpha is {ff5_mom['alpha_annualized']:.4f}.",
        }
    )

    summary = pd.concat([decomposition_summary, pd.DataFrame(rows)], ignore_index=True, sort=False)
    return summary
