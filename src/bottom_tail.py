"""Bottom-tail diagnostics for low IV spread underperformance."""

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


REQUIRED_COLUMNS = [
    "permno",
    "signal_month",
    "return_month",
    "iv_spread_adj",
    "ret_fwd_1m",
    "mktcap",
]

UNIVERSE_FILTERS = {
    "all": None,
    "mktcap_100m": 100,
    "mktcap_500m": 500,
    "mktcap_1b": 1000,
}

TAIL_GROUPS = {"decile": 10, "quintile": 5}


def load_bottom_tail_panel(processed_data_dir):
    """Load the monthly signal panel for bottom-tail diagnostics."""
    processed_data_dir = Path(processed_data_dir)
    panel_path = processed_data_dir / "monthly_signal_panel.parquet"
    panel = pd.read_parquet(panel_path)

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in panel.columns]

    print("\n" + "=" * 80)
    print("Loading Monthly Panel for Bottom-Tail Diagnostics")
    print("=" * 80)
    print(f"Loaded panel: {panel.shape}")
    print(f"Missing required columns: {missing_columns if missing_columns else 'None'}")

    if missing_columns:
        raise ValueError(f"monthly_signal_panel is missing required columns: {missing_columns}")

    panel = panel[REQUIRED_COLUMNS].copy()
    panel = panel.assign(
        permno=pd.to_numeric(panel["permno"], errors="coerce").astype("Int64"),
        signal_month=pd.PeriodIndex(panel["signal_month"].astype(str), freq="M"),
        return_month=pd.PeriodIndex(panel["return_month"].astype(str), freq="M"),
        iv_spread_adj=pd.to_numeric(panel["iv_spread_adj"], errors="coerce"),
        ret_fwd_1m=pd.to_numeric(panel["ret_fwd_1m"], errors="coerce"),
        mktcap=pd.to_numeric(panel["mktcap"], errors="coerce"),
    )
    panel = panel.dropna(subset=REQUIRED_COLUMNS)

    print(f"signal_month range: {panel['signal_month'].min()} to {panel['signal_month'].max()}")
    print(f"return_month range: {panel['return_month'].min()} to {panel['return_month'].max()}")
    print(f"Unique permnos: {panel['permno'].nunique():,}")

    return panel


def assign_bottom_tail_groups(panel, tail="decile", universe="all"):
    """Identify low-IV-spread bottom-tail stocks within each signal month."""
    if tail not in TAIL_GROUPS:
        raise ValueError(f"Unknown tail definition: {tail}")
    if universe not in UNIVERSE_FILTERS:
        raise ValueError(f"Unknown universe: {universe}")

    n_groups = TAIL_GROUPS[tail]
    mktcap_cutoff = UNIVERSE_FILTERS[universe]

    data = panel.copy()
    if mktcap_cutoff is not None:
        data = data.loc[data["mktcap"] >= mktcap_cutoff].copy()

    data = data.dropna(subset=["signal_month", "iv_spread_adj", "ret_fwd_1m", "mktcap"])
    data.loc[:, "tail_group"] = pd.NA
    data.loc[:, "bottom_tail"] = 0
    data.loc[:, "top_tail"] = 0
    data.loc[:, "tail"] = tail
    data.loc[:, "universe"] = universe

    for _, month_data in data.groupby("signal_month", sort=True):
        valid = month_data.dropna(subset=["iv_spread_adj"])
        if valid["iv_spread_adj"].nunique() < n_groups:
            continue

        try:
            group_codes = pd.qcut(
                valid["iv_spread_adj"],
                q=n_groups,
                labels=False,
                duplicates="drop",
            )
        except ValueError:
            continue

        if group_codes.nunique(dropna=True) < n_groups:
            continue

        group_numbers = (group_codes + 1).astype("Int64")
        data.loc[valid.index, "tail_group"] = group_numbers
        data.loc[valid.index, "bottom_tail"] = (group_numbers == 1).astype(int)
        data.loc[valid.index, "top_tail"] = (group_numbers == n_groups).astype(int)

    data = data.dropna(subset=["tail_group"]).copy()
    data.loc[:, "tail_group"] = data["tail_group"].astype("Int64")
    data.loc[:, "bottom_tail"] = data["bottom_tail"].astype(int)
    data.loc[:, "top_tail"] = data["top_tail"].astype(int)

    stocks_per_month = data.groupby("signal_month")["permno"].nunique()
    bottom_per_month = (
        data.loc[data["bottom_tail"] == 1]
        .groupby("signal_month")["permno"]
        .nunique()
    )

    print("\n" + "=" * 80)
    print(f"Assigned Bottom-Tail Groups: tail={tail}, universe={universe}")
    print("=" * 80)
    print(f"Rows after universe filter and assignment: {len(data):,}")
    print(f"Months: {data['signal_month'].nunique():,}")
    print(f"Average stocks per month: {stocks_per_month.mean():,.1f}")
    print(f"Average bottom-tail stocks per month: {bottom_per_month.mean():,.1f}")

    return data


def _weighted_return(df, value_weighted=False):
    """Compute equal-weighted or value-weighted return."""
    if df.empty:
        return np.nan

    returns = pd.to_numeric(df["ret_fwd_1m"], errors="coerce")
    if not value_weighted:
        return returns.mean()

    weights = pd.to_numeric(df["mktcap"], errors="coerce")
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
    std = series.std()
    if pd.isna(std) or std == 0:
        return np.nan
    return series.mean() / (std / np.sqrt(len(series)))


def compute_bottom_tail_returns(panel_groups):
    """Compute monthly bottom-tail underperformance legs."""
    rows = []
    tail = panel_groups["tail"].iloc[0]
    universe = panel_groups["universe"].iloc[0]

    for signal_month, month_data in panel_groups.groupby("signal_month", sort=True):
        bottom = month_data.loc[month_data["bottom_tail"] == 1]
        top = month_data.loc[month_data["top_tail"] == 1]

        for weighting, value_weighted in [("ew", False), ("vw", True)]:
            universe_ret = _weighted_return(month_data, value_weighted=value_weighted)
            bottom_ret = _weighted_return(bottom, value_weighted=value_weighted)
            top_ret = _weighted_return(top, value_weighted=value_weighted)

            rows.append(
                {
                    "signal_month": signal_month,
                    "return_month": month_data["return_month"].iloc[0],
                    "tail": tail,
                    "universe": universe,
                    "weighting": weighting,
                    "universe_ret": universe_ret,
                    "bottom_tail_ret": bottom_ret,
                    "top_tail_ret": top_ret,
                    "universe_minus_bottom": universe_ret - bottom_ret,
                    "top_minus_bottom": top_ret - bottom_ret,
                    "top_minus_universe": top_ret - universe_ret,
                    "n_universe": len(month_data),
                    "n_bottom": len(bottom),
                    "n_top": len(top),
                    "avg_mktcap_universe": month_data["mktcap"].mean(),
                    "avg_mktcap_bottom": bottom["mktcap"].mean(),
                    "avg_mktcap_top": top["mktcap"].mean(),
                    "avg_iv_spread_universe": month_data["iv_spread_adj"].mean(),
                    "avg_iv_spread_bottom": bottom["iv_spread_adj"].mean(),
                    "avg_iv_spread_top": top["iv_spread_adj"].mean(),
                }
            )

    monthly_returns = pd.DataFrame(rows)
    print(
        f"Computed bottom-tail monthly returns: tail={tail}, universe={universe}, "
        f"rows={len(monthly_returns):,}"
    )
    return monthly_returns


def summarize_bottom_tail_returns(monthly_returns):
    """Summarize bottom-tail relative returns by tail, universe, weighting, and leg."""
    legs = ["universe_minus_bottom", "top_minus_bottom", "top_minus_universe"]
    rows = []

    for (tail, universe, weighting), group in monthly_returns.groupby(
        ["tail", "universe", "weighting"],
        sort=True,
    ):
        for leg in legs:
            series = pd.to_numeric(group[leg], errors="coerce").dropna()
            monthly_vol = series.std()
            annual_vol = monthly_vol * np.sqrt(12)
            annual_return = series.mean() * 12
            rows.append(
                {
                    "tail": tail,
                    "universe": universe,
                    "weighting": weighting,
                    "leg": leg,
                    "mean_monthly_return": series.mean(),
                    "annualized_return": annual_return,
                    "monthly_volatility": monthly_vol,
                    "annualized_volatility": annual_vol,
                    "sharpe_ratio": annual_return / annual_vol
                    if pd.notna(annual_vol) and annual_vol != 0
                    else np.nan,
                    "t_stat": _t_stat(series),
                    "positive_month_pct": (series > 0).mean() if len(series) else np.nan,
                    "n_months": len(series),
                    "avg_n_universe": group["n_universe"].mean(),
                    "avg_n_bottom": group["n_bottom"].mean(),
                    "avg_n_top": group["n_top"].mean(),
                    "avg_mktcap_universe": group["avg_mktcap_universe"].mean(),
                    "avg_mktcap_bottom": group["avg_mktcap_bottom"].mean(),
                    "avg_mktcap_top": group["avg_mktcap_top"].mean(),
                    "avg_iv_spread_universe": group["avg_iv_spread_universe"].mean(),
                    "avg_iv_spread_bottom": group["avg_iv_spread_bottom"].mean(),
                    "avg_iv_spread_top": group["avg_iv_spread_top"].mean(),
                }
            )

    summary = pd.DataFrame(rows)
    print("\nBottom-tail summary:")
    print(
        summary.loc[summary["leg"] == "universe_minus_bottom"]
        .sort_values(["tail", "universe", "weighting"])
        .to_string(index=False)
    )
    return summary


def compute_bottom_tail_by_year(monthly_returns, tail="decile", universe="all", weighting="ew"):
    """Compute by-year performance for the main universe-minus-bottom leg."""
    data = monthly_returns.loc[
        (monthly_returns["tail"] == tail)
        & (monthly_returns["universe"] == universe)
        & (monthly_returns["weighting"] == weighting)
    ].copy()
    data.loc[:, "return_year"] = pd.PeriodIndex(data["return_month"].astype(str), freq="M").year

    rows = []
    for return_year, group in data.groupby("return_year", sort=True):
        series = pd.to_numeric(group["universe_minus_bottom"], errors="coerce").dropna()
        rows.append(
            {
                "return_year": int(return_year),
                "n_months": len(series),
                "mean_monthly_return": series.mean(),
                "annualized_return": series.mean() * 12,
                "t_stat": _t_stat(series),
                "positive_month_pct": (series > 0).mean() if len(series) else np.nan,
                "min": series.min(),
                "max": series.max(),
                "partial_year": bool(len(series) < 6),
            }
        )

    by_year = pd.DataFrame(rows)
    print("\nBottom-tail by-year results:")
    print(by_year.to_string(index=False))
    return by_year


def run_bottom_tail_factor_alpha(monthly_returns, raw_data_dir):
    """Run FF5 + Momentum alphas for selected bottom-tail strategies."""
    selected_configs = [
        ("decile", "all", "ew"),
        ("decile", "mktcap_100m", "ew"),
        ("quintile", "all", "ew"),
        ("quintile", "mktcap_100m", "ew"),
        ("decile", "all", "vw"),
        ("decile", "mktcap_100m", "vw"),
    ]
    legs = ["universe_minus_bottom", "top_minus_bottom", "top_minus_universe"]
    factor_cols = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "MOM"]

    print("\n" + "=" * 80)
    print("Running Bottom-Tail FF5 + Momentum Alpha Checks")
    print("=" * 80)

    try:
        from src.regressions import load_factor_data, run_factor_regression

        factor_df = load_factor_data(raw_data_dir)
    except Exception as exc:
        print(f"Skipping factor alpha checks because factor data could not be loaded: {exc}")
        return pd.DataFrame()

    rows = []
    for tail, universe, weighting in selected_configs:
        subset = monthly_returns.loc[
            (monthly_returns["tail"] == tail)
            & (monthly_returns["universe"] == universe)
            & (monthly_returns["weighting"] == weighting)
        ].copy()

        for leg in legs:
            returns = subset[["signal_month", "return_month", leg]].copy()
            returns = returns.rename(columns={leg: "LS"})
            returns = returns.dropna(subset=["signal_month", "return_month", "LS"])
            if returns.empty:
                continue

            label = f"{tail}_{universe}_{weighting}_{leg}"
            result = run_factor_regression(
                returns,
                factor_df,
                portfolio_label=label,
                model_name="FF5_MOM",
                factor_cols=factor_cols,
                nw_lags=4,
            )
            rows.append(
                {
                    "tail": tail,
                    "universe": universe,
                    "weighting": weighting,
                    "leg": leg,
                    "alpha_monthly": result["alpha_monthly"],
                    "alpha_annualized": result["alpha_annualized"],
                    "alpha_tstat": result["alpha_tstat"],
                    "alpha_pvalue": result["alpha_pvalue"],
                    "r_squared": result["r_squared"],
                    "n_months": result["n_months"],
                    "first_signal_month": result["first_signal_month"],
                    "first_return_month": result["first_return_month"],
                    "last_signal_month": result["last_signal_month"],
                    "last_return_month": result["last_return_month"],
                }
            )
            print(
                f"{label}: alpha={result['alpha_annualized']:.2%}, "
                f"t-stat={result['alpha_tstat']:.2f}"
            )

    return pd.DataFrame(rows)


def _prepare_cumulative_data(data, leg):
    """Prepare cumulative growth data for one return leg."""
    plot_data = data.dropna(subset=[leg]).copy()
    plot_data = plot_data.sort_values("return_month")
    plot_data.loc[:, "cumulative_growth"] = (1 + plot_data[leg]).cumprod()
    return plot_data


def plot_bottom_tail_cumulative(monthly_returns, charts_dir):
    """Plot cumulative universe-minus-bottom returns for key configurations."""
    charts_dir = Path(charts_dir)
    charts_dir.mkdir(parents=True, exist_ok=True)

    configs = [
        ("decile", "all", "ew", "Decile All EW"),
        ("decile", "mktcap_100m", "ew", "Decile MktCap100M EW"),
        ("quintile", "all", "ew", "Quintile All EW"),
    ]

    plt.figure(figsize=(10, 6))
    for tail, universe, weighting, label in configs:
        data = monthly_returns.loc[
            (monthly_returns["tail"] == tail)
            & (monthly_returns["universe"] == universe)
            & (monthly_returns["weighting"] == weighting)
        ]
        if data.empty:
            continue
        plot_data = _prepare_cumulative_data(data, "universe_minus_bottom")
        plt.plot(plot_data["return_month"].astype(str), plot_data["cumulative_growth"], label=label)

    plt.title("IV Spread Bottom-Tail Cumulative Relative Returns")
    plt.xlabel("Return month")
    plt.ylabel("Growth of $1")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    output_path = charts_dir / "iv_spread_bottom_tail_cumulative.png"
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved chart: {output_path}")


def plot_bottom_tail_by_year(by_year_df, charts_dir):
    """Plot annualized bottom-tail underperformance by return year."""
    charts_dir = Path(charts_dir)
    charts_dir.mkdir(parents=True, exist_ok=True)

    plot_data = by_year_df.copy()
    plot_data.loc[:, "year_label"] = plot_data["return_year"].astype(str)
    plot_data.loc[plot_data["partial_year"], "year_label"] = (
        plot_data.loc[plot_data["partial_year"], "year_label"] + "*"
    )

    ax = plot_data.set_index("year_label")["annualized_return"].plot(kind="bar", figsize=(9, 5))
    ax.set_title("IV Spread Bottom-Tail Underperformance by Year")
    ax.set_xlabel("Return year (* partial)")
    ax.set_ylabel("Annualized Universe - Bottom return")
    plt.xticks(rotation=0)
    plt.tight_layout()
    output_path = charts_dir / "iv_spread_bottom_tail_by_year.png"
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved chart: {output_path}")


def plot_bottom_tail_factor_alpha(alpha_df, charts_dir):
    """Plot FF5 + Momentum alphas for universe-minus-bottom strategies."""
    charts_dir = Path(charts_dir)
    charts_dir.mkdir(parents=True, exist_ok=True)

    plot_data = alpha_df.loc[alpha_df["leg"] == "universe_minus_bottom"].copy()
    if plot_data.empty:
        print("Skipping factor alpha chart: alpha table is empty.")
        return

    plot_data.loc[:, "label"] = (
        plot_data["tail"] + " / " + plot_data["universe"] + " / " + plot_data["weighting"]
    )
    ax = plot_data.set_index("label")["alpha_annualized"].plot(kind="bar", figsize=(11, 5))
    ax.set_title("IV Spread Bottom-Tail FF5 + Momentum Alpha")
    ax.set_xlabel("")
    ax.set_ylabel("Annualized alpha")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    output_path = charts_dir / "iv_spread_bottom_tail_factor_alpha.png"
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved chart: {output_path}")


def plot_bottom_tail_robustness(summary_df, charts_dir):
    """Plot annualized bottom-tail underperformance across key configurations."""
    charts_dir = Path(charts_dir)
    charts_dir.mkdir(parents=True, exist_ok=True)

    configs = [
        ("decile", "all", "ew"),
        ("decile", "mktcap_100m", "ew"),
        ("quintile", "all", "ew"),
        ("quintile", "mktcap_100m", "ew"),
        ("decile", "all", "vw"),
        ("decile", "mktcap_100m", "vw"),
    ]

    rows = []
    for tail, universe, weighting in configs:
        match = summary_df.loc[
            (summary_df["tail"] == tail)
            & (summary_df["universe"] == universe)
            & (summary_df["weighting"] == weighting)
            & (summary_df["leg"] == "universe_minus_bottom")
        ]
        if match.empty:
            continue
        row = match.iloc[0].copy()
        row["label"] = f"{tail}/{universe}/{weighting}"
        rows.append(row)

    plot_data = pd.DataFrame(rows)
    if plot_data.empty:
        print("Skipping robustness chart: no matching rows.")
        return

    ax = plot_data.set_index("label")["annualized_return"].plot(kind="bar", figsize=(11, 5))
    ax.set_title("IV Spread Bottom-Tail Robustness")
    ax.set_xlabel("")
    ax.set_ylabel("Annualized Universe - Bottom return")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    output_path = charts_dir / "iv_spread_bottom_tail_robustness_bars.png"
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved chart: {output_path}")
