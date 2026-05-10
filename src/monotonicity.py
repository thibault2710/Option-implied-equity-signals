"""Monotonicity diagnostics for the IV spread signal."""

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


def load_monotonicity_panel(processed_data_dir):
    """Load the monthly signal panel for IV spread monotonicity diagnostics."""
    processed_data_dir = Path(processed_data_dir)
    panel_path = processed_data_dir / "monthly_signal_panel.parquet"
    panel = pd.read_parquet(panel_path)

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in panel.columns]

    print("\n" + "=" * 80)
    print("Loading Monthly Panel for Monotonicity Diagnostics")
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

    print(f"signal_month range: {panel['signal_month'].min()} to {panel['signal_month'].max()}")
    print(f"return_month range: {panel['return_month'].min()} to {panel['return_month'].max()}")
    print(f"Unique permnos: {panel['permno'].nunique():,}")

    return panel


def assign_deciles(panel, signal_col="iv_spread_adj"):
    """Assign IV spread deciles within each signal month."""
    print("\n" + "=" * 80)
    print("Assigning IV Spread Deciles")
    print("=" * 80)

    data = panel.copy()
    data = data.dropna(subset=["signal_month", signal_col, "ret_fwd_1m"])
    data.loc[:, "iv_spread_decile"] = pd.NA

    for _, month_data in data.groupby("signal_month", sort=True):
        valid = month_data.dropna(subset=[signal_col])
        if valid[signal_col].nunique() < 10:
            continue

        try:
            decile_codes = pd.qcut(
                valid[signal_col],
                q=10,
                labels=False,
                duplicates="drop",
            )
        except ValueError:
            continue

        if decile_codes.nunique(dropna=True) < 10:
            continue

        data.loc[valid.index, "iv_spread_decile"] = (decile_codes + 1).astype("Int64")

    data = data.dropna(subset=["iv_spread_decile"]).copy()
    data.loc[:, "iv_spread_decile"] = data["iv_spread_decile"].astype("Int64")

    counts = (
        data.groupby(["signal_month", "iv_spread_decile"])["permno"]
        .nunique()
        .unstack("iv_spread_decile")
    )

    print(f"Rows with deciles: {len(data):,}")
    print(f"Number of months: {data['signal_month'].nunique():,}")
    print("\nAverage stocks per decile:")
    print(counts.mean().rename(lambda value: f"D{int(value)}").to_string())

    return data


def _compute_weighted_return(df, value_weighted=False):
    """Compute an equal-weighted or value-weighted return."""
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


def compute_decile_returns(panel_deciles):
    """Compute equal-weighted and value-weighted monthly returns by decile."""
    print("\n" + "=" * 80)
    print("Computing Decile Returns")
    print("=" * 80)

    rows = []
    for signal_month, month_data in panel_deciles.groupby("signal_month", sort=True):
        for weighting, value_weighted in [("ew", False), ("vw", True)]:
            row = {
                "signal_month": signal_month,
                "return_month": month_data["return_month"].iloc[0],
                "weighting": weighting,
            }

            for decile in range(1, 11):
                decile_data = month_data.loc[month_data["iv_spread_decile"] == decile]
                row[f"D{decile}"] = _compute_weighted_return(
                    decile_data,
                    value_weighted=value_weighted,
                )
                row[f"n_D{decile}"] = len(decile_data)

            row["LS"] = row["D10"] - row["D1"]
            rows.append(row)

    decile_monthly_returns = pd.DataFrame(rows)
    ordered_columns = (
        ["signal_month", "return_month", "weighting"]
        + [f"D{decile}" for decile in range(1, 11)]
        + ["LS"]
        + [f"n_D{decile}" for decile in range(1, 11)]
    )
    decile_monthly_returns = decile_monthly_returns[ordered_columns]

    summary_rows = []
    ew_returns = decile_monthly_returns.loc[decile_monthly_returns["weighting"] == "ew"]
    vw_returns = decile_monthly_returns.loc[decile_monthly_returns["weighting"] == "vw"]

    for decile in range(1, 11):
        decile_label = f"D{decile}"
        ew_mean = ew_returns[decile_label].mean()
        vw_mean = vw_returns[decile_label].mean()
        summary_rows.append(
            {
                "decile": decile_label,
                "decile_number": decile,
                "ew_mean_monthly_return": ew_mean,
                "ew_annualized_return": ew_mean * 12,
                "vw_mean_monthly_return": vw_mean,
                "vw_annualized_return": vw_mean * 12,
                "avg_n_stocks": ew_returns[f"n_D{decile}"].mean(),
            }
        )

    decile_summary = pd.DataFrame(summary_rows)

    print(f"Monthly return rows: {len(decile_monthly_returns):,}")
    print(f"Months covered: {decile_monthly_returns['signal_month'].nunique():,}")
    print("\nDecile average returns:")
    print(decile_summary.to_string(index=False))

    return decile_monthly_returns, decile_summary


def compute_monotonicity_tests(decile_summary, decile_monthly_returns):
    """Quantify whether IV spread decile returns are monotonic."""
    print("\n" + "=" * 80)
    print("Computing Monotonicity Tests")
    print("=" * 80)

    summary = decile_summary.sort_values("decile_number").copy()
    decile_numbers = summary["decile_number"].astype(float)
    ew_returns = summary["ew_annualized_return"].astype(float)
    vw_returns = summary["vw_annualized_return"].astype(float)

    ew_adjacent_increases = int((ew_returns.diff().iloc[1:] > 0).sum())
    vw_adjacent_increases = int((vw_returns.diff().iloc[1:] > 0).sum())
    ew_spearman = decile_numbers.corr(ew_returns, method="spearman")
    vw_spearman = decile_numbers.corr(vw_returns, method="spearman")

    ew_ls = decile_monthly_returns.loc[
        decile_monthly_returns["weighting"] == "ew",
        "LS",
    ]
    vw_ls = decile_monthly_returns.loc[
        decile_monthly_returns["weighting"] == "vw",
        "LS",
    ]

    rows = [
        {
            "metric": "EW D10 > D1",
            "value": bool(ew_returns.iloc[-1] > ew_returns.iloc[0]),
            "interpretation": "True means the highest IV spread decile beats the lowest decile.",
        },
        {
            "metric": "VW D10 > D1",
            "value": bool(vw_returns.iloc[-1] > vw_returns.iloc[0]),
            "interpretation": "True means the value-weighted highest IV spread decile beats the lowest decile.",
        },
        {
            "metric": "EW adjacent increases",
            "value": ew_adjacent_increases,
            "interpretation": "Number of adjacent decile return increases out of 9.",
        },
        {
            "metric": "VW adjacent increases",
            "value": vw_adjacent_increases,
            "interpretation": "Number of adjacent decile return increases out of 9.",
        },
        {
            "metric": "EW decile-return Spearman",
            "value": ew_spearman,
            "interpretation": "Spearman correlation between decile number and average EW return.",
        },
        {
            "metric": "VW decile-return Spearman",
            "value": vw_spearman,
            "interpretation": "Spearman correlation between decile number and average VW return.",
        },
        {
            "metric": "EW average monthly D10-D1",
            "value": ew_ls.mean(),
            "interpretation": "Average monthly equal-weighted D10-D1 return.",
        },
        {
            "metric": "EW D10-D1 t-stat",
            "value": _t_stat(ew_ls),
            "interpretation": "T-statistic of monthly equal-weighted D10-D1 returns.",
        },
        {
            "metric": "VW average monthly D10-D1",
            "value": vw_ls.mean(),
            "interpretation": "Average monthly value-weighted D10-D1 return.",
        },
        {
            "metric": "VW D10-D1 t-stat",
            "value": _t_stat(vw_ls),
            "interpretation": "T-statistic of monthly value-weighted D10-D1 returns.",
        },
    ]

    tests = pd.DataFrame(rows)
    print(tests.to_string(index=False))
    return tests


def compute_monthly_rank_ic(panel, signal_col="iv_spread_adj", return_col="ret_fwd_1m"):
    """Compute monthly cross-sectional rank information coefficients."""
    print("\n" + "=" * 80)
    print("Computing Monthly Rank IC")
    print("=" * 80)

    try:
        from scipy.stats import spearmanr
    except Exception:
        spearmanr = None

    data = panel.dropna(subset=["signal_month", "return_month", signal_col, return_col]).copy()
    rows = []

    for signal_month, month_data in data.groupby("signal_month", sort=True):
        valid = month_data.dropna(subset=[signal_col, return_col])
        if len(valid) < 2:
            spearman_ic = np.nan
            pearson_ic = np.nan
        elif valid[signal_col].nunique() < 2 or valid[return_col].nunique() < 2:
            spearman_ic = np.nan
            pearson_ic = np.nan
        else:
            if spearmanr is not None:
                spearman_ic = spearmanr(valid[signal_col], valid[return_col]).correlation
            else:
                spearman_ic = valid[signal_col].corr(valid[return_col], method="spearman")
            pearson_ic = valid[signal_col].corr(valid[return_col], method="pearson")

        rows.append(
            {
                "signal_month": signal_month,
                "return_month": valid["return_month"].iloc[0] if len(valid) else pd.NaT,
                "spearman_ic": spearman_ic,
                "pearson_ic": pearson_ic,
                "n_stocks": len(valid),
            }
        )

    rank_ic = pd.DataFrame(rows)
    print(f"Rank IC rows: {len(rank_ic):,}")
    print(f"Mean Spearman IC: {rank_ic['spearman_ic'].mean():.4f}")
    print(f"Mean Pearson IC: {rank_ic['pearson_ic'].mean():.4f}")
    return rank_ic


def summarize_rank_ic(rank_ic_df):
    """Summarize monthly rank IC overall and by return year."""
    print("\n" + "=" * 80)
    print("Summarizing Rank IC")
    print("=" * 80)

    spearman_ic = pd.to_numeric(rank_ic_df["spearman_ic"], errors="coerce").dropna()
    pearson_ic = pd.to_numeric(rank_ic_df["pearson_ic"], errors="coerce").dropna()

    rank_ic_summary = pd.DataFrame(
        [
            {"metric": "mean_spearman_ic", "value": spearman_ic.mean()},
            {"metric": "median_spearman_ic", "value": spearman_ic.median()},
            {"metric": "std_spearman_ic", "value": spearman_ic.std()},
            {"metric": "t_stat_spearman_ic", "value": _t_stat(spearman_ic)},
            {"metric": "percent_positive_spearman_ic", "value": (spearman_ic > 0).mean()},
            {"metric": "mean_pearson_ic", "value": pearson_ic.mean()},
            {"metric": "t_stat_pearson_ic", "value": _t_stat(pearson_ic)},
            {"metric": "n_months", "value": len(spearman_ic)},
        ]
    )

    by_year_data = rank_ic_df.copy()
    by_year_data.loc[:, "return_year"] = pd.PeriodIndex(
        by_year_data["return_month"].astype(str),
        freq="M",
    ).year

    by_year_rows = []
    for return_year, group in by_year_data.groupby("return_year", sort=True):
        values = pd.to_numeric(group["spearman_ic"], errors="coerce").dropna()
        by_year_rows.append(
            {
                "return_year": int(return_year),
                "mean_spearman_ic": values.mean(),
                "t_stat": _t_stat(values),
                "percent_positive": (values > 0).mean() if len(values) else np.nan,
                "n_months": len(values),
            }
        )
    rank_ic_by_year = pd.DataFrame(by_year_rows)

    print("\nRank IC summary:")
    print(rank_ic_summary.to_string(index=False))
    print("\nRank IC by year:")
    print(rank_ic_by_year.to_string(index=False))

    return rank_ic_summary, rank_ic_by_year


def compute_tail_vs_universe(panel_deciles):
    """Compare top and bottom IV spread tails with the full universe."""
    print("\n" + "=" * 80)
    print("Computing Tail-vs-Universe Decomposition")
    print("=" * 80)

    rows = []
    for signal_month, month_data in panel_deciles.groupby("signal_month", sort=True):
        d1 = month_data.loc[month_data["iv_spread_decile"] == 1]
        d10 = month_data.loc[month_data["iv_spread_decile"] == 10]

        universe_ew = _compute_weighted_return(month_data, value_weighted=False)
        d1_ew = _compute_weighted_return(d1, value_weighted=False)
        d10_ew = _compute_weighted_return(d10, value_weighted=False)

        universe_vw = _compute_weighted_return(month_data, value_weighted=True)
        d1_vw = _compute_weighted_return(d1, value_weighted=True)
        d10_vw = _compute_weighted_return(d10, value_weighted=True)

        rows.append(
            {
                "signal_month": signal_month,
                "return_month": month_data["return_month"].iloc[0],
                "universe_ew": universe_ew,
                "d1_ew": d1_ew,
                "d10_ew": d10_ew,
                "ew_d10_minus_d1": d10_ew - d1_ew,
                "ew_d10_minus_universe": d10_ew - universe_ew,
                "ew_universe_minus_d1": universe_ew - d1_ew,
                "universe_vw": universe_vw,
                "d1_vw": d1_vw,
                "d10_vw": d10_vw,
                "vw_d10_minus_d1": d10_vw - d1_vw,
                "vw_d10_minus_universe": d10_vw - universe_vw,
                "vw_universe_minus_d1": universe_vw - d1_vw,
            }
        )

    monthly = pd.DataFrame(rows)

    summary_rows = []
    leg_map = {
        "D10_minus_D1": "d10_minus_d1",
        "D10_minus_Universe": "d10_minus_universe",
        "Universe_minus_D1": "universe_minus_d1",
    }
    for weighting in ["ew", "vw"]:
        for leg, column_suffix in leg_map.items():
            column = f"{weighting}_{column_suffix}"
            series = pd.to_numeric(monthly[column], errors="coerce").dropna()
            monthly_vol = series.std()
            annual_vol = monthly_vol * np.sqrt(12)
            annual_return = series.mean() * 12
            summary_rows.append(
                {
                    "leg": leg,
                    "weighting": weighting,
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
                }
            )

    summary = pd.DataFrame(summary_rows)
    print("\nTail-vs-universe summary:")
    print(summary.to_string(index=False))

    return monthly, summary


def plot_decile_returns(decile_summary, charts_dir):
    """Plot annualized EW and VW returns by IV spread decile."""
    charts_dir = Path(charts_dir)
    charts_dir.mkdir(parents=True, exist_ok=True)

    plot_data = decile_summary.set_index("decile")[
        ["ew_annualized_return", "vw_annualized_return"]
    ]
    ax = plot_data.plot(kind="bar", figsize=(10, 5))
    ax.set_title("IV Spread Decile Annualized Returns")
    ax.set_xlabel("IV spread decile")
    ax.set_ylabel("Annualized return")
    ax.legend(["Equal-weighted", "Value-weighted"])
    plt.xticks(rotation=0)
    plt.tight_layout()
    output_path = charts_dir / "iv_spread_decile_returns.png"
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved chart: {output_path}")


def plot_rank_ic_by_year(rank_ic_by_year, charts_dir):
    """Plot mean Spearman rank IC by return year."""
    charts_dir = Path(charts_dir)
    charts_dir.mkdir(parents=True, exist_ok=True)

    ax = rank_ic_by_year.set_index("return_year")["mean_spearman_ic"].plot(
        kind="bar",
        figsize=(9, 5),
    )
    ax.set_title("IV Spread Mean Spearman Rank IC by Year")
    ax.set_xlabel("Return year")
    ax.set_ylabel("Mean Spearman IC")
    plt.xticks(rotation=0)
    plt.tight_layout()
    output_path = charts_dir / "iv_spread_rank_ic_by_year.png"
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved chart: {output_path}")


def plot_tail_vs_universe(tail_summary, charts_dir):
    """Plot annualized tail-vs-universe decomposition returns."""
    charts_dir = Path(charts_dir)
    charts_dir.mkdir(parents=True, exist_ok=True)

    plot_data = tail_summary.pivot(
        index="leg",
        columns="weighting",
        values="annualized_return",
    )
    plot_data = plot_data.reindex(
        ["D10_minus_D1", "D10_minus_Universe", "Universe_minus_D1"]
    )
    ax = plot_data.plot(kind="bar", figsize=(10, 5))
    ax.set_title("IV Spread Tail-vs-Universe Annualized Returns")
    ax.set_xlabel("")
    ax.set_ylabel("Annualized return")
    ax.legend(["Equal-weighted", "Value-weighted"])
    plt.xticks(rotation=20)
    plt.tight_layout()
    output_path = charts_dir / "iv_spread_tail_vs_universe.png"
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved chart: {output_path}")


def plot_monthly_ic_cumulative(rank_ic_df, charts_dir):
    """Plot cumulative sum of monthly Spearman rank IC."""
    charts_dir = Path(charts_dir)
    charts_dir.mkdir(parents=True, exist_ok=True)

    data = rank_ic_df.dropna(subset=["spearman_ic"]).copy()
    data = data.sort_values("return_month")
    data.loc[:, "cumulative_spearman_ic"] = data["spearman_ic"].cumsum()

    plt.figure(figsize=(10, 5))
    plt.plot(data["return_month"].astype(str), data["cumulative_spearman_ic"])
    plt.title("Cumulative Monthly Spearman IC")
    plt.xlabel("Return month")
    plt.ylabel("Cumulative Spearman IC")
    plt.xticks(rotation=45)
    plt.tight_layout()
    output_path = charts_dir / "iv_spread_monthly_ic_cumulative.png"
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved chart: {output_path}")
