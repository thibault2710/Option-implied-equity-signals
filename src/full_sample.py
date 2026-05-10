"""Helpers for the 2010-2023 sample-specific expansion pipeline."""

import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils import compute_weighted_return, safe_mean, summarize_return_series


MPLCONFIGDIR = Path(tempfile.gettempdir()) / "options_implied_signals_matplotlib"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick


SIGNAL_COLUMNS = ["iv_spread_adj", "iv_skew_adj", "vrp_adj", "composite_signal"]
UNIVERSE_FILTERS = {
    "all": None,
    "mktcap_100m": 100,
    "mktcap_500m": 500,
    "mktcap_1b": 1000,
}


def ensure_output_dirs(tables_dir, charts_dir):
    """Create sample-specific output folders."""
    tables_dir = Path(tables_dir)
    charts_dir = Path(charts_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)
    charts_dir.mkdir(parents=True, exist_ok=True)
    return tables_dir, charts_dir


def save_table(df, output_path):
    """Save a table and print its path."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved table: {output_path} shape={df.shape}")


def prepare_monthly_panel(panel):
    """Normalize monthly panel types for full-sample sorts."""
    data = panel.copy()
    data.loc[:, "signal_month"] = pd.PeriodIndex(data["signal_month"].astype(str), freq="M")
    data.loc[:, "return_month"] = pd.PeriodIndex(data["return_month"].astype(str), freq="M")
    for column in ["permno", "mktcap", "ret_fwd_1m"] + SIGNAL_COLUMNS:
        if column in data.columns:
            data.loc[:, column] = pd.to_numeric(data[column], errors="coerce")
    return data


def apply_universe_filter(panel, universe):
    """Apply a market-cap universe filter."""
    if universe not in UNIVERSE_FILTERS:
        raise ValueError(f"Unknown universe: {universe}")
    cutoff = UNIVERSE_FILTERS[universe]
    data = panel.copy()
    if cutoff is not None:
        data = data.loc[data["mktcap"] >= cutoff].copy()
    return data


def add_signal_transform(panel, signal_col, transform):
    """Add one transformed signal column within signal_month."""
    data = panel.copy()
    transformed_col = f"{signal_col}_{transform}"
    if transform == "raw":
        data.loc[:, transformed_col] = data[signal_col]
    elif transform == "rank":
        data.loc[:, transformed_col] = data.groupby("signal_month")[signal_col].rank(
            method="average",
            pct=True,
        )
    elif transform == "winsor_z":
        def winsor_z(series):
            q1 = series.quantile(0.01)
            q99 = series.quantile(0.99)
            clipped = series.clip(q1, q99)
            std = clipped.std()
            if pd.isna(std) or std == 0:
                return pd.Series(pd.NA, index=series.index, dtype="Float64")
            return (clipped - clipped.mean()) / std

        data.loc[:, transformed_col] = data.groupby("signal_month")[signal_col].transform(winsor_z)
    else:
        raise ValueError(f"Unknown transform: {transform}")
    return data, transformed_col


def assign_quantiles(group, signal_col, n_quantiles):
    """Assign quantile groups within one month."""
    group = group.dropna(subset=[signal_col]).copy()
    group.loc[:, "quantile"] = pd.NA
    if group.empty or group[signal_col].nunique() < n_quantiles:
        return group
    try:
        codes = pd.qcut(group[signal_col], q=n_quantiles, labels=False, duplicates="drop")
    except ValueError:
        return group
    if codes.nunique(dropna=True) < n_quantiles:
        return group
    group.loc[:, "quantile"] = (codes + 1).astype("Int64")
    return group


def run_quantile_sort(panel, signal_col, n_quantiles=5, value_weighted=False):
    """Run monthly quantile sorts for one signal."""
    data = panel[["signal_month", "return_month", signal_col, "ret_fwd_1m", "mktcap"]].copy()
    data = data.dropna(subset=["signal_month", "return_month", signal_col, "ret_fwd_1m"])

    rows = []
    for signal_month, month_data in data.groupby("signal_month", sort=True):
        month_data = assign_quantiles(month_data, signal_col, n_quantiles)
        month_data = month_data.dropna(subset=["quantile"])
        row = {
            "signal_month": signal_month,
            "return_month": month_data["return_month"].iloc[0] if not month_data.empty else pd.NaT,
            "n_stocks": len(month_data),
        }
        for quantile in range(1, n_quantiles + 1):
            bucket = month_data.loc[month_data["quantile"] == quantile]
            row[f"Q{quantile}"] = (
                compute_weighted_return(bucket["ret_fwd_1m"], bucket["mktcap"])
                if value_weighted
                else safe_mean(bucket["ret_fwd_1m"])
            )
            row[f"n_Q{quantile}"] = len(bucket)
        row["LS"] = row[f"Q{n_quantiles}"] - row["Q1"]
        rows.append(row)

    results = pd.DataFrame(rows)
    ordered = (
        ["signal_month", "return_month"]
        + [f"Q{i}" for i in range(1, n_quantiles + 1)]
        + ["LS", "n_stocks"]
        + [f"n_Q{i}" for i in range(1, n_quantiles + 1)]
    )
    return results[ordered]


def summarize_ls(results, metadata):
    """Summarize a long-short return series with raw and Newey-West t-stats."""
    summary = summarize_return_series(results["LS"])
    row = {
        **metadata,
        **summary,
        "min_monthly_ls": pd.to_numeric(results["LS"], errors="coerce").min(),
        "max_monthly_ls": pd.to_numeric(results["LS"], errors="coerce").max(),
        "avg_n_stocks": results["n_stocks"].mean(),
        "avg_n_Q1": results["n_Q1"].mean() if "n_Q1" in results.columns else np.nan,
    }
    q_cols = [col for col in results.columns if col.startswith("n_Q")]
    qmax_col = max(q_cols, key=lambda col: int(col.replace("n_Q", ""))) if q_cols else None
    row["avg_n_top"] = results[qmax_col].mean() if qmax_col is not None else np.nan
    return row


def run_signal_sort_suite(panel, tables_dir, charts_dir, sample_label):
    """Run full-sample quintile, decile, and robustness sorts."""
    tables_dir, charts_dir = ensure_output_dirs(tables_dir, charts_dir)
    panel = prepare_monthly_panel(panel)

    quintile_rows = []
    decile_rows = []
    results_for_chart = {}
    for signal_col in SIGNAL_COLUMNS:
        for weighting, value_weighted in [("vw", True), ("ew", False)]:
            quintile = run_quantile_sort(panel, signal_col, n_quantiles=5, value_weighted=value_weighted)
            path = tables_dir / f"quintile_returns_{signal_col}_{weighting}_{sample_label}.csv"
            save_table(quintile, path)
            quintile_rows.append(
                summarize_ls(
                    quintile,
                    {"signal": signal_col, "weighting": weighting, "n_quantiles": 5},
                )
            )
            if weighting == "ew":
                results_for_chart[signal_col] = quintile

            decile = run_quantile_sort(panel, signal_col, n_quantiles=10, value_weighted=value_weighted)
            decile_path = tables_dir / f"decile_returns_{signal_col}_{weighting}_{sample_label}.csv"
            save_table(decile, decile_path)
            decile_rows.append(
                summarize_ls(
                    decile,
                    {"signal": signal_col, "weighting": weighting, "n_quantiles": 10},
                )
            )

    quintile_summary = pd.DataFrame(quintile_rows)
    decile_summary = pd.DataFrame(decile_rows)
    save_table(quintile_summary, tables_dir / f"quintile_summary_{sample_label}.csv")
    save_table(decile_summary, tables_dir / f"decile_summary_{sample_label}.csv")

    robustness_rows = []
    transforms = ["raw", "rank", "winsor_z"]
    for universe in UNIVERSE_FILTERS:
        universe_panel = apply_universe_filter(panel, universe)
        for signal_col in SIGNAL_COLUMNS:
            for transform in transforms:
                transformed_panel, transformed_col = add_signal_transform(universe_panel, signal_col, transform)
                for weighting, value_weighted in [("vw", True), ("ew", False)]:
                    returns = run_quantile_sort(
                        transformed_panel,
                        transformed_col,
                        n_quantiles=5,
                        value_weighted=value_weighted,
                    )
                    output_path = (
                        tables_dir
                        / f"robustness_quintile_returns_{signal_col}_{transform}_{universe}_{weighting}_{sample_label}.csv"
                    )
                    save_table(returns, output_path)
                    robustness_rows.append(
                        summarize_ls(
                            returns,
                            {
                                "signal": signal_col,
                                "transform": transform,
                                "universe": universe,
                                "weighting": weighting,
                            },
                        )
                    )

    robustness_summary = pd.DataFrame(robustness_rows)
    save_table(robustness_summary, tables_dir / f"robustness_quintile_summary_{sample_label}.csv")
    plot_quintile_cumulative(results_for_chart, charts_dir, sample_label)
    return quintile_summary, decile_summary, robustness_summary


def assign_bottom_tail_groups(panel, tail="decile", universe="all", signal_col="iv_spread_adj"):
    """Assign bottom/top tail groups for a signal."""
    n_groups = 10 if tail == "decile" else 5
    data = apply_universe_filter(prepare_monthly_panel(panel), universe)
    data = data.dropna(subset=["signal_month", signal_col, "ret_fwd_1m", "mktcap"]).copy()
    data.loc[:, "tail_group"] = pd.NA

    for _, month_data in data.groupby("signal_month", sort=True):
        valid = month_data.dropna(subset=[signal_col])
        if valid[signal_col].nunique() < n_groups:
            continue
        try:
            codes = pd.qcut(valid[signal_col], q=n_groups, labels=False, duplicates="drop")
        except ValueError:
            continue
        if codes.nunique(dropna=True) < n_groups:
            continue
        data.loc[valid.index, "tail_group"] = (codes + 1).astype("Int64")

    data = data.dropna(subset=["tail_group"]).copy()
    data.loc[:, "tail_group"] = data["tail_group"].astype("Int64")
    data.loc[:, "bottom_tail"] = (data["tail_group"] == 1).astype(int)
    data.loc[:, "top_tail"] = (data["tail_group"] == n_groups).astype(int)
    data.loc[:, "tail"] = tail
    data.loc[:, "universe"] = universe
    data.loc[:, "signal"] = signal_col
    return data


def compute_bottom_tail_returns(panel_groups, value_weighted=False, signal_col=None):
    """Compute monthly universe-minus-bottom and top-minus-bottom returns."""
    if signal_col is None:
        signal_values = panel_groups["signal"].dropna().unique()
        if len(signal_values) != 1:
            raise ValueError(
                "compute_bottom_tail_returns needs an explicit signal_col when "
                f"panel_groups contains {len(signal_values)} signal values."
            )
        signal_col = signal_values[0]
    if signal_col not in panel_groups.columns:
        raise ValueError(f"Signal column not found in panel_groups: {signal_col}")

    rows = []
    weighting = "vw" if value_weighted else "ew"
    for signal_month, month_data in panel_groups.groupby("signal_month", sort=True):
        bottom = month_data.loc[month_data["bottom_tail"] == 1]
        top = month_data.loc[month_data["top_tail"] == 1]

        def ret(group):
            if value_weighted:
                return compute_weighted_return(group["ret_fwd_1m"], group["mktcap"])
            return safe_mean(group["ret_fwd_1m"])

        universe_ret = ret(month_data)
        bottom_ret = ret(bottom)
        top_ret = ret(top)
        rows.append(
            {
                "signal_month": signal_month,
                "return_month": month_data["return_month"].iloc[0],
                "signal": month_data["signal"].iloc[0],
                "tail": month_data["tail"].iloc[0],
                "universe": month_data["universe"].iloc[0],
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
                "avg_signal_universe": month_data[signal_col].mean(),
                "avg_signal_bottom": bottom[signal_col].mean(),
                "avg_signal_top": top[signal_col].mean(),
            }
        )
    return pd.DataFrame(rows)


def summarize_bottom_tail(monthly_returns):
    """Summarize bottom-tail relative-return legs with Newey-West t-stats."""
    legs = ["universe_minus_bottom", "top_minus_bottom", "top_minus_universe"]
    rows = []
    group_cols = ["signal", "tail", "universe", "weighting"]
    for keys, group in monthly_returns.groupby(group_cols, sort=True):
        metadata = dict(zip(group_cols, keys))
        for leg in legs:
            summary = summarize_return_series(group[leg])
            rows.append(
                {
                    **metadata,
                    "leg": leg,
                    **summary,
                    "avg_n_universe": group["n_universe"].mean(),
                    "avg_n_bottom": group["n_bottom"].mean(),
                    "avg_n_top": group["n_top"].mean(),
                    "avg_mktcap_universe": group["avg_mktcap_universe"].mean(),
                    "avg_mktcap_bottom": group["avg_mktcap_bottom"].mean(),
                    "avg_mktcap_top": group["avg_mktcap_top"].mean(),
                    "avg_signal_universe": group["avg_signal_universe"].mean(),
                    "avg_signal_bottom": group["avg_signal_bottom"].mean(),
                    "avg_signal_top": group["avg_signal_top"].mean(),
                }
            )
    return pd.DataFrame(rows)


def run_bottom_tail_suite(panel, tables_dir, charts_dir, sample_label):
    """Run bottom-tail diagnostics for IV spread and save sample-specific outputs."""
    tables_dir, charts_dir = ensure_output_dirs(tables_dir, charts_dir)
    monthly_frames = []
    for signal_col in SIGNAL_COLUMNS:
        tail_definitions = ["decile"] if signal_col != "iv_spread_adj" else ["decile", "quintile"]
        for tail in tail_definitions:
            for universe in UNIVERSE_FILTERS:
                groups = assign_bottom_tail_groups(panel, tail=tail, universe=universe, signal_col=signal_col)
                for value_weighted in [False, True]:
                    monthly_frames.append(
                        compute_bottom_tail_returns(
                            groups,
                            value_weighted=value_weighted,
                            signal_col=signal_col,
                        )
                    )

    monthly_returns = pd.concat(monthly_frames, ignore_index=True)
    summary = summarize_bottom_tail(monthly_returns)
    save_table(monthly_returns, tables_dir / f"bottom_tail_returns_{sample_label}.csv")
    save_table(summary, tables_dir / f"bottom_tail_summary_{sample_label}.csv")
    plot_bottom_tail_cumulative(monthly_returns, charts_dir, sample_label)
    return monthly_returns, summary


def plot_quintile_cumulative(results_dict, charts_dir, sample_label):
    """Plot EW quintile long-short cumulative returns."""
    charts_dir = Path(charts_dir)
    charts_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    for signal, returns in results_dict.items():
        data = returns.dropna(subset=["LS"]).copy()
        data.loc[:, "growth"] = (1 + data["LS"]).cumprod()
        ax.plot(data["return_month"].astype(str), data["growth"], label=signal)
    ax.set_title(f"Full Sample EW Quintile Long-Short Returns ({sample_label})")
    ax.set_xlabel("Return month")
    ax.set_ylabel("Growth of $1")
    ax.legend()
    ax.tick_params(axis="x", labelrotation=45)
    fig.tight_layout()
    output_path = charts_dir / f"quintile_cumulative_ls_{sample_label}.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved chart: {output_path}")


def plot_bottom_tail_cumulative(monthly_returns, charts_dir, sample_label):
    """Plot key IV spread bottom-tail cumulative returns."""
    charts_dir = Path(charts_dir)
    charts_dir.mkdir(parents=True, exist_ok=True)
    configs = [
        ("iv_spread_adj", "decile", "all", "ew", "Decile All EW"),
        ("iv_spread_adj", "decile", "mktcap_100m", "ew", "Decile $100M+ EW"),
        ("iv_spread_adj", "quintile", "all", "ew", "Quintile All EW"),
    ]
    fig, ax = plt.subplots(figsize=(10, 6))
    for signal, tail, universe, weighting, label in configs:
        data = monthly_returns.loc[
            (monthly_returns["signal"] == signal)
            & (monthly_returns["tail"] == tail)
            & (monthly_returns["universe"] == universe)
            & (monthly_returns["weighting"] == weighting)
        ].copy()
        if data.empty:
            continue
        data.loc[:, "growth"] = (1 + data["universe_minus_bottom"]).cumprod()
        ax.plot(data["return_month"].astype(str), data["growth"], label=label)
    ax.set_title(f"Full Sample Low IV-Spread Relative Returns ({sample_label})")
    ax.set_xlabel("Return month")
    ax.set_ylabel("Growth of $1")
    ax.legend()
    ax.tick_params(axis="x", labelrotation=45)
    fig.tight_layout()
    output_path = charts_dir / f"bottom_tail_cumulative_{sample_label}.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved chart: {output_path}")


def plot_bar_summary(summary_df, value_col, label_col, output_path, title):
    """Plot a simple bar summary."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = summary_df.copy()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(data[label_col], data[value_col])
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.set_ylabel(value_col)
    ax.set_title(title)
    ax.tick_params(axis="x", labelrotation=30)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved chart: {output_path}")
