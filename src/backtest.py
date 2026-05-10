"""Functions for portfolio sorts and simple backtests."""

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


def load_backtest_panel(processed_data_dir):
    """Load the monthly signal panel used for portfolio sorts."""
    processed_data_dir = Path(processed_data_dir)
    panel_path = processed_data_dir / "monthly_signal_panel.parquet"

    panel = pd.read_parquet(panel_path)
    print(f"Loaded monthly signal panel: {panel.shape}")

    return panel


def assign_quintiles(group, signal_col, n_quantiles=5):
    """Assign signal quintiles within one signal_month group."""
    group = group.dropna(subset=[signal_col]).copy()
    group.loc[:, "quintile"] = pd.NA

    if group.empty or group[signal_col].nunique() < n_quantiles:
        return group

    try:
        quintile_codes = pd.qcut(
            group[signal_col],
            q=n_quantiles,
            labels=False,
            duplicates="drop",
        )
    except ValueError:
        return group

    if quintile_codes.nunique(dropna=True) < n_quantiles:
        return group

    group.loc[:, "quintile"] = (quintile_codes + 1).astype("Int64")
    return group


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

    if value_weighted:
        weights = pd.to_numeric(df[weight_col], errors="coerce")
        valid = returns.notna() & weights.notna() & (weights > 0)
        if not valid.any():
            return np.nan

        returns = returns.loc[valid]
        weights = weights.loc[valid]
        weight_sum = weights.sum()
        if weight_sum <= 0:
            return np.nan

        normalized_weights = weights / weight_sum
        return (returns * normalized_weights).sum()

    return returns.mean()


def run_quintile_sort(
    panel,
    signal_col,
    return_col="ret_fwd_1m",
    weight_col="mktcap",
    value_weighted=True,
):
    """Run monthly quintile sorts for one signal."""
    print("\n" + "=" * 80)
    weighting_label = "value-weighted" if value_weighted else "equal-weighted"
    print(f"Running {weighting_label} quintile sort: {signal_col}")
    print("=" * 80)

    needed_columns = ["signal_month", signal_col, return_col, weight_col]
    available_columns = [column for column in needed_columns if column in panel.columns]
    sort_data = panel[available_columns].copy()
    sort_data = sort_data.dropna(subset=["signal_month", signal_col, return_col])

    if "mktcap" in sort_data.columns:
        sort_data.loc[:, "mktcap"] = pd.to_numeric(sort_data["mktcap"], errors="coerce")

    rows = []
    for signal_month, month_data in sort_data.groupby("signal_month", sort=True):
        month_data = assign_quintiles(month_data, signal_col)
        month_data = month_data.dropna(subset=["quintile"])

        result_row = {"signal_month": signal_month, "n_stocks": len(month_data)}

        for quintile in range(1, 6):
            quintile_data = month_data.loc[month_data["quintile"] == quintile]
            result_row[f"Q{quintile}"] = compute_weighted_return(
                quintile_data,
                return_col=return_col,
                weight_col=weight_col,
                value_weighted=value_weighted,
            )
            result_row[f"n_Q{quintile}"] = len(quintile_data)

        result_row["LS"] = result_row["Q5"] - result_row["Q1"]
        rows.append(result_row)

    results = pd.DataFrame(rows)
    ordered_columns = [
        "signal_month",
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
    ]
    results = results[ordered_columns]

    print(f"\nMonths processed: {len(results):,}")
    print(f"Average stocks per month: {results['n_stocks'].mean():,.1f}")

    return results


def summarize_long_short(results_df, signal_name, value_weighted=True):
    """Summarize monthly long-short returns for one signal."""
    ls_returns = pd.to_numeric(results_df["LS"], errors="coerce").dropna()
    n_months = len(ls_returns)

    mean_monthly_ls = ls_returns.mean()
    monthly_volatility = ls_returns.std()
    annualized_ls = mean_monthly_ls * 12
    annualized_volatility = monthly_volatility * np.sqrt(12)

    sharpe_ratio = (
        annualized_ls / annualized_volatility
        if pd.notna(annualized_volatility) and annualized_volatility != 0
        else np.nan
    )
    t_stat = (
        mean_monthly_ls / (monthly_volatility / np.sqrt(n_months))
        if n_months > 1 and pd.notna(monthly_volatility) and monthly_volatility != 0
        else np.nan
    )
    positive_month_pct = (ls_returns > 0).mean() if n_months else np.nan

    weighting = "vw" if value_weighted else "ew"
    summary = pd.DataFrame(
        [
            {
                "signal": signal_name,
                "weighting": weighting,
                "mean_monthly_ls": mean_monthly_ls,
                "annualized_ls": annualized_ls,
                "monthly_volatility": monthly_volatility,
                "annualized_volatility": annualized_volatility,
                "sharpe_ratio": sharpe_ratio,
                "t_stat": t_stat,
                "min_monthly_ls": ls_returns.min() if n_months else np.nan,
                "max_monthly_ls": ls_returns.max() if n_months else np.nan,
                "n_months": n_months,
                "positive_month_pct": positive_month_pct,
            }
        ]
    )

    return summary


def run_all_quintile_sorts(panel, signal_cols, output_tables_dir):
    """Run and save value-weighted and equal-weighted quintile sorts."""
    output_tables_dir = Path(output_tables_dir)
    output_tables_dir.mkdir(parents=True, exist_ok=True)

    results_dict = {}
    summaries = []

    for signal_col in signal_cols:
        vw_results = run_quintile_sort(panel, signal_col, value_weighted=True)
        vw_path = output_tables_dir / f"quintile_returns_{signal_col}_vw.csv"
        vw_results.to_csv(vw_path, index=False)
        print(f"Saved value-weighted returns: {vw_path}")

        ew_results = run_quintile_sort(panel, signal_col, value_weighted=False)
        ew_path = output_tables_dir / f"quintile_returns_{signal_col}_ew.csv"
        ew_results.to_csv(ew_path, index=False)
        print(f"Saved equal-weighted returns: {ew_path}")

        results_dict[(signal_col, "vw")] = vw_results
        results_dict[(signal_col, "ew")] = ew_results

        summaries.append(summarize_long_short(vw_results, signal_col, value_weighted=True))
        summaries.append(summarize_long_short(ew_results, signal_col, value_weighted=False))

    summary = pd.concat(summaries, ignore_index=True)
    summary_path = output_tables_dir / "quintile_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"\nSaved quintile summary: {summary_path}")

    return results_dict, summary


def plot_cumulative_long_short(results_dict, charts_dir, weighting="vw"):
    """Plot cumulative growth of one dollar in long-short portfolios."""
    charts_dir = Path(charts_dir)
    charts_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))

    for (signal_name, result_weighting), results in results_dict.items():
        if result_weighting != weighting:
            continue

        plot_data = results[["signal_month", "LS"]].copy()
        plot_data = plot_data.dropna(subset=["LS"])
        plot_data.loc[:, "cumulative_growth"] = (1 + plot_data["LS"]).cumprod()
        plt.plot(
            plot_data["signal_month"].astype(str),
            plot_data["cumulative_growth"],
            label=signal_name,
        )

    plt.title(f"Cumulative Long-Short Returns ({weighting.upper()})")
    plt.xlabel("Signal Month")
    plt.ylabel("Growth of $1")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()

    output_path = charts_dir / f"cumulative_long_short_returns_{weighting}.png"
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved cumulative long-short chart: {output_path}")


def plot_quintile_average_returns(results_dict, charts_dir, weighting="vw"):
    """Plot annualized average returns for Q1-Q5 portfolios."""
    charts_dir = Path(charts_dir)
    charts_dir.mkdir(parents=True, exist_ok=True)

    quintile_columns = ["Q1", "Q2", "Q3", "Q4", "Q5"]
    rows = []
    for (signal_name, result_weighting), results in results_dict.items():
        if result_weighting != weighting:
            continue

        annualized_returns = results[quintile_columns].mean() * 12
        for quintile in quintile_columns:
            rows.append(
                {
                    "signal": signal_name,
                    "quintile": quintile,
                    "annualized_return": annualized_returns[quintile],
                }
            )

    plot_data = pd.DataFrame(rows)
    pivot = plot_data.pivot(index="signal", columns="quintile", values="annualized_return")

    ax = pivot.plot(kind="bar", figsize=(10, 6))
    ax.set_title(f"Annualized Average Quintile Returns ({weighting.upper()})")
    ax.set_xlabel("Signal")
    ax.set_ylabel("Annualized average return")
    ax.legend(title="Quintile")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    output_path = charts_dir / f"quintile_average_returns_{weighting}.png"
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved quintile average return chart: {output_path}")


def validate_backtest_inputs(panel, signal_cols):
    """Print validation details for the monthly backtest panel."""
    print("\n" + "=" * 80)
    print("Backtest Input Validation")
    print("=" * 80)

    print(f"\nShape: {panel.shape}")
    print(f"signal_month range: {panel['signal_month'].min()} to {panel['signal_month'].max()}")
    print(f"Unique permnos: {panel['permno'].nunique():,}")

    check_columns = signal_cols + ["ret_fwd_1m", "mktcap"]
    print("\nMissing values for backtest columns:")
    print(panel[check_columns].isna().sum().to_string())

    summary_columns = signal_cols + ["ret_fwd_1m"]
    print("\nSignal and return summary stats:")
    print(panel[summary_columns].describe().to_string())

    stocks_per_month = panel.groupby("signal_month")["permno"].nunique()
    print("\nNumber of stocks per month summary:")
    print(stocks_per_month.describe().to_string())


def apply_universe_filter(panel, universe_name):
    """Apply a pre-specified market-cap universe filter."""
    thresholds = {
        "all": None,
        "mktcap_100m": 100,
        "mktcap_500m": 500,
        "mktcap_1b": 1000,
    }
    if universe_name not in thresholds:
        raise ValueError(f"Unknown universe filter: {universe_name}")

    rows_before = len(panel)
    permnos_before = panel["permno"].nunique()

    filtered = panel.copy()
    if thresholds[universe_name] is not None:
        filtered = filtered.loc[filtered["mktcap"] >= thresholds[universe_name]].copy()

    rows_after = len(filtered)
    permnos_after = filtered["permno"].nunique()
    avg_stocks_per_month = filtered.groupby("signal_month")["permno"].nunique().mean()

    print("\n" + "-" * 80)
    print(f"Universe filter: {universe_name}")
    print(f"Rows before/after: {rows_before:,} -> {rows_after:,}")
    print(f"Unique permnos before/after: {permnos_before:,} -> {permnos_after:,}")
    print(f"Average stocks per month: {avg_stocks_per_month:,.1f}")

    return filtered


def add_signal_transform(panel, signal_col, transform):
    """Add one transformed signal column for robustness portfolio sorts."""
    if transform not in ["raw", "rank", "winsor_z"]:
        raise ValueError(f"Unknown signal transform: {transform}")

    transformed_col = f"{signal_col}_{transform}"
    transformed = panel.copy()

    if transform == "raw":
        transformed.loc[:, transformed_col] = transformed[signal_col]
        return transformed, transformed_col

    if transform == "rank":
        transformed.loc[:, transformed_col] = transformed.groupby("signal_month")[signal_col].transform(
            lambda series: series.rank(method="average", pct=True)
        )
        return transformed, transformed_col

    def winsorized_zscore(series):
        lower = series.quantile(0.01)
        upper = series.quantile(0.99)
        clipped = series.clip(lower=lower, upper=upper)
        std = clipped.std()
        if pd.isna(std) or std == 0:
            return pd.Series(pd.NA, index=series.index, dtype="Float64")
        return (clipped - clipped.mean()) / std

    transformed.loc[:, transformed_col] = transformed.groupby("signal_month")[signal_col].transform(
        winsorized_zscore
    )
    return transformed, transformed_col


def run_robustness_quintile_sorts(panel, signal_cols, universe_filters, transforms, output_tables_dir):
    """Run quintile sorts across universe filters and signal transformations."""
    output_tables_dir = Path(output_tables_dir)
    output_tables_dir.mkdir(parents=True, exist_ok=True)

    results_dict = {}
    summary_rows = []

    for universe_name in universe_filters:
        universe_panel = apply_universe_filter(panel, universe_name)

        for signal_col in signal_cols:
            for transform in transforms:
                transformed_panel, transformed_col = add_signal_transform(
                    universe_panel,
                    signal_col,
                    transform,
                )

                for weighting in ["vw", "ew"]:
                    value_weighted = weighting == "vw"
                    print(
                        "\nRobustness sort: "
                        f"signal={signal_col}, transform={transform}, "
                        f"universe={universe_name}, weighting={weighting}"
                    )

                    results = run_quintile_sort(
                        transformed_panel,
                        transformed_col,
                        value_weighted=value_weighted,
                    )
                    output_path = (
                        output_tables_dir
                        / (
                            "robustness_quintile_returns_"
                            f"{signal_col}_{transform}_{universe_name}_{weighting}.csv"
                        )
                    )
                    results.to_csv(output_path, index=False)
                    print(f"Saved robustness returns: {output_path}")

                    results_key = (signal_col, transform, universe_name, weighting)
                    results_dict[results_key] = results

                    summary = summarize_long_short(
                        results,
                        signal_col,
                        value_weighted=value_weighted,
                    )
                    summary.loc[:, "transform"] = transform
                    summary.loc[:, "universe"] = universe_name
                    summary.loc[:, "avg_n_stocks"] = results["n_stocks"].mean()
                    summary.loc[:, "avg_n_Q1"] = results["n_Q1"].mean()
                    summary.loc[:, "avg_n_Q5"] = results["n_Q5"].mean()

                    ordered_summary_columns = [
                        "signal",
                        "transform",
                        "universe",
                        "weighting",
                        "mean_monthly_ls",
                        "annualized_ls",
                        "monthly_volatility",
                        "annualized_volatility",
                        "sharpe_ratio",
                        "t_stat",
                        "min_monthly_ls",
                        "max_monthly_ls",
                        "n_months",
                        "positive_month_pct",
                        "avg_n_stocks",
                        "avg_n_Q1",
                        "avg_n_Q5",
                    ]
                    summary_rows.append(summary[ordered_summary_columns])

    summary_df = pd.concat(summary_rows, ignore_index=True)
    summary_path = output_tables_dir / "robustness_quintile_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSaved robustness quintile summary: {summary_path}")

    return results_dict, summary_df


def plot_robustness_summary(summary_df, charts_dir):
    """Plot value-weighted Sharpe ratios across robustness configurations."""
    charts_dir = Path(charts_dir)
    charts_dir.mkdir(parents=True, exist_ok=True)

    vw_summary = summary_df.loc[summary_df["weighting"] == "vw"].copy()
    vw_summary.loc[:, "combo"] = vw_summary["signal"] + "\n" + vw_summary["transform"]
    pivot = vw_summary.pivot_table(
        index="combo",
        columns="universe",
        values="sharpe_ratio",
        aggfunc="first",
    )
    desired_universe_order = ["all", "mktcap_100m", "mktcap_500m", "mktcap_1b"]
    pivot = pivot[[column for column in desired_universe_order if column in pivot.columns]]

    fig, ax = plt.subplots(figsize=(10, 8))
    image = ax.imshow(pivot.to_numpy(), aspect="auto")
    fig.colorbar(image, ax=ax, label="Sharpe ratio")

    ax.set_title("Value-Weighted Robustness Sharpe Ratios")
    ax.set_xlabel("Universe")
    ax.set_ylabel("Signal and Transform")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)

    for row_idx in range(len(pivot.index)):
        for col_idx in range(len(pivot.columns)):
            value = pivot.iloc[row_idx, col_idx]
            if pd.notna(value):
                ax.text(col_idx, row_idx, f"{value:.2f}", ha="center", va="center")

    plt.tight_layout()
    output_path = charts_dir / "robustness_summary_heatmap.png"
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved robustness summary chart: {output_path}")


def plot_selected_cumulative_ls(results_dict, summary_df, charts_dir):
    """Plot cumulative LS returns for the top value-weighted robustness configs."""
    charts_dir = Path(charts_dir)
    charts_dir.mkdir(parents=True, exist_ok=True)

    top_configs = (
        summary_df.loc[summary_df["weighting"] == "vw"]
        .dropna(subset=["sharpe_ratio"])
        .sort_values("sharpe_ratio", ascending=False)
        .head(4)
    )

    plt.figure(figsize=(10, 6))

    for _, row in top_configs.iterrows():
        key = (row["signal"], row["transform"], row["universe"], row["weighting"])
        results = results_dict[key]
        plot_data = results[["signal_month", "LS"]].dropna(subset=["LS"]).copy()
        plot_data.loc[:, "cumulative_growth"] = (1 + plot_data["LS"]).cumprod()
        label = f"{row['signal']} | {row['transform']} | {row['universe']}"
        plt.plot(
            plot_data["signal_month"].astype(str),
            plot_data["cumulative_growth"],
            label=label,
        )

    plt.title("Selected Robustness Long-Short Returns")
    plt.xlabel("Signal Month")
    plt.ylabel("Growth of $1")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()

    output_path = charts_dir / "robustness_cumulative_ls_selected.png"
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved selected robustness cumulative LS chart: {output_path}")
