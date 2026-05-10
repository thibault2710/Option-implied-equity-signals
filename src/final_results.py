"""Create final cleaned result tables and charts for the project write-up."""

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
import matplotlib.dates as mdates
import matplotlib.ticker as mtick
import matplotlib.pyplot as plt


FINAL_CONFIGS = [
    ("decile", "all", "ew", "Bottom Decile vs Universe, EW, All"),
    ("decile", "mktcap_100m", "ew", "Bottom Decile vs Universe, EW, MktCap >= $100M"),
    ("quintile", "all", "ew", "Bottom Quintile vs Universe, EW, All"),
    ("quintile", "mktcap_100m", "ew", "Bottom Quintile vs Universe, EW, MktCap >= $100M"),
    ("decile", "all", "vw", "Bottom Decile vs Universe, VW, All"),
    ("decile", "mktcap_100m", "vw", "Bottom Decile vs Universe, VW, MktCap >= $100M"),
]


def _universe_label(universe):
    """Convert universe code to a presentation label."""
    labels = {
        "all": "All",
        "mktcap_100m": "MktCap >= $100M",
        "mktcap_500m": "MktCap >= $500M",
        "mktcap_1b": "MktCap >= $1B",
    }
    return labels.get(universe, universe)


def _tail_label(tail):
    """Convert tail code to a presentation label."""
    labels = {"decile": "Bottom decile", "quintile": "Bottom quintile"}
    return labels.get(tail, tail)


def _strategy_label(tail, universe, weighting):
    """Build a compact strategy label."""
    tail_part = "Decile" if tail == "decile" else "Quintile"
    universe_part = {
        "all": "All",
        "mktcap_100m": "$100M+",
        "mktcap_500m": "$500M+",
        "mktcap_1b": "$1B+",
    }.get(universe, universe)
    return f"{tail_part} {universe_part} {weighting.upper()}"


def _percent(value):
    """Convert a decimal return to percent units."""
    return value * 100 if pd.notna(value) else np.nan


def _save_table(df, output_path):
    """Save a dataframe and print its path and shape."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved table: {output_path} shape={df.shape}")


def load_final_result_inputs(tables_dir, charts_dir, processed_data_dir):
    """Load all saved diagnostics needed for final results."""
    tables_dir = Path(tables_dir)
    charts_dir = Path(charts_dir)
    processed_data_dir = Path(processed_data_dir)

    table_files = {
        "bottom_tail_summary": "iv_spread_bottom_tail_summary.csv",
        "bottom_tail_factor_alpha": "iv_spread_bottom_tail_factor_alpha.csv",
        "bottom_tail_by_year": "iv_spread_bottom_tail_by_year.csv",
        "bottom_tail_robustness": "iv_spread_bottom_tail_robustness.csv",
        "bottom_tail_returns": "iv_spread_bottom_tail_returns.csv",
        "decile_summary": "iv_spread_monotonicity_decile_summary.csv",
        "rank_ic_summary": "iv_spread_rank_ic_summary.csv",
        "tail_vs_universe": "iv_spread_tail_vs_universe.csv",
        "audit_summary": "research_audit_summary.csv",
        "audit_warnings": "research_audit_warnings.csv",
    }

    inputs = {}
    print("\n" + "=" * 80)
    print("Loading Final Result Inputs")
    print("=" * 80)
    for key, file_name in table_files.items():
        path = tables_dir / file_name
        inputs[key] = pd.read_csv(path)
        print(f"{key}: {inputs[key].shape} from {path}")

    panel_path = processed_data_dir / "monthly_signal_panel.parquet"
    inputs["monthly_panel"] = pd.read_parquet(
        panel_path,
        columns=["permno", "signal_month", "return_month"],
    )
    print(f"monthly_panel timing columns: {inputs['monthly_panel'].shape} from {panel_path}")
    print(f"Charts directory: {charts_dir}")

    return inputs


def _merge_robustness_and_alpha(bottom_tail_robustness_df, factor_alpha_df):
    """Merge robustness rows with alpha rows for universe-minus-bottom."""
    robustness = bottom_tail_robustness_df.copy()
    alpha = factor_alpha_df.loc[
        factor_alpha_df["leg"] == "universe_minus_bottom",
        [
            "tail",
            "universe",
            "weighting",
            "alpha_annualized",
            "alpha_tstat",
            "alpha_pvalue",
            "r_squared",
        ],
    ].copy()

    merged = robustness.merge(
        alpha,
        on=["tail", "universe", "weighting"],
        how="left",
        suffixes=("", "_from_alpha_file"),
    )

    for column in ["alpha_annualized", "alpha_tstat", "alpha_pvalue", "r_squared"]:
        from_alpha_column = f"{column}_from_alpha_file"
        if from_alpha_column in merged.columns:
            merged.loc[:, column] = merged[column].combine_first(merged[from_alpha_column])
            merged = merged.drop(columns=[from_alpha_column])

    return merged


def create_final_bottom_tail_main_table(
    bottom_tail_robustness_df,
    factor_alpha_df,
    output_path=None,
):
    """Create the main cleaned bottom-tail result table."""
    merged = _merge_robustness_and_alpha(bottom_tail_robustness_df, factor_alpha_df)
    rows = []

    for tail, universe, weighting, strategy in FINAL_CONFIGS:
        match = merged.loc[
            (merged["tail"] == tail)
            & (merged["universe"] == universe)
            & (merged["weighting"] == weighting)
            & (merged["leg"] == "universe_minus_bottom")
        ]
        if match.empty:
            continue

        row = match.iloc[0]
        rows.append(
            {
                "Strategy": strategy,
                "Weighting": weighting.upper(),
                "Universe": _universe_label(universe),
                "Tail definition": _tail_label(tail),
                "Annualized return": row["annualized_return"],
                "Annualized return (%)": _percent(row["annualized_return"]),
                "Return t-stat": row["t_stat"],
                "Sharpe": row["sharpe_ratio"],
                "Positive months %": _percent(row["positive_month_pct"]),
                "FF5+MOM alpha": row["alpha_annualized"],
                "FF5+MOM alpha (%)": _percent(row["alpha_annualized"]),
                "Alpha t-stat": row["alpha_tstat"],
                "N months": int(row["n_months"]),
            }
        )

    table = pd.DataFrame(rows)
    if output_path is not None:
        _save_table(table, output_path)
    return table


def create_final_factor_alpha_table(factor_alpha_df, output_path=None):
    """Create a clean FF5 + Momentum alpha table."""
    rows = []
    alpha = factor_alpha_df.loc[factor_alpha_df["leg"] == "universe_minus_bottom"].copy()

    for tail, universe, weighting, _ in FINAL_CONFIGS:
        match = alpha.loc[
            (alpha["tail"] == tail)
            & (alpha["universe"] == universe)
            & (alpha["weighting"] == weighting)
        ]
        if match.empty:
            continue
        row = match.iloc[0]
        rows.append(
            {
                "Strategy": _strategy_label(tail, universe, weighting),
                "Annualized alpha": row["alpha_annualized"],
                "Annualized alpha (%)": _percent(row["alpha_annualized"]),
                "Alpha t-stat": row["alpha_tstat"],
                "Alpha p-value": row["alpha_pvalue"],
                "R-squared": row["r_squared"],
                "N months": int(row["n_months"]),
            }
        )

    table = pd.DataFrame(rows)
    if output_path is not None:
        _save_table(table, output_path)
    return table


def create_final_robustness_table(bottom_tail_robustness_df, output_path=None):
    """Create a concise robustness table from saved bottom-tail diagnostics."""
    rows = []
    for tail, universe, weighting, _ in FINAL_CONFIGS:
        match = bottom_tail_robustness_df.loc[
            (bottom_tail_robustness_df["tail"] == tail)
            & (bottom_tail_robustness_df["universe"] == universe)
            & (bottom_tail_robustness_df["weighting"] == weighting)
            & (bottom_tail_robustness_df["leg"] == "universe_minus_bottom")
        ]
        if match.empty:
            continue
        row = match.iloc[0]
        rows.append(
            {
                "Tail definition": _tail_label(tail),
                "Universe": _universe_label(universe),
                "Weighting": weighting.upper(),
                "Annualized return": row["annualized_return"],
                "Annualized return (%)": _percent(row["annualized_return"]),
                "t-stat": row["t_stat"],
                "Sharpe": row["sharpe_ratio"],
                "FF5+MOM alpha": row.get("alpha_annualized", np.nan),
                "FF5+MOM alpha (%)": _percent(row.get("alpha_annualized", np.nan)),
                "alpha t-stat": row.get("alpha_tstat", np.nan),
            }
        )

    table = pd.DataFrame(rows)
    if output_path is not None:
        _save_table(table, output_path)
    return table


def create_final_results_summary(
    bottom_tail_robustness_df,
    factor_alpha_df,
    rank_ic_summary_df,
    audit_summary_df,
    monthly_panel_df,
    output_path=None,
):
    """Create a compact final research summary table."""
    merged = _merge_robustness_and_alpha(bottom_tail_robustness_df, factor_alpha_df)
    main = merged.loc[
        (merged["tail"] == "decile")
        & (merged["universe"] == "all")
        & (merged["weighting"] == "ew")
        & (merged["leg"] == "universe_minus_bottom")
    ].iloc[0]

    rank_ic = rank_ic_summary_df.set_index("metric")["value"]
    pass_count = int((audit_summary_df["status"] == "PASS").sum())
    warn_count = int((audit_summary_df["status"] == "WARN").sum())
    fail_count = int((audit_summary_df["status"] == "FAIL").sum())

    monthly = monthly_panel_df.copy()
    monthly.loc[:, "signal_month"] = pd.PeriodIndex(monthly["signal_month"].astype(str), freq="M")
    monthly.loc[:, "return_month"] = pd.PeriodIndex(monthly["return_month"].astype(str), freq="M")

    table = pd.DataFrame(
        [
            {
                "research_question": "Does call-minus-put implied volatility predict future stock returns?",
                "main_finding": "Unusually low IV spread stocks underperform the broader optionable-stock universe.",
                "sample_period": (
                    f"Signals {monthly['signal_month'].min()} to {monthly['signal_month'].max()}; "
                    f"returns {monthly['return_month'].min()} to {monthly['return_month'].max()}"
                ),
                "observations_months": int(main["n_months"]),
                "main_strategy": "Universe minus bottom IV-spread decile, equal-weighted, all stocks",
                "main_annualized_return": main["annualized_return"],
                "main_annualized_return_pct": _percent(main["annualized_return"]),
                "main_t_stat": main["t_stat"],
                "main_ff5_mom_alpha": main["alpha_annualized"],
                "main_ff5_mom_alpha_pct": _percent(main["alpha_annualized"]),
                "main_alpha_t_stat": main["alpha_tstat"],
                "rank_ic_mean": rank_ic.get("mean_spearman_ic", np.nan),
                "rank_ic_t_stat": rank_ic.get("t_stat_spearman_ic", np.nan),
                "audit_status": f"PASS={pass_count}, WARN={warn_count}, FAIL={fail_count}",
                "key_caveat": "The signal is not monotonic; the effect is concentrated in the bottom tail.",
                "final_interpretation": (
                    "Frame IV spread as a bottom-tail negative-selection signal, "
                    "not as a broad monotonic ranking factor."
                ),
            }
        ]
    )

    if output_path is not None:
        _save_table(table, output_path)
    return table


def create_final_audit_summary(audit_summary_df, audit_warnings_df, output_path=None):
    """Summarize the saved research audit results."""
    pass_count = int((audit_summary_df["status"] == "PASS").sum())
    warn_count = int((audit_summary_df["status"] == "WARN").sum())
    fail_count = int((audit_summary_df["status"] == "FAIL").sum())
    recommendation = "Proceed" if fail_count == 0 else "Review before proceeding"

    warnings = (
        "; ".join(
            audit_warnings_df["check_name"].astype(str)
            + ": "
            + audit_warnings_df["message"].astype(str)
        )
        if len(audit_warnings_df)
        else "None"
    )

    table = pd.DataFrame(
        [
            {
                "PASS checks": pass_count,
                "WARN checks": warn_count,
                "FAIL checks": fail_count,
                "final recommendation": recommendation,
                "remaining warnings": warnings,
            }
        ]
    )

    if output_path is not None:
        _save_table(table, output_path)
    return table


def _period_to_timestamp(series):
    """Convert period-like values to timestamps for plotting."""
    return pd.PeriodIndex(series.astype(str), freq="M").to_timestamp()


def _format_time_axis(ax):
    """Use readable annual date ticks."""
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))


def _add_bar_labels(ax, values, label_format="{:.2f}"):
    """Add simple labels above bars."""
    y_min, y_max = ax.get_ylim()
    offset = (y_max - y_min) * 0.02
    for patch, value in zip(ax.patches, values):
        if pd.isna(value):
            continue
        height = patch.get_height()
        y = height + offset if height >= 0 else height - offset
        va = "bottom" if height >= 0 else "top"
        ax.text(
            patch.get_x() + patch.get_width() / 2,
            y,
            label_format.format(value),
            ha="center",
            va=va,
            fontsize=8,
        )


def plot_final_bottom_tail_cumulative(bottom_tail_returns_df, charts_dir):
    """Plot cumulative relative returns for the main bottom-tail strategies."""
    charts_dir = Path(charts_dir)
    charts_dir.mkdir(parents=True, exist_ok=True)

    configs = [
        ("decile", "all", "ew", "Decile All EW"),
        ("decile", "mktcap_100m", "ew", "Decile $100M+ EW"),
        ("quintile", "all", "ew", "Quintile All EW"),
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    for tail, universe, weighting, label in configs:
        data = bottom_tail_returns_df.loc[
            (bottom_tail_returns_df["tail"] == tail)
            & (bottom_tail_returns_df["universe"] == universe)
            & (bottom_tail_returns_df["weighting"] == weighting)
        ].copy()
        data = data.sort_values("return_month")
        data.loc[:, "date"] = _period_to_timestamp(data["return_month"])
        data.loc[:, "growth"] = (1 + data["universe_minus_bottom"]).cumprod()
        ax.plot(data["date"], data["growth"], label=label, linewidth=2)

    ax.set_title("Low IV-Spread Stocks Underperform: Cumulative Relative Returns")
    ax.set_xlabel("Return month")
    ax.set_ylabel("Growth of $1")
    _format_time_axis(ax)
    ax.legend()
    fig.tight_layout()
    output_path = charts_dir / "final_bottom_tail_cumulative.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved chart: {output_path}")


def plot_final_bottom_tail_robustness(final_robustness_table, charts_dir):
    """Plot annualized relative returns for selected bottom-tail strategies."""
    charts_dir = Path(charts_dir)
    charts_dir.mkdir(parents=True, exist_ok=True)

    table = final_robustness_table.copy()
    table.loc[:, "label"] = (
        table["Tail definition"].str.replace("Bottom ", "", regex=False).str.title()
        + " "
        + table["Universe"].replace({"MktCap >= $100M": "$100M+"})
        + " "
        + table["Weighting"]
    )

    fig, ax = plt.subplots(figsize=(11, 5.5))
    values = table["Annualized return"].astype(float)
    ax.bar(table["label"], values)
    ax.set_title("Bottom-Tail Robustness: Universe Minus Low IV-Spread Stocks")
    ax.set_xlabel("")
    ax.set_ylabel("Annualized return")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.tick_params(axis="x", rotation=25)
    _add_bar_labels(ax, table["t-stat"], label_format="t={:.2f}")
    fig.tight_layout()
    output_path = charts_dir / "final_bottom_tail_robustness.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved chart: {output_path}")


def plot_final_bottom_tail_factor_alpha(final_factor_alpha_table, charts_dir):
    """Plot annualized FF5 + Momentum alphas."""
    charts_dir = Path(charts_dir)
    charts_dir.mkdir(parents=True, exist_ok=True)

    table = final_factor_alpha_table.copy()
    fig, ax = plt.subplots(figsize=(11, 5.5))
    values = table["Annualized alpha"].astype(float)
    ax.bar(table["Strategy"], values)
    ax.set_title("Factor-Adjusted Performance: FF5 + Momentum Alpha")
    ax.set_xlabel("")
    ax.set_ylabel("Annualized alpha")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.tick_params(axis="x", rotation=25)
    _add_bar_labels(ax, table["Alpha t-stat"], label_format="t={:.2f}")
    fig.tight_layout()
    output_path = charts_dir / "final_bottom_tail_factor_alpha.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved chart: {output_path}")


def plot_final_bottom_tail_by_year(by_year_df, charts_dir):
    """Plot annualized bottom-tail relative returns by full return year."""
    charts_dir = Path(charts_dir)
    charts_dir.mkdir(parents=True, exist_ok=True)

    data = by_year_df.loc[~by_year_df["partial_year"].astype(bool)].copy()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(data["return_year"].astype(str), data["annualized_return"])
    ax.set_title("Universe Minus Bottom Decile by Year\nFull return years only")
    ax.set_xlabel("Return year")
    ax.set_ylabel("Annualized return")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    _add_bar_labels(ax, data["t_stat"], label_format="t={:.2f}")
    fig.tight_layout()
    output_path = charts_dir / "final_bottom_tail_by_year.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved chart: {output_path}")


def plot_final_decile_returns(decile_summary_df, charts_dir):
    """Plot annualized equal-weighted returns by IV spread decile."""
    charts_dir = Path(charts_dir)
    charts_dir.mkdir(parents=True, exist_ok=True)

    data = decile_summary_df.copy()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(data["decile"], data["ew_annualized_return"])
    ax.axhline(0, linewidth=1)
    ax.set_title("IV Spread Decile Returns\nPattern is not monotonic; bottom decile underperforms")
    ax.set_xlabel("IV spread decile")
    ax.set_ylabel("Equal-weighted annualized return")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.annotate(
        "Bottom decile",
        xy=(0, data["ew_annualized_return"].iloc[0]),
        xytext=(0.5, data["ew_annualized_return"].min() - 0.03),
        arrowprops={"arrowstyle": "->"},
        fontsize=9,
    )
    fig.tight_layout()
    output_path = charts_dir / "final_decile_returns.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved chart: {output_path}")


def plot_final_tail_decomposition(tail_vs_universe_df, charts_dir):
    """Plot the decile tail-vs-universe decomposition."""
    charts_dir = Path(charts_dir)
    charts_dir.mkdir(parents=True, exist_ok=True)

    labels = {
        "D10_minus_D1": "Top - Bottom",
        "D10_minus_Universe": "Top - Universe",
        "Universe_minus_D1": "Universe - Bottom",
    }
    data = tail_vs_universe_df.loc[tail_vs_universe_df["weighting"] == "ew"].copy()
    data.loc[:, "label"] = data["leg"].map(labels)
    data = data.set_index("label").loc[
        ["Top - Bottom", "Top - Universe", "Universe - Bottom"]
    ].reset_index()

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(data["label"], data["annualized_return"])
    ax.axhline(0, linewidth=1)
    ax.set_title("Tail Decomposition: What Drives the IV Spread Result?")
    ax.set_xlabel("")
    ax.set_ylabel("Equal-weighted annualized return")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    _add_bar_labels(ax, data["t_stat"], label_format="t={:.2f}")
    fig.tight_layout()
    output_path = charts_dir / "final_tail_decomposition.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved chart: {output_path}")
