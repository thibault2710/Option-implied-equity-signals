"""Run standalone public main portfolio results for the 2010-2023 sample.

This file does not call old numbered files or import them as modules. It uses
reusable portfolio helpers from src.full_sample and writes results to
outputs/public_2010_2023/ and does not depend on private development-output
folders that are excluded from the public release.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "options_implied_signals_matplotlib"))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from src.config import FULL_EXPANSION_SAMPLE_LABEL, sample_processed_path  # noqa: E402
from src.full_sample import (  # noqa: E402
    SIGNAL_COLUMNS,
    UNIVERSE_FILTERS,
    assign_bottom_tail_groups as full_sample_assign_bottom_tail_groups,
    assign_quantiles as full_sample_assign_quantiles,
    compute_bottom_tail_returns as full_sample_compute_bottom_tail_returns,
    prepare_monthly_panel,
    run_quantile_sort,
    summarize_bottom_tail,
    summarize_ls,
)


PUBLIC_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "public_2010_2023"
PUBLIC_TABLES_DIR = PUBLIC_OUTPUT_DIR / "tables"
PUBLIC_CHARTS_DIR = PUBLIC_OUTPUT_DIR / "charts"

REQUIRED_PANEL_COLUMNS = [
    "permno",
    "signal_month",
    "return_month",
    "ret_fwd_1m",
    "mktcap",
    "iv_spread_adj",
]

COMPARISON_ROWS = [
    ("iv_spread_adj", "decile", "all", "ew", "universe_minus_bottom"),
    ("iv_spread_adj", "decile", "mktcap_100m", "ew", "universe_minus_bottom"),
    ("iv_spread_adj", "quintile", "all", "ew", "universe_minus_bottom"),
    ("iv_spread_adj", "quintile", "mktcap_100m", "ew", "universe_minus_bottom"),
    ("iv_spread_adj", "decile", "all", "vw", "universe_minus_bottom"),
    ("iv_spread_adj", "decile", "mktcap_100m", "vw", "universe_minus_bottom"),
]


def print_header(title: str) -> None:
    """Print a readable section header."""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def save_outputs(df: pd.DataFrame, output_path: Path) -> None:
    """Save a CSV output with parent directory creation."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved {output_path} shape={df.shape}")


def load_panel() -> pd.DataFrame:
    """Load and validate the full-sample monthly panel."""
    panel_path = sample_processed_path("monthly_signal_panel", FULL_EXPANSION_SAMPLE_LABEL)
    if not panel_path.exists():
        raise FileNotFoundError(f"Missing monthly panel: {panel_path}")

    panel = pd.read_parquet(panel_path)
    missing = [column for column in REQUIRED_PANEL_COLUMNS if column not in panel.columns]
    if missing:
        raise ValueError(f"Monthly panel is missing required columns: {missing}")

    print(f"Loaded monthly panel: {panel_path} shape={panel.shape}")
    return prepare_monthly_panel(panel)


def apply_universe_filter(panel: pd.DataFrame, universe: str) -> pd.DataFrame:
    """Apply a market-cap universe filter."""
    if universe not in UNIVERSE_FILTERS:
        raise ValueError(f"Unknown universe: {universe}")
    cutoff = UNIVERSE_FILTERS[universe]
    data = panel.copy()
    if cutoff is not None:
        data = data.loc[data["mktcap"] >= cutoff].copy()
    return data


def assign_quantiles(group: pd.DataFrame, signal_col: str, n_quantiles: int) -> pd.DataFrame:
    """Assign quantiles using the shared project implementation."""
    return full_sample_assign_quantiles(group, signal_col, n_quantiles)


def compute_quantile_returns(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute quintile and decile long-short summaries for all main signals."""
    quintile_rows = []
    decile_rows = []

    for signal_col in SIGNAL_COLUMNS:
        if signal_col not in panel.columns:
            print(f"Skipping missing signal column: {signal_col}")
            continue
        for weighting, value_weighted in [("vw", True), ("ew", False)]:
            quintile = run_quantile_sort(panel, signal_col, n_quantiles=5, value_weighted=value_weighted)
            save_outputs(
                quintile,
                PUBLIC_TABLES_DIR / f"quintile_returns_{signal_col}_{weighting}_{FULL_EXPANSION_SAMPLE_LABEL}.csv",
            )
            quintile_rows.append(
                summarize_ls(
                    quintile,
                    {"signal": signal_col, "weighting": weighting, "n_quantiles": 5},
                )
            )

            decile = run_quantile_sort(panel, signal_col, n_quantiles=10, value_weighted=value_weighted)
            save_outputs(
                decile,
                PUBLIC_TABLES_DIR / f"decile_returns_{signal_col}_{weighting}_{FULL_EXPANSION_SAMPLE_LABEL}.csv",
            )
            decile_rows.append(
                summarize_ls(
                    decile,
                    {"signal": signal_col, "weighting": weighting, "n_quantiles": 10},
                )
            )

    quintile_summary = pd.DataFrame(quintile_rows)
    decile_summary = pd.DataFrame(decile_rows)
    save_outputs(quintile_summary, PUBLIC_TABLES_DIR / f"quintile_summary_{FULL_EXPANSION_SAMPLE_LABEL}.csv")
    save_outputs(decile_summary, PUBLIC_TABLES_DIR / f"decile_summary_{FULL_EXPANSION_SAMPLE_LABEL}.csv")
    return quintile_summary, decile_summary


def compute_bottom_tail_returns(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute bottom-tail returns and summaries for the main signals."""
    monthly_frames = []
    for signal_col in SIGNAL_COLUMNS:
        if signal_col not in panel.columns:
            print(f"Skipping missing signal column: {signal_col}")
            continue
        tail_definitions = ["decile", "quintile"] if signal_col == "iv_spread_adj" else ["decile"]
        for tail in tail_definitions:
            for universe in UNIVERSE_FILTERS:
                groups = full_sample_assign_bottom_tail_groups(
                    panel,
                    tail=tail,
                    universe=universe,
                    signal_col=signal_col,
                )
                for weighting, value_weighted in [("ew", False), ("vw", True)]:
                    returns = full_sample_compute_bottom_tail_returns(
                        groups,
                        value_weighted=value_weighted,
                        signal_col=signal_col,
                    )
                    if returns.empty:
                        print(f"No bottom-tail returns for {signal_col} {tail} {universe} {weighting}")
                    monthly_frames.append(returns)

    if not monthly_frames:
        raise RuntimeError("No bottom-tail monthly returns were computed.")

    monthly_returns = pd.concat(monthly_frames, ignore_index=True)
    summary = summarize_bottom_tail(monthly_returns)
    save_outputs(monthly_returns, PUBLIC_TABLES_DIR / f"bottom_tail_returns_{FULL_EXPANSION_SAMPLE_LABEL}.csv")
    save_outputs(summary, PUBLIC_TABLES_DIR / f"bottom_tail_summary_{FULL_EXPANSION_SAMPLE_LABEL}.csv")
    return monthly_returns, summary


def make_charts(bottom_tail_returns: pd.DataFrame) -> None:
    """Create public main-result charts."""
    PUBLIC_CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_decile_returns()
    plot_cumulative_performance(bottom_tail_returns)


def plot_decile_returns() -> None:
    """Plot annualized EW decile returns for IV spread."""
    decile_path = PUBLIC_TABLES_DIR / f"decile_returns_iv_spread_adj_ew_{FULL_EXPANSION_SAMPLE_LABEL}.csv"
    decile = pd.read_csv(decile_path)
    quantile_cols = [f"Q{i}" for i in range(1, 11)]
    annualized = decile[quantile_cols].mean() * 12

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar([f"D{i}" for i in range(1, 11)], annualized.to_numpy())
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.set_title("IV Spread Decile Annualized Returns, 2010-2023")
    ax.set_xlabel("IV-spread decile")
    ax.set_ylabel("Annualized return")
    fig.tight_layout()
    output_path = PUBLIC_CHARTS_DIR / "main_decile_returns.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved chart: {output_path}")


def plot_cumulative_performance(bottom_tail_returns: pd.DataFrame) -> None:
    """Plot cumulative universe-minus-bottom relative performance."""
    configs = [
        ("decile", "all", "ew", "Bottom Decile U-B EW All"),
        ("decile", "mktcap_100m", "ew", "Bottom Decile U-B EW $100M+"),
        ("quintile", "all", "ew", "Bottom Quintile U-B EW All"),
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    for tail, universe, weighting, label in configs:
        data = bottom_tail_returns.loc[
            (bottom_tail_returns["signal"] == "iv_spread_adj")
            & (bottom_tail_returns["tail"] == tail)
            & (bottom_tail_returns["universe"] == universe)
            & (bottom_tail_returns["weighting"] == weighting)
        ].copy()
        if data.empty:
            raise ValueError(f"Missing cumulative performance series for {label}")
        data.loc[:, "return_month"] = pd.PeriodIndex(data["return_month"].astype(str), freq="M")
        data = data.sort_values("return_month")
        data.loc[:, "growth"] = (1 + pd.to_numeric(data["universe_minus_bottom"], errors="coerce")).cumprod()
        ax.plot(data["return_month"].astype(str), data["growth"], label=label)

    ax.set_title("Low IV-Spread Relative Performance, 2010-2023")
    ax.set_xlabel("Return month")
    ax.set_ylabel("Growth of $1")
    ax.legend()
    ax.tick_params(axis="x", labelrotation=45)
    fig.tight_layout()
    output_path = PUBLIC_CHARTS_DIR / "main_cumulative_performance.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved chart: {output_path}")


def strict_summary_row(
    summary: pd.DataFrame,
    signal: str,
    tail: str,
    universe: str,
    weighting: str,
    leg: str,
    source_name: str,
) -> pd.Series:
    """Return one exact bottom-tail summary row or fail loudly."""
    row = summary.loc[
        (summary["signal"] == signal)
        & (summary["tail"] == tail)
        & (summary["universe"] == universe)
        & (summary["weighting"] == weighting)
        & (summary["leg"] == leg)
    ]
    if len(row) != 1:
        raise ValueError(
            f"Expected exactly one row in {source_name} for "
            f"{signal}/{tail}/{universe}/{weighting}/{leg}; found {len(row)}"
        )
    return row.iloc[0]


def create_public_main_results_self_check(public_summary: pd.DataFrame) -> pd.DataFrame:
    """Check that the key public main-results rows are present and valid."""
    rows = []
    metrics = ["annualized_return", "nw_t_stat", "sharpe_ratio"]

    for signal, tail, universe, weighting, leg in COMPARISON_ROWS:
        strategy = f"{signal} {tail} {universe} {weighting} {leg}"
        row = {
            "strategy": strategy,
            "signal": signal,
            "tail": tail,
            "universe": universe,
            "weighting": weighting,
            "leg": leg,
        }
        try:
            public_row = strict_summary_row(public_summary, signal, tail, universe, weighting, leg, "public summary")
            missing_metrics = [metric for metric in metrics if pd.isna(public_row.get(metric, pd.NA))]
            for metric in metrics:
                row[f"{metric}_public"] = public_row.get(metric, pd.NA)
            row["status"] = "PASS" if not missing_metrics else "REVIEW"
            row["message"] = "" if not missing_metrics else f"missing metrics: {missing_metrics}"
        except ValueError as exc:
            for metric in metrics:
                row[f"{metric}_public"] = pd.NA
            row["status"] = "MISSING"
            row["message"] = str(exc)
        rows.append(row)

    comparison = pd.DataFrame(rows)
    save_outputs(comparison, PUBLIC_TABLES_DIR / "public_main_results_comparison.csv")
    return comparison


def main() -> None:
    """Run public main-results workflow."""
    print_header("Public Main Results: 2010-2023")
    PUBLIC_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    PUBLIC_CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    panel = load_panel()
    print(f"Signals available: {[signal for signal in SIGNAL_COLUMNS if signal in panel.columns]}")
    print(f"Signal months: {panel['signal_month'].min()} to {panel['signal_month'].max()}")
    print(f"Return months: {panel['return_month'].min()} to {panel['return_month'].max()}")

    compute_quantile_returns(panel)
    bottom_tail_returns, bottom_tail_summary = compute_bottom_tail_returns(panel)
    make_charts(bottom_tail_returns)
    comparison = create_public_main_results_self_check(bottom_tail_summary)

    print_header("Key public main-results self-check")
    display_cols = [
        "strategy",
        "annualized_return_public",
        "nw_t_stat_public",
        "sharpe_ratio_public",
        "status",
        "message",
    ]
    print(comparison[display_cols].to_string(index=False))

    if (comparison["status"] != "PASS").any():
        raise RuntimeError("One or more public main-results self-check rows require review.")

    print("\nPASS: public main results include the required key rows.")
    print(f"Public tables: {PUBLIC_TABLES_DIR}")
    print(f"Public charts: {PUBLIC_CHARTS_DIR}")


if __name__ == "__main__":
    main()
