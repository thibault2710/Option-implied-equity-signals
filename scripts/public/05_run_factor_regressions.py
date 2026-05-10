"""Run standalone public factor regressions for the 2010-2023 sample.

This script uses public main-result outputs and reusable functions from src/.
It does not call old numbered files and does not import them as modules.
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

from src.config import (  # noqa: E402
    FULL_EXPANSION_SAMPLE_LABEL,
    RAW_DATA_DIR,
    sample_processed_path,
)
from src.full_sample import apply_universe_filter, prepare_monthly_panel, run_quantile_sort  # noqa: E402
from src.regressions import FACTOR_MODELS, load_factor_data, run_factor_regression  # noqa: E402


PUBLIC_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "public_2010_2023"
PUBLIC_TABLES_DIR = PUBLIC_OUTPUT_DIR / "tables"
PUBLIC_CHARTS_DIR = PUBLIC_OUTPUT_DIR / "charts"
REPORT_PATH = PROJECT_ROOT / "docs" / "public_pipeline_step2_factor_regression_report.md"

PORTFOLIO_SPECS = [
    {
        "label": "IV Spread Bottom Decile U-B EW All",
        "kind": "bottom_tail",
        "tail": "decile",
        "universe": "all",
        "weighting": "ew",
        "leg": "universe_minus_bottom",
    },
    {
        "label": "IV Spread Bottom Decile U-B EW MktCap100M",
        "kind": "bottom_tail",
        "tail": "decile",
        "universe": "mktcap_100m",
        "weighting": "ew",
        "leg": "universe_minus_bottom",
    },
    {
        "label": "IV Spread Bottom Quintile U-B EW All",
        "kind": "bottom_tail",
        "tail": "quintile",
        "universe": "all",
        "weighting": "ew",
        "leg": "universe_minus_bottom",
    },
    {
        "label": "IV Spread Bottom Quintile U-B EW MktCap100M",
        "kind": "bottom_tail",
        "tail": "quintile",
        "universe": "mktcap_100m",
        "weighting": "ew",
        "leg": "universe_minus_bottom",
    },
    {
        "label": "IV Spread Bottom Decile U-B VW All",
        "kind": "bottom_tail",
        "tail": "decile",
        "universe": "all",
        "weighting": "vw",
        "leg": "universe_minus_bottom",
    },
    {
        "label": "IV Spread Bottom Decile U-B VW MktCap100M",
        "kind": "bottom_tail",
        "tail": "decile",
        "universe": "mktcap_100m",
        "weighting": "vw",
        "leg": "universe_minus_bottom",
    },
    {
        "label": "IV Spread Q5-Q1 EW All",
        "kind": "q5_q1",
        "universe": "all",
        "weighting": "ew",
    },
    {
        "label": "IV Spread Q5-Q1 EW MktCap100M",
        "kind": "q5_q1",
        "universe": "mktcap_100m",
        "weighting": "ew",
    },
]

COMPARISON_NUMERIC_COLUMNS = [
    "alpha_annualized",
    "alpha_tstat",
    "r_squared",
    "n_months",
    "beta_Mkt-RF",
    "beta_SMB",
    "beta_HML",
    "beta_RMW",
    "beta_CMA",
    "beta_MOM",
]


def print_header(title: str) -> None:
    """Print a readable section header."""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def save_table(df: pd.DataFrame, output_path: Path) -> None:
    """Save a dataframe to CSV and print the path."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved {output_path} shape={df.shape}")


def strict_one_row(df: pd.DataFrame, mask: pd.Series, label: str) -> pd.Series:
    """Return one row or fail loudly."""
    rows = df.loc[mask]
    if len(rows) != 1:
        raise ValueError(f"Expected exactly one row for {label}; found {len(rows)}")
    return rows.iloc[0]


def clean_regression_return_frame(frame: pd.DataFrame, return_col: str, label: str) -> pd.DataFrame:
    """Return a regression-ready signal_month/return_month/LS dataframe."""
    required = ["signal_month", "return_month", return_col]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")

    cleaned = frame[required].rename(columns={return_col: "LS"}).copy()
    cleaned.loc[:, "signal_month"] = pd.PeriodIndex(cleaned["signal_month"].astype(str), freq="M")
    cleaned.loc[:, "return_month"] = pd.PeriodIndex(cleaned["return_month"].astype(str), freq="M")
    cleaned.loc[:, "LS"] = pd.to_numeric(cleaned["LS"], errors="coerce")
    cleaned = cleaned.dropna(subset=["signal_month", "return_month", "LS"]).reset_index(drop=True)
    if cleaned.empty:
        raise ValueError(f"{label} has no valid regression returns.")
    return cleaned


def load_bottom_tail_portfolio(bottom_tail: pd.DataFrame, spec: dict[str, str]) -> pd.DataFrame:
    """Load one public bottom-tail return series."""
    label = spec["label"]
    filtered = bottom_tail.loc[
        (bottom_tail["signal"] == "iv_spread_adj")
        & (bottom_tail["tail"] == spec["tail"])
        & (bottom_tail["universe"] == spec["universe"])
        & (bottom_tail["weighting"] == spec["weighting"])
    ].copy()
    if filtered.empty:
        raise ValueError(f"No public bottom-tail returns found for {label}")
    return clean_regression_return_frame(filtered, spec["leg"], label)


def load_or_create_q5_q1_returns(universe: str, weighting: str) -> pd.DataFrame:
    """Load or create a public IV-spread Q5-Q1 return series."""
    if weighting != "ew":
        raise ValueError("Public Q5-Q1 factor regressions currently require EW weighting.")

    if universe == "all":
        path = PUBLIC_TABLES_DIR / f"quintile_returns_iv_spread_adj_ew_{FULL_EXPANSION_SAMPLE_LABEL}.csv"
    else:
        path = (
            PUBLIC_TABLES_DIR
            / f"robustness_quintile_returns_iv_spread_adj_raw_{universe}_ew_{FULL_EXPANSION_SAMPLE_LABEL}.csv"
        )

    if path.exists():
        returns = pd.read_csv(path)
        print(f"Loaded public Q5-Q1 returns: {path}")
        return clean_regression_return_frame(returns, "LS", f"IV Spread Q5-Q1 EW {universe}")

    print(f"Public Q5-Q1 return file missing; creating from monthly panel: {path}")
    panel_path = sample_processed_path("monthly_signal_panel", FULL_EXPANSION_SAMPLE_LABEL)
    if not panel_path.exists():
        raise FileNotFoundError(f"Missing monthly panel needed to create Q5-Q1 returns: {panel_path}")

    panel = prepare_monthly_panel(pd.read_parquet(panel_path))
    panel = apply_universe_filter(panel, universe)
    returns = run_quantile_sort(panel, "iv_spread_adj", n_quantiles=5, value_weighted=False)
    save_table(returns, path)
    return clean_regression_return_frame(returns, "LS", f"IV Spread Q5-Q1 EW {universe}")


def load_public_portfolios() -> dict[str, pd.DataFrame]:
    """Load the eight public return series used for factor regressions."""
    bottom_tail_path = PUBLIC_TABLES_DIR / f"bottom_tail_returns_{FULL_EXPANSION_SAMPLE_LABEL}.csv"
    if not bottom_tail_path.exists():
        raise FileNotFoundError(
            f"Missing public bottom-tail returns: {bottom_tail_path}. "
            "Run scripts/public/04_run_main_results.py first."
        )

    bottom_tail = pd.read_csv(bottom_tail_path)
    portfolios = {}

    for spec in PORTFOLIO_SPECS:
        if spec["kind"] == "bottom_tail":
            frame = load_bottom_tail_portfolio(bottom_tail, spec)
        elif spec["kind"] == "q5_q1":
            frame = load_or_create_q5_q1_returns(spec["universe"], spec["weighting"])
        else:
            raise ValueError(f"Unknown portfolio kind: {spec['kind']}")

        portfolios[spec["label"]] = frame
        print(
            f"Loaded {spec['label']}: n={len(frame)}, "
            f"return_month={frame['return_month'].min()} to {frame['return_month'].max()}"
        )

    return portfolios


def run_public_factor_regressions(portfolios: dict[str, pd.DataFrame], factors: pd.DataFrame) -> pd.DataFrame:
    """Run all factor models for all public portfolios."""
    rows = []
    for label, returns in portfolios.items():
        for model_name, factor_cols in FACTOR_MODELS.items():
            print(f"Running {model_name} for {label}")
            rows.append(
                run_factor_regression(
                    returns,
                    factors,
                    portfolio_label=label,
                    model_name=model_name,
                    factor_cols=factor_cols,
                    nw_lags=4,
                )
            )
    summary = pd.DataFrame(rows)
    output_path = PUBLIC_TABLES_DIR / f"factor_regression_summary_{FULL_EXPANSION_SAMPLE_LABEL}.csv"
    save_table(summary, output_path)
    return summary


def create_public_regression_self_check(public_summary: pd.DataFrame) -> pd.DataFrame:
    """Check that the public regression table has the expected rows and metrics."""
    rows = []
    expected_keys = [(spec["label"], model_name) for spec in PORTFOLIO_SPECS for model_name in FACTOR_MODELS]

    for portfolio, model in expected_keys:
        public_rows = public_summary.loc[
            (public_summary["portfolio"] == portfolio) & (public_summary["model"] == model)
        ]

        row = {"portfolio": portfolio, "model": model}
        if public_rows.empty:
            row["status"] = "MISSING"
            row["message"] = "public regression row is missing"
            rows.append(row)
            continue

        if len(public_rows) != 1:
            row["status"] = "REVIEW"
            row["message"] = f"expected one public row, found {len(public_rows)}"
            rows.append(row)
            continue

        public_row = public_rows.iloc[0]
        messages = []

        for column in COMPARISON_NUMERIC_COLUMNS:
            if column not in public_summary.columns:
                continue
            public_value = pd.to_numeric(pd.Series([public_row.get(column, pd.NA)]), errors="coerce").iloc[0]

            if pd.isna(public_value):
                messages.append(f"{column}: value missing")

            row[f"{column}_public"] = public_value

        row["status"] = "PASS" if not messages else "REVIEW"
        row["message"] = "; ".join(messages)
        rows.append(row)

    comparison = pd.DataFrame(rows)
    save_table(comparison, PUBLIC_TABLES_DIR / "public_factor_regression_comparison.csv")
    return comparison


def create_ff5_mom_headline(public_summary: pd.DataFrame, comparison: pd.DataFrame) -> pd.DataFrame:
    """Create a compact FF5+MOM headline alpha table."""
    headline = public_summary.loc[public_summary["model"] == "FF5_MOM"].copy()
    status = comparison.loc[comparison["model"] == "FF5_MOM", ["portfolio", "status"]]
    headline = headline.merge(status, on="portfolio", how="left", suffixes=("", "_comparison"))
    columns = [
        "portfolio",
        "alpha_annualized",
        "alpha_tstat",
        "alpha_pvalue",
        "r_squared",
        "n_months",
        "status",
    ]
    headline = headline[columns].sort_values("alpha_tstat", ascending=False)
    save_table(headline, PUBLIC_TABLES_DIR / "public_factor_regression_ff5_mom_headline.csv")
    return headline


def plot_factor_alpha_headline(headline: pd.DataFrame) -> None:
    """Plot FF5+MOM annualized alphas for public portfolios."""
    PUBLIC_CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    data = headline.sort_values("alpha_annualized", ascending=False).copy()
    labels = [
        label.replace("IV Spread ", "").replace("MktCap100M", "$100M+")
        for label in data["portfolio"]
    ]

    fig, ax = plt.subplots(figsize=(11, 6))
    bars = ax.bar(labels, data["alpha_annualized"])
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.set_ylabel("Annualized FF5+MOM alpha")
    ax.set_title("Public Pipeline FF5+Momentum Alpha, 2010-2023")
    ax.tick_params(axis="x", labelrotation=35)

    for bar, t_stat in zip(bars, data["alpha_tstat"]):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"t={t_stat:.2f}",
            ha="center",
            va="bottom" if height >= 0 else "top",
            fontsize=8,
        )

    fig.tight_layout()
    output_path = PUBLIC_CHARTS_DIR / "factor_alpha_headline.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved chart: {output_path}")


def write_report(
    factor_paths: list[Path],
    public_summary: pd.DataFrame,
    comparison: pd.DataFrame,
    headline: pd.DataFrame,
) -> None:
    """Write a markdown report for this public pipeline step."""
    status_counts = comparison["status"].value_counts().to_dict()
    ff5_lines = []
    for _, row in headline.iterrows():
        ff5_lines.append(
            f"| {row['portfolio']} | {row['alpha_annualized']:.6f} | "
            f"{row['alpha_tstat']:.6f} | {row['r_squared']:.6f} | "
            f"{int(row['n_months'])} | {row['status']} |"
        )

    report = f"""# Public Pipeline Step 2: Factor Regression Report

This step creates `scripts/public/05_run_factor_regressions.py`, a standalone public factor-regression script.

## Files Created

- `scripts/public/05_run_factor_regressions.py`
- `docs/public_pipeline_step2_factor_regression_report.md`
- `outputs/public_2010_2023/tables/factor_regression_summary_2010_2023.csv`
- `outputs/public_2010_2023/tables/public_factor_regression_comparison.csv`
- `outputs/public_2010_2023/tables/public_factor_regression_ff5_mom_headline.csv`
- `outputs/public_2010_2023/charts/factor_alpha_headline.png`

## Source Files Modified

None.

## Old Numbered Scripts

The public regression script does not depend on legacy numbered files. It uses reusable functions from `src.regressions`, `src.full_sample`, and `src.config`.

## Regression Inputs

- `outputs/public_2010_2023/tables/bottom_tail_returns_2010_2023.csv`
- `outputs/public_2010_2023/tables/quintile_returns_iv_spread_adj_ew_2010_2023.csv`
- `outputs/public_2010_2023/tables/robustness_quintile_returns_iv_spread_adj_raw_mktcap_100m_ew_2010_2023.csv`
- `data/processed/monthly_signal_panel_2010_2023.parquet` only if the public `$100M+` Q5-Q1 support series must be created

## Factor Files Used

{chr(10).join(f'- `{path}`' for path in factor_paths)}

## Models Run

- CAPM
- FF3
- FF5
- FF5_MOM

Total regression rows created: {len(public_summary)}

## FF5+MOM Headline Rows

| Portfolio | Alpha annualized | Alpha t-stat | R-squared | N months | Status |
|---|---:|---:|---:|---:|---|
{chr(10).join(ff5_lines)}

## Comparison Summary

Public regression self-check:

- PASS: {status_counts.get('PASS', 0)}
- REVIEW: {status_counts.get('REVIEW', 0)}
- MISSING: {status_counts.get('MISSING', 0)}

Self-check table:

- `outputs/public_2010_2023/tables/public_factor_regression_comparison.csv`

## Discrepancies

{"None. All required public regression rows are present." if status_counts.get('REVIEW', 0) == 0 and status_counts.get('MISSING', 0) == 0 else "Review the comparison CSV for non-PASS rows."}

## GitHub Readiness

The public factor regression script is safe for the public workflow. It is readable, uses project-root-relative paths, does not use WRDS, and does not depend on old numbered scripts.

## Recommended Next Public Script

Build `scripts/public/09_audit_results.py` next after extracting the full-sample audit logic into `src.audit`, or build `scripts/public/03_build_monthly_panel.py` if the goal is to complete the core construction pipeline first.
"""
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"Saved report: {REPORT_PATH}")


def main() -> None:
    """Run public factor regressions and comparisons."""
    print_header("Public Factor Regressions: 2010-2023")
    PUBLIC_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    PUBLIC_CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    factors = load_factor_data(RAW_DATA_DIR)
    factor_paths = sorted((RAW_DATA_DIR / "factors").glob("*.csv")) + sorted(
        (RAW_DATA_DIR / "factors").glob("*.CSV")
    )
    portfolios = load_public_portfolios()
    public_summary = run_public_factor_regressions(portfolios, factors)
    comparison = create_public_regression_self_check(public_summary)
    headline = create_ff5_mom_headline(public_summary, comparison)
    plot_factor_alpha_headline(headline)
    write_report(factor_paths, public_summary, comparison, headline)

    print_header("Public Factor Regression Summary")
    print(f"Regression rows created: {len(public_summary)}")
    print("Comparison status counts:")
    print(comparison["status"].value_counts().to_string())
    print("\nFF5+MOM headline alpha table:")
    print(
        headline[
            ["portfolio", "alpha_annualized", "alpha_tstat", "r_squared", "n_months", "status"]
        ].to_string(index=False)
    )

    if (comparison["status"] != "PASS").any():
        raise RuntimeError("One or more factor regression self-check rows require review.")

    print("\nPASS: public factor regressions include all required rows.")
    print(f"Public tables: {PUBLIC_TABLES_DIR}")
    print(f"Public charts: {PUBLIC_CHARTS_DIR}")


if __name__ == "__main__":
    main()
