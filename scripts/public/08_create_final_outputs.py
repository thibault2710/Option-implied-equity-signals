"""Create public final tables, figures, and results pack for 2010-2023.

This script uses only public pipeline outputs when building final results. It
does not call legacy files and does not import them as modules.
"""

from __future__ import annotations

import math
import os
import shutil
import sys
import tempfile
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "options_implied_signals_matplotlib"))
os.environ.setdefault("MPLBACKEND", "Agg")

from src.utils import summarize_return_series  # noqa: E402


PUBLIC_ROOT = PROJECT_ROOT / "outputs/public_2010_2023"
PUBLIC_TABLES_DIR = PUBLIC_ROOT / "tables"
PUBLIC_CHARTS_DIR = PUBLIC_ROOT / "charts"
FINAL_TABLES_DIR = PUBLIC_ROOT / "final_tables"
FINAL_FIGURES_DIR = PUBLIC_ROOT / "final_figures"
FINAL_PACK_PATH = PUBLIC_ROOT / "final_results_pack.md"
DOC_REPORT_PATH = PROJECT_ROOT / "docs/public_pipeline_step4_final_outputs_report.md"

SAMPLE_LABEL = "2010_2023"
TOLERANCE = 2e-6


def print_header(title: str) -> None:
    """Print a readable section header."""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def read_public_csv(file_name: str) -> pd.DataFrame:
    """Read a required public output table."""
    path = PUBLIC_TABLES_DIR / file_name
    if not path.exists():
        raise FileNotFoundError(f"Missing required public table: {path}")
    return pd.read_csv(path)


def save_table(df: pd.DataFrame, output_path: Path) -> None:
    """Save a CSV table and print its path."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved {output_path} shape={df.shape}")


def strict_row(df: pd.DataFrame, filters: dict[str, object], label: str) -> pd.Series:
    """Select exactly one row from a dataframe."""
    mask = pd.Series(True, index=df.index)
    for column, value in filters.items():
        if column not in df.columns:
            raise ValueError(f"{label}: missing filter column {column}")
        mask = mask & (df[column] == value)
    rows = df.loc[mask]
    if len(rows) != 1:
        raise ValueError(f"{label}: expected exactly one row, found {len(rows)} with {filters}")
    return rows.iloc[0]


def pct(value: object, digits: int = 2) -> str:
    """Format a decimal return as a percentage."""
    value = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(value):
        return ""
    return f"{value * 100:.{digits}f}%"


def num(value: object, digits: int = 2) -> str:
    """Format a number."""
    value = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(value):
        return ""
    return f"{value:.{digits}f}"


def markdown_table(df: pd.DataFrame, formatters: dict[str, callable] | None = None) -> str:
    """Build a simple GitHub-flavored markdown table without optional dependencies."""
    formatters = formatters or {}
    columns = list(df.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in df.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if column in formatters:
                values.append(formatters[column](value))
            elif pd.isna(value):
                values.append("")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def factor_row(factor: pd.DataFrame, portfolio: str) -> pd.Series:
    """Get an FF5+MOM factor regression row."""
    return strict_row(factor, {"portfolio": portfolio, "model": "FF5_MOM"}, f"factor row {portfolio}")


def bottom_row(
    bottom: pd.DataFrame,
    tail: str,
    universe: str,
    weighting: str,
    leg: str = "universe_minus_bottom",
) -> pd.Series:
    """Get one public bottom-tail summary row."""
    return strict_row(
        bottom,
        {
            "signal": "iv_spread_adj",
            "tail": tail,
            "universe": universe,
            "weighting": weighting,
            "leg": leg,
        },
        f"bottom row {tail}/{universe}/{weighting}/{leg}",
    )


def q5_q1_summary(universe: str) -> dict[str, float]:
    """Summarize a public IV-spread Q5-Q1 monthly return series."""
    if universe == "all":
        path = PUBLIC_TABLES_DIR / f"quintile_returns_iv_spread_adj_ew_{SAMPLE_LABEL}.csv"
    else:
        path = PUBLIC_TABLES_DIR / f"robustness_quintile_returns_iv_spread_adj_raw_{universe}_ew_{SAMPLE_LABEL}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing public Q5-Q1 return file: {path}")
    returns = pd.read_csv(path)
    if "LS" not in returns.columns:
        raise ValueError(f"Public Q5-Q1 return file has no LS column: {path}")
    return summarize_return_series(returns["LS"])


def build_headline_table(bottom: pd.DataFrame, factor: pd.DataFrame) -> pd.DataFrame:
    """Build the public headline table."""
    specs = [
        ("Bottom Decile U-B EW All", "bottom", "decile", "all", "ew", "IV Spread Bottom Decile U-B EW All"),
        (
            "Bottom Decile U-B EW $100M+",
            "bottom",
            "decile",
            "mktcap_100m",
            "ew",
            "IV Spread Bottom Decile U-B EW MktCap100M",
        ),
        ("Bottom Quintile U-B EW All", "bottom", "quintile", "all", "ew", "IV Spread Bottom Quintile U-B EW All"),
        (
            "Bottom Quintile U-B EW $100M+",
            "bottom",
            "quintile",
            "mktcap_100m",
            "ew",
            "IV Spread Bottom Quintile U-B EW MktCap100M",
        ),
        ("Q5-Q1 EW All", "q5_q1", "", "all", "ew", "IV Spread Q5-Q1 EW All"),
        ("Q5-Q1 EW $100M+", "q5_q1", "", "mktcap_100m", "ew", "IV Spread Q5-Q1 EW MktCap100M"),
    ]

    rows = []
    for strategy, source, tail, universe, weighting, portfolio in specs:
        perf = bottom_row(bottom, tail, universe, weighting) if source == "bottom" else q5_q1_summary(universe)
        alpha = factor_row(factor, portfolio)
        rows.append(
            {
                "strategy": strategy,
                "annualized_return": perf["annualized_return"],
                "annualized_volatility": perf["annualized_volatility"],
                "sharpe_ratio": perf["sharpe_ratio"],
                "raw_t_stat": perf["raw_t_stat"],
                "nw_t_stat": perf["nw_t_stat"],
                "ff5_mom_alpha": alpha["alpha_annualized"],
                "ff5_mom_alpha_tstat": alpha["alpha_tstat"],
                "n_months": int(perf["n_months"]),
            }
        )
    table = pd.DataFrame(rows)
    save_table(table, FINAL_TABLES_DIR / f"headline_table_{SAMPLE_LABEL}.csv")
    return table


def build_value_weighted_table(bottom: pd.DataFrame, factor: pd.DataFrame) -> pd.DataFrame:
    """Build the public value-weighted robustness table."""
    specs = [
        ("Bottom Decile U-B VW All", "all", "IV Spread Bottom Decile U-B VW All"),
        ("Bottom Decile U-B VW $100M+", "mktcap_100m", "IV Spread Bottom Decile U-B VW MktCap100M"),
    ]
    rows = []
    for strategy, universe, portfolio in specs:
        perf = bottom_row(bottom, "decile", universe, "vw")
        alpha = factor_row(factor, portfolio)
        rows.append(
            {
                "strategy": strategy,
                "annualized_return": perf["annualized_return"],
                "annualized_volatility": perf["annualized_volatility"],
                "sharpe_ratio": perf["sharpe_ratio"],
                "raw_t_stat": perf["raw_t_stat"],
                "nw_t_stat": perf["nw_t_stat"],
                "ff5_mom_alpha": alpha["alpha_annualized"],
                "ff5_mom_alpha_tstat": alpha["alpha_tstat"],
                "n_months": int(perf["n_months"]),
            }
        )
    table = pd.DataFrame(rows)
    save_table(table, FINAL_TABLES_DIR / f"value_weighted_table_{SAMPLE_LABEL}.csv")
    return table


def build_decile_leg_table(bottom_returns: pd.DataFrame) -> pd.DataFrame:
    """Build decile return and leg decomposition table."""
    decile_path = PUBLIC_TABLES_DIR / f"decile_returns_iv_spread_adj_ew_{SAMPLE_LABEL}.csv"
    if not decile_path.exists():
        raise FileNotFoundError(f"Missing public decile returns: {decile_path}")
    decile = pd.read_csv(decile_path)
    rows = []
    for decile_number in range(1, 11):
        col = f"Q{decile_number}"
        rows.append(
            {
                "portfolio_or_leg": f"D{decile_number}",
                "annualized_ew_return": pd.to_numeric(decile[col], errors="coerce").mean() * 12,
                "source": "decile_returns",
            }
        )

    data = bottom_returns.loc[
        (bottom_returns["signal"] == "iv_spread_adj")
        & (bottom_returns["tail"] == "decile")
        & (bottom_returns["universe"] == "all")
        & (bottom_returns["weighting"] == "ew")
    ].copy()
    if data.empty:
        raise ValueError("Missing public bottom-tail monthly returns for decile/all/ew.")

    leg_specs = [
        ("Universe", "universe_ret"),
        ("Bottom decile", "bottom_tail_ret"),
        ("Top decile", "top_tail_ret"),
        ("Universe - Bottom", "universe_minus_bottom"),
        ("Top - Universe", "top_minus_universe"),
        ("Top - Bottom", "top_minus_bottom"),
    ]
    for label, col in leg_specs:
        rows.append(
            {
                "portfolio_or_leg": label,
                "annualized_ew_return": pd.to_numeric(data[col], errors="coerce").mean() * 12,
                "source": "bottom_tail_returns",
            }
        )

    table = pd.DataFrame(rows)
    save_table(table, FINAL_TABLES_DIR / f"decile_leg_table_{SAMPLE_LABEL}.csv")
    return table


def build_audit_status_table(audit: pd.DataFrame) -> pd.DataFrame:
    """Build audit status-count table."""
    counts = audit["status"].value_counts().to_dict()
    table = pd.DataFrame(
        [{"status": status, "count": int(counts.get(status, 0))} for status in ["PASS", "WARN", "FAIL", "INFO"]]
    )
    save_table(table, FINAL_TABLES_DIR / f"audit_status_table_{SAMPLE_LABEL}.csv")
    return table


def copy_final_figures() -> pd.DataFrame:
    """Copy selected public charts into final_figures."""
    FINAL_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    figure_specs = [
        {
            "figure_number": 1,
            "filename": "01_main_cumulative_performance.png",
            "title": "Main Cumulative Performance",
            "source_path": PUBLIC_CHARTS_DIR / "main_cumulative_performance.png",
            "takeaway": "Low IV-spread bottom-tail strategies generate persistent positive relative returns.",
            "suggested_caption": "Cumulative growth of $1 for universe-minus-bottom-tail IV-spread strategies, 2010-2023.",
        },
        {
            "figure_number": 2,
            "filename": "02_decile_returns.png",
            "title": "IV-Spread Decile Returns",
            "source_path": PUBLIC_CHARTS_DIR / "main_decile_returns.png",
            "takeaway": "The result is concentrated in the bottom tail rather than smoothly monotonic across deciles.",
            "suggested_caption": "Annualized equal-weighted returns by IV-spread decile, 2010-2023.",
        },
        {
            "figure_number": 3,
            "filename": "03_factor_alpha_headline.png",
            "title": "FF5+Momentum Alpha",
            "source_path": PUBLIC_CHARTS_DIR / "factor_alpha_headline.png",
            "takeaway": "The main bottom-tail portfolios retain positive FF5+Momentum alpha.",
            "suggested_caption": "Annualized FF5+Momentum alphas for selected IV-spread strategies, 2010-2023.",
        },
    ]

    rows = []
    for spec in figure_specs:
        source = Path(spec["source_path"])
        final_path = FINAL_FIGURES_DIR / spec["filename"]
        if source.exists():
            shutil.copy2(source, final_path)
            include = "yes"
            print(f"Copied figure: {source} -> {final_path}")
        else:
            include = "no"
            print(f"Missing source figure, not copied: {source}")
        rows.append(
            {
                "figure_number": spec["figure_number"],
                "filename": spec["filename"],
                "title": spec["title"],
                "source_path": str(source),
                "final_path": str(final_path),
                "include_in_results_doc": include,
                "takeaway": spec["takeaway"],
                "suggested_caption": spec["suggested_caption"],
            }
        )

    table = pd.DataFrame(rows)
    save_table(table, FINAL_TABLES_DIR / f"figure_inventory_{SAMPLE_LABEL}.csv")
    return table


def factor_alpha_summary(factor: pd.DataFrame) -> pd.DataFrame:
    """Return public FF5+MOM factor rows for markdown."""
    columns = ["portfolio", "alpha_annualized", "alpha_tstat", "r_squared", "n_months"]
    ff5 = factor.loc[factor["model"] == "FF5_MOM", columns].copy()
    return ff5.sort_values("alpha_tstat", ascending=False)


def format_perf_table(df: pd.DataFrame) -> pd.DataFrame:
    """Format performance columns for markdown display."""
    formatted = df.copy(deep=True).astype(object)
    for column in ["annualized_return", "annualized_volatility", "ff5_mom_alpha"]:
        if column in formatted.columns:
            formatted.loc[:, column] = formatted[column].map(pct)
    for column in ["sharpe_ratio", "raw_t_stat", "nw_t_stat", "ff5_mom_alpha_tstat"]:
        if column in formatted.columns:
            formatted.loc[:, column] = formatted[column].map(num)
    return formatted


def write_final_results_pack(
    headline: pd.DataFrame,
    value_weighted: pd.DataFrame,
    decile_leg: pd.DataFrame,
    factor: pd.DataFrame,
    audit_status: pd.DataFrame,
    figure_inventory: pd.DataFrame,
) -> None:
    """Write public final markdown pack."""
    formatted_headline = format_perf_table(headline)
    formatted_vw = format_perf_table(value_weighted)
    formatted_decile = decile_leg.copy(deep=True).astype(object)
    formatted_decile.loc[:, "annualized_ew_return"] = formatted_decile["annualized_ew_return"].map(pct)
    formatted_factor = factor_alpha_summary(factor).copy(deep=True).astype(object)
    formatted_factor.loc[:, "alpha_annualized"] = formatted_factor["alpha_annualized"].map(pct)
    formatted_factor.loc[:, "alpha_tstat"] = formatted_factor["alpha_tstat"].map(num)
    formatted_factor.loc[:, "r_squared"] = formatted_factor["r_squared"].map(lambda value: num(value, 3))

    figure_lines = []
    for _, row in figure_inventory.iterrows():
        if row["include_in_results_doc"] == "yes":
            figure_lines.append(
                f"- Figure {row['figure_number']}: `{row['final_path']}` - {row['suggested_caption']}"
            )

    final_table_paths = sorted(FINAL_TABLES_DIR.glob("*.csv"))
    final_figure_paths = sorted(FINAL_FIGURES_DIR.glob("*.png"))

    markdown = f"""# Final Results Pack: Public 2010-2023 Pipeline

## Main Interpretation

Low call-minus-put implied volatility identifies bottom-tail underperformance within the optionable-stock universe. The evidence is best interpreted as a negative-selection signal rather than a symmetric long-short sentiment factor. Results are gross of transaction costs.

## Headline Table

{markdown_table(formatted_headline)}

## Value-Weighted Robustness Table

{markdown_table(formatted_vw)}

## Decile and Leg Decomposition

{markdown_table(formatted_decile)}

## Factor Alpha Summary

{markdown_table(formatted_factor)}

## Final Figures

{chr(10).join(figure_lines)}

## Public Audit Status

{markdown_table(audit_status)}

## Reproducibility Note

These public final outputs are generated from the standalone public pipeline scripts:

- `scripts/public/04_run_main_results.py`
- `scripts/public/05_run_factor_regressions.py`
- `scripts/public/08_create_final_outputs.py`
- `scripts/public/09_audit_results.py`

Raw WRDS, OptionMetrics, and CRSP data are not included in the public GitHub repository. Users with the required data access can regenerate raw and processed files locally.

## Output File Inventory

Final public tables:

{chr(10).join(f'- `{path}`' for path in final_table_paths)}

Final public figures:

{chr(10).join(f'- `{path}`' for path in final_figure_paths)}
"""
    FINAL_PACK_PATH.write_text(markdown, encoding="utf-8")
    print(f"Saved final results pack: {FINAL_PACK_PATH}")


def add_validation(rows: list[dict[str, object]], check: str, status: str, details: str) -> None:
    """Append one validation check."""
    rows.append({"check": check, "status": status, "details": details})
    print(f"[{status}] {check}: {details}")


def create_validation_table(
    headline: pd.DataFrame,
    value_weighted: pd.DataFrame,
    audit_status: pd.DataFrame,
    figure_inventory: pd.DataFrame,
    bottom: pd.DataFrame,
    factor: pd.DataFrame,
) -> pd.DataFrame:
    """Validate the public final pack and tables."""
    rows: list[dict[str, object]] = []
    add_validation(rows, "headline_table_has_6_rows", "PASS" if len(headline) == 6 else "FAIL", f"rows={len(headline)}")
    add_validation(rows, "value_weighted_table_has_2_rows", "PASS" if len(value_weighted) == 2 else "FAIL", f"rows={len(value_weighted)}")

    statuses = set(audit_status["status"])
    required_statuses = {"PASS", "WARN", "FAIL"}
    add_validation(
        rows,
        "audit_status_includes_pass_warn_fail",
        "PASS" if required_statuses.issubset(statuses) else "FAIL",
        f"statuses={sorted(statuses)}",
    )

    copied = figure_inventory.loc[figure_inventory["include_in_results_doc"] == "yes"].copy()
    missing_figures = [path for path in copied["final_path"] if not Path(path).exists()]
    add_validation(rows, "required_final_figures_exist", "PASS" if not missing_figures else "FAIL", f"missing={missing_figures}")

    pack_ok = FINAL_PACK_PATH.exists() and FINAL_PACK_PATH.stat().st_size > 0
    add_validation(rows, "final_results_pack_nonempty", "PASS" if pack_ok else "FAIL", f"path={FINAL_PACK_PATH}")

    headline_row = strict_row(headline, {"strategy": "Bottom Decile U-B EW All"}, "headline main row")
    bottom_main = bottom_row(bottom, "decile", "all", "ew")
    diff_return = abs(float(headline_row["annualized_return"]) - float(bottom_main["annualized_return"]))
    add_validation(
        rows,
        "headline_main_return_matches_bottom_summary",
        "PASS" if diff_return <= TOLERANCE else "FAIL",
        f"headline={headline_row['annualized_return']}, bottom={bottom_main['annualized_return']}, diff={diff_return}",
    )

    alpha_headline = float(headline_row["ff5_mom_alpha"])
    alpha_factor = float(factor_row(factor, "IV Spread Bottom Decile U-B EW All")["alpha_annualized"])
    diff_alpha = abs(alpha_headline - alpha_factor)
    add_validation(
        rows,
        "headline_main_alpha_matches_factor_summary",
        "PASS" if diff_alpha <= TOLERANCE else "FAIL",
        f"headline={alpha_headline}, factor={alpha_factor}, diff={diff_alpha}",
    )

    validation = pd.DataFrame(rows)
    save_table(validation, FINAL_TABLES_DIR / "public_final_pack_validation.csv")
    return validation


def write_step_report(
    headline: pd.DataFrame,
    value_weighted: pd.DataFrame,
    figure_inventory: pd.DataFrame,
    validation: pd.DataFrame,
) -> None:
    """Write documentation report for this public pipeline step."""
    status_counts = validation["status"].value_counts().to_dict()
    copied_figures = figure_inventory.loc[figure_inventory["include_in_results_doc"] == "yes", "filename"].tolist()
    report = f"""# Public Pipeline Step 4: Final Outputs Report

This step creates `scripts/public/08_create_final_outputs.py`, a standalone public reporting script.

## Files Created

- `scripts/public/08_create_final_outputs.py`
- `docs/public_pipeline_step4_final_outputs_report.md`
- `outputs/public_2010_2023/final_results_pack.md`
- `outputs/public_2010_2023/final_tables/headline_table_2010_2023.csv`
- `outputs/public_2010_2023/final_tables/value_weighted_table_2010_2023.csv`
- `outputs/public_2010_2023/final_tables/decile_leg_table_2010_2023.csv`
- `outputs/public_2010_2023/final_tables/audit_status_table_2010_2023.csv`
- `outputs/public_2010_2023/final_tables/figure_inventory_2010_2023.csv`
- `outputs/public_2010_2023/final_tables/public_final_pack_validation.csv`

## Source Files Modified

None.

## Legacy Script Dependency

The public final-output script does not call or import legacy numbered files.

## Public Inputs Used

- `outputs/public_2010_2023/tables/bottom_tail_summary_2010_2023.csv`
- `outputs/public_2010_2023/tables/bottom_tail_returns_2010_2023.csv`
- `outputs/public_2010_2023/tables/quintile_summary_2010_2023.csv`
- `outputs/public_2010_2023/tables/decile_summary_2010_2023.csv`
- `outputs/public_2010_2023/tables/factor_regression_summary_2010_2023.csv`
- `outputs/public_2010_2023/tables/public_research_audit_summary_2010_2023.csv`
- Public chart files under `outputs/public_2010_2023/charts/`

## Final Tables Created

- Headline rows: {len(headline)}
- Value-weighted rows: {len(value_weighted)}
- Validation checks: {len(validation)}

## Final Figures Copied

{chr(10).join(f'- `{filename}`' for filename in copied_figures)}

## Validation Summary

- PASS: {status_counts.get('PASS', 0)}
- WARN: {status_counts.get('WARN', 0)}
- FAIL: {status_counts.get('FAIL', 0)}

## Missing Public Outputs

None required for this step.

## GitHub Safety

The public final-output script is safe for GitHub: it uses public generated tables/charts, project-root-relative paths, and no WRDS access.

## Recommended Next Public Script

Build `scripts/public/03_build_monthly_panel.py` next to continue filling in the construction side of the standalone public pipeline, or build `scripts/public/06_run_long_only_exclusion.py` if the priority is final-analysis coverage.
"""
    DOC_REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"Saved documentation report: {DOC_REPORT_PATH}")


def main() -> None:
    """Create final public output pack."""
    print_header("Public Final Outputs: 2010-2023")
    FINAL_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FINAL_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    bottom = read_public_csv(f"bottom_tail_summary_{SAMPLE_LABEL}.csv")
    bottom_returns = read_public_csv(f"bottom_tail_returns_{SAMPLE_LABEL}.csv")
    read_public_csv(f"quintile_summary_{SAMPLE_LABEL}.csv")
    read_public_csv(f"decile_summary_{SAMPLE_LABEL}.csv")
    factor = read_public_csv(f"factor_regression_summary_{SAMPLE_LABEL}.csv")
    audit = read_public_csv(f"public_research_audit_summary_{SAMPLE_LABEL}.csv")

    headline = build_headline_table(bottom, factor)
    value_weighted = build_value_weighted_table(bottom, factor)
    decile_leg = build_decile_leg_table(bottom_returns)
    audit_status = build_audit_status_table(audit)
    figure_inventory = copy_final_figures()
    write_final_results_pack(headline, value_weighted, decile_leg, factor, audit_status, figure_inventory)
    validation = create_validation_table(headline, value_weighted, audit_status, figure_inventory, bottom, factor)
    write_step_report(headline, value_weighted, figure_inventory, validation)

    status_counts = validation["status"].value_counts().to_dict()
    print_header("Public Final Outputs Summary")
    print(f"Final tables directory: {FINAL_TABLES_DIR}")
    print(f"Final figures directory: {FINAL_FIGURES_DIR}")
    print(f"Final pack: {FINAL_PACK_PATH}")
    print(f"Validation PASS: {status_counts.get('PASS', 0)}")
    print(f"Validation WARN: {status_counts.get('WARN', 0)}")
    print(f"Validation FAIL: {status_counts.get('FAIL', 0)}")
    if status_counts.get("FAIL", 0) > 0:
        raise RuntimeError("Public final output validation has FAIL checks.")
    print("\nPASS: public final outputs created successfully.")


if __name__ == "__main__":
    main()
