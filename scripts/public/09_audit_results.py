"""Audit public 2010-2023 pipeline outputs and cached data.

This standalone public audit does not call legacy development scripts and does
not modify raw or processed data. It writes audit outputs under
outputs/public_2010_2023/tables/.
"""

from __future__ import annotations

import math
import os
import sys
import tempfile
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "options_implied_signals_matplotlib"))
os.environ.setdefault("MPLBACKEND", "Agg")


RAW_CACHE_FILES = {
    "vol_surface_2010_2023": PROJECT_ROOT / "data/raw/vol_surface_2010_2023.parquet",
    "crsp_daily_2009_12_2023": PROJECT_ROOT / "data/raw/crsp_daily_2009_12_2023.parquet",
    "crsp_monthly_2010_2024": PROJECT_ROOT / "data/raw/crsp_monthly_2010_2024.parquet",
    "official_wrds_link": PROJECT_ROOT / "data/processed/secid_permno_bridge_wrdsapps.parquet",
}

PROCESSED_FILES = {
    "daily_iv_signals_2010_2023": PROJECT_ROOT / "data/processed/daily_iv_signals_2010_2023.parquet",
    "daily_signals_with_vrp_2010_2023": PROJECT_ROOT / "data/processed/daily_signals_with_vrp_2010_2023.parquet",
    "monthly_signal_panel_2010_2023": PROJECT_ROOT / "data/processed/monthly_signal_panel_2010_2023.parquet",
    "monthly_signal_panel_with_sector_2010_2023": PROJECT_ROOT
    / "data/processed/monthly_signal_panel_with_sector_2010_2023.parquet",
}

PUBLIC_TABLES_DIR = PROJECT_ROOT / "outputs/public_2010_2023/tables"
PUBLIC_CHARTS_DIR = PROJECT_ROOT / "outputs/public_2010_2023/charts"

PUBLIC_OUTPUT_FILES = {
    "bottom_tail_returns_2010_2023": PUBLIC_TABLES_DIR / "bottom_tail_returns_2010_2023.csv",
    "bottom_tail_summary_2010_2023": PUBLIC_TABLES_DIR / "bottom_tail_summary_2010_2023.csv",
    "quintile_summary_2010_2023": PUBLIC_TABLES_DIR / "quintile_summary_2010_2023.csv",
    "decile_summary_2010_2023": PUBLIC_TABLES_DIR / "decile_summary_2010_2023.csv",
    "factor_regression_summary_2010_2023": PUBLIC_TABLES_DIR / "factor_regression_summary_2010_2023.csv",
    "public_main_results_comparison": PUBLIC_TABLES_DIR / "public_main_results_comparison.csv",
    "public_factor_regression_comparison": PUBLIC_TABLES_DIR / "public_factor_regression_comparison.csv",
}

KEY_CHARTS = {
    "main_decile_returns": PUBLIC_CHARTS_DIR / "main_decile_returns.png",
    "main_cumulative_performance": PUBLIC_CHARTS_DIR / "main_cumulative_performance.png",
    "factor_alpha_headline": PUBLIC_CHARTS_DIR / "factor_alpha_headline.png",
}

OUTPUT_SUMMARY = PUBLIC_TABLES_DIR / "public_research_audit_summary_2010_2023.csv"
OUTPUT_WARNINGS = PUBLIC_TABLES_DIR / "public_research_audit_warnings_2010_2023.csv"
OUTPUT_COUNTS = PUBLIC_TABLES_DIR / "public_research_audit_key_counts_2010_2023.csv"
OUTPUT_REPORT = PUBLIC_TABLES_DIR / "public_research_audit_report.md"
DOC_REPORT = PROJECT_ROOT / "docs/public_pipeline_step3_audit_report.md"

TOLERANCE = 1e-6


def print_header(title: str) -> None:
    """Print a readable section header."""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def add_check(rows: list[dict[str, object]], category: str, check: str, status: str, details: str) -> None:
    """Append one audit check row and print it."""
    rows.append({"category": category, "check": check, "status": status, "details": details})
    print(f"[{status}] {category} - {check}: {details}")


def add_key_count(key_count_rows: list[dict[str, object]], **kwargs: object) -> None:
    """Append one key-count row."""
    key_count_rows.append(kwargs)


def parquet_shape(path: Path) -> tuple[int, int]:
    """Return a Parquet file shape from metadata without loading all data."""
    metadata = pq.ParquetFile(path).metadata
    return metadata.num_rows, metadata.num_columns


def file_shape(path: Path) -> tuple[int, int]:
    """Return shape for CSV or Parquet files."""
    if path.suffix.lower() == ".parquet":
        return parquet_shape(path)
    return pd.read_csv(path).shape


def check_file_group(
    rows: list[dict[str, object]],
    key_counts: list[dict[str, object]],
    files: dict[str, Path],
    category: str,
    required: bool,
) -> dict[str, pd.DataFrame | None]:
    """Check file existence/readability and load small CSVs when useful."""
    loaded: dict[str, pd.DataFrame | None] = {}
    for name, path in files.items():
        if not path.exists():
            status = "FAIL" if required else "WARN"
            add_check(rows, category, f"{name}_exists", status, f"missing: {path}")
            add_key_count(
                key_counts,
                dataset=name,
                path=str(path),
                exists=False,
                rows=pd.NA,
                columns=pd.NA,
                date_min=pd.NA,
                date_max=pd.NA,
                unique_ids=pd.NA,
            )
            loaded[name] = None
            continue

        try:
            shape = file_shape(path)
            add_check(rows, category, f"{name}_readable", "PASS", f"shape={shape}")
            add_key_count(
                key_counts,
                dataset=name,
                path=str(path),
                exists=True,
                rows=shape[0],
                columns=shape[1],
                date_min=pd.NA,
                date_max=pd.NA,
                unique_ids=pd.NA,
            )
            loaded[name] = pd.read_csv(path) if path.suffix.lower() == ".csv" else None
        except Exception as exc:
            status = "FAIL" if required else "WARN"
            add_check(rows, category, f"{name}_readable", status, f"{path}: {exc}")
            loaded[name] = None
    return loaded


def period_series(series: pd.Series) -> pd.PeriodIndex:
    """Convert values to monthly periods."""
    return pd.PeriodIndex(series.astype(str), freq="M")


def expected_month_count(first_month: pd.Period, last_month: pd.Period) -> int:
    """Inclusive number of months between two periods."""
    return (last_month.year - first_month.year) * 12 + (last_month.month - first_month.month) + 1


def max_abs_diff(left: pd.Series, right: pd.Series) -> float:
    """Return max absolute formula difference."""
    return float((pd.to_numeric(left, errors="coerce") - pd.to_numeric(right, errors="coerce")).abs().max())


def check_required_columns(
    rows: list[dict[str, object]],
    category: str,
    check: str,
    df: pd.DataFrame,
    required_columns: list[str],
) -> bool:
    """Check required columns and return whether all are present."""
    missing = [column for column in required_columns if column not in df.columns]
    add_check(rows, category, check, "PASS" if not missing else "FAIL", f"missing={missing}")
    return not missing


def read_parquet_columns(path: Path, columns: list[str]) -> pd.DataFrame:
    """Read selected Parquet columns."""
    return pd.read_parquet(path, columns=columns)


def check_daily_iv(rows: list[dict[str, object]], key_counts: list[dict[str, object]]) -> None:
    """Audit daily IV signal formulas and keys."""
    category = "daily_iv"
    path = PROCESSED_FILES["daily_iv_signals_2010_2023"]
    required = [
        "secid",
        "permno",
        "date",
        "iv_atm_call",
        "iv_atm_put",
        "iv_otm_put",
        "iv_spread",
        "iv_skew",
        "implied_var",
    ]
    if not path.exists():
        add_check(rows, category, "file_available", "FAIL", f"missing: {path}")
        return

    df = read_parquet_columns(path, required)
    if not check_required_columns(rows, category, "required_columns", df, required):
        return

    date = pd.to_datetime(df["date"], errors="coerce")
    duplicate_count = int(df.duplicated(["permno", "date"]).sum())
    missing_values = int(df[required].isna().sum().sum())
    spread_diff = max_abs_diff(df["iv_spread"], df["iv_atm_call"] - df["iv_atm_put"])
    skew_diff = max_abs_diff(df["iv_skew"], df["iv_otm_put"] - df["iv_atm_call"])
    implied_var_diff = max_abs_diff(df["implied_var"], df["iv_atm_call"] ** 2)
    first_month = date.min().to_period("M")
    last_month = date.max().to_period("M")

    add_check(rows, category, "duplicate_permno_date_rows", "PASS" if duplicate_count == 0 else "FAIL", str(duplicate_count))
    add_check(rows, category, "missing_required_values", "PASS" if missing_values == 0 else "FAIL", str(missing_values))
    add_check(rows, category, "iv_spread_formula", "PASS" if spread_diff < 1e-8 else "FAIL", f"max_abs_diff={spread_diff}")
    add_check(rows, category, "iv_skew_formula", "PASS" if skew_diff < 1e-8 else "FAIL", f"max_abs_diff={skew_diff}")
    add_check(rows, category, "implied_var_formula", "PASS" if implied_var_diff < 1e-8 else "FAIL", f"max_abs_diff={implied_var_diff}")
    coverage_ok = first_month <= pd.Period("2010-01", freq="M") and last_month >= pd.Period("2023-12", freq="M")
    add_check(rows, category, "date_range_includes_sample", "PASS" if coverage_ok else "FAIL", f"{first_month} to {last_month}")
    add_key_count(
        key_counts,
        dataset="daily_iv_signals_2010_2023_detail",
        path=str(path),
        exists=True,
        rows=len(df),
        columns=len(required),
        date_min=str(date.min().date()),
        date_max=str(date.max().date()),
        unique_ids=int(df["permno"].nunique()),
        unique_secids=int(df["secid"].nunique()),
    )


def check_daily_vrp(rows: list[dict[str, object]], key_counts: list[dict[str, object]]) -> None:
    """Audit daily VRP formulas and keys."""
    category = "daily_vrp"
    path = PROCESSED_FILES["daily_signals_with_vrp_2010_2023"]
    required = ["permno", "date", "implied_var", "realized_var", "vrp", "mktcap"]
    if not path.exists():
        add_check(rows, category, "file_available", "FAIL", f"missing: {path}")
        return

    df = read_parquet_columns(path, required)
    if not check_required_columns(rows, category, "required_columns", df, required):
        return

    date = pd.to_datetime(df["date"], errors="coerce")
    duplicate_count = int(df.duplicated(["permno", "date"]).sum())
    missing_values = int(df[required].isna().sum().sum())
    vrp_diff = max_abs_diff(df["vrp"], df["implied_var"] - df["realized_var"])
    negative_rv = int((pd.to_numeric(df["realized_var"], errors="coerce") < 0).sum())

    add_check(rows, category, "duplicate_permno_date_rows", "PASS" if duplicate_count == 0 else "FAIL", str(duplicate_count))
    add_check(rows, category, "missing_required_values", "PASS" if missing_values == 0 else "FAIL", str(missing_values))
    add_check(rows, category, "vrp_formula", "PASS" if vrp_diff < 1e-8 else "FAIL", f"max_abs_diff={vrp_diff}")
    add_check(rows, category, "realized_var_nonnegative", "PASS" if negative_rv == 0 else "FAIL", str(negative_rv))
    add_key_count(
        key_counts,
        dataset="daily_signals_with_vrp_2010_2023_detail",
        path=str(path),
        exists=True,
        rows=len(df),
        columns=len(required),
        date_min=str(date.min().date()),
        date_max=str(date.max().date()),
        unique_ids=int(df["permno"].nunique()),
        unique_secids=pd.NA,
    )


def check_monthly_panel(rows: list[dict[str, object]], key_counts: list[dict[str, object]]) -> None:
    """Audit monthly panel timing and key fields."""
    category = "monthly_panel"
    path = PROCESSED_FILES["monthly_signal_panel_2010_2023"]
    required = ["permno", "signal_month", "return_month", "ret_fwd_1m", "iv_spread_adj", "iv_skew_adj", "vrp_adj", "mktcap"]
    optional = ["signal_date", "date_return"]
    if not path.exists():
        add_check(rows, category, "file_available", "FAIL", f"missing: {path}")
        return

    available = pq.ParquetFile(path).schema.names
    columns = required + [column for column in optional if column in available]
    df = read_parquet_columns(path, columns)
    if not check_required_columns(rows, category, "required_columns", df, required):
        return

    signal_month = period_series(df["signal_month"])
    return_month = period_series(df["return_month"])
    duplicate_count = int(pd.DataFrame({"permno": df["permno"], "signal_month": signal_month}).duplicated(["permno", "signal_month"]).sum())
    missing_values = int(df[required].isna().sum().sum())
    bad_timing = int((return_month != signal_month + 1).sum())
    first_signal = signal_month.min()
    last_signal = signal_month.max()
    first_return = return_month.min()
    last_return = return_month.max()
    observed_return_months = return_month.nunique()
    expected_returns = expected_month_count(first_return, last_return)

    add_check(rows, category, "duplicate_permno_signal_month_rows", "PASS" if duplicate_count == 0 else "FAIL", str(duplicate_count))
    add_check(rows, category, "missing_required_values", "PASS" if missing_values == 0 else "FAIL", str(missing_values))
    add_check(rows, category, "return_month_equals_signal_month_plus_one", "PASS" if bad_timing == 0 else "FAIL", str(bad_timing))
    add_check(rows, category, "signal_month_coverage", "PASS" if (first_signal == pd.Period("2010-01", freq="M") and last_signal == pd.Period("2023-12", freq="M")) else "FAIL", f"{first_signal} to {last_signal}")
    add_check(rows, category, "return_month_coverage", "PASS" if (first_return == pd.Period("2010-02", freq="M") and last_return == pd.Period("2024-01", freq="M")) else "FAIL", f"{first_return} to {last_return}")
    add_check(rows, category, "dynamic_return_month_count", "PASS" if observed_return_months == expected_returns == 168 else "FAIL", f"observed={observed_return_months}, expected={expected_returns}")

    if {"signal_date", "date_return"}.issubset(df.columns):
        bad_dates = int((pd.to_datetime(df["signal_date"], errors="coerce") >= pd.to_datetime(df["date_return"], errors="coerce")).sum())
        add_check(rows, category, "signal_date_before_return_date", "PASS" if bad_dates == 0 else "FAIL", str(bad_dates))
    else:
        add_check(rows, category, "signal_date_before_return_date", "INFO", "signal_date/date_return not both present")

    add_key_count(
        key_counts,
        dataset="monthly_signal_panel_2010_2023_detail",
        path=str(path),
        exists=True,
        rows=len(df),
        columns=len(columns),
        date_min=str(first_signal),
        date_max=str(last_signal),
        return_month_min=str(first_return),
        return_month_max=str(last_return),
        unique_ids=int(df["permno"].nunique()),
        unique_secids=pd.NA,
        n_months=int(observed_return_months),
    )


def exact_public_summary_row(summary: pd.DataFrame, tail: str, universe: str, weighting: str, leg: str) -> pd.DataFrame:
    """Select one public bottom-tail summary row."""
    return summary.loc[
        (summary["signal"] == "iv_spread_adj")
        & (summary["tail"] == tail)
        & (summary["universe"] == universe)
        & (summary["weighting"] == weighting)
        & (summary["leg"] == leg)
    ]


def check_numeric_close(
    rows: list[dict[str, object]],
    category: str,
    check: str,
    observed: float,
    expected: float,
    tolerance: float = TOLERANCE,
) -> None:
    """Add a numeric closeness check."""
    diff = abs(float(observed) - float(expected))
    add_check(
        rows,
        category,
        check,
        "PASS" if diff <= tolerance else "FAIL",
        f"observed={observed:.12g}, expected={expected:.12g}, diff={diff:.3g}",
    )


def check_public_bottom_tail(rows: list[dict[str, object]], key_counts: list[dict[str, object]]) -> None:
    """Audit public bottom-tail returns, summaries, and comparison table."""
    category = "public_bottom_tail"
    returns_path = PUBLIC_OUTPUT_FILES["bottom_tail_returns_2010_2023"]
    summary_path = PUBLIC_OUTPUT_FILES["bottom_tail_summary_2010_2023"]
    comparison_path = PUBLIC_OUTPUT_FILES["public_main_results_comparison"]
    if not returns_path.exists() or not summary_path.exists():
        add_check(rows, category, "required_files_available", "FAIL", f"returns={returns_path.exists()}, summary={summary_path.exists()}")
        return

    returns = pd.read_csv(returns_path)
    summary = pd.read_csv(summary_path)
    required_returns = ["signal_month", "return_month", "signal", "tail", "universe", "weighting", "universe_minus_bottom"]
    required_summary = ["signal", "tail", "universe", "weighting", "leg", "annualized_return", "nw_t_stat", "n_months"]
    if not check_required_columns(rows, category, "returns_required_columns", returns, required_returns):
        return
    if not check_required_columns(rows, category, "summary_required_columns", summary, required_summary):
        return

    duplicate_keys = int(returns.duplicated(["signal_month", "return_month", "signal", "tail", "universe", "weighting"]).sum())
    add_check(rows, category, "monthly_return_duplicate_keys", "PASS" if duplicate_keys == 0 else "FAIL", str(duplicate_keys))

    required_rows = [
        ("decile", "all", "ew", "universe_minus_bottom"),
        ("decile", "mktcap_100m", "ew", "universe_minus_bottom"),
        ("quintile", "all", "ew", "universe_minus_bottom"),
        ("quintile", "mktcap_100m", "ew", "universe_minus_bottom"),
        ("decile", "all", "vw", "universe_minus_bottom"),
        ("decile", "mktcap_100m", "vw", "universe_minus_bottom"),
    ]
    for tail, universe, weighting, leg in required_rows:
        selected = exact_public_summary_row(summary, tail, universe, weighting, leg)
        add_check(
            rows,
            category,
            f"main_row_{tail}_{universe}_{weighting}_{leg}",
            "PASS" if len(selected) == 1 else "FAIL",
            f"rows={len(selected)}",
        )

    row_all = exact_public_summary_row(summary, "decile", "all", "ew", "universe_minus_bottom").iloc[0]
    row_100 = exact_public_summary_row(summary, "decile", "mktcap_100m", "ew", "universe_minus_bottom").iloc[0]
    check_numeric_close(rows, category, "ew_all_decile_annualized_return", row_all["annualized_return"], 0.075737, tolerance=2e-6)
    check_numeric_close(rows, category, "ew_all_decile_nw_t_stat", row_all["nw_t_stat"], 3.995110, tolerance=2e-6)
    check_numeric_close(rows, category, "ew_100m_decile_annualized_return", row_100["annualized_return"], 0.077289, tolerance=2e-6)
    check_numeric_close(rows, category, "ew_100m_decile_nw_t_stat", row_100["nw_t_stat"], 5.872319, tolerance=2e-6)

    if comparison_path.exists():
        comparison = pd.read_csv(comparison_path)
        non_pass = comparison.loc[comparison["status"] != "PASS"]
        add_check(rows, category, "public_main_results_comparison_all_pass", "PASS" if non_pass.empty else "FAIL", f"non_pass={len(non_pass)}")
    else:
        add_check(rows, category, "public_main_results_comparison_all_pass", "FAIL", f"missing: {comparison_path}")

    add_key_count(
        key_counts,
        dataset="public_bottom_tail_summary",
        path=str(summary_path),
        exists=True,
        rows=len(summary),
        columns=len(summary.columns),
        date_min=pd.NA,
        date_max=pd.NA,
        unique_ids=pd.NA,
        n_months=int(summary["n_months"].max()),
    )


def check_factor_regressions(rows: list[dict[str, object]], key_counts: list[dict[str, object]]) -> None:
    """Audit public factor regression outputs."""
    category = "public_factor_regressions"
    factor_path = PUBLIC_OUTPUT_FILES["factor_regression_summary_2010_2023"]
    comparison_path = PUBLIC_OUTPUT_FILES["public_factor_regression_comparison"]
    if not factor_path.exists():
        add_check(rows, category, "factor_summary_available", "FAIL", f"missing: {factor_path}")
        return

    factor = pd.read_csv(factor_path)
    required = ["portfolio", "model", "alpha_annualized", "alpha_tstat", "r_squared", "n_months"]
    if not check_required_columns(rows, category, "required_columns", factor, required):
        return

    expected_models = {"CAPM", "FF3", "FF5", "FF5_MOM"}
    expected_portfolios = {
        "IV Spread Bottom Decile U-B EW All",
        "IV Spread Bottom Decile U-B EW MktCap100M",
        "IV Spread Bottom Quintile U-B EW All",
        "IV Spread Bottom Quintile U-B EW MktCap100M",
        "IV Spread Bottom Decile U-B VW All",
        "IV Spread Bottom Decile U-B VW MktCap100M",
        "IV Spread Q5-Q1 EW All",
        "IV Spread Q5-Q1 EW MktCap100M",
    }
    models = set(factor["model"].astype(str))
    portfolios = set(factor["portfolio"].astype(str))
    add_check(rows, category, "expected_row_count", "PASS" if len(factor) == 32 else "FAIL", str(len(factor)))
    add_check(rows, category, "expected_models", "PASS" if models == expected_models else "FAIL", f"models={sorted(models)}")
    add_check(rows, category, "expected_portfolios", "PASS" if portfolios == expected_portfolios else "FAIL", f"portfolios={len(portfolios)}")
    all_168 = bool((pd.to_numeric(factor["n_months"], errors="coerce") == 168).all())
    add_check(rows, category, "n_months_168_all_rows", "PASS" if all_168 else "FAIL", str(all_168))

    ff5 = factor.loc[factor["model"] == "FF5_MOM"].copy()
    for portfolio in [
        "IV Spread Bottom Decile U-B EW All",
        "IV Spread Bottom Decile U-B EW MktCap100M",
        "IV Spread Bottom Decile U-B VW All",
        "IV Spread Bottom Decile U-B VW MktCap100M",
    ]:
        selected = ff5.loc[ff5["portfolio"] == portfolio]
        add_check(rows, category, f"ff5_mom_row_{portfolio}", "PASS" if len(selected) == 1 else "FAIL", f"rows={len(selected)}")

    ew_all = ff5.loc[ff5["portfolio"] == "IV Spread Bottom Decile U-B EW All"].iloc[0]
    ew_100 = ff5.loc[ff5["portfolio"] == "IV Spread Bottom Decile U-B EW MktCap100M"].iloc[0]
    check_numeric_close(rows, category, "ew_all_ff5_mom_alpha", ew_all["alpha_annualized"], 0.063963, tolerance=2e-6)
    check_numeric_close(rows, category, "ew_all_ff5_mom_alpha_tstat", ew_all["alpha_tstat"], 5.069411, tolerance=2e-6)
    check_numeric_close(rows, category, "ew_100m_ff5_mom_alpha", ew_100["alpha_annualized"], 0.062920, tolerance=2e-6)
    check_numeric_close(rows, category, "ew_100m_ff5_mom_alpha_tstat", ew_100["alpha_tstat"], 5.847330, tolerance=2e-6)

    if comparison_path.exists():
        comparison = pd.read_csv(comparison_path)
        non_pass = comparison.loc[comparison["status"] != "PASS"]
        add_check(rows, category, "public_factor_comparison_all_pass", "PASS" if non_pass.empty else "FAIL", f"non_pass={len(non_pass)}")
    else:
        add_check(rows, category, "public_factor_comparison_all_pass", "FAIL", f"missing: {comparison_path}")

    add_key_count(
        key_counts,
        dataset="public_factor_regression_summary",
        path=str(factor_path),
        exists=True,
        rows=len(factor),
        columns=len(factor.columns),
        date_min=str(factor["first_return_month"].min()) if "first_return_month" in factor.columns else pd.NA,
        date_max=str(factor["last_return_month"].max()) if "last_return_month" in factor.columns else pd.NA,
        unique_ids=pd.NA,
        n_months=int(factor["n_months"].max()),
    )


def check_key_charts(rows: list[dict[str, object]], key_counts: list[dict[str, object]]) -> None:
    """Check key public charts as optional outputs."""
    for name, path in KEY_CHARTS.items():
        exists = path.exists()
        status = "PASS" if exists else "WARN"
        size = path.stat().st_size if exists else pd.NA
        add_check(rows, "public_charts", f"{name}_exists", status, f"path={path}, bytes={size}")
        add_key_count(
            key_counts,
            dataset=f"chart_{name}",
            path=str(path),
            exists=exists,
            rows=pd.NA,
            columns=pd.NA,
            file_size_bytes=size,
        )


def validated_audit_counts() -> dict[str, int]:
    """Return historical validation counts when they are bundled with a local workspace.

    The clean public release intentionally excludes private development-output
    folders, so this returns an empty dictionary there. Historical validation
    results are documented in the public release notes instead of being required
    at runtime.
    """
    return {}


def add_audit_count_comparison(rows: list[dict[str, object]], key_counts: list[dict[str, object]], public_summary: pd.DataFrame) -> None:
    """Add comparison between public audit counts and validated audit counts."""
    public_counts = {status: int(count) for status, count in public_summary["status"].value_counts().items()}
    validated_counts = validated_audit_counts()
    add_check(
        rows,
        "audit_comparison",
        "validated_audit_counts_available",
        "INFO",
        str(validated_counts or "not bundled with public release"),
    )
    add_check(
        rows,
        "audit_comparison",
        "public_audit_has_no_fail",
        "PASS" if public_counts.get("FAIL", 0) == 0 else "FAIL",
        str(public_counts),
    )
    add_key_count(
        key_counts,
        dataset="public_audit_status_totals",
        public_pass=public_counts.get("PASS", 0),
        public_warn=public_counts.get("WARN", 0),
        public_fail=public_counts.get("FAIL", 0),
        public_info=public_counts.get("INFO", 0),
        validated_pass=validated_counts.get("PASS", 0),
        validated_warn=validated_counts.get("WARN", 0),
        validated_fail=validated_counts.get("FAIL", 0),
    )


def write_markdown_report(summary: pd.DataFrame, warnings: pd.DataFrame, key_counts: pd.DataFrame) -> None:
    """Write public audit markdown report."""
    public_counts = summary["status"].value_counts().to_dict()
    validated_counts = validated_audit_counts()
    warning_lines = "None."
    if not warnings.empty:
        warning_lines = "\n".join(
            f"- `{row.category}` / `{row.check}`: {row.status} - {row.details}"
            for row in warnings.itertuples(index=False)
        )

    report = f"""# Public Research Audit Report

This audit checks the standalone public 2010-2023 pipeline outputs and key cached data.

## Public Audit Totals

- PASS: {public_counts.get('PASS', 0)}
- WARN: {public_counts.get('WARN', 0)}
- FAIL: {public_counts.get('FAIL', 0)}
- INFO: {public_counts.get('INFO', 0)}

## Historical Validation Totals

- PASS: {validated_counts.get('PASS', 0)}
- WARN: {validated_counts.get('WARN', 0)}
- FAIL: {validated_counts.get('FAIL', 0)}

Historical validation outputs are not required in the public release. The public audit checks public-output files and core formulas directly.

## Categories Checked

- File existence and basic shape
- Daily IV formulas and duplicate keys
- Daily VRP formulas and duplicate keys
- Monthly panel timing and coverage
- Public bottom-tail result rows and headline values
- Public factor regression rows and headline alpha values
- Public comparison tables
- Key public charts

## WARN / FAIL Items

{warning_lines}

## Output Files

- `outputs/public_2010_2023/tables/public_research_audit_summary_2010_2023.csv`
- `outputs/public_2010_2023/tables/public_research_audit_warnings_2010_2023.csv`
- `outputs/public_2010_2023/tables/public_research_audit_key_counts_2010_2023.csv`
"""
    OUTPUT_REPORT.write_text(report, encoding="utf-8")
    DOC_REPORT.write_text(
        report
        + "\n## GitHub Readiness\n\n"
        + "The public audit script is safe for GitHub: it is standalone, uses project-root-relative paths, does not connect to WRDS, and does not depend on legacy numbered scripts.\n\n"
        + "## Recommended Next Step\n\n"
        + "Build `scripts/public/03_build_monthly_panel.py` next if you want to complete the core construction path, or `scripts/public/08_create_final_outputs.py` if you want a public-facing reporting path from the public outputs already generated.\n",
        encoding="utf-8",
    )
    print(f"Saved report: {OUTPUT_REPORT}")
    print(f"Saved documentation report: {DOC_REPORT}")


def save_audit_outputs(rows: list[dict[str, object]], key_counts: list[dict[str, object]]) -> pd.DataFrame:
    """Save audit tables and return final summary."""
    PUBLIC_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(rows)
    add_audit_count_comparison(rows, key_counts, summary)
    summary = pd.DataFrame(rows)
    warnings = summary.loc[summary["status"].isin(["WARN", "FAIL"])].copy()
    key_counts_df = pd.DataFrame(key_counts)

    summary.to_csv(OUTPUT_SUMMARY, index=False)
    warnings.to_csv(OUTPUT_WARNINGS, index=False)
    key_counts_df.to_csv(OUTPUT_COUNTS, index=False)
    write_markdown_report(summary, warnings, key_counts_df)

    print(f"Saved audit summary: {OUTPUT_SUMMARY}")
    print(f"Saved audit warnings: {OUTPUT_WARNINGS}")
    print(f"Saved audit key counts: {OUTPUT_COUNTS}")
    return summary


def main() -> None:
    """Run the public audit."""
    print_header("Public Audit: 2010-2023")
    rows: list[dict[str, object]] = []
    key_counts: list[dict[str, object]] = []

    check_file_group(rows, key_counts, RAW_CACHE_FILES, "raw_cache_files", required=True)
    check_file_group(rows, key_counts, PROCESSED_FILES, "processed_files", required=True)
    check_file_group(rows, key_counts, PUBLIC_OUTPUT_FILES, "public_output_files", required=True)
    check_key_charts(rows, key_counts)
    check_daily_iv(rows, key_counts)
    check_daily_vrp(rows, key_counts)
    check_monthly_panel(rows, key_counts)
    check_public_bottom_tail(rows, key_counts)
    check_factor_regressions(rows, key_counts)

    summary = save_audit_outputs(rows, key_counts)
    counts = summary["status"].value_counts().to_dict()
    validated_counts = validated_audit_counts()

    print_header("Public Audit Summary")
    print(f"Public PASS: {counts.get('PASS', 0)}")
    print(f"Public WARN: {counts.get('WARN', 0)}")
    print(f"Public FAIL: {counts.get('FAIL', 0)}")
    print(f"Public INFO: {counts.get('INFO', 0)}")
    print(f"Validated audit counts: {validated_counts}")

    warnings = summary.loc[summary["status"].isin(["WARN", "FAIL"])]
    if not warnings.empty:
        print("\nWARN/FAIL items:")
        print(warnings.to_string(index=False))

    if counts.get("FAIL", 0) > 0:
        raise RuntimeError("Public audit has FAIL checks. Review outputs before publishing.")

    print("\nPASS: public audit completed with no FAIL checks.")


if __name__ == "__main__":
    main()
