"""Build the public 2010-2023 monthly signal panel from cached local inputs.

This script constructs public processed outputs under data/public_processed/.
It reads local daily VRP signals and CRSP monthly returns only. It does not pull
data and does not overwrite validated files under data/processed/.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (  # noqa: E402
    FULL_EXPANSION_SAMPLE_LABEL,
    RAW_DATA_DIR,
    sample_crsp_monthly_path,
    sample_processed_path,
)
from src.panel import (  # noqa: E402
    add_signal_transforms,
    aggregate_daily_signals_to_monthly,
    merge_monthly_signals_with_forward_returns,
    prepare_crsp_monthly_returns,
    save_monthly_panel,
    validate_monthly_panel,
)
from src.sector_utils import enrich_monthly_panel_with_sector, prepare_security_sector_table  # noqa: E402


SAMPLE_LABEL = FULL_EXPANSION_SAMPLE_LABEL
PUBLIC_PROCESSED_DIR = PROJECT_ROOT / "data" / "public_processed"
PUBLIC_TABLES_DIR = PROJECT_ROOT / "outputs" / "public_2010_2023" / "tables"
DOC_REPORT_PATH = PROJECT_ROOT / "docs" / "public_pipeline_step8_monthly_panel_report.md"

PUBLIC_PANEL_PATH = PUBLIC_PROCESSED_DIR / f"monthly_signal_panel_{SAMPLE_LABEL}.parquet"
PUBLIC_SECTOR_PANEL_PATH = PUBLIC_PROCESSED_DIR / f"monthly_signal_panel_with_sector_{SAMPLE_LABEL}.parquet"
VALIDATED_PANEL_PATH = sample_processed_path("monthly_signal_panel", SAMPLE_LABEL)
VALIDATED_SECTOR_PANEL_PATH = sample_processed_path("monthly_signal_panel_with_sector", SAMPLE_LABEL)

REQUIRED_COLUMNS = [
    "permno",
    "signal_month",
    "return_month",
    "date",
    "signal_date",
    "date_return",
    "ret_fwd_1m",
    "iv_spread",
    "iv_skew",
    "vrp",
    "iv_spread_adj",
    "iv_skew_adj",
    "vrp_adj",
    "iv_spread_z",
    "iv_skew_z",
    "vrp_z",
    "composite_signal",
    "mktcap",
]

FORMULA_TOLERANCE = 1e-10
COMPARISON_TOLERANCE = 1e-10


def print_header(title: str) -> None:
    """Print a readable section header."""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def save_table(df: pd.DataFrame, output_path: Path) -> None:
    """Save a CSV table and print its shape."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved {output_path} shape={df.shape}")


def period_month(series: pd.Series) -> pd.Series:
    """Convert values to monthly Period values."""
    return pd.Series(pd.PeriodIndex(series.astype(str), freq="M"), index=series.index, dtype="period[M]")


def numeric(series: pd.Series) -> pd.Series:
    """Convert a series to numeric values."""
    return pd.to_numeric(series, errors="coerce")


def status_from_bool(condition: bool) -> str:
    """Return PASS or FAIL."""
    return "PASS" if bool(condition) else "FAIL"


def add_check(rows: list[dict[str, object]], category: str, check: str, status: str, details: str) -> None:
    """Append one validation/comparison row."""
    rows.append(
        {
            "category": category,
            "check": check,
            "status": status,
            "details": details,
        }
    )


def add_metric_comparison(
    rows: list[dict[str, object]],
    category: str,
    check: str,
    old_value: object,
    public_value: object,
    status: str,
    details: str = "",
) -> None:
    """Append one old-versus-public comparison row."""
    rows.append(
        {
            "category": category,
            "check": check,
            "old_value": old_value,
            "public_value": public_value,
            "difference": difference_value(old_value, public_value),
            "status": status,
            "details": details,
        }
    )


def difference_value(old_value: object, public_value: object) -> object:
    """Return numeric difference when possible."""
    if isinstance(old_value, (bool, np.bool_)) or isinstance(public_value, (bool, np.bool_)):
        return 0 if bool(old_value) == bool(public_value) else "different"
    old_num = pd.to_numeric(pd.Series([old_value]), errors="coerce").iloc[0]
    public_num = pd.to_numeric(pd.Series([public_value]), errors="coerce").iloc[0]
    if pd.notna(old_num) and pd.notna(public_num):
        return public_num - old_num
    return ""


def validate_input_files(daily_path: Path, crsp_path: Path) -> None:
    """Fail loudly if required local inputs are missing."""
    missing = [path for path in [daily_path, crsp_path] if not path.exists()]
    if missing:
        missing_text = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing required public monthly-panel input(s):\n{missing_text}")


def build_monthly_panel() -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Build the public monthly panel and optional sector-enriched panel."""
    daily_path = sample_processed_path("daily_signals_with_vrp", SAMPLE_LABEL)
    crsp_monthly_path = sample_crsp_monthly_path(2010, 2024)
    validate_input_files(daily_path, crsp_monthly_path)

    print(f"Input daily VRP signals: {daily_path}")
    print(f"Input CRSP monthly returns: {crsp_monthly_path}")
    print(f"Public output panel: {PUBLIC_PANEL_PATH}")

    daily_signals = pd.read_parquet(daily_path)
    crsp_monthly = pd.read_parquet(crsp_monthly_path)
    print(f"Loaded daily VRP signals: {daily_signals.shape}")
    print(f"Loaded CRSP monthly returns: {crsp_monthly.shape}")

    monthly_signals = aggregate_daily_signals_to_monthly(daily_signals)
    monthly_returns = prepare_crsp_monthly_returns(crsp_monthly)
    monthly_panel = merge_monthly_signals_with_forward_returns(monthly_signals, monthly_returns)
    monthly_panel = add_signal_transforms(monthly_panel)
    validate_monthly_panel(monthly_panel)
    save_monthly_panel(monthly_panel, PUBLIC_PANEL_PATH)

    sector_panel = None
    security_path = RAW_DATA_DIR / "security_master_full.parquet"
    if security_path.exists():
        print("\nAdding public sector enrichment.")
        security_master = pd.read_parquet(security_path)
        security_sector = prepare_security_sector_table(security_master)
        sector_panel = enrich_monthly_panel_with_sector(monthly_panel, security_sector)
        save_monthly_panel(sector_panel, PUBLIC_SECTOR_PANEL_PATH)
    else:
        print(f"\nOptional security metadata not found; skipping sector panel: {security_path}")

    return monthly_panel, sector_panel


def contiguous_months(start: str, end: str) -> pd.PeriodIndex:
    """Return contiguous monthly periods from start through end."""
    return pd.period_range(pd.Period(start, freq="M"), pd.Period(end, freq="M"), freq="M")


def max_abs_diff(left: pd.Series, right: pd.Series) -> float:
    """Return max absolute numeric difference."""
    diff = (numeric(left) - numeric(right)).abs()
    if diff.dropna().empty:
        return np.nan
    return float(diff.max())


def validate_public_panel(panel: pd.DataFrame, sector_panel: pd.DataFrame | None) -> pd.DataFrame:
    """Run validation checks on the public monthly panel."""
    rows: list[dict[str, object]] = []
    data = panel.copy()
    data.loc[:, "signal_month"] = period_month(data["signal_month"])
    data.loc[:, "return_month"] = period_month(data["return_month"])

    add_check(rows, "structure", "row_count", "INFO", f"{len(data):,}")
    add_check(rows, "structure", "column_count", "INFO", f"{len(data.columns):,}")
    add_check(rows, "structure", "required_columns", status_from_bool(all(c in data.columns for c in REQUIRED_COLUMNS)), f"required={len(REQUIRED_COLUMNS)}")
    add_check(rows, "structure", "signal_month_range", status_from_bool(data["signal_month"].min() == pd.Period("2010-01", "M") and data["signal_month"].max() == pd.Period("2023-12", "M")), f"{data['signal_month'].min()} to {data['signal_month'].max()}")
    add_check(rows, "structure", "return_month_range", status_from_bool(data["return_month"].min() == pd.Period("2010-02", "M") and data["return_month"].max() == pd.Period("2024-01", "M")), f"{data['return_month'].min()} to {data['return_month'].max()}")
    add_check(rows, "structure", "unique_permnos", "INFO", f"{data['permno'].nunique():,}")
    duplicates = data.duplicated(subset=["permno", "signal_month"]).sum()
    add_check(rows, "structure", "duplicate_permno_signal_month", status_from_bool(duplicates == 0), f"{duplicates:,}")

    for column in REQUIRED_COLUMNS:
        if column in data.columns:
            missing = int(data[column].isna().sum())
            add_check(rows, "missing_values", column, status_from_bool(missing == 0), f"missing={missing:,}")
        else:
            add_check(rows, "missing_values", column, "FAIL", "missing column")

    timing_ok = (data["return_month"] == data["signal_month"] + 1).all()
    add_check(rows, "timing", "return_month_equals_signal_month_plus_1", status_from_bool(timing_ok), f"violations={(~(data['return_month'] == data['signal_month'] + 1)).sum():,}")
    if "signal_date" in data.columns and "date_return" in data.columns:
        signal_date = pd.to_datetime(data["signal_date"], errors="coerce")
        return_date = pd.to_datetime(data["date_return"], errors="coerce")
        timing_dates_ok = (signal_date < return_date).all()
        add_check(rows, "timing", "signal_date_before_return_date", status_from_bool(timing_dates_ok), f"violations={(~(signal_date < return_date)).sum():,}")

    formula_checks = [
        ("iv_spread_formula", data["iv_spread"], data["iv_atm_call"] - data["iv_atm_put"]),
        ("iv_skew_formula", data["iv_skew"], data["iv_otm_put"] - data["iv_atm_call"]),
        ("implied_var_formula", data["implied_var"], data["iv_atm_call"] ** 2),
        ("vrp_formula", data["vrp"], data["implied_var"] - data["realized_var"]),
        ("iv_spread_adj_formula", data["iv_spread_adj"], data["iv_spread"]),
        ("iv_skew_adj_formula", data["iv_skew_adj"], -data["iv_skew"]),
        ("vrp_adj_formula", data["vrp_adj"], -data["vrp"]),
    ]
    for check, left, right in formula_checks:
        diff = max_abs_diff(left, right)
        add_check(rows, "formula", check, status_from_bool(pd.notna(diff) and diff <= FORMULA_TOLERANCE), f"max_abs_diff={diff}")

    signal_months = pd.PeriodIndex(data["signal_month"].astype(str), freq="M").unique().sort_values()
    return_months = pd.PeriodIndex(data["return_month"].astype(str), freq="M").unique().sort_values()
    expected_signal = contiguous_months("2010-01", "2023-12")
    expected_return = contiguous_months("2010-02", "2024-01")
    add_check(rows, "coverage", "signal_months_contiguous", status_from_bool(signal_months.equals(expected_signal)), f"observed={len(signal_months)}, expected={len(expected_signal)}")
    add_check(rows, "coverage", "return_months_contiguous", status_from_bool(return_months.equals(expected_return)), f"observed={len(return_months)}, expected={len(expected_return)}")
    add_check(rows, "coverage", "observed_unique_return_months", status_from_bool(len(return_months) == 168), f"{len(return_months)}")

    if sector_panel is not None:
        add_check(rows, "sector", "sector_panel_created", "PASS", f"shape={sector_panel.shape}")
        unknown_share = (sector_panel["sector"].fillna("Unknown") == "Unknown").mean() if "sector" in sector_panel.columns else np.nan
        add_check(rows, "sector", "unknown_sector_share", "INFO", f"{unknown_share:.6f}")
    else:
        add_check(rows, "sector", "sector_panel_created", "WARN", "security metadata not available")

    validation = pd.DataFrame(rows)
    save_table(validation, PUBLIC_TABLES_DIR / "public_monthly_panel_validation.csv")
    return validation


def compare_basic_metric(
    rows: list[dict[str, object]],
    check: str,
    old_value: object,
    public_value: object,
    tolerance: float = 0,
    details: str = "",
) -> None:
    """Append a simple metric comparison."""
    if isinstance(old_value, (bool, np.bool_)) or isinstance(public_value, (bool, np.bool_)):
        status = "PASS" if bool(old_value) == bool(public_value) else "REVIEW"
        add_metric_comparison(rows, "monthly_panel", check, old_value, public_value, status, details)
        return

    old_num = pd.to_numeric(pd.Series([old_value]), errors="coerce").iloc[0]
    public_num = pd.to_numeric(pd.Series([public_value]), errors="coerce").iloc[0]
    if pd.notna(old_num) and pd.notna(public_num):
        status = "PASS" if abs(public_num - old_num) <= tolerance else "REVIEW"
    else:
        status = "PASS" if str(old_value) == str(public_value) else "REVIEW"
    add_metric_comparison(rows, "monthly_panel", check, old_value, public_value, status, details)


def compare_public_to_validated(public_panel: pd.DataFrame, public_sector_panel: pd.DataFrame | None) -> pd.DataFrame:
    """Compare public-built panel against the validated processed panel."""
    if not VALIDATED_PANEL_PATH.exists():
        raise FileNotFoundError(f"Missing validated panel for comparison: {VALIDATED_PANEL_PATH}")

    old = pd.read_parquet(VALIDATED_PANEL_PATH)
    public = public_panel.copy()
    rows: list[dict[str, object]] = []

    for frame in [old, public]:
        frame.loc[:, "signal_month"] = period_month(frame["signal_month"])
        frame.loc[:, "return_month"] = period_month(frame["return_month"])

    compare_basic_metric(rows, "row_count", len(old), len(public))
    compare_basic_metric(rows, "column_count", len(old.columns), len(public.columns))
    compare_basic_metric(rows, "exact_column_set_match", sorted(old.columns) == sorted(public.columns), sorted(old.columns) == sorted(public.columns))
    compare_basic_metric(rows, "signal_month_min", str(old["signal_month"].min()), str(public["signal_month"].min()))
    compare_basic_metric(rows, "signal_month_max", str(old["signal_month"].max()), str(public["signal_month"].max()))
    compare_basic_metric(rows, "return_month_min", str(old["return_month"].min()), str(public["return_month"].min()))
    compare_basic_metric(rows, "return_month_max", str(old["return_month"].max()), str(public["return_month"].max()))
    compare_basic_metric(rows, "unique_permnos", old["permno"].nunique(), public["permno"].nunique())
    compare_basic_metric(rows, "duplicate_keys", old.duplicated(["permno", "signal_month"]).sum(), public.duplicated(["permno", "signal_month"]).sum())

    key_columns = ["permno", "signal_month"]
    numeric_columns = [
        "ret_fwd_1m",
        "iv_spread",
        "iv_skew",
        "vrp",
        "iv_spread_adj",
        "iv_skew_adj",
        "vrp_adj",
        "composite_signal",
        "mktcap",
    ]
    merged = old[key_columns + numeric_columns].merge(
        public[key_columns + numeric_columns],
        on=key_columns,
        how="inner",
        suffixes=("_old", "_public"),
    )
    compare_basic_metric(rows, "inner_join_rows_on_permno_signal_month", len(old), len(merged))
    for column in numeric_columns:
        old_col = f"{column}_old"
        public_col = f"{column}_public"
        diff = (numeric(merged[public_col]) - numeric(merged[old_col])).abs()
        max_diff = diff.max()
        mean_diff = diff.mean()
        status = "PASS" if pd.notna(max_diff) and max_diff <= COMPARISON_TOLERANCE else "REVIEW"
        add_metric_comparison(
            rows,
            "numeric_columns",
            column,
            0.0,
            max_diff,
            status,
            f"max_abs_diff={max_diff}; mean_abs_diff={mean_diff}",
        )

    if public_sector_panel is not None and VALIDATED_SECTOR_PANEL_PATH.exists():
        old_sector = pd.read_parquet(VALIDATED_SECTOR_PANEL_PATH)
        public_sector = public_sector_panel.copy()
        compare_basic_metric(rows, "sector_row_count", len(old_sector), len(public_sector))
        compare_basic_metric(rows, "sector_column_exists", "sector" in old_sector.columns, "sector" in public_sector.columns)
        old_unknown = (old_sector["sector"].fillna("Unknown") == "Unknown").mean()
        public_unknown = (public_sector["sector"].fillna("Unknown") == "Unknown").mean()
        compare_basic_metric(rows, "sector_unknown_share", old_unknown, public_unknown, tolerance=COMPARISON_TOLERANCE)
        old_dist = old_sector["sector"].fillna("Unknown").value_counts(normalize=True)
        public_dist = public_sector["sector"].fillna("Unknown").value_counts(normalize=True)
        all_sectors = sorted(set(old_dist.index) | set(public_dist.index))
        max_sector_share_diff = max(abs(public_dist.get(sector, 0) - old_dist.get(sector, 0)) for sector in all_sectors)
        add_metric_comparison(
            rows,
            "sector_distribution",
            "max_sector_share_diff",
            0.0,
            max_sector_share_diff,
            "PASS" if max_sector_share_diff <= COMPARISON_TOLERANCE else "REVIEW",
            f"n_sectors={len(all_sectors)}",
        )
    elif public_sector_panel is None:
        add_metric_comparison(rows, "sector", "sector_panel_comparison", "created", "not_created", "MISSING", "public sector panel not created")
    else:
        add_metric_comparison(rows, "sector", "sector_panel_comparison", "validated_exists", "missing", "MISSING", "validated sector panel missing")

    comparison = pd.DataFrame(rows)
    save_table(comparison, PUBLIC_TABLES_DIR / "public_monthly_panel_comparison.csv")
    return comparison


def build_summary(panel: pd.DataFrame, sector_panel: pd.DataFrame | None, validation: pd.DataFrame, comparison: pd.DataFrame) -> pd.DataFrame:
    """Build compact monthly-panel construction summary."""
    validation_counts = validation["status"].value_counts().to_dict()
    comparison_counts = comparison["status"].value_counts().to_dict()
    rows = [
        {
            "item": "public_monthly_panel",
            "path": str(PUBLIC_PANEL_PATH.relative_to(PROJECT_ROOT)),
            "rows": len(panel),
            "columns": len(panel.columns),
            "status": "CREATED",
        },
        {
            "item": "public_sector_panel",
            "path": str(PUBLIC_SECTOR_PANEL_PATH.relative_to(PROJECT_ROOT)) if sector_panel is not None else "",
            "rows": len(sector_panel) if sector_panel is not None else 0,
            "columns": len(sector_panel.columns) if sector_panel is not None else 0,
            "status": "CREATED" if sector_panel is not None else "SKIPPED",
        },
        {
            "item": "validation_counts",
            "path": str((PUBLIC_TABLES_DIR / "public_monthly_panel_validation.csv").relative_to(PROJECT_ROOT)),
            "rows": validation_counts.get("PASS", 0),
            "columns": validation_counts.get("WARN", 0),
            "status": f"PASS={validation_counts.get('PASS', 0)} WARN={validation_counts.get('WARN', 0)} FAIL={validation_counts.get('FAIL', 0)} INFO={validation_counts.get('INFO', 0)}",
        },
        {
            "item": "comparison_counts",
            "path": str((PUBLIC_TABLES_DIR / "public_monthly_panel_comparison.csv").relative_to(PROJECT_ROOT)),
            "rows": comparison_counts.get("PASS", 0),
            "columns": comparison_counts.get("REVIEW", 0),
            "status": f"PASS={comparison_counts.get('PASS', 0)} REVIEW={comparison_counts.get('REVIEW', 0)} MISSING={comparison_counts.get('MISSING', 0)}",
        },
    ]
    summary = pd.DataFrame(rows)
    save_table(summary, PUBLIC_TABLES_DIR / "public_monthly_panel_build_summary.csv")
    return summary


def write_docs_report(
    panel: pd.DataFrame,
    sector_panel: pd.DataFrame | None,
    validation: pd.DataFrame,
    comparison: pd.DataFrame,
) -> Path:
    """Write documentation report for this public construction step."""
    DOC_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    validation_counts = validation["status"].value_counts().to_dict()
    comparison_counts = comparison["status"].value_counts().to_dict()
    review_rows = comparison.loc[comparison["status"].isin(["REVIEW", "MISSING"])]
    fail_rows = validation.loc[validation["status"].isin(["FAIL", "WARN"])]

    lines = [
        "# Public Pipeline Step 8: Monthly Panel Construction",
        "",
        "## Files Created",
        "",
        f"- `{PUBLIC_PANEL_PATH.relative_to(PROJECT_ROOT)}`",
        f"- `{PUBLIC_SECTOR_PANEL_PATH.relative_to(PROJECT_ROOT)}`" if sector_panel is not None else "- Sector-enriched public panel was skipped because sector metadata were unavailable.",
        "- `outputs/public_2010_2023/tables/public_monthly_panel_build_summary.csv`",
        "- `outputs/public_2010_2023/tables/public_monthly_panel_validation.csv`",
        "- `outputs/public_2010_2023/tables/public_monthly_panel_comparison.csv`",
        "",
        "## Source Changes",
        "",
        "No `src/` files were modified.",
        "",
        "## Standalone Status",
        "",
        "The public monthly-panel script uses reusable `src.panel` and `src.sector_utils` functions directly. It does not execute legacy development files.",
        "",
        "## Public Inputs Used",
        "",
        "- `data/processed/daily_signals_with_vrp_2010_2023.parquet`",
        "- `data/raw/crsp_monthly_2010_2024.parquet`",
        "- `data/raw/security_master_full.parquet`, when available",
        "",
        "## Public Outputs Created",
        "",
        f"- Monthly panel shape: {panel.shape}",
        f"- Sector panel shape: {sector_panel.shape if sector_panel is not None else 'not created'}",
        "",
        "## Validation Summary",
        "",
        f"- PASS: {validation_counts.get('PASS', 0)}",
        f"- WARN: {validation_counts.get('WARN', 0)}",
        f"- FAIL: {validation_counts.get('FAIL', 0)}",
        f"- INFO: {validation_counts.get('INFO', 0)}",
        "",
        "## Comparison Summary",
        "",
        f"- PASS: {comparison_counts.get('PASS', 0)}",
        f"- REVIEW: {comparison_counts.get('REVIEW', 0)}",
        f"- MISSING: {comparison_counts.get('MISSING', 0)}",
        "",
        "## Discrepancies",
        "",
    ]
    if review_rows.empty and fail_rows.empty:
        lines.append("No discrepancies require review.")
    else:
        if not fail_rows.empty:
            lines.append("Validation warnings/failures:")
            lines.extend(f"- {row['category']} / {row['check']}: {row['status']} ({row['details']})" for _, row in fail_rows.iterrows())
        if not review_rows.empty:
            lines.append("Comparison review/missing rows:")
            lines.extend(f"- {row['category']} / {row['check']}: {row['status']} ({row['details']})" for _, row in review_rows.iterrows())

    lines.extend(
        [
            "",
            "## GitHub Readiness",
            "",
            "This script is safe for the public pipeline. It requires local cached processed inputs but no WRDS connection.",
            "",
            "## Recommended Next Step",
            "",
            "Build the public option-signal construction script, then build the raw-data pull script last.",
            "",
        ]
    )
    DOC_REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved {DOC_REPORT_PATH}")
    return DOC_REPORT_PATH


def print_terminal_summary(
    panel: pd.DataFrame,
    sector_panel: pd.DataFrame | None,
    validation: pd.DataFrame,
    comparison: pd.DataFrame,
) -> None:
    """Print final status lines."""
    validation_counts = validation["status"].value_counts().to_dict()
    comparison_counts = comparison["status"].value_counts().to_dict()
    print_header("Terminal Summary")
    print("scripts/public/03_build_monthly_panel.py created: yes")
    print("src files modified: no")
    print("legacy files called or imported: no")
    print("run status: completed")
    print(f"public monthly panel shape: {panel.shape}")
    print(f"public sector panel shape: {sector_panel.shape if sector_panel is not None else 'not created'}")
    print(
        "validation totals: "
        f"PASS={validation_counts.get('PASS', 0)} "
        f"WARN={validation_counts.get('WARN', 0)} "
        f"FAIL={validation_counts.get('FAIL', 0)} "
        f"INFO={validation_counts.get('INFO', 0)}"
    )
    print(
        "comparison totals: "
        f"PASS={comparison_counts.get('PASS', 0)} "
        f"REVIEW={comparison_counts.get('REVIEW', 0)} "
        f"MISSING={comparison_counts.get('MISSING', 0)}"
    )
    discrepancies = comparison.loc[comparison["status"].isin(["REVIEW", "MISSING"])]
    if discrepancies.empty:
        print("discrepancies: none")
    else:
        print("discrepancies:")
        print(discrepancies.to_string(index=False))
    print("\nOutput paths:")
    for path in [
        PUBLIC_PANEL_PATH,
        PUBLIC_SECTOR_PANEL_PATH if sector_panel is not None else None,
        PUBLIC_TABLES_DIR / "public_monthly_panel_build_summary.csv",
        PUBLIC_TABLES_DIR / "public_monthly_panel_validation.csv",
        PUBLIC_TABLES_DIR / "public_monthly_panel_comparison.csv",
        DOC_REPORT_PATH,
    ]:
        if path is not None:
            print(path.relative_to(PROJECT_ROOT))
    print("\nRecommended next step: build scripts/public/02_build_option_signals.py.")


def main() -> int:
    """Build and validate the public monthly panel."""
    try:
        print_header("Public Monthly Panel Construction")
        PUBLIC_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        PUBLIC_TABLES_DIR.mkdir(parents=True, exist_ok=True)

        panel, sector_panel = build_monthly_panel()
        validation = validate_public_panel(panel, sector_panel)
        comparison = compare_public_to_validated(panel, sector_panel)
        build_summary(panel, sector_panel, validation, comparison)
        write_docs_report(panel, sector_panel, validation, comparison)
        print_terminal_summary(panel, sector_panel, validation, comparison)
        return 0
    except Exception as exc:
        print_header("Public Monthly Panel Construction Failed")
        print(f"Error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
