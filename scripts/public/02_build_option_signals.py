"""Build public 2010-2023 daily option signals from cached local data.

This script reads cached OptionMetrics, WRDS link, and CRSP daily files. It
does not pull data and does not overwrite validated files under data/processed/.
Public construction outputs are written under data/public_processed/.
"""

from __future__ import annotations

import gc
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (  # noqa: E402
    FULL_EXPANSION_SAMPLE_LABEL,
    PROCESSED_DATA_DIR,
    sample_crsp_daily_path,
    sample_processed_path,
    sample_raw_vol_surface_path,
)
from src.linking import link_signals_to_permno_time_aware  # noqa: E402
from src.signals import (  # noqa: E402
    compute_daily_iv_signals,
    compute_realized_variance,
    deduplicate_permno_date_signals,
    merge_vrp_signals,
    pivot_iv_surface,
    save_daily_signals,
    save_vrp_panel,
    validate_daily_signals,
    validate_vol_surface,
    validate_vrp_panel,
)


SAMPLE_LABEL = FULL_EXPANSION_SAMPLE_LABEL
PUBLIC_PROCESSED_DIR = PROJECT_ROOT / "data" / "public_processed"
PUBLIC_TABLES_DIR = PROJECT_ROOT / "outputs" / "public_2010_2023" / "tables"
DOC_REPORT_PATH = PROJECT_ROOT / "docs" / "public_pipeline_step9_option_signals_report.md"

VOL_SURFACE_PATH = sample_raw_vol_surface_path(2010, 2023)
LINK_TABLE_PATH = PROCESSED_DATA_DIR / "secid_permno_bridge_wrdsapps.parquet"
CRSP_DAILY_PATH = sample_crsp_daily_path(2010, 2023)

PUBLIC_DAILY_IV_PATH = PUBLIC_PROCESSED_DIR / f"daily_iv_signals_{SAMPLE_LABEL}.parquet"
PUBLIC_DAILY_VRP_PATH = PUBLIC_PROCESSED_DIR / f"daily_signals_with_vrp_{SAMPLE_LABEL}.parquet"

VALIDATED_DAILY_IV_PATH = sample_processed_path("daily_iv_signals", SAMPLE_LABEL)
VALIDATED_DAILY_VRP_PATH = sample_processed_path("daily_signals_with_vrp", SAMPLE_LABEL)

DAILY_IV_REQUIRED_COLUMNS = [
    "secid",
    "permno",
    "date",
    "iv_atm_call",
    "iv_atm_put",
    "iv_otm_put",
    "iv_spread",
    "iv_skew",
    "implied_var",
    "score",
]

DAILY_VRP_REQUIRED_COLUMNS = DAILY_IV_REQUIRED_COLUMNS + [
    "realized_var",
    "mktcap",
    "exchcd",
    "shrcd",
    "vrp",
]

DAILY_IV_COMPARE_COLUMNS = [
    "iv_atm_call",
    "iv_atm_put",
    "iv_otm_put",
    "iv_spread",
    "iv_skew",
    "implied_var",
    "score",
]

DAILY_VRP_COMPARE_COLUMNS = [
    "realized_var",
    "mktcap",
    "vrp",
    "implied_var",
    "iv_spread",
    "iv_skew",
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


def status_from_bool(condition: bool) -> str:
    """Return PASS or FAIL."""
    return "PASS" if bool(condition) else "FAIL"


def add_check(rows: list[dict[str, object]], category: str, check: str, status: str, details: str) -> None:
    """Append one validation row."""
    rows.append(
        {
            "category": category,
            "check": check,
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


def add_comparison(
    rows: list[dict[str, object]],
    dataset: str,
    check: str,
    old_value: object,
    public_value: object,
    status: str,
    details: str = "",
) -> None:
    """Append one old-versus-public comparison row."""
    rows.append(
        {
            "dataset": dataset,
            "check": check,
            "old_value": old_value,
            "public_value": public_value,
            "difference": difference_value(old_value, public_value),
            "status": status,
            "details": details,
        }
    )


def numeric(series: pd.Series) -> pd.Series:
    """Convert a series to numeric values."""
    return pd.to_numeric(series, errors="coerce")


def max_abs_diff(left: pd.Series, right: pd.Series) -> float:
    """Return the maximum absolute numeric difference."""
    diff = (numeric(left) - numeric(right)).abs()
    if diff.dropna().empty:
        return np.nan
    return float(diff.max())


def mean_abs_diff(left: pd.Series, right: pd.Series) -> float:
    """Return the mean absolute numeric difference."""
    diff = (numeric(left) - numeric(right)).abs()
    if diff.dropna().empty:
        return np.nan
    return float(diff.mean())


def parquet_columns(path: Path) -> list[str]:
    """Return parquet column names without loading the full file."""
    return list(pq.ParquetFile(path).schema_arrow.names)


def parquet_shape(path: Path) -> tuple[int, int]:
    """Return parquet row and column counts without loading the full file."""
    parquet_file = pq.ParquetFile(path)
    return parquet_file.metadata.num_rows, len(parquet_file.schema_arrow.names)


def validate_input_files() -> None:
    """Fail loudly if required cached inputs are missing."""
    required = [VOL_SURFACE_PATH, LINK_TABLE_PATH, CRSP_DAILY_PATH]
    missing = [path for path in required if not path.exists()]
    if missing:
        missing_text = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing required cached option-signal input(s):\n{missing_text}")


def build_public_daily_iv_signals() -> pd.DataFrame:
    """Build and save public daily IV signals."""
    print_header("Build Public Daily IV Signals")
    print(f"Input volatility surface: {VOL_SURFACE_PATH}")
    print(f"Input official link table: {LINK_TABLE_PATH}")
    print(f"Public daily IV output: {PUBLIC_DAILY_IV_PATH}")

    vol_surface = pd.read_parquet(
        VOL_SURFACE_PATH,
        columns=["secid", "date", "days", "delta", "impl_volatility", "cp_flag"],
    )
    official_link = pd.read_parquet(
        LINK_TABLE_PATH,
        columns=["secid", "sdate", "edate", "permno", "score"],
    )
    print(f"Loaded volatility surface: {vol_surface.shape}")
    print(f"Loaded official link table: {official_link.shape}")

    validate_vol_surface(vol_surface)
    iv_wide = pivot_iv_surface(vol_surface)
    del vol_surface
    gc.collect()

    daily_signals = compute_daily_iv_signals(iv_wide)
    del iv_wide
    gc.collect()

    linked = link_signals_to_permno_time_aware(daily_signals, official_link)
    del daily_signals, official_link
    gc.collect()

    linked = deduplicate_permno_date_signals(linked)
    validate_daily_signals(linked)
    save_daily_signals(linked, PUBLIC_DAILY_IV_PATH)
    return linked


def build_public_daily_vrp_panel(daily_iv: pd.DataFrame) -> pd.DataFrame:
    """Build and save public daily VRP signals."""
    print_header("Build Public Daily VRP Panel")
    print(f"Input CRSP daily: {CRSP_DAILY_PATH}")
    print(f"Input public daily IV signals: {PUBLIC_DAILY_IV_PATH}")
    print(f"Public daily VRP output: {PUBLIC_DAILY_VRP_PATH}")

    crsp_daily = pd.read_parquet(
        CRSP_DAILY_PATH,
        columns=["permno", "date", "ret", "vol", "prc", "shrout", "cusip", "exchcd", "shrcd"],
    )
    print(f"Loaded CRSP daily: {crsp_daily.shape}")
    print(f"Loaded public daily IV signals: {daily_iv.shape}")

    realized = compute_realized_variance(crsp_daily, window=21, min_periods=15)
    del crsp_daily
    gc.collect()

    vrp_panel = merge_vrp_signals(daily_iv, realized)
    del realized
    gc.collect()

    validate_vrp_panel(vrp_panel)
    save_vrp_panel(vrp_panel, PUBLIC_DAILY_VRP_PATH)
    return vrp_panel


def validate_daily_iv(daily_iv: pd.DataFrame) -> pd.DataFrame:
    """Validate public daily IV signals and return a check table."""
    rows: list[dict[str, object]] = []
    data = daily_iv.copy()
    data.loc[:, "date"] = pd.to_datetime(data["date"])

    add_check(rows, "structure", "row_count", "INFO", f"{len(data):,}")
    add_check(rows, "structure", "column_count", "INFO", f"{len(data.columns):,}")
    add_check(
        rows,
        "structure",
        "required_columns",
        status_from_bool(all(column in data.columns for column in DAILY_IV_REQUIRED_COLUMNS)),
        f"required={len(DAILY_IV_REQUIRED_COLUMNS)}",
    )

    duplicates = data.duplicated(subset=["permno", "date"]).sum()
    add_check(rows, "duplicates", "duplicate_permno_date_rows", status_from_bool(duplicates == 0), f"{duplicates:,}")

    for column in DAILY_IV_REQUIRED_COLUMNS:
        missing = data[column].isna().sum() if column in data.columns else len(data)
        add_check(rows, "missing", f"missing_{column}", status_from_bool(missing == 0), f"{missing:,}")

    iv_spread_diff = max_abs_diff(data["iv_spread"], data["iv_atm_call"] - data["iv_atm_put"])
    iv_skew_diff = max_abs_diff(data["iv_skew"], data["iv_otm_put"] - data["iv_atm_call"])
    implied_var_diff = max_abs_diff(data["implied_var"], data["iv_atm_call"] ** 2)
    add_check(rows, "formula", "iv_spread_equals_call_minus_put", status_from_bool(iv_spread_diff <= FORMULA_TOLERANCE), f"max_abs_diff={iv_spread_diff:.12g}")
    add_check(rows, "formula", "iv_skew_equals_otm_put_minus_call", status_from_bool(iv_skew_diff <= FORMULA_TOLERANCE), f"max_abs_diff={iv_skew_diff:.12g}")
    add_check(rows, "formula", "implied_var_equals_call_squared", status_from_bool(implied_var_diff <= FORMULA_TOLERANCE), f"max_abs_diff={implied_var_diff:.12g}")

    add_check(rows, "coverage", "date_range", status_from_bool(data["date"].min() <= pd.Timestamp("2010-01-04") and data["date"].max() >= pd.Timestamp("2023-12-29")), f"{data['date'].min()} to {data['date'].max()}")
    add_check(rows, "coverage", "unique_permnos", "INFO", f"{data['permno'].nunique():,}")
    add_check(rows, "coverage", "unique_secids", "INFO", f"{data['secid'].nunique():,}")
    add_check(rows, "coverage", "unique_dates", "INFO", f"{data['date'].nunique():,}")

    return pd.DataFrame(rows)


def validate_daily_vrp(vrp_panel: pd.DataFrame) -> pd.DataFrame:
    """Validate public daily VRP panel and return a check table."""
    rows: list[dict[str, object]] = []
    data = vrp_panel.copy()
    data.loc[:, "date"] = pd.to_datetime(data["date"])

    add_check(rows, "structure", "row_count", "INFO", f"{len(data):,}")
    add_check(rows, "structure", "column_count", "INFO", f"{len(data.columns):,}")
    add_check(
        rows,
        "structure",
        "required_columns",
        status_from_bool(all(column in data.columns for column in DAILY_VRP_REQUIRED_COLUMNS)),
        f"required={len(DAILY_VRP_REQUIRED_COLUMNS)}",
    )

    duplicates = data.duplicated(subset=["permno", "date"]).sum()
    add_check(rows, "duplicates", "duplicate_permno_date_rows", status_from_bool(duplicates == 0), f"{duplicates:,}")

    for column in DAILY_VRP_REQUIRED_COLUMNS:
        missing = data[column].isna().sum() if column in data.columns else len(data)
        add_check(rows, "missing", f"missing_{column}", status_from_bool(missing == 0), f"{missing:,}")

    vrp_diff = max_abs_diff(data["vrp"], data["implied_var"] - data["realized_var"])
    min_realized_var = numeric(data["realized_var"]).min()
    add_check(rows, "formula", "vrp_equals_implied_minus_realized", status_from_bool(vrp_diff <= FORMULA_TOLERANCE), f"max_abs_diff={vrp_diff:.12g}")
    add_check(rows, "formula", "realized_var_nonnegative", status_from_bool(min_realized_var >= 0), f"min_realized_var={min_realized_var:.12g}")

    add_check(rows, "coverage", "date_range", status_from_bool(data["date"].min() <= pd.Timestamp("2010-01-04") and data["date"].max() >= pd.Timestamp("2023-12-29")), f"{data['date'].min()} to {data['date'].max()}")
    add_check(rows, "coverage", "unique_permnos", "INFO", f"{data['permno'].nunique():,}")
    add_check(rows, "coverage", "unique_dates", "INFO", f"{data['date'].nunique():,}")

    return pd.DataFrame(rows)


def frame_summary(
    dataset: str,
    path: Path,
    df: pd.DataFrame,
    notes: str,
    input_paths: str = "",
    match_rate: str = "",
) -> dict[str, object]:
    """Return a compact summary row for one dataset."""
    date_min = ""
    date_max = ""
    if "date" in df.columns:
        dates = pd.to_datetime(df["date"])
        date_min = dates.min()
        date_max = dates.max()
    return {
        "dataset": dataset,
        "input_paths": input_paths,
        "output_path": str(path),
        "rows": len(df),
        "columns": len(df.columns),
        "min_date": date_min,
        "max_date": date_max,
        "unique_secids": df["secid"].nunique() if "secid" in df.columns else "",
        "unique_permnos": df["permno"].nunique() if "permno" in df.columns else "",
        "duplicate_permno_date_rows": df.duplicated(subset=["permno", "date"]).sum() if {"permno", "date"}.issubset(df.columns) else "",
        "merge_match_rate": match_rate,
        "notes": notes,
    }


def create_daily_iv_build_summary(daily_iv: pd.DataFrame) -> pd.DataFrame:
    """Create the public daily IV build summary table."""
    formula_details = (
        "iv_spread, iv_skew, and implied_var are recomputed by src.signals; "
        "time-aware linking uses the official local link table."
    )
    return pd.DataFrame(
        [
            frame_summary(
                "daily_iv_signals",
                PUBLIC_DAILY_IV_PATH,
                daily_iv,
                formula_details,
                input_paths=f"{VOL_SURFACE_PATH}; {LINK_TABLE_PATH}",
            )
        ]
    )


def create_daily_vrp_build_summary(vrp_panel: pd.DataFrame, daily_iv_rows: int) -> pd.DataFrame:
    """Create the public daily VRP build summary table."""
    match_rate = f"{len(vrp_panel) / daily_iv_rows:.6f}" if daily_iv_rows else ""
    notes = "realized_var uses a 21-trading-day rolling variance with min_periods=15, annualized by 252."
    return pd.DataFrame(
        [
            frame_summary(
                "daily_signals_with_vrp",
                PUBLIC_DAILY_VRP_PATH,
                vrp_panel,
                notes,
                input_paths=f"{PUBLIC_DAILY_IV_PATH}; {CRSP_DAILY_PATH}",
                match_rate=match_rate,
            )
        ]
    )


def basic_file_comparisons(
    rows: list[dict[str, object]],
    dataset: str,
    old_path: Path,
    public_path: Path,
    key_columns: list[str],
    include_secid: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, bool]:
    """Compare metadata and key ordering for two parquet files."""
    old_rows, old_cols = parquet_shape(old_path)
    public_rows, public_cols = parquet_shape(public_path)
    old_columns = parquet_columns(old_path)
    public_columns = parquet_columns(public_path)

    add_comparison(rows, dataset, "row_count", old_rows, public_rows, status_from_bool(old_rows == public_rows))
    add_comparison(rows, dataset, "column_count", old_cols, public_cols, status_from_bool(old_cols == public_cols))
    add_comparison(
        rows,
        dataset,
        "exact_column_set_match",
        set(old_columns) == set(public_columns),
        set(old_columns) == set(public_columns),
        status_from_bool(set(old_columns) == set(public_columns)),
        f"old_only={sorted(set(old_columns) - set(public_columns))}; public_only={sorted(set(public_columns) - set(old_columns))}",
    )

    read_columns = key_columns + (["secid"] if include_secid and "secid" in old_columns else [])
    read_columns = list(dict.fromkeys(read_columns))
    old_keys = pd.read_parquet(old_path, columns=read_columns)
    public_keys = pd.read_parquet(public_path, columns=read_columns)
    old_keys.loc[:, "date"] = pd.to_datetime(old_keys["date"])
    public_keys.loc[:, "date"] = pd.to_datetime(public_keys["date"])

    add_comparison(rows, dataset, "date_min", old_keys["date"].min(), public_keys["date"].min(), status_from_bool(old_keys["date"].min() == public_keys["date"].min()))
    add_comparison(rows, dataset, "date_max", old_keys["date"].max(), public_keys["date"].max(), status_from_bool(old_keys["date"].max() == public_keys["date"].max()))
    add_comparison(rows, dataset, "unique_permnos", old_keys["permno"].nunique(), public_keys["permno"].nunique(), status_from_bool(old_keys["permno"].nunique() == public_keys["permno"].nunique()))
    if include_secid and "secid" in old_keys.columns and "secid" in public_keys.columns:
        add_comparison(rows, dataset, "unique_secids", old_keys["secid"].nunique(), public_keys["secid"].nunique(), status_from_bool(old_keys["secid"].nunique() == public_keys["secid"].nunique()))

    old_duplicates = old_keys.duplicated(subset=key_columns).sum()
    public_duplicates = public_keys.duplicated(subset=key_columns).sum()
    add_comparison(rows, dataset, "duplicate_keys", old_duplicates, public_duplicates, status_from_bool(old_duplicates == public_duplicates == 0))

    same_order = (
        len(old_keys) == len(public_keys)
        and old_keys[key_columns].reset_index(drop=True).equals(public_keys[key_columns].reset_index(drop=True))
    )
    add_comparison(rows, dataset, "key_order_match", True, same_order, status_from_bool(same_order))

    return old_keys[key_columns], public_keys[key_columns], same_order


def compare_numeric_column_same_order(
    rows: list[dict[str, object]],
    dataset: str,
    old_path: Path,
    public_path: Path,
    column: str,
) -> None:
    """Compare one numeric column when row order is already known to match."""
    old_values = pd.read_parquet(old_path, columns=[column])[column]
    public_values = pd.read_parquet(public_path, columns=[column])[column]
    max_diff = max_abs_diff(old_values, public_values)
    mean_diff = mean_abs_diff(old_values, public_values)
    status = "PASS" if pd.notna(max_diff) and max_diff <= COMPARISON_TOLERANCE else "REVIEW"
    add_comparison(
        rows,
        dataset,
        f"{column}_max_abs_diff",
        0.0,
        max_diff,
        status,
        f"mean_abs_diff={mean_diff:.12g}",
    )


def compare_numeric_column_with_merge(
    rows: list[dict[str, object]],
    dataset: str,
    old_path: Path,
    public_path: Path,
    key_columns: list[str],
    column: str,
) -> None:
    """Compare one numeric column by joining on keys."""
    old_values = pd.read_parquet(old_path, columns=key_columns + [column])
    public_values = pd.read_parquet(public_path, columns=key_columns + [column])
    old_values.loc[:, "date"] = pd.to_datetime(old_values["date"])
    public_values.loc[:, "date"] = pd.to_datetime(public_values["date"])
    merged = old_values.merge(public_values, on=key_columns, how="inner", suffixes=("_old", "_public"))
    if len(merged) == 0:
        add_comparison(rows, dataset, f"{column}_max_abs_diff", "", "", "MISSING", "no matching keys")
        return
    max_diff = max_abs_diff(merged[f"{column}_old"], merged[f"{column}_public"])
    mean_diff = mean_abs_diff(merged[f"{column}_old"], merged[f"{column}_public"])
    status = "PASS" if pd.notna(max_diff) and max_diff <= COMPARISON_TOLERANCE else "REVIEW"
    add_comparison(
        rows,
        dataset,
        f"{column}_max_abs_diff",
        0.0,
        max_diff,
        status,
        f"mean_abs_diff={mean_diff:.12g}; matched_rows={len(merged):,}",
    )


def compare_numeric_columns(
    rows: list[dict[str, object]],
    dataset: str,
    old_path: Path,
    public_path: Path,
    key_columns: list[str],
    compare_columns: list[str],
    same_order: bool,
) -> None:
    """Compare selected numeric columns using the safest available method."""
    old_columns = set(parquet_columns(old_path))
    public_columns = set(parquet_columns(public_path))
    for column in compare_columns:
        if column not in old_columns or column not in public_columns:
            add_comparison(rows, dataset, f"{column}_max_abs_diff", "", "", "MISSING", "column missing")
            continue
        if same_order:
            compare_numeric_column_same_order(rows, dataset, old_path, public_path, column)
        else:
            compare_numeric_column_with_merge(rows, dataset, old_path, public_path, key_columns, column)
        gc.collect()


def compare_public_outputs() -> pd.DataFrame:
    """Compare public daily outputs against validated processed outputs."""
    rows: list[dict[str, object]] = []

    if not VALIDATED_DAILY_IV_PATH.exists() or not PUBLIC_DAILY_IV_PATH.exists():
        add_comparison(rows, "daily_iv", "file_exists", VALIDATED_DAILY_IV_PATH.exists(), PUBLIC_DAILY_IV_PATH.exists(), "MISSING")
    else:
        _, _, same_order = basic_file_comparisons(
            rows,
            "daily_iv",
            VALIDATED_DAILY_IV_PATH,
            PUBLIC_DAILY_IV_PATH,
            key_columns=["permno", "date"],
            include_secid=True,
        )
        compare_numeric_columns(
            rows,
            "daily_iv",
            VALIDATED_DAILY_IV_PATH,
            PUBLIC_DAILY_IV_PATH,
            key_columns=["permno", "date"],
            compare_columns=DAILY_IV_COMPARE_COLUMNS,
            same_order=same_order,
        )

    if not VALIDATED_DAILY_VRP_PATH.exists() or not PUBLIC_DAILY_VRP_PATH.exists():
        add_comparison(rows, "daily_vrp", "file_exists", VALIDATED_DAILY_VRP_PATH.exists(), PUBLIC_DAILY_VRP_PATH.exists(), "MISSING")
    else:
        _, _, same_order = basic_file_comparisons(
            rows,
            "daily_vrp",
            VALIDATED_DAILY_VRP_PATH,
            PUBLIC_DAILY_VRP_PATH,
            key_columns=["permno", "date"],
            include_secid=True,
        )
        compare_numeric_columns(
            rows,
            "daily_vrp",
            VALIDATED_DAILY_VRP_PATH,
            PUBLIC_DAILY_VRP_PATH,
            key_columns=["permno", "date"],
            compare_columns=DAILY_VRP_COMPARE_COLUMNS,
            same_order=same_order,
        )

    return pd.DataFrame(rows)


def status_counts(df: pd.DataFrame) -> dict[str, int]:
    """Return status counts as a plain dictionary."""
    if df.empty or "status" not in df.columns:
        return {}
    return {str(key): int(value) for key, value in df["status"].value_counts().sort_index().items()}


def format_counts(counts: dict[str, int]) -> str:
    """Format PASS/WARN/FAIL/INFO style counts."""
    return ", ".join(f"{key}={counts.get(key, 0)}" for key in ["PASS", "WARN", "FAIL", "INFO", "REVIEW", "MISSING"])


def write_documentation_report(
    iv_validation: pd.DataFrame,
    vrp_validation: pd.DataFrame,
    comparison: pd.DataFrame,
) -> None:
    """Write the public option-signal construction report."""
    iv_counts = status_counts(iv_validation)
    vrp_counts = status_counts(vrp_validation)
    comparison_counts = status_counts(comparison)
    non_pass = comparison[comparison["status"] != "PASS"] if not comparison.empty else pd.DataFrame()

    lines = [
        "# Public Pipeline Step 9: Option Signal Construction",
        "",
        "## Summary",
        "",
        "Created a standalone public option-signal construction script that builds daily IV signals and daily VRP signals from cached local data.",
        "",
        "## Files Created",
        "",
        "- `scripts/public/02_build_option_signals.py`",
        "- `data/public_processed/daily_iv_signals_2010_2023.parquet`",
        "- `data/public_processed/daily_signals_with_vrp_2010_2023.parquet`",
        "- `outputs/public_2010_2023/tables/public_daily_iv_build_summary.csv`",
        "- `outputs/public_2010_2023/tables/public_daily_vrp_build_summary.csv`",
        "- `outputs/public_2010_2023/tables/public_daily_iv_validation.csv`",
        "- `outputs/public_2010_2023/tables/public_daily_vrp_validation.csv`",
        "- `outputs/public_2010_2023/tables/public_option_signal_comparison.csv`",
        "",
        "## Source Changes",
        "",
        "No `src/` files were modified.",
        "",
        "## Legacy Development Files",
        "",
        "The public script uses reusable functions from `src.signals` and `src.linking` directly. It does not execute legacy development files.",
        "",
        "## Public Inputs Used",
        "",
        f"- `{VOL_SURFACE_PATH.relative_to(PROJECT_ROOT)}`",
        f"- `{LINK_TABLE_PATH.relative_to(PROJECT_ROOT)}`",
        f"- `{CRSP_DAILY_PATH.relative_to(PROJECT_ROOT)}`",
        "",
        "## Validation Totals",
        "",
        f"- Daily IV validation: {format_counts(iv_counts)}",
        f"- Daily VRP validation: {format_counts(vrp_counts)}",
        "",
        "## Comparison Against Validated Outputs",
        "",
        f"- Comparison totals: {format_counts(comparison_counts)}",
        "",
    ]

    if non_pass.empty:
        lines.extend(["No comparison discrepancies were found.", ""])
    else:
        lines.extend(
            [
                "Comparison rows requiring review:",
                "",
                non_pass.to_markdown(index=False),
                "",
            ]
        )

    lines.extend(
        [
            "## GitHub Safety",
            "",
            "The script is safe for the public code path because it reads cached local data, writes only public processed outputs, and does not require WRDS.",
            "",
            "## Recommended Next Step",
            "",
            "Build `scripts/public/01_pull_data.py` last, because it is the only remaining public script that will need WRDS access and careful cache behavior.",
            "",
        ]
    )

    DOC_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    DOC_REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved {DOC_REPORT_PATH}")


def main() -> None:
    """Run public daily option-signal construction, validation, and comparison."""
    print_header("Public Option Signal Construction")
    validate_input_files()
    PUBLIC_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    PUBLIC_TABLES_DIR.mkdir(parents=True, exist_ok=True)

    daily_iv = build_public_daily_iv_signals()
    iv_validation = validate_daily_iv(daily_iv)
    iv_build_summary = create_daily_iv_build_summary(daily_iv)
    daily_iv_rows = len(daily_iv)

    daily_vrp = build_public_daily_vrp_panel(daily_iv)
    del daily_iv
    gc.collect()

    vrp_validation = validate_daily_vrp(daily_vrp)
    vrp_build_summary = create_daily_vrp_build_summary(daily_vrp, daily_iv_rows)
    del daily_vrp
    gc.collect()

    save_table(iv_build_summary, PUBLIC_TABLES_DIR / "public_daily_iv_build_summary.csv")
    save_table(vrp_build_summary, PUBLIC_TABLES_DIR / "public_daily_vrp_build_summary.csv")
    save_table(iv_validation, PUBLIC_TABLES_DIR / "public_daily_iv_validation.csv")
    save_table(vrp_validation, PUBLIC_TABLES_DIR / "public_daily_vrp_validation.csv")

    print_header("Compare Public Outputs Against Validated Outputs")
    comparison = compare_public_outputs()
    save_table(comparison, PUBLIC_TABLES_DIR / "public_option_signal_comparison.csv")

    write_documentation_report(iv_validation, vrp_validation, comparison)

    iv_counts = status_counts(iv_validation)
    vrp_counts = status_counts(vrp_validation)
    comparison_counts = status_counts(comparison)
    iv_shape = parquet_shape(PUBLIC_DAILY_IV_PATH)
    vrp_shape = parquet_shape(PUBLIC_DAILY_VRP_PATH)
    non_pass = comparison[comparison["status"] != "PASS"] if not comparison.empty else pd.DataFrame()

    print_header("Terminal Summary")
    print("scripts/public/02_build_option_signals.py created: yes")
    print("src files modified: no")
    print("legacy files called or loaded: no")
    print("run status: completed")
    print(f"public daily IV signal shape: {iv_shape}")
    print(f"public daily VRP shape: {vrp_shape}")
    print(f"daily IV validation totals: {format_counts(iv_counts)}")
    print(f"daily VRP validation totals: {format_counts(vrp_counts)}")
    print(f"comparison totals: {format_counts(comparison_counts)}")
    if non_pass.empty:
        print("discrepancies: none")
    else:
        print("discrepancies:")
        print(non_pass.to_string(index=False))

    print("\nOutput paths:")
    for path in [
        PUBLIC_DAILY_IV_PATH,
        PUBLIC_DAILY_VRP_PATH,
        PUBLIC_TABLES_DIR / "public_daily_iv_build_summary.csv",
        PUBLIC_TABLES_DIR / "public_daily_vrp_build_summary.csv",
        PUBLIC_TABLES_DIR / "public_daily_iv_validation.csv",
        PUBLIC_TABLES_DIR / "public_daily_vrp_validation.csv",
        PUBLIC_TABLES_DIR / "public_option_signal_comparison.csv",
        DOC_REPORT_PATH,
    ]:
        print(path.relative_to(PROJECT_ROOT))

    print("\nRecommended next step: build scripts/public/01_pull_data.py.")


if __name__ == "__main__":
    main()
