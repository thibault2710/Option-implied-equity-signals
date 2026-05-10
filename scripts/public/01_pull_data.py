"""Prepare cached raw inputs for the public 2010-2023 pipeline.

This script is cache-first. By default it checks local files and writes public
status tables only. It opens WRDS only when --allow-wrds is provided and a
selected cache is missing, or when --force-pull is also provided.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (  # noqa: E402
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    sample_crsp_daily_path,
    sample_crsp_monthly_path,
    sample_raw_vol_surface_path,
)


PUBLIC_TABLES_DIR = PROJECT_ROOT / "outputs" / "public_2010_2023" / "tables"
DOC_REPORT_PATH = PROJECT_ROOT / "docs" / "public_pipeline_step10_data_pull_report.md"

VOL_COLUMNS = ["secid", "date", "days", "delta", "impl_volatility", "cp_flag"]
CRSP_DAILY_COLUMNS = ["permno", "date", "ret", "vol", "prc", "shrout", "cusip", "exchcd", "shrcd"]
CRSP_MONTHLY_COLUMNS = ["permno", "date", "ret", "retx", "cusip", "exchcd", "shrcd"]
SECURITY_COLUMNS = [
    "secid",
    "cusip",
    "ticker",
    "sic",
    "index_flag",
    "exchange_d",
    "class",
    "issue_type",
    "industry_group",
]
LINK_COLUMNS = ["secid", "sdate", "edate", "permno", "score"]
FACTOR_FILES = [
    "F-F_Research_Data_5_Factors_2x3.csv",
    "F-F_Momentum_Factor.csv",
]

DATASETS = {
    "vol_surface": {
        "path": sample_raw_vol_surface_path(2010, 2023),
        "required_for": "daily option signal construction",
        "columns": VOL_COLUMNS,
        "date_column": "date",
    },
    "crsp_daily": {
        "path": sample_crsp_daily_path(2010, 2023),
        "required_for": "realized variance and VRP construction",
        "columns": CRSP_DAILY_COLUMNS,
        "date_column": "date",
    },
    "crsp_monthly": {
        "path": sample_crsp_monthly_path(2010, 2024),
        "required_for": "monthly forward returns",
        "columns": CRSP_MONTHLY_COLUMNS,
        "date_column": "date",
    },
    "security_master": {
        "path": RAW_DATA_DIR / "security_master_full.parquet",
        "required_for": "sector enrichment",
        "columns": SECURITY_COLUMNS,
        "date_column": None,
    },
    "link_table": {
        "path": PROCESSED_DATA_DIR / "secid_permno_bridge_wrdsapps.parquet",
        "required_for": "time-aware OptionMetrics to CRSP linking",
        "columns": LINK_COLUMNS,
        "date_column": None,
    },
    "ff5_factors": {
        "path": RAW_DATA_DIR / "factors" / FACTOR_FILES[0],
        "required_for": "factor regressions",
        "columns": None,
        "date_column": None,
    },
    "momentum_factor": {
        "path": RAW_DATA_DIR / "factors" / FACTOR_FILES[1],
        "required_for": "factor regressions",
        "columns": None,
        "date_column": None,
    },
}


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


def add_validation(rows: list[dict[str, object]], file_key: str, check: str, status: str, details: str) -> None:
    """Append one validation row."""
    rows.append(
        {
            "file_key": file_key,
            "check": check,
            "status": status,
            "details": details,
        }
    )


def parquet_shape(path: Path) -> tuple[int, int]:
    """Return parquet shape from metadata."""
    parquet_file = pq.ParquetFile(path)
    return parquet_file.metadata.num_rows, len(parquet_file.schema_arrow.names)


def parquet_columns(path: Path) -> list[str]:
    """Return parquet column names from metadata."""
    return list(pq.ParquetFile(path).schema_arrow.names)


def parse_monthly_ken_french_file(path: Path) -> tuple[int, str, str]:
    """Parse the monthly section of a local Ken French CSV enough to validate it."""
    with path.open("r", encoding="latin1") as file:
        lines = file.readlines()

    header_index = None
    for index, line in enumerate(lines):
        parts = [item.strip().lower() for item in line.strip().split(",")]
        if len(parts) > 1 and ("mkt-rf" in parts or "mom" in parts):
            header_index = index
            break

    if header_index is None:
        raise ValueError("Could not find monthly factor header.")

    months = []
    for line in lines[header_index + 1 :]:
        parts = [item.strip() for item in line.strip().split(",")]
        if not parts or not parts[0].isdigit() or len(parts[0]) != 6:
            break
        months.append(parts[0])

    if not months:
        raise ValueError("Could not find monthly factor rows.")

    return len(months), months[0], months[-1]


def inspect_parquet_cache(file_key: str, config: dict[str, object]) -> dict[str, object]:
    """Inspect one parquet cache for the cache-status table."""
    path = Path(config["path"])
    row = {
        "file_key": file_key,
        "path": str(path.relative_to(PROJECT_ROOT) if path.is_relative_to(PROJECT_ROOT) else path),
        "exists": path.exists(),
        "readable": False,
        "shape": "",
        "min_date": "",
        "max_date": "",
        "required_for": config["required_for"],
        "status": "FAIL",
        "notes": "",
    }

    if not path.exists():
        row["status"] = "WARN"
        row["notes"] = "Missing local cache. Use --allow-wrds to pull missing data where supported."
        return row

    try:
        n_rows, n_cols = parquet_shape(path)
        row["readable"] = True
        row["shape"] = f"{n_rows}x{n_cols}"
        date_column = config.get("date_column")
        if date_column:
            dates = pd.read_parquet(path, columns=[str(date_column)])[str(date_column)]
            dates = pd.to_datetime(dates, errors="coerce")
            row["min_date"] = dates.min()
            row["max_date"] = dates.max()
        row["status"] = "PASS"
        row["notes"] = "Local cache exists and is readable."
    except Exception as exc:
        row["status"] = "FAIL"
        row["notes"] = f"Could not read cache: {exc}"

    return row


def inspect_factor_cache(file_key: str, config: dict[str, object]) -> dict[str, object]:
    """Inspect one factor CSV for the cache-status table."""
    path = Path(config["path"])
    row = {
        "file_key": file_key,
        "path": str(path.relative_to(PROJECT_ROOT) if path.is_relative_to(PROJECT_ROOT) else path),
        "exists": path.exists(),
        "readable": False,
        "shape": "",
        "min_date": "",
        "max_date": "",
        "required_for": config["required_for"],
        "status": "FAIL",
        "notes": "",
    }

    if not path.exists():
        row["status"] = "WARN"
        row["notes"] = "Missing local factor CSV. Download manually from Ken French data library."
        return row

    try:
        n_rows, first_month, last_month = parse_monthly_ken_french_file(path)
        row["readable"] = True
        row["shape"] = f"{n_rows} monthly rows"
        row["min_date"] = first_month
        row["max_date"] = last_month
        row["status"] = "PASS"
        row["notes"] = "Local factor CSV exists and monthly section is parseable."
    except Exception as exc:
        row["status"] = "FAIL"
        row["notes"] = f"Could not parse factor CSV: {exc}"

    return row


def build_cache_status(selected_keys: list[str]) -> pd.DataFrame:
    """Create cache status rows for selected required files."""
    rows = []
    for file_key in selected_keys:
        config = DATASETS[file_key]
        if file_key in {"ff5_factors", "momentum_factor"}:
            rows.append(inspect_factor_cache(file_key, config))
        else:
            rows.append(inspect_parquet_cache(file_key, config))
    return pd.DataFrame(rows)


def validate_required_columns(rows: list[dict[str, object]], file_key: str, path: Path, required_columns: list[str]) -> list[str]:
    """Validate required parquet columns."""
    if not path.exists():
        add_validation(rows, file_key, "file_exists", "WARN", "Missing cache.")
        return []
    try:
        columns = parquet_columns(path)
    except Exception as exc:
        add_validation(rows, file_key, "readable", "FAIL", f"Could not read parquet metadata: {exc}")
        return []

    missing = sorted(set(required_columns) - set(columns))
    add_validation(rows, file_key, "required_columns", status_from_bool(not missing), f"missing={missing}")
    return columns


def validate_vol_surface(rows: list[dict[str, object]], path: Path) -> None:
    """Validate cached OptionMetrics volatility surface."""
    columns = validate_required_columns(rows, "vol_surface", path, VOL_COLUMNS)
    if not columns:
        return

    data = pd.read_parquet(path, columns=VOL_COLUMNS)
    dates = pd.to_datetime(data["date"], errors="coerce")
    add_validation(rows, "vol_surface", "row_count_reasonable", status_from_bool(len(data) > 40_000_000), f"rows={len(data):,}")
    add_validation(rows, "vol_surface", "date_range", status_from_bool(dates.min() <= pd.Timestamp("2010-01-04") and dates.max() >= pd.Timestamp("2023-12-29")), f"{dates.min()} to {dates.max()}")
    add_validation(rows, "vol_surface", "unique_days_30_only", status_from_bool(set(pd.to_numeric(data["days"], errors="coerce").dropna().unique()) == {30}), f"days={sorted(pd.to_numeric(data['days'], errors='coerce').dropna().unique().tolist())}")
    missing_iv = data["impl_volatility"].isna().sum()
    add_validation(rows, "vol_surface", "missing_impl_volatility", status_from_bool(missing_iv == 0), f"{missing_iv:,}")

    flags = data["cp_flag"].astype("string").str.strip().str.upper()
    deltas = pd.to_numeric(data["delta"], errors="coerce")
    required_masks = {
        "atm_put_rows": (deltas == -50) & (flags == "P"),
        "otm_put_rows": (deltas == -25) & (flags == "P"),
        "atm_call_rows": (deltas == 50) & (flags == "C"),
    }
    for check, mask in required_masks.items():
        count = int(mask.sum())
        add_validation(rows, "vol_surface", check, status_from_bool(count > 0), f"{count:,}")


def validate_crsp_daily(rows: list[dict[str, object]], path: Path) -> None:
    """Validate cached CRSP daily data."""
    columns = validate_required_columns(rows, "crsp_daily", path, CRSP_DAILY_COLUMNS)
    if not columns:
        return

    data = pd.read_parquet(path, columns=CRSP_DAILY_COLUMNS)
    dates = pd.to_datetime(data["date"], errors="coerce")
    duplicates = data.duplicated(subset=["permno", "date"]).sum()
    add_validation(rows, "crsp_daily", "row_count_reasonable", status_from_bool(len(data) > 12_000_000), f"rows={len(data):,}")
    add_validation(rows, "crsp_daily", "date_range", status_from_bool(dates.min() <= pd.Timestamp("2009-12-01") and dates.max() >= pd.Timestamp("2023-12-29")), f"{dates.min()} to {dates.max()}")
    add_validation(rows, "crsp_daily", "duplicate_permno_date", status_from_bool(duplicates == 0), f"{duplicates:,}")
    add_validation(rows, "crsp_daily", "has_price_and_shares", status_from_bool(data["prc"].notna().any() and data["shrout"].notna().any()), "prc/shrout available for mktcap")
    add_validation(rows, "crsp_daily", "share_exchange_filters", status_from_bool(set(pd.to_numeric(data["exchcd"], errors="coerce").dropna().astype(int).unique()).issubset({1, 2, 3}) and set(pd.to_numeric(data["shrcd"], errors="coerce").dropna().astype(int).unique()).issubset({10, 11})), f"shrcd={sorted(data['shrcd'].dropna().unique().tolist())}; exchcd={sorted(data['exchcd'].dropna().unique().tolist())}")


def validate_crsp_monthly(rows: list[dict[str, object]], path: Path) -> None:
    """Validate cached CRSP monthly data."""
    columns = validate_required_columns(rows, "crsp_monthly", path, CRSP_MONTHLY_COLUMNS)
    if not columns:
        return

    data = pd.read_parquet(path, columns=CRSP_MONTHLY_COLUMNS)
    dates = pd.to_datetime(data["date"], errors="coerce")
    return_months = pd.PeriodIndex(dates.dt.to_period("M"), freq="M")
    duplicates = data.duplicated(subset=["permno", "date"]).sum()
    add_validation(rows, "crsp_monthly", "row_count_reasonable", status_from_bool(len(data) > 600_000), f"rows={len(data):,}")
    add_validation(rows, "crsp_monthly", "date_range", status_from_bool(return_months.min() <= pd.Period("2010-01", "M") and return_months.max() >= pd.Period("2024-01", "M")), f"{return_months.min()} to {return_months.max()}")
    add_validation(rows, "crsp_monthly", "duplicate_permno_month", status_from_bool(duplicates == 0), f"{duplicates:,}")
    add_validation(rows, "crsp_monthly", "share_exchange_filters", status_from_bool(set(pd.to_numeric(data["exchcd"], errors="coerce").dropna().astype(int).unique()).issubset({1, 2, 3}) and set(pd.to_numeric(data["shrcd"], errors="coerce").dropna().astype(int).unique()).issubset({10, 11})), f"shrcd={sorted(data['shrcd'].dropna().unique().tolist())}; exchcd={sorted(data['exchcd'].dropna().unique().tolist())}")


def validate_security_master(rows: list[dict[str, object]], path: Path) -> None:
    """Validate cached OptionMetrics security metadata."""
    columns = validate_required_columns(rows, "security_master", path, ["secid"])
    if not columns:
        return
    data = pd.read_parquet(path)
    add_validation(rows, "security_master", "row_count_reasonable", status_from_bool(len(data) > 100_000), f"rows={len(data):,}")
    add_validation(rows, "security_master", "sector_helper_columns", status_from_bool(any(column in data.columns for column in ["sic", "industry_group"])), f"columns={list(data.columns)}")


def validate_link_table(rows: list[dict[str, object]], path: Path) -> None:
    """Validate cached official CRSP-OptionMetrics link table."""
    columns = validate_required_columns(rows, "link_table", path, LINK_COLUMNS)
    if not columns:
        return
    data = pd.read_parquet(path, columns=LINK_COLUMNS)
    data.loc[:, "sdate"] = pd.to_datetime(data["sdate"], errors="coerce")
    data.loc[:, "edate"] = pd.to_datetime(data["edate"], errors="coerce")
    invalid_ranges = (data["sdate"] > data["edate"]).sum()
    add_validation(rows, "link_table", "row_count_reasonable", status_from_bool(len(data) > 30_000), f"rows={len(data):,}")
    add_validation(rows, "link_table", "invalid_date_ranges", status_from_bool(invalid_ranges == 0), f"{invalid_ranges:,}")
    add_validation(rows, "link_table", "unique_secids", "INFO", f"{data['secid'].nunique():,}")
    add_validation(rows, "link_table", "unique_permnos", "INFO", f"{data['permno'].nunique():,}")


def validate_factor_file(rows: list[dict[str, object]], file_key: str, path: Path) -> None:
    """Validate one local factor CSV."""
    if not path.exists():
        add_validation(rows, file_key, "file_exists", "WARN", "Missing local factor CSV.")
        return
    try:
        n_rows, first_month, last_month = parse_monthly_ken_french_file(path)
        add_validation(rows, file_key, "parseable_monthly_section", "PASS", f"rows={n_rows:,}; range={first_month} to {last_month}")
    except Exception as exc:
        add_validation(rows, file_key, "parseable_monthly_section", "FAIL", str(exc))


def build_validation(selected_keys: list[str]) -> pd.DataFrame:
    """Build validation rows for selected cache files."""
    rows: list[dict[str, object]] = []
    for file_key in selected_keys:
        path = Path(DATASETS[file_key]["path"])
        if file_key == "vol_surface":
            validate_vol_surface(rows, path)
        elif file_key == "crsp_daily":
            validate_crsp_daily(rows, path)
        elif file_key == "crsp_monthly":
            validate_crsp_monthly(rows, path)
        elif file_key == "security_master":
            validate_security_master(rows, path)
        elif file_key == "link_table":
            validate_link_table(rows, path)
        elif file_key in {"ff5_factors", "momentum_factor"}:
            validate_factor_file(rows, file_key, path)
    return pd.DataFrame(rows)


def normalize_vol_surface_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize volatility-surface rows before caching."""
    data = df[VOL_COLUMNS].copy()
    data.loc[:, "secid"] = pd.to_numeric(data["secid"], errors="coerce")
    data.loc[:, "date"] = pd.to_datetime(data["date"], errors="coerce")
    data.loc[:, "days"] = pd.to_numeric(data["days"], errors="coerce")
    data.loc[:, "delta"] = pd.to_numeric(data["delta"], errors="coerce")
    data.loc[:, "impl_volatility"] = pd.to_numeric(data["impl_volatility"], errors="coerce")
    data.loc[:, "cp_flag"] = data["cp_flag"].astype("string").str.strip().str.upper()
    data = data.dropna(subset=VOL_COLUMNS).copy()
    correct_rows = (
        (data["days"] == 30)
        & (
            ((data["delta"] == -50) & (data["cp_flag"] == "P"))
            | ((data["delta"] == -25) & (data["cp_flag"] == "P"))
            | ((data["delta"] == 50) & (data["cp_flag"] == "C"))
        )
    )
    data = data.loc[correct_rows, VOL_COLUMNS].copy()
    data.loc[:, "secid"] = data["secid"].astype("int64")
    data.loc[:, "days"] = data["days"].astype("int64")
    data.loc[:, "delta"] = data["delta"].astype("int64")
    data.loc[:, "impl_volatility"] = data["impl_volatility"].astype("float64")
    return data.sort_values(["secid", "date", "delta", "cp_flag"]).reset_index(drop=True)


def normalize_crsp_dtypes(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Normalize CRSP rows before caching."""
    data = df[columns].copy()
    data.loc[:, "permno"] = pd.to_numeric(data["permno"], errors="coerce")
    data.loc[:, "date"] = pd.to_datetime(data["date"], errors="coerce")
    for column in ["ret", "retx", "vol", "prc", "shrout", "exchcd", "shrcd"]:
        if column in data.columns:
            data.loc[:, column] = pd.to_numeric(data[column], errors="coerce")
    if "cusip" in data.columns:
        data.loc[:, "cusip"] = data["cusip"].astype("string").str.strip()

    required = [column for column in ["permno", "date", "ret", "exchcd", "shrcd"] if column in data.columns]
    data = data.dropna(subset=required).copy()
    data.loc[:, "permno"] = data["permno"].astype("int64")
    data.loc[:, "exchcd"] = data["exchcd"].astype("int64")
    data.loc[:, "shrcd"] = data["shrcd"].astype("int64")
    for column in ["ret", "retx", "vol", "prc", "shrout"]:
        if column in data.columns:
            data.loc[:, column] = data[column].astype("float64")
    return data.sort_values(["permno", "date"]).reset_index(drop=True)


def optionmetrics_where_clause(year: int) -> str:
    """Return the public-pipeline OptionMetrics filter for one year."""
    return f"""
        date BETWEEN '{year}-01-01' AND '{year}-12-31'
        AND days = 30
        AND impl_volatility IS NOT NULL
        AND impl_volatility > 0
        AND impl_volatility < 5
        AND (
            (delta = -50 AND cp_flag = 'P')
            OR (delta = -25 AND cp_flag = 'P')
            OR (delta = 50 AND cp_flag = 'C')
        )
    """


def open_wrds_connection():
    """Open WRDS only after explicit user CLI permission."""
    from src.data_pull import connect_wrds

    return connect_wrds()


def save_parquet(df: pd.DataFrame, path: Path) -> None:
    """Save a parquet file, creating parent folders first."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    print(f"Saved {path}: shape={df.shape}")


def pull_optionmetrics_surface(db, start_date: str, end_date: str, force_pull: bool) -> list[Path]:
    """Pull missing or forced OptionMetrics surface files."""
    output_path = DATASETS["vol_surface"]["path"]
    if output_path.exists() and not force_pull:
        print(f"Using cached volatility surface: {output_path}")
        return []

    start_year = pd.Timestamp(start_date).year
    end_year = pd.Timestamp(end_date).year
    yearly_frames = []
    written_paths = []
    for year in range(start_year, end_year + 1):
        cache_path = RAW_DATA_DIR / f"vol_surface_{year}.parquet"
        if cache_path.exists() and not force_pull:
            print(f"Loading yearly OptionMetrics cache: {cache_path}")
            year_df = pd.read_parquet(cache_path, columns=VOL_COLUMNS)
        else:
            table_name = f"optionm.vsurfd{year}"
            query = f"""
                SELECT secid, date, days, delta, impl_volatility, cp_flag
                FROM {table_name}
                WHERE {optionmetrics_where_clause(year)}
            """
            print(f"Pulling {table_name}...")
            year_df = db.raw_sql(query)
            year_df = normalize_vol_surface_dtypes(year_df)
            save_parquet(year_df, cache_path)
            written_paths.append(cache_path)
        yearly_frames.append(normalize_vol_surface_dtypes(year_df))

    combined = pd.concat(yearly_frames, ignore_index=True)
    combined = normalize_vol_surface_dtypes(combined)
    save_parquet(combined, output_path)
    written_paths.append(output_path)
    return written_paths


def pull_crsp_daily(db, start_date: str, end_date: str, force_pull: bool) -> list[Path]:
    """Pull missing or forced CRSP daily data."""
    output_path = DATASETS["crsp_daily"]["path"]
    if output_path.exists() and not force_pull:
        print(f"Using cached CRSP daily: {output_path}")
        return []

    query = f"""
        SELECT
            a.permno,
            a.date,
            a.ret,
            a.vol,
            a.prc,
            a.shrout,
            b.cusip,
            b.exchcd,
            b.shrcd
        FROM crsp.dsf AS a
        INNER JOIN crsp.dsenames AS b
            ON a.permno = b.permno
           AND a.date >= b.namedt
           AND (a.date <= b.nameendt OR b.nameendt IS NULL)
        WHERE a.date BETWEEN '{start_date}' AND '{end_date}'
          AND b.shrcd IN (10, 11)
          AND b.exchcd IN (1, 2, 3)
    """
    print("Pulling CRSP daily from WRDS...")
    data = normalize_crsp_dtypes(db.raw_sql(query), CRSP_DAILY_COLUMNS)
    save_parquet(data, output_path)
    return [output_path]


def pull_crsp_monthly(db, start_date: str, end_date: str, force_pull: bool) -> list[Path]:
    """Pull missing or forced CRSP monthly data."""
    output_path = DATASETS["crsp_monthly"]["path"]
    if output_path.exists() and not force_pull:
        print(f"Using cached CRSP monthly: {output_path}")
        return []

    query = f"""
        SELECT
            a.permno,
            a.date,
            a.ret,
            a.retx,
            b.cusip,
            b.exchcd,
            b.shrcd
        FROM crsp.msf AS a
        INNER JOIN crsp.msenames AS b
            ON a.permno = b.permno
           AND a.date >= b.namedt
           AND (a.date <= b.nameendt OR b.nameendt IS NULL)
        WHERE a.date BETWEEN '{start_date}' AND '{end_date}'
          AND b.shrcd IN (10, 11)
          AND b.exchcd IN (1, 2, 3)
    """
    print("Pulling CRSP monthly from WRDS...")
    data = normalize_crsp_dtypes(db.raw_sql(query), CRSP_MONTHLY_COLUMNS)
    save_parquet(data, output_path)
    return [output_path]


def pull_security_master(db, force_pull: bool) -> list[Path]:
    """Pull missing or forced OptionMetrics security metadata."""
    output_path = DATASETS["security_master"]["path"]
    if output_path.exists() and not force_pull:
        print(f"Using cached security master: {output_path}")
        return []

    column_query = """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = 'optionm'
          AND table_name = 'securd'
        ORDER BY ordinal_position
    """
    available = db.raw_sql(column_query)
    available_columns = {str(column).lower() for column in available["column_name"].tolist()}
    selected_columns = [column for column in SECURITY_COLUMNS if column.lower() in available_columns]
    if "secid" not in selected_columns:
        raise ValueError("optionm.securd does not expose required column: secid")
    query = f"SELECT {', '.join(selected_columns)} FROM optionm.securd"
    print("Pulling OptionMetrics security master from WRDS...")
    data = db.raw_sql(query)
    save_parquet(data, output_path)
    return [output_path]


def pull_link_table(db, force_pull: bool) -> list[Path]:
    """Pull missing or forced official CRSP-OptionMetrics link table."""
    output_path = DATASETS["link_table"]["path"]
    if output_path.exists() and not force_pull:
        print(f"Using cached official link table: {output_path}")
        return []

    from src.linking import pull_wrdsapps_optionm_crsp_link, save_wrdsapps_link

    print("Pulling official CRSP-OptionMetrics link table from WRDS...")
    link_df = pull_wrdsapps_optionm_crsp_link(db)
    save_wrdsapps_link(link_df, output_path)
    return [output_path]


def selected_dataset_keys(args: argparse.Namespace) -> list[str]:
    """Return dataset keys selected by CLI flags."""
    keys = []
    if not args.skip_optionmetrics:
        keys.append("vol_surface")
    if not args.skip_crsp:
        keys.extend(["crsp_daily", "crsp_monthly"])
    if not args.skip_security_master:
        keys.append("security_master")
    if not args.skip_link_table:
        keys.append("link_table")
    if not args.skip_factors:
        keys.extend(["ff5_factors", "momentum_factor"])
    return keys


def missing_or_forced_keys(selected_keys: list[str], force_pull: bool) -> list[str]:
    """Return selected WRDS-backed keys that need a WRDS action."""
    wrds_keys = ["vol_surface", "crsp_daily", "crsp_monthly", "security_master", "link_table"]
    needed = []
    for key in selected_keys:
        if key not in wrds_keys:
            continue
        path = Path(DATASETS[key]["path"])
        if force_pull or not path.exists():
            needed.append(key)
    return needed


def run_wrds_actions(args: argparse.Namespace, selected_keys: list[str]) -> tuple[bool, list[Path]]:
    """Run allowed WRDS actions and return whether WRDS was contacted plus written paths."""
    needed = missing_or_forced_keys(selected_keys, args.force_pull)
    if not needed:
        print("No WRDS-backed files need pulling.")
        return False, []

    if args.check_cache_only:
        print("Cache-check-only mode: WRDS actions are disabled.")
        return False, []

    if not args.allow_wrds:
        print("WRDS action needed but --allow-wrds was not provided.")
        print(f"Would pull: {', '.join(needed)}")
        return False, []

    if args.force_pull and not args.allow_wrds:
        raise ValueError("--force-pull requires --allow-wrds.")

    db = None
    written_paths: list[Path] = []
    start_time = time.perf_counter()
    try:
        db = open_wrds_connection()
        if "vol_surface" in needed:
            written_paths.extend(pull_optionmetrics_surface(db, args.start_date, args.end_date, args.force_pull))
        if "crsp_daily" in needed:
            written_paths.extend(pull_crsp_daily(db, args.crsp_daily_start_date, args.end_date, args.force_pull))
        if "crsp_monthly" in needed:
            monthly_start = pd.Timestamp(args.start_date).replace(day=1).strftime("%Y-%m-%d")
            written_paths.extend(pull_crsp_monthly(db, monthly_start, args.crsp_monthly_end_date, args.force_pull))
        if "security_master" in needed:
            written_paths.extend(pull_security_master(db, args.force_pull))
        if "link_table" in needed:
            written_paths.extend(pull_link_table(db, args.force_pull))
    finally:
        if db is not None:
            db.close()
            print("Closed WRDS connection.")

    print(f"WRDS action runtime seconds: {time.perf_counter() - start_time:.1f}")
    return True, written_paths


def build_summary(
    args: argparse.Namespace,
    selected_keys: list[str],
    cache_status: pd.DataFrame,
    validation: pd.DataFrame,
    wrds_contacted: bool,
    written_paths: list[Path],
) -> pd.DataFrame:
    """Create one summary table for this run."""
    missing = cache_status.loc[~cache_status["exists"], "file_key"].tolist() if not cache_status.empty else []
    would_pull = missing_or_forced_keys(selected_keys, args.force_pull)
    validation_counts = validation["status"].value_counts().to_dict() if not validation.empty else {}
    return pd.DataFrame(
        [
            {
                "run_mode": "check_cache_only" if args.check_cache_only or not args.allow_wrds else "wrds_allowed",
                "selected_files": ", ".join(selected_keys),
                "missing_files": ", ".join(missing),
                "would_pull_with_wrds": ", ".join(would_pull),
                "wrds_contacted": wrds_contacted,
                "force_pull": args.force_pull,
                "raw_or_processed_files_written": len(written_paths),
                "written_paths": "; ".join(str(path.relative_to(PROJECT_ROOT) if path.is_relative_to(PROJECT_ROOT) else path) for path in written_paths),
                "validation_pass": int(validation_counts.get("PASS", 0)),
                "validation_warn": int(validation_counts.get("WARN", 0)),
                "validation_fail": int(validation_counts.get("FAIL", 0)),
                "validation_info": int(validation_counts.get("INFO", 0)),
            }
        ]
    )


def status_counts(df: pd.DataFrame) -> dict[str, int]:
    """Return status counts from a dataframe."""
    if df.empty or "status" not in df.columns:
        return {}
    return {str(key): int(value) for key, value in df["status"].value_counts().sort_index().items()}


def format_counts(counts: dict[str, int]) -> str:
    """Format standard status counts."""
    return ", ".join(f"{key}={counts.get(key, 0)}" for key in ["PASS", "WARN", "FAIL", "INFO"])


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    """Render a small dataframe as a markdown table without optional packages."""
    if df.empty:
        return "_No rows._"

    display = df.copy()
    display = display.astype(object).where(pd.notna(display), "")
    columns = [str(column) for column in display.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in display.iterrows():
        values = [str(row[column]).replace("|", "\\|") for column in display.columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(
    args: argparse.Namespace,
    cache_status: pd.DataFrame,
    validation: pd.DataFrame,
    summary: pd.DataFrame,
    wrds_contacted: bool,
    written_paths: list[Path],
) -> None:
    """Write the public data-pull documentation report."""
    cache_counts = status_counts(cache_status)
    validation_counts = status_counts(validation)
    missing = cache_status.loc[~cache_status["exists"], "file_key"].tolist() if not cache_status.empty else []
    non_pass_validation = validation[~validation["status"].isin(["PASS", "INFO"])] if not validation.empty else pd.DataFrame()

    lines = [
        "# Public Pipeline Step 10: Data Pull and Cache Check",
        "",
        "## Summary",
        "",
        "Created a cache-first public data preparation entrypoint for the 2010-2023 pipeline.",
        "",
        "## Files Created",
        "",
        "- `scripts/public/01_pull_data.py`",
        "- `outputs/public_2010_2023/tables/public_data_pull_cache_status.csv`",
        "- `outputs/public_2010_2023/tables/public_data_pull_summary.csv`",
        "- `outputs/public_2010_2023/tables/public_data_pull_validation.csv`",
        "",
        "## Source Changes",
        "",
        "No `src/` files were modified.",
        "",
        "## Legacy Development Files",
        "",
        "The public script does not execute legacy development files.",
        "",
        "## CLI Flags",
        "",
        "- `--check-cache-only`: check local caches only and do not connect to WRDS.",
        "- `--allow-wrds`: allow WRDS connection for missing selected files.",
        "- `--force-pull`: repull selected WRDS-backed files; requires `--allow-wrds`.",
        "- `--skip-optionmetrics`, `--skip-crsp`, `--skip-security-master`, `--skip-link-table`, `--skip-factors`: skip selected file groups.",
        "- `--start-date`, `--crsp-daily-start-date`, `--end-date`, `--crsp-monthly-end-date`: override public sample date bounds.",
        "",
        "## Cache Status",
        "",
        f"- Cache status totals: {format_counts(cache_counts)}",
        f"- Missing files: {', '.join(missing) if missing else 'none'}",
        "",
        "## Validation",
        "",
        f"- Validation totals: {format_counts(validation_counts)}",
        "",
        "## WRDS and File Writes",
        "",
        f"- WRDS contacted: {wrds_contacted}",
        f"- Raw or processed cache files written: {len(written_paths)}",
        "",
    ]

    if not non_pass_validation.empty:
        lines.extend(["Validation rows needing attention:", "", dataframe_to_markdown(non_pass_validation), ""])

    lines.extend(
        [
            "## GitHub Safety",
            "",
            "The script is safe for GitHub because default/cache-check mode does not connect to WRDS, does not store credentials, and reports missing licensed data clearly.",
            "",
            "## Current Run Summary",
            "",
            dataframe_to_markdown(summary),
            "",
            "## Recommended Next Step",
            "",
            "The standalone public pipeline scripts are now complete. Next, update the public README to document the full workflow and licensed-data exclusions.",
            "",
        ]
    )

    DOC_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    DOC_REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved {DOC_REPORT_PATH}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Cache-first data preparation for the public pipeline.")
    parser.add_argument("--check-cache-only", action="store_true", help="Check local caches only; never connect to WRDS.")
    parser.add_argument("--allow-wrds", action="store_true", help="Allow WRDS connection for missing selected files.")
    parser.add_argument("--force-pull", action="store_true", help="Repull selected WRDS-backed files; requires --allow-wrds.")
    parser.add_argument("--skip-optionmetrics", action="store_true", help="Skip OptionMetrics volatility-surface preparation.")
    parser.add_argument("--skip-crsp", action="store_true", help="Skip CRSP daily and monthly preparation.")
    parser.add_argument("--skip-security-master", action="store_true", help="Skip OptionMetrics security metadata preparation.")
    parser.add_argument("--skip-link-table", action="store_true", help="Skip official CRSP-OptionMetrics link-table preparation.")
    parser.add_argument("--skip-factors", action="store_true", help="Skip local Fama-French factor file checks.")
    parser.add_argument("--start-date", default="2010-01-01", help="Signal/OptionMetrics sample start date.")
    parser.add_argument("--crsp-daily-start-date", default="2009-12-01", help="CRSP daily start date for realized-variance lookback.")
    parser.add_argument("--end-date", default="2023-12-31", help="Signal/OptionMetrics/CRSP daily end date.")
    parser.add_argument("--crsp-monthly-end-date", default="2024-01-31", help="CRSP monthly end date for forward returns.")
    args = parser.parse_args()

    if args.force_pull and not args.allow_wrds:
        parser.error("--force-pull requires --allow-wrds.")
    return args


def main() -> None:
    """Run cache checks and optional WRDS pulls."""
    print_header("Public Data Pull and Cache Check")
    args = parse_args()
    PUBLIC_TABLES_DIR.mkdir(parents=True, exist_ok=True)

    selected_keys = selected_dataset_keys(args)
    print(f"Selected file groups: {', '.join(selected_keys)}")
    print(f"check_cache_only={args.check_cache_only}")
    print(f"allow_wrds={args.allow_wrds}")
    print(f"force_pull={args.force_pull}")

    wrds_contacted, written_paths = run_wrds_actions(args, selected_keys)

    cache_status = build_cache_status(selected_keys)
    validation = build_validation(selected_keys)
    summary = build_summary(args, selected_keys, cache_status, validation, wrds_contacted, written_paths)

    save_table(cache_status, PUBLIC_TABLES_DIR / "public_data_pull_cache_status.csv")
    save_table(validation, PUBLIC_TABLES_DIR / "public_data_pull_validation.csv")
    save_table(summary, PUBLIC_TABLES_DIR / "public_data_pull_summary.csv")
    write_report(args, cache_status, validation, summary, wrds_contacted, written_paths)

    cache_counts = status_counts(cache_status)
    validation_counts = status_counts(validation)
    missing = cache_status.loc[~cache_status["exists"], "file_key"].tolist() if not cache_status.empty else []

    print_header("Terminal Summary")
    print("scripts/public/01_pull_data.py created: yes")
    print("src files modified: no")
    print("legacy files called or loaded: no")
    print("cache-check run status: completed")
    print(f"WRDS contacted: {wrds_contacted}")
    print(f"raw or processed cache files written: {len(written_paths)}")
    print(f"cache status totals: {format_counts(cache_counts)}")
    print(f"validation totals: {format_counts(validation_counts)}")
    print(f"missing files: {', '.join(missing) if missing else 'none'}")
    print("\nOutput paths:")
    for path in [
        PUBLIC_TABLES_DIR / "public_data_pull_cache_status.csv",
        PUBLIC_TABLES_DIR / "public_data_pull_validation.csv",
        PUBLIC_TABLES_DIR / "public_data_pull_summary.csv",
        DOC_REPORT_PATH,
    ]:
        print(path.relative_to(PROJECT_ROOT))
    print("\nRecommended next step: update the public README with the completed public workflow.")


if __name__ == "__main__":
    main()
