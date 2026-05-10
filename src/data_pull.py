"""Reusable WRDS data-pull functions.

This module only pulls raw sample data and saves it locally. Analysis,
signal construction, linking, and regressions will be added later.
"""

from pathlib import Path
import time

import pandas as pd
import wrds


def format_elapsed(seconds):
    """Format elapsed seconds as a compact human-readable string."""
    minutes, seconds = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)

    if hours:
        return f"{hours}h {minutes}m {seconds}s"
    if minutes:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


def connect_wrds():
    """Open a WRDS connection."""
    print("Connecting to WRDS...")
    db = wrds.Connection()
    print("Connected to WRDS.")
    return db


def print_dataset_summary(df, name):
    """Print a small summary for a pulled dataset."""
    print(f"{name}: {df.shape[0]:,} rows x {df.shape[1]:,} columns")

    if "date" in df.columns and not df.empty:
        print(f"{name} date range: {df['date'].min().date()} to {df['date'].max().date()}")


def save_parquet(df, path):
    """Save a dataframe to parquet, creating the parent folder if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    print(f"Saved {len(df):,} rows to {path}")


def _year_range(start_date, end_date):
    """Return calendar years covered by a start and end date."""
    start_year = pd.to_datetime(start_date).year
    end_year = pd.to_datetime(end_date).year
    return range(start_year, end_year + 1)


def _optionmetrics_vol_surface_filter(start_date, end_date):
    """Return the standard WHERE clause for volatility surface sample pulls."""
    # In this WRDS OptionMetrics version, deltas are stored as whole numbers,
    # not decimals: 50 means 0.50 delta call, -50 means -0.50 delta put, and
    # -25 means -0.25 delta put.
    return f"""
        days = 30
        AND delta IN (50, -50, -25)
        AND date BETWEEN '{start_date}' AND '{end_date}'
        AND impl_volatility IS NOT NULL
        AND impl_volatility > 0
        AND impl_volatility < 5
    """


def _load_cached_year(path):
    """Load a yearly parquet cache if it exists and contains rows."""
    if not path.exists():
        return None

    cached_df = pd.read_parquet(path)
    if cached_df.empty:
        print(f"    Existing cache has 0 rows and will be overwritten: {path}")
        return None

    print(f"    Loaded {len(cached_df):,} cached rows from {path}")
    return cached_df


def pull_optionmetrics_vol_surface(db, start_date, end_date, raw_data_dir=None):
    """Pull a sample of OptionMetrics 30-day volatility surface data."""
    print("\nPulling OptionMetrics volatility surface...")

    yearly_frames = []
    years = list(_year_range(start_date, end_date))
    raw_data_dir = Path(raw_data_dir) if raw_data_dir is not None else None

    for year_number, year in enumerate(years, start=1):
        print(f"  Year {year} ({year_number}/{len(years)})...")
        year_start_time = time.perf_counter()
        yearly_cache_path = None

        if raw_data_dir is not None:
            yearly_cache_path = raw_data_dir / f"vol_surface_{year}.parquet"
            cached_df = _load_cached_year(yearly_cache_path)
            if cached_df is not None:
                yearly_frames.append(cached_df.copy())
                continue

        where_clause = _optionmetrics_vol_surface_filter(start_date, end_date)

        count_query = f"""
            SELECT COUNT(*) AS n
            FROM optionm.vsurfd{year}
            WHERE {where_clause}
        """

        count_df = db.raw_sql(count_query)
        expected_rows = int(count_df["n"].iloc[0])
        print(f"    Matching rows before pull: {expected_rows:,}")

        query = f"""
            SELECT
                secid,
                date,
                days,
                delta,
                impl_volatility,
                cp_flag
            FROM optionm.vsurfd{year}
            WHERE {where_clause}
        """

        year_df = db.raw_sql(query).copy()
        year_df.loc[:, "date"] = pd.to_datetime(year_df["date"])
        yearly_frames.append(year_df)

        if yearly_cache_path is not None:
            save_parquet(year_df, yearly_cache_path)

        year_elapsed = time.perf_counter() - year_start_time
        print(f"    Retrieved {len(year_df):,} rows in {format_elapsed(year_elapsed)}.")

    if yearly_frames:
        df = pd.concat(yearly_frames, ignore_index=True)
    else:
        df = pd.DataFrame()

    print(f"Total combined volatility surface rows: {len(df):,}")
    print_dataset_summary(df, "OptionMetrics volatility surface")
    return df


def pull_optionmetrics_security_master(db):
    """Pull OptionMetrics equity security master records."""
    print("\nPulling OptionMetrics security master...")

    query = """
        SELECT
            secid,
            cusip,
            ticker,
            issue_type
        FROM optionm.securd
        WHERE issue_type = '0'
    """

    df = db.raw_sql(query)
    print_dataset_summary(df, "OptionMetrics security master")
    return df


def pull_crsp_daily(db, start_date, end_date):
    """Pull CRSP daily stock data with valid-name filters."""
    print("\nPulling CRSP daily stock data...")

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

    df = db.raw_sql(query)
    df["date"] = pd.to_datetime(df["date"])
    print_dataset_summary(df, "CRSP daily")
    return df


def pull_crsp_monthly(db, start_date, end_date):
    """Pull CRSP monthly stock data with valid-name filters."""
    print("\nPulling CRSP monthly stock data...")

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

    df = db.raw_sql(query)
    df["date"] = pd.to_datetime(df["date"])
    print_dataset_summary(df, "CRSP monthly")
    return df
