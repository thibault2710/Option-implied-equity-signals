"""Research audit checks for the options-implied signals project."""

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


REQUIRED_FILES = [
    ("vol_surface", "data/raw/vol_surface_2018_2023.parquet"),
    ("crsp_daily", "data/raw/crsp_daily_2017_2023.parquet"),
    ("crsp_monthly", "data/raw/crsp_monthly_2018_2024.parquet"),
    ("official_link", "data/processed/secid_permno_bridge_wrdsapps.parquet"),
    ("daily_iv_signals", "data/processed/daily_iv_signals.parquet"),
    ("daily_signals_with_vrp", "data/processed/daily_signals_with_vrp.parquet"),
    ("monthly_signal_panel", "data/processed/monthly_signal_panel.parquet"),
    ("quintile_summary", "outputs/tables/quintile_summary.csv"),
    ("robustness_quintile_summary", "outputs/tables/robustness_quintile_summary.csv"),
    ("factor_regression_summary", "outputs/tables/factor_regression_summary.csv"),
]


def add_check(records, section, check_name, status, value="", message=""):
    """Append one audit check result."""
    records.append(
        {
            "section": section,
            "check_name": check_name,
            "status": status,
            "value": value,
            "message": message,
        }
    )


def print_section(title):
    """Print a consistent section header."""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def format_value(value):
    """Format audit values for compact CSV output."""
    if isinstance(value, float):
        return f"{value:.8g}"
    return str(value)


def period_count(first_period, last_period):
    """Count calendar months in a closed monthly Period range."""
    if pd.isna(first_period) or pd.isna(last_period):
        return np.nan
    first = pd.Period(str(first_period), freq="M")
    last = pd.Period(str(last_period), freq="M")
    return len(pd.period_range(first, last, freq="M"))


def period_range_summary(months):
    """Return first, last, observed count, and contiguous expected count."""
    periods = pd.PeriodIndex(pd.Series(months).dropna().astype(str), freq="M")
    if len(periods) == 0:
        return {
            "first_month": "",
            "last_month": "",
            "observed_months": 0,
            "expected_contiguous_months": 0,
        }

    unique_periods = pd.PeriodIndex(periods.unique()).sort_values()
    first = unique_periods.min()
    last = unique_periods.max()
    return {
        "first_month": first,
        "last_month": last,
        "observed_months": len(unique_periods),
        "expected_contiguous_months": period_count(first, last),
    }


def robustness_return_file_months(tables_dir, row):
    """Infer expected months for one robustness summary row from its return file."""
    file_name = (
        f"robustness_quintile_returns_{row['signal']}_{row['transform']}_"
        f"{row['universe']}_{row['weighting']}.csv"
    )
    path = tables_dir / file_name
    if not path.exists():
        return None

    returns = pd.read_csv(path)
    month_col = "return_month" if "return_month" in returns.columns else "signal_month"
    if month_col not in returns.columns:
        return None

    summary = period_range_summary(returns[month_col])
    summary["file_name"] = file_name
    summary["month_col"] = month_col
    return summary


def parquet_metadata(path):
    """Read Parquet metadata without loading the full dataset."""
    parquet_file = pq.ParquetFile(path)
    metadata = parquet_file.metadata
    compressed_mb = path.stat().st_size / 1024**2
    uncompressed_mb = sum(
        metadata.row_group(index).total_byte_size for index in range(metadata.num_row_groups)
    ) / 1024**2

    return {
        "rows": metadata.num_rows,
        "columns": parquet_file.schema.names,
        "file_size_mb": compressed_mb,
        "memory_usage_mb": uncompressed_mb,
    }


def file_existence_and_shape_checks(project_root, records):
    """Check required files and collect file-level counts."""
    print_section("1. File Existence and Basic Shape Checks")
    key_counts = []

    for label, relative_path in REQUIRED_FILES:
        path = project_root / relative_path
        exists = path.exists()

        if not exists:
            add_check(
                records,
                "File checks",
                f"{label} exists",
                "FAIL",
                relative_path,
                "Required file is missing.",
            )
            key_counts.append(
                {
                    "file": label,
                    "path": relative_path,
                    "exists": False,
                    "rows": pd.NA,
                    "n_columns": pd.NA,
                    "file_size_mb": pd.NA,
                    "memory_usage_mb": pd.NA,
                    "columns": "",
                }
            )
            print(f"[FAIL] Missing required file: {relative_path}")
            continue

        if path.suffix == ".parquet":
            metadata = parquet_metadata(path)
            rows = metadata["rows"]
            columns = metadata["columns"]
            file_size_mb = metadata["file_size_mb"]
            memory_usage_mb = metadata["memory_usage_mb"]
        else:
            data = pd.read_csv(path)
            rows = len(data)
            columns = list(data.columns)
            file_size_mb = path.stat().st_size / 1024**2
            memory_usage_mb = data.memory_usage(deep=True).sum() / 1024**2

        print(f"[PASS] {relative_path}")
        print(f"  shape: ({rows:,}, {len(columns):,})")
        print(f"  columns: {columns}")
        print(f"  memory usage estimate: {memory_usage_mb:,.2f} MB")

        add_check(
            records,
            "File checks",
            f"{label} exists",
            "PASS",
            relative_path,
            f"Loaded metadata with shape ({rows}, {len(columns)}).",
        )
        key_counts.append(
            {
                "file": label,
                "path": relative_path,
                "exists": True,
                "rows": rows,
                "n_columns": len(columns),
                "file_size_mb": file_size_mb,
                "memory_usage_mb": memory_usage_mb,
                "columns": ", ".join(columns),
            }
        )

    return pd.DataFrame(key_counts)


def check_required_columns(df, required_columns, section, records):
    """Audit whether a dataframe contains required columns."""
    missing = sorted(set(required_columns) - set(df.columns))
    status = "PASS" if not missing else "FAIL"
    add_check(
        records,
        section,
        "required columns exist",
        status,
        ", ".join(missing) if missing else "all present",
        "Missing required columns." if missing else "All required columns are present.",
    )
    return not missing


def audit_vol_surface(raw_data_dir, records):
    """Audit the OptionMetrics volatility surface."""
    print_section("2. OptionMetrics Volatility Surface Checks")
    path = raw_data_dir / "vol_surface_2018_2023.parquet"
    if not path.exists():
        add_check(records, "Vol surface", "file available", "FAIL", path, "Cannot audit missing file.")
        return None

    columns = ["secid", "date", "days", "delta", "impl_volatility", "cp_flag"]
    vol = pd.read_parquet(path, columns=columns)
    vol = vol.assign(
        date=pd.to_datetime(vol["date"]),
        days=pd.to_numeric(vol["days"], errors="coerce"),
        delta=pd.to_numeric(vol["delta"], errors="coerce"),
        impl_volatility=pd.to_numeric(vol["impl_volatility"], errors="coerce"),
        cp_flag=vol["cp_flag"].astype("string").str.upper(),
    )

    days_values = sorted(vol["days"].dropna().unique().tolist())
    add_check(
        records,
        "Vol surface",
        "days only 30",
        "PASS" if days_values == [30] or days_values == [30.0] else "FAIL",
        days_values,
        "days should only contain 30.",
    )

    delta_values = sorted(vol["delta"].dropna().unique().tolist())
    add_check(
        records,
        "Vol surface",
        "delta values",
        "PASS" if delta_values == [-50, -25, 50] or delta_values == [-50.0, -25.0, 50.0] else "FAIL",
        delta_values,
        "Expected deltas are -50, -25, and 50.",
    )

    cp_values = sorted(vol["cp_flag"].dropna().unique().tolist())
    add_check(
        records,
        "Vol surface",
        "cp_flag values",
        "PASS" if cp_values == ["C", "P"] else "FAIL",
        cp_values,
        "Expected cp_flag values are C and P.",
    )

    invalid_call = ((vol["delta"] == 50) & (vol["cp_flag"] != "C")).sum()
    add_check(
        records,
        "Vol surface",
        "delta 50 maps to C",
        "PASS" if invalid_call == 0 else "FAIL",
        invalid_call,
        "Rows with delta 50 should be calls.",
    )

    invalid_put = (vol["delta"].isin([-50, -25]) & (vol["cp_flag"] != "P")).sum()
    add_check(
        records,
        "Vol surface",
        "negative deltas map to P",
        "PASS" if invalid_put == 0 else "FAIL",
        invalid_put,
        "Rows with delta -50 or -25 should be puts.",
    )

    missing_iv = vol["impl_volatility"].isna().sum()
    add_check(
        records,
        "Vol surface",
        "impl_volatility nonmissing",
        "PASS" if missing_iv == 0 else "FAIL",
        missing_iv,
        "impl_volatility should be non-missing.",
    )

    outside_iv = (~vol["impl_volatility"].between(0, 5)).sum()
    add_check(
        records,
        "Vol surface",
        "impl_volatility between 0 and 5",
        "PASS" if outside_iv == 0 else "FAIL",
        outside_iv,
        "impl_volatility should be between 0 and 5.",
    )

    duplicate_rows = vol.duplicated(subset=["secid", "date", "delta", "cp_flag"]).sum()
    add_check(
        records,
        "Vol surface",
        "duplicate secid-date-delta-cp_flag rows",
        "PASS" if duplicate_rows == 0 else "FAIL",
        duplicate_rows,
        "No duplicate surface grid rows expected.",
    )

    date_min = vol["date"].min()
    date_max = vol["date"].max()
    expected_start = pd.Timestamp("2018-01-02")
    expected_end = pd.Timestamp("2023-12-29")
    date_status = "PASS" if date_min == expected_start and date_max == expected_end else "WARN"
    add_check(
        records,
        "Vol surface",
        "date range",
        date_status,
        f"{date_min.date()} to {date_max.date()}",
        "Expected range is close to 2018-01-02 to 2023-12-29.",
    )

    add_check(
        records,
        "Vol surface",
        "unique secids",
        "PASS",
        vol["secid"].nunique(),
        "Informational count.",
    )
    secid_date_pairs = vol[["secid", "date"]].drop_duplicates()
    add_check(
        records,
        "Vol surface",
        "unique secid-date pairs",
        "PASS",
        len(secid_date_pairs),
        "Informational count.",
    )

    print("Vol surface audit complete.")
    return vol[["secid", "date"]].drop_duplicates().reset_index(drop=True)


def audit_official_link(processed_data_dir, vol_secid_dates, records):
    """Audit the official WRDS CRSP-OptionMetrics link table."""
    print_section("3. Official Link Table Checks")
    path = processed_data_dir / "secid_permno_bridge_wrdsapps.parquet"
    if not path.exists():
        add_check(records, "Official link", "file available", "FAIL", path, "Cannot audit missing file.")
        return None

    link = pd.read_parquet(path)
    required = ["secid", "sdate", "edate", "permno", "score"]
    check_required_columns(link, required, "Official link", records)
    if not set(required).issubset(link.columns):
        return link

    link = link.assign(
        secid=pd.to_numeric(link["secid"], errors="coerce").astype("Int64"),
        permno=pd.to_numeric(link["permno"], errors="coerce").astype("Int64"),
        sdate=pd.to_datetime(link["sdate"]),
        edate=pd.to_datetime(link["edate"]),
    )

    bad_date_order = (link["sdate"] > link["edate"]).sum()
    add_check(records, "Official link", "sdate <= edate", "PASS" if bad_date_order == 0 else "FAIL", bad_date_order)

    missing_required = link[["secid", "permno", "sdate", "edate"]].isna().sum().sum()
    add_check(
        records,
        "Official link",
        "no missing key fields",
        "PASS" if missing_required == 0 else "FAIL",
        missing_required,
    )

    score_distribution = link["score"].value_counts(dropna=False).sort_index().to_dict()
    add_check(records, "Official link", "score distribution", "PASS", score_distribution, "Informational.")

    duplicate_rows = link.duplicated().sum()
    add_check(records, "Official link", "duplicate rows", "PASS" if duplicate_rows == 0 else "WARN", duplicate_rows)

    add_check(
        records,
        "Official link",
        "link date ranges",
        "PASS",
        f"sdate {link['sdate'].min().date()} to {link['sdate'].max().date()}, "
        f"edate {link['edate'].min().date()} to {link['edate'].max().date()}",
    )

    if vol_secid_dates is not None and not vol_secid_dates.empty:
        vol_secids = set(pd.to_numeric(vol_secid_dates["secid"], errors="coerce").dropna().astype(int))
        link_secids = set(link["secid"].dropna().astype(int))
        coverage = len(vol_secids & link_secids) / len(vol_secids) if vol_secids else np.nan
        add_check(
            records,
            "Official link",
            "vol_surface secid coverage",
            "PASS" if coverage >= 0.99 else "WARN",
            coverage,
            "Share of vol surface secids covered by official link table.",
        )

        sample_size = min(100_000, len(vol_secid_dates))
        sample = vol_secid_dates.sample(n=sample_size, random_state=42).copy()
        sample = sample.assign(
            secid=pd.to_numeric(sample["secid"], errors="coerce").astype("Int64"),
            date=pd.to_datetime(sample["date"]),
        )
        sample.loc[:, "_row_id"] = range(len(sample))
        merged = sample.merge(link, on="secid", how="left")
        valid = merged[(merged["sdate"] <= merged["date"]) & (merged["date"] <= merged["edate"])]
        sample_match_rate = valid["_row_id"].nunique() / len(sample) if len(sample) else np.nan
        add_check(
            records,
            "Official link",
            "time-aware sample coverage",
            "PASS" if sample_match_rate >= 0.95 else "WARN",
            sample_match_rate,
            "Sample match rate using sdate <= date <= edate.",
        )

    print("Official link audit complete.")
    return link


def audit_daily_iv_signals(processed_data_dir, records):
    """Audit daily IV signal construction."""
    print_section("4. Daily IV Signal Checks")
    path = processed_data_dir / "daily_iv_signals.parquet"
    if not path.exists():
        add_check(records, "Daily IV signals", "file available", "FAIL", path, "Cannot audit missing file.")
        return None

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
        "score",
    ]
    daily = pd.read_parquet(path, columns=required)
    check_required_columns(daily, required, "Daily IV signals", records)

    missing = daily.isna().sum().sum()
    add_check(records, "Daily IV signals", "no missing values", "PASS" if missing == 0 else "FAIL", missing)

    duplicate_rows = daily.duplicated(subset=["permno", "date"]).sum()
    add_check(
        records,
        "Daily IV signals",
        "no duplicate permno-date rows",
        "PASS" if duplicate_rows == 0 else "FAIL",
        duplicate_rows,
    )

    spread_diff = (daily["iv_spread"] - (daily["iv_atm_call"] - daily["iv_atm_put"])).abs().max()
    skew_diff = (daily["iv_skew"] - (daily["iv_otm_put"] - daily["iv_atm_call"])).abs().max()
    var_diff = (daily["implied_var"] - daily["iv_atm_call"] ** 2).abs().max()
    add_check(
        records,
        "Daily IV signals",
        "iv_spread formula",
        "PASS" if spread_diff <= 1e-8 else "FAIL",
        spread_diff,
        "Maximum absolute difference from iv_atm_call - iv_atm_put.",
    )
    add_check(
        records,
        "Daily IV signals",
        "iv_skew formula",
        "PASS" if skew_diff <= 1e-8 else "FAIL",
        skew_diff,
        "Maximum absolute difference from iv_otm_put - iv_atm_call.",
    )
    add_check(
        records,
        "Daily IV signals",
        "implied_var formula",
        "PASS" if var_diff <= 1e-8 else "FAIL",
        var_diff,
        "Maximum absolute difference from iv_atm_call squared.",
    )

    iv_cols = ["iv_atm_call", "iv_atm_put", "iv_otm_put"]
    out_of_range = (~daily[iv_cols].apply(lambda col: col.between(0, 5))).sum().sum()
    add_check(records, "Daily IV signals", "IV columns between 0 and 5", "PASS" if out_of_range == 0 else "FAIL", out_of_range)

    dates = pd.to_datetime(daily["date"])
    add_check(records, "Daily IV signals", "date range", "PASS", f"{dates.min().date()} to {dates.max().date()}")
    add_check(records, "Daily IV signals", "unique permnos", "PASS", daily["permno"].nunique())
    avg_permnos = daily.groupby("date")["permno"].nunique().mean()
    add_check(records, "Daily IV signals", "average permnos per date", "PASS", avg_permnos)

    print("Daily IV signal audit complete.")
    return daily


def audit_vrp(raw_data_dir, processed_data_dir, records):
    """Audit realized variance and VRP construction."""
    print_section("5. Realized Variance and VRP Checks")
    path = processed_data_dir / "daily_signals_with_vrp.parquet"
    if not path.exists():
        add_check(records, "Daily VRP", "file available", "FAIL", path, "Cannot audit missing file.")
        return None

    required = ["permno", "date", "implied_var", "realized_var", "vrp"]
    optional = ["mktcap", "exchcd", "shrcd"]
    columns = required + [col for col in optional if col in pq.ParquetFile(path).schema.names]
    vrp = pd.read_parquet(path, columns=columns)
    check_required_columns(vrp, required, "Daily VRP", records)

    missing = vrp.isna().sum().sum()
    add_check(records, "Daily VRP", "no missing values", "PASS" if missing == 0 else "FAIL", missing)

    duplicate_rows = vrp.duplicated(subset=["permno", "date"]).sum()
    add_check(records, "Daily VRP", "no duplicate permno-date rows", "PASS" if duplicate_rows == 0 else "FAIL", duplicate_rows)

    negative_rv = (vrp["realized_var"] < 0).sum()
    add_check(records, "Daily VRP", "realized_var nonnegative", "PASS" if negative_rv == 0 else "FAIL", negative_rv)

    vrp_diff = (vrp["vrp"] - (vrp["implied_var"] - vrp["realized_var"])).abs().max()
    add_check(records, "Daily VRP", "vrp formula", "PASS" if vrp_diff <= 1e-8 else "FAIL", vrp_diff)

    dates = pd.to_datetime(vrp["date"])
    add_check(records, "Daily VRP", "date range", "PASS", f"{dates.min().date()} to {dates.max().date()}")
    add_check(records, "Daily VRP", "unique permnos", "PASS", vrp["permno"].nunique())
    avg_permnos = vrp.groupby("date")["permno"].nunique().mean()
    add_check(records, "Daily VRP", "average permnos per date", "PASS", avg_permnos)

    max_diff = recompute_realized_variance_sample(raw_data_dir, vrp)
    status = "PASS" if pd.notna(max_diff) and max_diff <= 1e-10 else "WARN"
    add_check(
        records,
        "Daily VRP",
        "sample realized_var recomputation",
        status,
        max_diff,
        "Recomputed using trailing CRSP returns with date <= signal date.",
    )

    print("Daily VRP audit complete.")
    return vrp


def recompute_realized_variance_sample(raw_data_dir, vrp):
    """Recompute realized variance for a small sample of permno-date rows."""
    crsp_path = raw_data_dir / "crsp_daily_2017_2023.parquet"
    if not crsp_path.exists():
        return np.nan

    rng = np.random.default_rng(42)
    unique_permnos = pd.Series(vrp["permno"].dropna().unique())
    sample_permnos = unique_permnos.sample(n=min(50, len(unique_permnos)), random_state=42)

    candidate_rows = vrp.loc[vrp["permno"].isin(sample_permnos), ["permno", "date", "realized_var"]]
    sample_rows = candidate_rows.sample(n=min(250, len(candidate_rows)), random_state=42)

    crsp = pd.read_parquet(crsp_path, columns=["permno", "date", "ret"])
    crsp = crsp.loc[crsp["permno"].isin(sample_permnos)].copy()
    crsp = crsp.assign(
        permno=pd.to_numeric(crsp["permno"], errors="coerce").astype("Int64"),
        date=pd.to_datetime(crsp["date"]),
        ret=pd.to_numeric(crsp["ret"], errors="coerce"),
    )
    crsp = crsp.dropna(subset=["permno", "date", "ret"]).sort_values(["permno", "date"])
    crsp.loc[:, "ret_clean"] = crsp["ret"].clip(lower=-0.5, upper=0.5)
    crsp.loc[:, "realized_var_recomputed"] = (
        crsp.groupby("permno")["ret_clean"]
        .rolling(window=21, min_periods=15)
        .var()
        .reset_index(level=0, drop=True)
        * 252
    )

    sample_rows = sample_rows.assign(
        permno=pd.to_numeric(sample_rows["permno"], errors="coerce").astype("Int64"),
        date=pd.to_datetime(sample_rows["date"]),
    )
    merged = sample_rows.merge(
        crsp[["permno", "date", "realized_var_recomputed"]],
        on=["permno", "date"],
        how="inner",
    )
    if merged.empty:
        return np.nan
    return (merged["realized_var"] - merged["realized_var_recomputed"]).abs().max()


def audit_monthly_panel(processed_data_dir, records):
    """Audit monthly panel timing and signal transforms."""
    print_section("6. Monthly Panel Timing Checks")
    path = processed_data_dir / "monthly_signal_panel.parquet"
    if not path.exists():
        add_check(records, "Monthly panel timing", "file available", "FAIL", path, "Cannot audit missing file.")
        return None

    monthly = pd.read_parquet(path)
    monthly = monthly.assign(
        signal_month=pd.PeriodIndex(monthly["signal_month"].astype(str), freq="M"),
        return_month=pd.PeriodIndex(monthly["return_month"].astype(str), freq="M"),
        signal_date=pd.to_datetime(monthly["signal_date"]),
        date_return=pd.to_datetime(monthly["date_return"]),
    )

    missing = monthly.isna().sum().sum()
    add_check(records, "Monthly panel timing", "no missing values", "PASS" if missing == 0 else "FAIL", missing)

    duplicate_signal = monthly.duplicated(subset=["permno", "signal_month"]).sum()
    add_check(records, "Monthly panel timing", "no duplicate permno-signal_month rows", "PASS" if duplicate_signal == 0 else "FAIL", duplicate_signal)

    duplicate_return = monthly.duplicated(subset=["permno", "return_month"]).sum()
    add_check(records, "Monthly panel timing", "no duplicate permno-return_month rows", "PASS" if duplicate_return == 0 else "FAIL", duplicate_return)

    signal_month_summary = period_range_summary(monthly["signal_month"])
    return_month_summary = period_range_summary(monthly["return_month"])
    add_check(
        records,
        "Monthly panel timing",
        "signal_month range is contiguous",
        "PASS"
        if signal_month_summary["observed_months"]
        == signal_month_summary["expected_contiguous_months"]
        else "WARN",
        (
            f"{signal_month_summary['first_month']} to {signal_month_summary['last_month']}; "
            f"observed={signal_month_summary['observed_months']}; "
            f"expected_contiguous={signal_month_summary['expected_contiguous_months']}"
        ),
    )
    add_check(
        records,
        "Monthly panel timing",
        "return_month range is contiguous",
        "PASS"
        if return_month_summary["observed_months"]
        == return_month_summary["expected_contiguous_months"]
        else "WARN",
        (
            f"{return_month_summary['first_month']} to {return_month_summary['last_month']}; "
            f"observed={return_month_summary['observed_months']}; "
            f"expected_contiguous={return_month_summary['expected_contiguous_months']}"
        ),
    )

    month_shift_bad = (monthly["return_month"] != monthly["signal_month"] + 1).sum()
    add_check(records, "Monthly panel timing", "return_month = signal_month + 1", "PASS" if month_shift_bad == 0 else "FAIL", month_shift_bad)

    bad_signal_date = (monthly["signal_date"].dt.to_period("M") != monthly["signal_month"]).sum()
    bad_return_date = (monthly["date_return"].dt.to_period("M") != monthly["return_month"]).sum()
    add_check(records, "Monthly panel timing", "signal_date belongs to signal_month", "PASS" if bad_signal_date == 0 else "FAIL", bad_signal_date)
    add_check(records, "Monthly panel timing", "date_return belongs to return_month", "PASS" if bad_return_date == 0 else "FAIL", bad_return_date)

    bad_order = (monthly["signal_date"] >= monthly["date_return"]).sum()
    add_check(records, "Monthly panel timing", "signal_date < date_return", "PASS" if bad_order == 0 else "FAIL", bad_order)

    add_check(records, "Monthly panel timing", "ret_fwd_1m nonmissing", "PASS" if monthly["ret_fwd_1m"].isna().sum() == 0 else "FAIL", monthly["ret_fwd_1m"].isna().sum())

    signal_cols = ["iv_spread_adj", "iv_skew_adj", "vrp_adj", "composite_signal"]
    check_required_columns(monthly, signal_cols, "Monthly panel timing", records)

    spread_adj_diff = (monthly["iv_spread_adj"] - monthly["iv_spread"]).abs().max()
    skew_adj_diff = (monthly["iv_skew_adj"] + monthly["iv_skew"]).abs().max()
    vrp_adj_diff = (monthly["vrp_adj"] + monthly["vrp"]).abs().max()
    composite_diff = (
        monthly["composite_signal"] - (monthly["iv_spread_z"] - monthly["iv_skew_z"] - monthly["vrp_z"]) / 3
    ).abs().max()
    add_check(records, "Monthly panel timing", "iv_spread_adj formula", "PASS" if spread_adj_diff <= 1e-8 else "FAIL", spread_adj_diff)
    add_check(records, "Monthly panel timing", "iv_skew_adj formula", "PASS" if skew_adj_diff <= 1e-8 else "FAIL", skew_adj_diff)
    add_check(records, "Monthly panel timing", "vrp_adj formula", "PASS" if vrp_adj_diff <= 1e-8 else "FAIL", vrp_adj_diff)
    add_check(records, "Monthly panel timing", "composite_signal formula", "PASS" if composite_diff <= 1e-8 else "FAIL", composite_diff)

    print("Monthly panel audit complete.")
    return monthly


def audit_portfolio_outputs(tables_dir, records):
    """Audit baseline and robustness quintile outputs."""
    print_section("7. Portfolio Sort Output Checks")
    quintile_path = tables_dir / "quintile_summary.csv"
    robustness_path = tables_dir / "robustness_quintile_summary.csv"
    if not quintile_path.exists() or not robustness_path.exists():
        add_check(records, "Portfolio outputs", "summary files available", "FAIL", "", "Missing portfolio summary file.")
        return

    quintile = pd.read_csv(quintile_path)
    robustness = pd.read_csv(robustness_path)
    expected_signals = {"iv_spread_adj", "iv_skew_adj", "vrp_adj", "composite_signal"}
    expected_weightings = {"vw", "ew"}
    expected_universes = {"all", "mktcap_100m", "mktcap_500m", "mktcap_1b"}
    expected_transforms = {"raw", "rank", "winsor_z"}

    signal_ok = expected_signals.issubset(set(quintile["signal"])) and expected_signals.issubset(set(robustness["signal"]))
    add_check(records, "Portfolio outputs", "expected signals present", "PASS" if signal_ok else "FAIL", expected_signals)

    weighting_ok = expected_weightings.issubset(set(quintile["weighting"])) and expected_weightings.issubset(set(robustness["weighting"]))
    add_check(records, "Portfolio outputs", "expected weightings present", "PASS" if weighting_ok else "FAIL", expected_weightings)

    add_check(records, "Portfolio outputs", "expected robustness universes", "PASS" if expected_universes.issubset(set(robustness["universe"])) else "FAIL", expected_universes)
    add_check(records, "Portfolio outputs", "expected robustness transforms", "PASS" if expected_transforms.issubset(set(robustness["transform"])) else "FAIL", expected_transforms)

    month_diagnostics = []
    missing_month_files = 0
    for _, row in robustness.iterrows():
        inferred = robustness_return_file_months(tables_dir, row)
        observed_n_months = int(row["n_months"])
        if inferred is None:
            missing_month_files += 1
            month_diagnostics.append(False)
            continue

        month_diagnostics.append(
            observed_n_months == inferred["observed_months"]
            and inferred["observed_months"] == inferred["expected_contiguous_months"]
        )

    n_months_bad = int((~pd.Series(month_diagnostics, dtype=bool)).sum())
    add_check(
        records,
        "Portfolio outputs",
        "robustness n_months match return files",
        "PASS" if n_months_bad == 0 else "WARN",
        f"bad_rows={n_months_bad}; missing_return_files={missing_month_files}",
        "Expected months are inferred from each robustness return file, not hardcoded.",
    )

    annualized_diff = (robustness["annualized_ls"] - robustness["mean_monthly_ls"] * 12).abs().max()
    sharpe_diff = (
        robustness["sharpe_ratio"] - robustness["annualized_ls"] / robustness["annualized_volatility"]
    ).abs().replace([np.inf, -np.inf], np.nan).max()
    add_check(records, "Portfolio outputs", "annualized_ls formula", "PASS" if annualized_diff <= 1e-10 else "FAIL", annualized_diff)
    add_check(records, "Portfolio outputs", "sharpe_ratio formula", "PASS" if sharpe_diff <= 1e-10 else "FAIL", sharpe_diff)

    identical_count = 0
    for (_, group) in robustness.groupby(["signal", "universe", "weighting"]):
        rounded = group.set_index("transform")["LS" if "LS" in group.columns else "annualized_ls"]
        if expected_transforms.issubset(set(rounded.index)) and rounded.nunique(dropna=False) == 1:
            identical_count += 1
    add_check(
        records,
        "Portfolio outputs",
        "raw/rank/winsor_z identical note",
        "PASS",
        identical_count,
        "Identical results can happen because quintile sorts depend only on ranks.",
    )

    print("Portfolio output audit complete.")


def audit_factor_regressions(tables_dir, records):
    """Audit factor regression output."""
    print_section("8. Factor Regression Checks")
    path = tables_dir / "factor_regression_summary.csv"
    if not path.exists():
        add_check(records, "Factor regressions", "file available", "FAIL", path, "Cannot audit missing file.")
        return

    summary = pd.read_csv(path)
    expected_portfolios = {
        "IV Spread EW All",
        "IV Spread EW MktCap100M",
        "Composite EW All",
        "Composite EW MktCap100M",
        "VRP VW All",
        "VRP VW MktCap100M",
    }
    expected_models = {"CAPM", "FF3", "FF5", "FF5_MOM"}
    add_check(records, "Factor regressions", "selected portfolios present", "PASS" if expected_portfolios.issubset(set(summary["portfolio"])) else "FAIL", expected_portfolios)
    add_check(records, "Factor regressions", "models present", "PASS" if expected_models.issubset(set(summary["model"])) else "FAIL", expected_models)

    factor_month_checks = []
    timing_checks = []
    for _, row in summary.iterrows():
        first_signal = pd.Period(str(row["first_signal_month"]), freq="M")
        first_return = pd.Period(str(row["first_return_month"]), freq="M")
        last_signal = pd.Period(str(row["last_signal_month"]), freq="M")
        last_return = pd.Period(str(row["last_return_month"]), freq="M")
        expected_months = period_count(first_return, last_return)
        factor_month_checks.append(int(row["n_months"]) == expected_months)
        timing_checks.append((first_return == first_signal + 1) and (last_return == last_signal + 1))

    n_months_bad = int((~pd.Series(factor_month_checks, dtype=bool)).sum())
    add_check(
        records,
        "Factor regressions",
        "n_months match return_month range",
        "PASS" if n_months_bad == 0 else "FAIL",
        n_months_bad,
        "Expected months are inferred from each row's first/last return_month.",
    )

    timing_bad = int((~pd.Series(timing_checks, dtype=bool)).sum())
    add_check(
        records,
        "Factor regressions",
        "factor regression timing",
        "PASS" if timing_bad == 0 else "FAIL",
        timing_bad,
        "Checks return_month = signal_month + 1 at the first and last months.",
    )

    alpha_diff = (summary["alpha_annualized"] - summary["alpha_monthly"] * 12).abs().max()
    add_check(records, "Factor regressions", "alpha annualization formula", "PASS" if alpha_diff <= 1e-10 else "FAIL", alpha_diff)

    ff5_mom = summary.loc[summary["model"] == "FF5_MOM"].copy()
    iv_all_alpha = ff5_mom.loc[ff5_mom["portfolio"] == "IV Spread EW All", "alpha_annualized"].iloc[0]
    iv_100_alpha = ff5_mom.loc[ff5_mom["portfolio"] == "IV Spread EW MktCap100M", "alpha_annualized"].iloc[0]
    add_check(records, "Factor regressions", "IV Spread EW All FF5_MOM alpha positive", "PASS" if iv_all_alpha > 0 else "FAIL", iv_all_alpha)
    add_check(records, "Factor regressions", "IV Spread EW MktCap100M FF5_MOM alpha positive", "PASS" if iv_100_alpha > 0 else "FAIL", iv_100_alpha)

    print("\nFF5_MOM alpha table:")
    print(ff5_mom[["portfolio", "alpha_annualized", "alpha_tstat", "r_squared", "n_months"]].to_string(index=False))
    print("Factor regression audit complete.")


def audit_extreme_values(monthly, records):
    """Audit suspicious values in the monthly signal panel."""
    print_section("9. Suspicious/Extreme Value Checks")
    if monthly is None:
        add_check(records, "Extreme values", "monthly panel available", "FAIL", "", "Cannot audit missing monthly panel.")
        return

    columns = ["ret_fwd_1m", "iv_spread", "iv_skew", "vrp", "composite_signal"]
    for column in columns:
        stats = monthly[column].quantile([0.01, 0.99]).to_dict()
        value = (
            f"min={monthly[column].min():.6g}, p1={stats[0.01]:.6g}, "
            f"p99={stats[0.99]:.6g}, max={monthly[column].max():.6g}"
        )
        add_check(records, "Extreme values", f"{column} distribution", "PASS", value)

    thresholds = {
        "abs(ret_fwd_1m) > 1": monthly["ret_fwd_1m"].abs() > 1,
        "abs(iv_spread) > 2": monthly["iv_spread"].abs() > 2,
        "abs(iv_skew) > 2": monthly["iv_skew"].abs() > 2,
        "abs(vrp) > 10": monthly["vrp"].abs() > 10,
    }
    for check_name, mask in thresholds.items():
        count = int(mask.sum())
        share = count / len(monthly) if len(monthly) else 0
        status = "WARN" if share > 0.001 else "PASS"
        add_check(
            records,
            "Extreme values",
            check_name,
            status,
            count,
            f"Share of monthly panel: {share:.4%}. No rows are modified.",
        )

    print("Extreme value audit complete.")


def save_audit_outputs(records, key_counts, tables_dir):
    """Save final audit CSV outputs."""
    tables_dir = Path(tables_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)

    summary = pd.DataFrame(records)
    summary.loc[:, "value"] = summary["value"].map(format_value)
    warnings = summary.loc[summary["status"].isin(["WARN", "FAIL"])].copy()
    timing = summary.loc[
        summary["section"].isin(["Monthly panel timing", "Factor regressions"])
        | summary["check_name"].str.contains("month|date|timing", case=False, na=False)
    ].copy()

    summary_path = tables_dir / "research_audit_summary.csv"
    warnings_path = tables_dir / "research_audit_warnings.csv"
    key_counts_path = tables_dir / "research_audit_key_counts.csv"
    timing_path = tables_dir / "research_audit_timing_checks.csv"

    summary.to_csv(summary_path, index=False)
    warnings.to_csv(warnings_path, index=False)
    key_counts.to_csv(key_counts_path, index=False)
    timing.to_csv(timing_path, index=False)

    print_section("10. Final Audit Report")
    pass_count = (summary["status"] == "PASS").sum()
    warn_count = (summary["status"] == "WARN").sum()
    fail_count = (summary["status"] == "FAIL").sum()

    print(f"PASS checks: {pass_count}")
    print(f"WARN checks: {warn_count}")
    print(f"FAIL checks: {fail_count}")

    fail_rows = summary.loc[summary["status"] == "FAIL", ["section", "check_name", "message"]]
    warn_rows = summary.loc[summary["status"] == "WARN", ["section", "check_name", "message"]]

    print("\nFAIL checks:")
    print("None" if fail_rows.empty else fail_rows.to_string(index=False))

    print("\nWARN checks:")
    print("None" if warn_rows.empty else warn_rows.to_string(index=False))

    recommendation = "Proceed" if fail_count == 0 else "Review before proceeding"
    print(f"\nFinal recommendation: {recommendation}")

    print(f"\nSaved audit summary: {summary_path}")
    print(f"Saved audit warnings: {warnings_path}")
    print(f"Saved audit key counts: {key_counts_path}")
    print(f"Saved audit timing checks: {timing_path}")

    return summary, warnings


def run_research_audit(project_root, raw_data_dir, processed_data_dir, tables_dir):
    """Run the full read-only research audit."""
    records = []

    key_counts = file_existence_and_shape_checks(project_root, records)
    vol_secid_dates = audit_vol_surface(raw_data_dir, records)
    audit_official_link(processed_data_dir, vol_secid_dates, records)
    audit_daily_iv_signals(processed_data_dir, records)
    audit_vrp(raw_data_dir, processed_data_dir, records)
    monthly = audit_monthly_panel(processed_data_dir, records)
    audit_portfolio_outputs(tables_dir, records)
    audit_factor_regressions(tables_dir, records)
    audit_extreme_values(monthly, records)

    return save_audit_outputs(records, key_counts, tables_dir)
