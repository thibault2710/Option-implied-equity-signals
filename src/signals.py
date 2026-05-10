"""Functions for constructing daily options-implied signals."""

from pathlib import Path

import pandas as pd


IV_COLUMNS = ["iv_atm_call", "iv_atm_put", "iv_otm_put"]
SIGNAL_COLUMNS = IV_COLUMNS + ["iv_spread", "iv_skew", "implied_var"]


def load_signal_inputs(raw_data_dir, processed_data_dir):
    """Load the volatility surface and official WRDS link table."""
    raw_data_dir = Path(raw_data_dir)
    processed_data_dir = Path(processed_data_dir)

    vol_surface_path = raw_data_dir / "vol_surface_2018_2023.parquet"
    link_path = processed_data_dir / "secid_permno_bridge_wrdsapps.parquet"

    vol_surface = pd.read_parquet(
        vol_surface_path,
        columns=["secid", "date", "days", "delta", "impl_volatility", "cp_flag"],
    )
    official_link = pd.read_parquet(
        link_path,
        columns=["secid", "sdate", "edate", "permno", "score"],
    )

    print(f"Loaded volatility surface: {vol_surface.shape}")
    print(f"Loaded official WRDS link table: {official_link.shape}")

    return vol_surface, official_link


def validate_vol_surface(vol_surface):
    """Print a validation report for the volatility surface."""
    print("\n" + "=" * 80)
    print("Volatility Surface Validation")
    print("=" * 80)

    print(f"\nShape: {vol_surface.shape}")
    print("\nColumns:")
    print(list(vol_surface.columns))

    date_series = pd.to_datetime(vol_surface["date"])
    print(f"\nDate range: {date_series.min()} to {date_series.max()}")
    print(f"Unique secids: {vol_surface['secid'].nunique():,}")

    print("\nUnique days values:")
    print(vol_surface["days"].dropna().sort_values().unique())

    print("\nUnique delta values:")
    print(vol_surface["delta"].dropna().sort_values().unique())

    print("\ncp_flag counts:")
    print(vol_surface["cp_flag"].value_counts(dropna=False).sort_index().to_string())

    print(f"\nMissing impl_volatility count: {vol_surface['impl_volatility'].isna().sum():,}")

    print("\nimpl_volatility summary stats:")
    print(vol_surface["impl_volatility"].describe().to_string())

    print("\nRows per delta:")
    print(vol_surface["delta"].value_counts(dropna=False).sort_index().to_string())

    secid_date_pairs = vol_surface[["secid", "date"]].drop_duplicates()
    print(f"\nUnique secid-date pairs: {len(secid_date_pairs):,}")


def pivot_iv_surface(vol_surface):
    """Pivot the selected 30-day IV surface rows to one row per secid-date."""
    print("\n" + "=" * 80)
    print("Pivoting IV Surface")
    print("=" * 80)
    print(f"\nInput shape: {vol_surface.shape}")

    needed_columns = ["secid", "date", "days", "delta", "impl_volatility", "cp_flag"]
    surface = vol_surface[needed_columns].copy()

    surface = surface.assign(
        date=pd.to_datetime(surface["date"]),
        secid=pd.to_numeric(surface["secid"], errors="coerce").astype("Int64"),
        delta=pd.to_numeric(surface["delta"], errors="coerce"),
        days=pd.to_numeric(surface["days"], errors="coerce"),
        cp_flag=surface["cp_flag"].astype("string").str.strip().str.upper(),
        impl_volatility=pd.to_numeric(surface["impl_volatility"], errors="coerce"),
    )

    surface = surface.dropna(subset=["secid", "date", "delta", "impl_volatility", "cp_flag"])

    atm_call_mask = (surface["days"] == 30) & (surface["delta"] == 50) & (
        surface["cp_flag"] == "C"
    )
    atm_put_mask = (surface["days"] == 30) & (surface["delta"] == -50) & (
        surface["cp_flag"] == "P"
    )
    otm_put_mask = (surface["days"] == 30) & (surface["delta"] == -25) & (
        surface["cp_flag"] == "P"
    )

    surface = surface.loc[atm_call_mask | atm_put_mask | otm_put_mask].copy()
    print(f"Shape after keeping required 30-day delta/cp_flag rows: {surface.shape}")

    surface.loc[:, "iv_label"] = pd.NA
    surface.loc[atm_call_mask.loc[surface.index], "iv_label"] = "iv_atm_call"
    surface.loc[atm_put_mask.loc[surface.index], "iv_label"] = "iv_atm_put"
    surface.loc[otm_put_mask.loc[surface.index], "iv_label"] = "iv_otm_put"

    duplicate_count = surface.duplicated(subset=["secid", "date", "iv_label"]).sum()
    print(f"Duplicate secid-date-label rows before averaging: {duplicate_count:,}")

    grouped = (
        surface.groupby(["secid", "date", "iv_label"], observed=True)["impl_volatility"]
        .mean()
        .reset_index()
    )

    iv_wide = grouped.pivot(
        index=["secid", "date"],
        columns="iv_label",
        values="impl_volatility",
    ).reset_index()
    iv_wide.columns.name = None

    for column in IV_COLUMNS:
        if column not in iv_wide.columns:
            iv_wide.loc[:, column] = pd.NA

    before_complete_filter = len(iv_wide)
    iv_wide = iv_wide.dropna(subset=IV_COLUMNS)
    iv_wide = iv_wide.assign(
        secid=pd.to_numeric(iv_wide["secid"], errors="coerce").astype("Int64")
    )
    print(f"Wide rows before requiring all three IVs: {before_complete_filter:,}")
    print(f"Wide rows after requiring all three IVs: {len(iv_wide):,}")

    iv_wide = iv_wide[["secid", "date"] + IV_COLUMNS].reset_index(drop=True)
    print(f"Output shape: {iv_wide.shape}")

    return iv_wide


def compute_daily_iv_signals(iv_wide):
    """Compute daily IV spread, skew, and implied variance."""
    daily_signals = iv_wide.copy()

    daily_signals.loc[:, "iv_spread"] = (
        daily_signals["iv_atm_call"] - daily_signals["iv_atm_put"]
    )
    daily_signals.loc[:, "iv_skew"] = (
        daily_signals["iv_otm_put"] - daily_signals["iv_atm_call"]
    )
    daily_signals.loc[:, "implied_var"] = daily_signals["iv_atm_call"] ** 2

    print("\nComputed daily IV signals.")
    print(f"Daily signal shape before linking: {daily_signals.shape}")

    return daily_signals


def link_signals_to_permno_time_aware(signals_df, link_df):
    """
    Deprecated wrapper kept only for backward compatibility.

    Use src.linking.link_signals_to_permno_time_aware for the canonical
    row-id-based time-aware CRSP-OptionMetrics link implementation.
    """
    from src.linking import link_signals_to_permno_time_aware as _preferred_linker

    return _preferred_linker(signals_df, link_df)


def deduplicate_permno_date_signals(daily_signals):
    """Collapse duplicate permno-date rows after time-aware linking."""
    print("\n" + "=" * 80)
    print("Deduplicating PERMNO-Date Signals")
    print("=" * 80)

    required_columns = [
        "secid",
        "permno",
        "date",
        "iv_atm_call",
        "iv_atm_put",
        "iv_otm_put",
        "score",
    ]
    missing_columns = sorted(set(required_columns) - set(daily_signals.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns for deduplication: {missing_columns}")

    possible_columns = [
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
        "realized_var",
        "mktcap",
        "exchcd",
        "shrcd",
        "vrp",
    ]
    input_columns = [column for column in possible_columns if column in daily_signals.columns]

    rows_before = len(daily_signals)
    duplicate_rows_before = daily_signals.duplicated(subset=["permno", "date"]).sum()

    print(f"\nDuplicate permno-date rows before cleaning: {duplicate_rows_before:,}")
    print(f"Rows before cleaning: {rows_before:,}")

    cleaned = daily_signals[input_columns].copy()
    cleaned = cleaned.assign(
        secid=pd.to_numeric(cleaned["secid"], errors="coerce").astype("Int64"),
        permno=pd.to_numeric(cleaned["permno"], errors="coerce").astype("Int64"),
        date=pd.to_datetime(cleaned["date"]),
        iv_atm_call=pd.to_numeric(cleaned["iv_atm_call"], errors="coerce"),
        iv_atm_put=pd.to_numeric(cleaned["iv_atm_put"], errors="coerce"),
        iv_otm_put=pd.to_numeric(cleaned["iv_otm_put"], errors="coerce"),
        score=pd.to_numeric(cleaned["score"], errors="coerce"),
    )

    for optional_numeric in ["realized_var", "mktcap"]:
        if optional_numeric in cleaned.columns:
            cleaned.loc[:, optional_numeric] = pd.to_numeric(
                cleaned[optional_numeric],
                errors="coerce",
            )

    if duplicate_rows_before > 0:
        cleaned = cleaned.sort_values(["permno", "date", "score", "secid"])

        aggregation = {
            "secid": ("secid", "first"),
            "iv_atm_call": ("iv_atm_call", "mean"),
            "iv_atm_put": ("iv_atm_put", "mean"),
            "iv_otm_put": ("iv_otm_put", "mean"),
            "score": ("score", "min"),
        }
        if "realized_var" in cleaned.columns:
            aggregation["realized_var"] = ("realized_var", "mean")
        if "mktcap" in cleaned.columns:
            aggregation["mktcap"] = ("mktcap", "mean")
        if "exchcd" in cleaned.columns:
            aggregation["exchcd"] = ("exchcd", "first")
        if "shrcd" in cleaned.columns:
            aggregation["shrcd"] = ("shrcd", "first")
        if "vrp" in cleaned.columns and "realized_var" not in cleaned.columns:
            aggregation["vrp"] = ("vrp", "mean")

        cleaned = (
            cleaned.groupby(["permno", "date"], as_index=False)
            .agg(**aggregation)
            .reset_index(drop=True)
        )

    cleaned = cleaned.assign(
        secid=pd.to_numeric(cleaned["secid"], errors="coerce").astype("Int64"),
        permno=pd.to_numeric(cleaned["permno"], errors="coerce").astype("Int64"),
    )

    cleaned.loc[:, "iv_spread"] = cleaned["iv_atm_call"] - cleaned["iv_atm_put"]
    cleaned.loc[:, "iv_skew"] = cleaned["iv_otm_put"] - cleaned["iv_atm_call"]
    cleaned.loc[:, "implied_var"] = cleaned["iv_atm_call"] ** 2
    if "realized_var" in cleaned.columns:
        cleaned.loc[:, "vrp"] = cleaned["implied_var"] - cleaned["realized_var"]

    keep_columns = [
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
        "realized_var",
        "mktcap",
        "exchcd",
        "shrcd",
        "vrp",
    ]
    keep_columns = [column for column in keep_columns if column in cleaned.columns]
    cleaned = cleaned[keep_columns].reset_index(drop=True)

    duplicate_rows_after = cleaned.duplicated(subset=["permno", "date"]).sum()
    iv_spread_diff = (cleaned["iv_spread"] - (cleaned["iv_atm_call"] - cleaned["iv_atm_put"])).abs().max()
    iv_skew_diff = (cleaned["iv_skew"] - (cleaned["iv_otm_put"] - cleaned["iv_atm_call"])).abs().max()
    implied_var_diff = (cleaned["implied_var"] - cleaned["iv_atm_call"] ** 2).abs().max()

    print(f"Rows after cleaning: {len(cleaned):,}")
    print(f"Duplicate permno-date rows after cleaning: {duplicate_rows_after:,}")
    print(f"Max formula difference after cleaning, iv_spread: {iv_spread_diff:.12g}")
    print(f"Max formula difference after cleaning, iv_skew: {iv_skew_diff:.12g}")
    print(f"Max formula difference after cleaning, implied_var: {implied_var_diff:.12g}")
    if "vrp" in cleaned.columns and "realized_var" in cleaned.columns:
        vrp_diff = (cleaned["vrp"] - (cleaned["implied_var"] - cleaned["realized_var"])).abs().max()
        print(f"Max formula difference after cleaning, vrp: {vrp_diff:.12g}")

    return cleaned


def validate_daily_signals(daily_signals):
    """Print a validation report for the final daily signal panel."""
    print("\n" + "=" * 80)
    print("Daily IV Signals Validation")
    print("=" * 80)

    print(f"\nShape: {daily_signals.shape}")
    print(f"Date range: {daily_signals['date'].min()} to {daily_signals['date'].max()}")
    print(f"Unique secids: {daily_signals['secid'].nunique():,}")
    print(f"Unique permnos: {daily_signals['permno'].nunique():,}")

    permnos_per_date = daily_signals.groupby("date")["permno"].nunique()
    print("\nNumber of unique permnos per date summary:")
    print(permnos_per_date.describe().to_string())

    print("\nMissing values by column:")
    print(daily_signals.isna().sum().to_string())

    print("\nSignal summary stats:")
    print(daily_signals[SIGNAL_COLUMNS].describe().to_string())

    print("\nSignal correlation matrix:")
    print(daily_signals[SIGNAL_COLUMNS].corr().to_string())

    duplicate_permno_dates = daily_signals.duplicated(subset=["permno", "date"]).sum()
    print(f"\nDuplicate permno-date rows: {duplicate_permno_dates:,}")
    if duplicate_permno_dates > 0:
        print("WARNING: duplicate permno-date rows remain after cleaning.")

    print("\nSample rows:")
    print(daily_signals.head(10).to_string(index=False))


def save_daily_signals(daily_signals, output_path):
    """Save daily IV signals to parquet."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    daily_signals.to_parquet(output_path, index=False)

    print(f"\nSaved daily IV signals to {output_path}")
    print(f"Saved shape: {daily_signals.shape}")


def load_vrp_inputs(raw_data_dir, processed_data_dir):
    """Load CRSP daily returns and the daily IV signal panel."""
    raw_data_dir = Path(raw_data_dir)
    processed_data_dir = Path(processed_data_dir)

    crsp_daily_path = raw_data_dir / "crsp_daily_2017_2023.parquet"
    daily_iv_signals_path = processed_data_dir / "daily_iv_signals.parquet"

    crsp_daily_columns = ["permno", "date", "ret", "prc", "shrout", "exchcd", "shrcd"]
    crsp_daily = pd.read_parquet(crsp_daily_path, columns=crsp_daily_columns)
    daily_iv_signals = pd.read_parquet(daily_iv_signals_path)

    print(f"Loaded CRSP daily: {crsp_daily.shape}")
    print(f"Loaded daily IV signals: {daily_iv_signals.shape}")

    return crsp_daily, daily_iv_signals


def compute_realized_variance(crsp_daily, window=21, min_periods=15):
    """Compute trailing annualized realized variance from CRSP daily returns."""
    print("\n" + "=" * 80)
    print("Computing Realized Variance")
    print("=" * 80)

    needed_columns = ["permno", "date", "ret", "prc", "shrout", "exchcd", "shrcd"]
    available_columns = [column for column in needed_columns if column in crsp_daily.columns]
    crsp = crsp_daily[available_columns].copy()

    crsp = crsp.assign(
        date=pd.to_datetime(crsp["date"]),
        permno=pd.to_numeric(crsp["permno"], errors="coerce").astype("Int64"),
        ret=pd.to_numeric(crsp["ret"], errors="coerce"),
    )
    crsp = crsp.dropna(subset=["permno", "date", "ret"])
    crsp = crsp.sort_values(["permno", "date"]).reset_index(drop=True)

    crsp.loc[:, "ret_clean"] = crsp["ret"].clip(lower=-0.5, upper=0.5)
    crsp.loc[:, "realized_var"] = (
        crsp.groupby("permno")["ret_clean"]
        .rolling(window=window, min_periods=min_periods)
        .var()
        .reset_index(level=0, drop=True)
        * 252
    )

    if {"prc", "shrout"}.issubset(crsp.columns):
        crsp.loc[:, "prc"] = pd.to_numeric(crsp["prc"], errors="coerce")
        crsp.loc[:, "shrout"] = pd.to_numeric(crsp["shrout"], errors="coerce")
        # CRSP shrout is in thousands, so this gives market cap in millions.
        crsp.loc[:, "mktcap"] = crsp["prc"].abs() * crsp["shrout"] / 1000

    keep_columns = ["permno", "date", "realized_var", "mktcap", "exchcd", "shrcd"]
    keep_columns = [column for column in keep_columns if column in crsp.columns]

    realized_var = crsp[keep_columns].dropna(subset=["realized_var"]).reset_index(drop=True)

    print(f"\nRealized variance shape: {realized_var.shape}")
    print(f"Date range: {realized_var['date'].min()} to {realized_var['date'].max()}")
    print(f"Unique permnos: {realized_var['permno'].nunique():,}")
    print("\nrealized_var summary stats:")
    print(realized_var["realized_var"].describe().to_string())

    return realized_var


def merge_vrp_signals(daily_iv_signals, realized_var_df):
    """Merge realized variance with daily IV signals and compute VRP."""
    print("\n" + "=" * 80)
    print("Merging Daily IV Signals with Realized Variance")
    print("=" * 80)

    iv_signals = daily_iv_signals.copy()
    realized_var = realized_var_df.copy()

    iv_signals = iv_signals.assign(
        date=pd.to_datetime(iv_signals["date"]),
        permno=pd.to_numeric(iv_signals["permno"], errors="coerce").astype("Int64"),
    )
    realized_var = realized_var.assign(
        date=pd.to_datetime(realized_var["date"]),
        permno=pd.to_numeric(realized_var["permno"], errors="coerce").astype("Int64"),
    )

    iv_signals = iv_signals.dropna(subset=["permno", "date"])
    realized_var = realized_var.dropna(subset=["permno", "date", "realized_var"])

    merged = iv_signals.merge(realized_var, on=["permno", "date"], how="inner")
    merged.loc[:, "vrp"] = merged["implied_var"] - merged["realized_var"]

    match_rate = len(merged) / len(daily_iv_signals) if len(daily_iv_signals) else 0

    print(f"\nRows in daily_iv_signals: {len(daily_iv_signals):,}")
    print(f"Rows in realized_var_df: {len(realized_var_df):,}")
    print(f"Rows after merge: {len(merged):,}")
    print(f"Merge match rate: {match_rate:.2%}")
    print(f"Date range: {merged['date'].min()} to {merged['date'].max()}")
    print(f"Unique permnos: {merged['permno'].nunique():,}")

    return merged


def validate_vrp_panel(vrp_panel):
    """Print a validation report for the daily VRP panel."""
    print("\n" + "=" * 80)
    print("Daily VRP Panel Validation")
    print("=" * 80)

    print(f"\nShape: {vrp_panel.shape}")
    print(f"Date range: {vrp_panel['date'].min()} to {vrp_panel['date'].max()}")
    print(f"Unique permnos: {vrp_panel['permno'].nunique():,}")

    print("\nMissing values by column:")
    print(vrp_panel.isna().sum().to_string())

    duplicate_permno_dates = vrp_panel.duplicated(subset=["permno", "date"]).sum()
    print(f"\nDuplicate permno-date rows: {duplicate_permno_dates:,}")

    summary_columns = ["implied_var", "realized_var", "vrp", "iv_spread", "iv_skew"]
    print("\nVRP signal summary stats:")
    print(vrp_panel[summary_columns].describe().to_string())

    corr_columns = ["iv_spread", "iv_skew", "implied_var", "realized_var", "vrp"]
    print("\nVRP signal correlation matrix:")
    print(vrp_panel[corr_columns].corr().to_string())

    permnos_per_date = vrp_panel.groupby("date")["permno"].nunique()
    print("\nDistribution of number of permnos per date:")
    print(permnos_per_date.describe().to_string())

    print("\nSample rows:")
    print(vrp_panel.head(10).to_string(index=False))


def save_vrp_panel(vrp_panel, output_path):
    """Save the daily VRP panel to parquet."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    vrp_panel.to_parquet(output_path, index=False)

    print(f"\nSaved daily VRP panel to {output_path}")
    print(f"Saved shape: {vrp_panel.shape}")
