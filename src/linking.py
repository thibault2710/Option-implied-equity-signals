"""Functions for linking OptionMetrics securities to CRSP identifiers."""

from pathlib import Path

import pandas as pd


def clean_cusip(series, length=8):
    """Clean CUSIP-like identifiers for linking."""
    cleaned = series.astype("string").str.strip().str.upper()

    invalid_values = {"", "NAN", "NONE", "<NA>", "00000000"}
    cleaned = cleaned.mask(cleaned.isna() | cleaned.isin(invalid_values))
    cleaned = cleaned.str.slice(0, length)
    cleaned = cleaned.mask(cleaned.isna() | cleaned.isin(invalid_values))
    cleaned = cleaned.mask(cleaned.str.len().fillna(0) < 6)

    return cleaned


def load_raw_linking_inputs(raw_data_dir):
    """Load raw files needed to build the first static CUSIP bridge."""
    raw_data_dir = Path(raw_data_dir)

    security_master_path = raw_data_dir / "security_master.parquet"
    crsp_daily_path = raw_data_dir / "crsp_daily_2017_2023.parquet"
    vol_surface_path = raw_data_dir / "vol_surface_2018_2023.parquet"

    security_master = pd.read_parquet(security_master_path)
    crsp_daily = pd.read_parquet(
        crsp_daily_path,
        columns=["permno", "cusip", "exchcd", "shrcd"],
    )
    vol_surface = pd.read_parquet(vol_surface_path, columns=["secid"])

    print(f"Loaded security_master: {security_master.shape}")
    print(f"Loaded crsp_daily linking columns: {crsp_daily.shape}")
    print(f"Loaded vol_surface secids: {vol_surface.shape}")

    return security_master, crsp_daily, vol_surface


def build_secid_permno_bridge(security_master, crsp_daily):
    """Build a static secid-to-permno bridge through CUSIP8."""
    security = security_master.copy()
    crsp = crsp_daily.copy()

    security.loc[:, "cusip8"] = clean_cusip(security["cusip"])
    crsp.loc[:, "cusip8"] = clean_cusip(crsp["cusip"])

    security = security.dropna(subset=["cusip8"])
    crsp = crsp.dropna(subset=["cusip8"])

    crsp_link = (
        crsp[["permno", "cusip8", "exchcd", "shrcd"]]
        .dropna(subset=["permno"])
        .drop_duplicates()
    )

    bridge = security.merge(crsp_link, on="cusip8", how="inner")

    desired_columns = ["secid", "permno", "cusip8", "ticker", "issue_type", "exchcd", "shrcd"]
    existing_columns = [column for column in desired_columns if column in bridge.columns]

    bridge = (
        bridge[existing_columns]
        .drop_duplicates()
        .sort_values(["secid", "permno"])
        .reset_index(drop=True)
    )

    return bridge


def validate_bridge(bridge, security_master=None, crsp_daily=None, vol_surface=None):
    """Print a validation report for the static secid-permno bridge."""
    print("\n" + "=" * 80)
    print("SECID-PERMNO Bridge Validation")
    print("=" * 80)
    print("Note: this is a static CUSIP bridge because optionm.securd does not")
    print("include effect_date or expir_date in this WRDS subscription/version.")

    print(f"\nBridge rows: {len(bridge):,}")
    print(f"Unique bridge secids: {bridge['secid'].nunique():,}")
    print(f"Unique bridge permnos: {bridge['permno'].nunique():,}")
    print(f"Unique bridge cusip8s: {bridge['cusip8'].nunique():,}")

    if security_master is not None and "secid" in security_master.columns:
        print(f"Unique secids in security_master: {security_master['secid'].nunique():,}")

    if vol_surface is not None and "secid" in vol_surface.columns:
        vol_secids = set(vol_surface["secid"].dropna().unique())
        bridge_secids = set(bridge["secid"].dropna().unique())
        matched_vol_secids = len(vol_secids & bridge_secids)
        total_vol_secids = len(vol_secids)
        match_rate = matched_vol_secids / total_vol_secids if total_vol_secids else 0

        print(f"Unique secids in vol_surface: {total_vol_secids:,}")
        print(f"Unique vol_surface secids matched in bridge: {matched_vol_secids:,}")
        print(f"Vol_surface secid match rate: {match_rate:.2%}")

    if crsp_daily is not None and "permno" in crsp_daily.columns:
        print(f"Unique CRSP permnos in crsp_daily: {crsp_daily['permno'].nunique():,}")

    duplicate_secid_permno = bridge.duplicated(subset=["secid", "permno"]).sum()
    print(f"Duplicate secid-permno rows: {duplicate_secid_permno:,}")

    permnos_per_secid = bridge.groupby("secid")["permno"].nunique()
    secids_per_permno = bridge.groupby("permno")["secid"].nunique()

    print("\nDistribution of number of permnos per secid:")
    print(permnos_per_secid.value_counts().sort_index().to_string())

    print("\nDistribution of number of secids per permno:")
    print(secids_per_permno.value_counts().sort_index().to_string())

    if "ticker" in bridge.columns:
        ticker_links = (
            bridge.groupby("ticker")["permno"]
            .nunique()
            .sort_values(ascending=False)
            .head(10)
        )
        print("\nTop 10 tickers with the most permno links:")
        print(ticker_links.to_string())


def save_bridge(bridge, output_path):
    """Save the secid-permno bridge to parquet."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bridge.to_parquet(output_path, index=False)
    print(f"\nSaved bridge to {output_path}")
    print(f"Bridge shape: {bridge.shape}")


def pull_wrdsapps_optionm_crsp_link(db):
    """Pull the official WRDS CRSP-OptionMetrics link table."""
    print("\nPulling official WRDS CRSP-OptionMetrics link table...")

    query = """
        SELECT secid, sdate, edate, permno, score
        FROM wrdsapps_link_crsp_optionm.opcrsphist
    """

    link_df = db.raw_sql(query).copy()
    link_df = link_df.dropna(subset=["secid", "permno"])

    link_df.loc[:, "secid"] = pd.to_numeric(link_df["secid"], errors="coerce").astype("Int64")
    link_df.loc[:, "permno"] = pd.to_numeric(link_df["permno"], errors="coerce").astype("Int64")
    link_df = link_df.dropna(subset=["secid", "permno"])

    link_df.loc[:, "sdate"] = pd.to_datetime(link_df["sdate"])
    link_df.loc[:, "edate"] = pd.to_datetime(link_df["edate"])

    link_df = (
        link_df[["secid", "sdate", "edate", "permno", "score"]]
        .sort_values(["secid", "sdate", "edate", "score"])
        .reset_index(drop=True)
    )

    print(f"Pulled official link table: {link_df.shape}")
    return link_df


def validate_wrdsapps_link(link_df, vol_surface=None, crsp_daily=None):
    """Print a validation report for the official WRDS link table."""
    print("\n" + "=" * 80)
    print("Official WRDS CRSP-OptionMetrics Link Validation")
    print("=" * 80)

    print(f"\nRows: {len(link_df):,}")
    print(f"Unique secids: {link_df['secid'].nunique():,}")
    print(f"Unique permnos: {link_df['permno'].nunique():,}")
    print(f"sdate range: {link_df['sdate'].min()} to {link_df['sdate'].max()}")
    print(f"edate range: {link_df['edate'].min()} to {link_df['edate'].max()}")

    print("\nScore distribution:")
    print(link_df["score"].value_counts(dropna=False).sort_index().to_string())

    print("\nMissing value counts:")
    print(link_df.isna().sum().to_string())

    duplicate_rows = link_df.duplicated().sum()
    print(f"\nDuplicate rows: {duplicate_rows:,}")

    if vol_surface is not None and {"secid", "date"}.issubset(vol_surface.columns):
        vol = vol_surface[["secid", "date"]].copy()
        vol.loc[:, "secid"] = pd.to_numeric(vol["secid"], errors="coerce").astype("Int64")
        vol.loc[:, "date"] = pd.to_datetime(vol["date"])
        vol = vol.dropna(subset=["secid", "date"])

        vol_secids = set(vol["secid"].dropna().unique())
        link_secids = set(link_df["secid"].dropna().unique())
        matched_secids = len(vol_secids & link_secids)
        match_rate = matched_secids / len(vol_secids) if vol_secids else 0

        print(f"\nUnique vol_surface secids: {len(vol_secids):,}")
        print(f"Vol_surface secids matched by official link: {matched_secids:,}")
        print(f"Vol_surface secid match rate: {match_rate:.2%}")

        sample_size = min(100_000, len(vol))
        sample = vol.sample(n=sample_size, random_state=42) if len(vol) > sample_size else vol
        sample = sample.drop_duplicates(["secid", "date"]).reset_index(drop=True)
        sample.loc[:, "_sample_row_id"] = sample.index

        time_test = sample.merge(link_df, on="secid", how="left")
        valid_time_test = time_test[
            (time_test["sdate"] <= time_test["date"]) & (time_test["date"] <= time_test["edate"])
        ]

        matched_rows = valid_time_test["_sample_row_id"].nunique()
        time_match_rate = matched_rows / len(sample) if len(sample) else 0

        print("\nTime-aware sample match test:")
        print(f"Sampled secid-date rows: {len(sample):,}")
        print(f"Rows with valid date-range permno: {matched_rows:,}")
        print(f"Time-aware sample match rate: {time_match_rate:.2%}")

    if crsp_daily is not None and "permno" in crsp_daily.columns:
        crsp_permnos = set(pd.to_numeric(crsp_daily["permno"], errors="coerce").dropna().astype(int))
        linked_permnos = set(link_df["permno"].dropna().astype(int))
        matched_permnos = len(linked_permnos & crsp_permnos)
        permno_match_rate = matched_permnos / len(linked_permnos) if linked_permnos else 0

        print(f"\nLinked permnos appearing in CRSP daily: {matched_permnos:,}")
        print(f"Linked permno match rate to CRSP daily: {permno_match_rate:.2%}")


def save_wrdsapps_link(link_df, output_path):
    """Save the official WRDS link table to parquet."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    link_df.to_parquet(output_path, index=False)
    print(f"\nSaved official WRDS link to {output_path}")
    print(f"Official link shape: {link_df.shape}")


def link_signals_to_permno_time_aware(signals_df, link_df):
    """Link future signal rows to CRSP permnos using WRDS date ranges.

    If this becomes slow on very large signal panels, optimize the time-aware
    merge later with chunking or a more specialized interval join.
    """
    required_signal_columns = {"secid", "date"}
    required_link_columns = {"secid", "sdate", "edate", "permno", "score"}

    if not required_signal_columns.issubset(signals_df.columns):
        missing = required_signal_columns - set(signals_df.columns)
        raise ValueError(f"signals_df is missing required columns: {sorted(missing)}")

    if not required_link_columns.issubset(link_df.columns):
        missing = required_link_columns - set(link_df.columns)
        raise ValueError(f"link_df is missing required columns: {sorted(missing)}")

    signals = signals_df.copy()
    links = link_df.copy()

    signals.loc[:, "secid"] = pd.to_numeric(signals["secid"], errors="coerce").astype("Int64")
    signals.loc[:, "date"] = pd.to_datetime(signals["date"])
    links.loc[:, "secid"] = pd.to_numeric(links["secid"], errors="coerce").astype("Int64")
    links.loc[:, "sdate"] = pd.to_datetime(links["sdate"])
    links.loc[:, "edate"] = pd.to_datetime(links["edate"])

    signals = signals.reset_index(drop=True)
    signals.loc[:, "_signal_row_id"] = signals.index

    original_rows = len(signals)
    merged = signals.merge(links, on="secid", how="left")
    linked = merged[(merged["sdate"] <= merged["date"]) & (merged["date"] <= merged["edate"])]

    linked = (
        linked.sort_values(["_signal_row_id", "score"])
        .drop_duplicates(subset=["_signal_row_id"], keep="first")
        .drop(columns=["_signal_row_id"])
        .reset_index(drop=True)
    )

    match_rate = len(linked) / original_rows if original_rows else 0
    print(f"Signals before linking: {original_rows:,}")
    print(f"Signals after time-aware linking: {len(linked):,}")
    print(f"Time-aware linking match rate: {match_rate:.2%}")

    return linked
