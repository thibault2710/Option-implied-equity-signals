"""Functions for building the monthly research panel."""

from pathlib import Path

import pandas as pd


DAILY_SIGNAL_COLUMNS = [
    "secid",
    "permno",
    "date",
    "iv_atm_call",
    "iv_atm_put",
    "iv_otm_put",
    "iv_spread",
    "iv_skew",
    "implied_var",
    "realized_var",
    "vrp",
    "mktcap",
    "exchcd",
    "shrcd",
    "score",
]

MONTHLY_RETURN_COLUMNS = ["permno", "date", "ret", "retx", "exchcd", "shrcd"]


def load_monthly_panel_inputs(raw_data_dir, processed_data_dir):
    """Load daily VRP signals and CRSP monthly returns."""
    raw_data_dir = Path(raw_data_dir)
    processed_data_dir = Path(processed_data_dir)

    daily_signals_path = processed_data_dir / "daily_signals_with_vrp.parquet"
    crsp_monthly_path = raw_data_dir / "crsp_monthly_2018_2024.parquet"

    daily_signals_with_vrp = pd.read_parquet(daily_signals_path, columns=DAILY_SIGNAL_COLUMNS)
    crsp_monthly = pd.read_parquet(crsp_monthly_path, columns=MONTHLY_RETURN_COLUMNS)

    print(f"Loaded daily signals with VRP: {daily_signals_with_vrp.shape}")
    print(f"Loaded CRSP monthly returns: {crsp_monthly.shape}")

    return daily_signals_with_vrp, crsp_monthly


def aggregate_daily_signals_to_monthly(daily_signals):
    """Use the latest available daily signal in each permno-month."""
    print("\n" + "=" * 80)
    print("Aggregating Daily Signals to Monthly")
    print("=" * 80)

    available_columns = [column for column in DAILY_SIGNAL_COLUMNS if column in daily_signals.columns]
    signals = daily_signals[available_columns].copy()
    input_rows = len(signals)

    signals = signals.assign(
        date=pd.to_datetime(signals["date"]),
        permno=pd.to_numeric(signals["permno"], errors="coerce").astype("Int64"),
    )
    signals = signals.dropna(subset=["permno", "date"])
    signals.loc[:, "signal_month"] = signals["date"].dt.to_period("M")

    signals = signals.sort_values(["permno", "signal_month", "date"])
    monthly_signals = (
        signals.groupby(["permno", "signal_month"], as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )

    monthly_signals.loc[:, "signal_date"] = monthly_signals["date"]
    monthly_signals.loc[:, "return_month"] = monthly_signals["signal_month"] + 1

    print(f"\nInput rows: {input_rows:,}")
    print(f"Output rows: {len(monthly_signals):,}")
    print(f"Date range: {monthly_signals['signal_date'].min()} to {monthly_signals['signal_date'].max()}")
    print(
        "signal_month range: "
        f"{monthly_signals['signal_month'].min()} to {monthly_signals['signal_month'].max()}"
    )
    print(f"Unique permnos: {monthly_signals['permno'].nunique():,}")

    permnos_per_month = monthly_signals.groupby("signal_month")["permno"].nunique()
    print("\nNumber of permnos per signal_month summary:")
    print(permnos_per_month.describe().to_string())

    print("\nSample rows:")
    print(monthly_signals.head(10).to_string(index=False))

    return monthly_signals


def prepare_crsp_monthly_returns(crsp_monthly):
    """Prepare one CRSP monthly return row per permno-return_month."""
    print("\n" + "=" * 80)
    print("Preparing CRSP Monthly Returns")
    print("=" * 80)

    available_columns = [
        column for column in MONTHLY_RETURN_COLUMNS if column in crsp_monthly.columns
    ]
    returns = crsp_monthly[available_columns].copy()

    returns = returns.assign(
        date=pd.to_datetime(returns["date"]),
        permno=pd.to_numeric(returns["permno"], errors="coerce").astype("Int64"),
        ret=pd.to_numeric(returns["ret"], errors="coerce"),
    )
    if "retx" in returns.columns:
        returns.loc[:, "retx"] = pd.to_numeric(returns["retx"], errors="coerce")

    returns.loc[:, "return_month"] = returns["date"].dt.to_period("M")
    missing_ret_before_drop = returns["ret"].isna().sum()
    duplicate_rows_before = returns.duplicated(subset=["permno", "return_month"]).sum()

    returns = returns.dropna(subset=["permno", "return_month", "ret"])
    returns = returns.sort_values(["permno", "return_month", "date"])
    returns = returns.drop_duplicates(subset=["permno", "return_month"], keep="last")
    duplicate_rows_after = returns.duplicated(subset=["permno", "return_month"]).sum()

    rename_columns = {"ret": "ret_fwd_1m"}
    if "retx" in returns.columns:
        rename_columns["retx"] = "retx_fwd_1m"
    returns = returns.rename(columns=rename_columns)

    print(f"\nShape: {returns.shape}")
    print(f"return_month range: {returns['return_month'].min()} to {returns['return_month'].max()}")
    print(f"Unique permnos: {returns['permno'].nunique():,}")
    print(f"Missing ret count before dropping: {missing_ret_before_drop:,}")
    print(f"Duplicate permno-return_month rows before cleaning: {duplicate_rows_before:,}")
    print(f"Duplicate permno-return_month rows after cleaning: {duplicate_rows_after:,}")

    return returns.reset_index(drop=True)


def merge_monthly_signals_with_forward_returns(monthly_signals, monthly_returns):
    """Merge month-t signals with CRSP returns for month t+1."""
    print("\n" + "=" * 80)
    print("Merging Monthly Signals with Forward Returns")
    print("=" * 80)

    signals = monthly_signals.copy()
    returns = monthly_returns.copy()

    signals = signals.assign(
        permno=pd.to_numeric(signals["permno"], errors="coerce").astype("Int64"),
        signal_month=signals["signal_month"].astype("period[M]"),
        return_month=signals["return_month"].astype("period[M]"),
    )
    returns = returns.assign(
        permno=pd.to_numeric(returns["permno"], errors="coerce").astype("Int64"),
        return_month=returns["return_month"].astype("period[M]"),
    )

    monthly_panel = signals.merge(
        returns,
        on=["permno", "return_month"],
        how="inner",
        suffixes=("", "_return"),
    )

    match_rate = len(monthly_panel) / len(monthly_signals) if len(monthly_signals) else 0

    print(f"\nmonthly_signals rows: {len(monthly_signals):,}")
    print(f"monthly_returns rows: {len(monthly_returns):,}")
    print(f"Merged rows: {len(monthly_panel):,}")
    print(f"Merge match rate: {match_rate:.2%}")
    print(
        "signal_month range: "
        f"{monthly_panel['signal_month'].min()} to {monthly_panel['signal_month'].max()}"
    )
    print(
        "return_month range: "
        f"{monthly_panel['return_month'].min()} to {monthly_panel['return_month'].max()}"
    )
    print(f"Unique permnos: {monthly_panel['permno'].nunique():,}")

    return monthly_panel


def add_signal_transforms(monthly_panel):
    """Add sign-adjusted raw signals, z-scores, and a composite signal."""
    print("\n" + "=" * 80)
    print("Adding Signal Transforms")
    print("=" * 80)

    panel = monthly_panel.copy()

    panel.loc[:, "iv_spread_adj"] = panel["iv_spread"]
    panel.loc[:, "iv_skew_adj"] = -panel["iv_skew"]
    panel.loc[:, "vrp_adj"] = -panel["vrp"]

    def cross_sectional_zscore(series):
        std = series.std()
        if pd.isna(std) or std == 0:
            return pd.Series(pd.NA, index=series.index, dtype="Float64")
        return (series - series.mean()) / std

    panel.loc[:, "iv_spread_z"] = panel.groupby("signal_month")["iv_spread"].transform(
        cross_sectional_zscore
    )
    panel.loc[:, "iv_skew_z"] = panel.groupby("signal_month")["iv_skew"].transform(
        cross_sectional_zscore
    )
    panel.loc[:, "vrp_z"] = panel.groupby("signal_month")["vrp"].transform(
        cross_sectional_zscore
    )
    panel.loc[:, "composite_signal"] = (
        panel["iv_spread_z"] - panel["iv_skew_z"] - panel["vrp_z"]
    ) / 3

    transform_columns = ["iv_spread_z", "iv_skew_z", "vrp_z", "composite_signal"]
    print("\nSignal transform summary stats:")
    print(panel[transform_columns].describe().to_string())

    return panel


def validate_monthly_panel(monthly_panel):
    """Print a validation report for the final monthly signal panel."""
    print("\n" + "=" * 80)
    print("Monthly Signal Panel Validation")
    print("=" * 80)

    print(f"\nShape: {monthly_panel.shape}")
    print(
        "signal_month range: "
        f"{monthly_panel['signal_month'].min()} to {monthly_panel['signal_month'].max()}"
    )
    print(
        "return_month range: "
        f"{monthly_panel['return_month'].min()} to {monthly_panel['return_month'].max()}"
    )
    print(f"Unique permnos: {monthly_panel['permno'].nunique():,}")

    print("\nMissing values by column:")
    print(monthly_panel.isna().sum().to_string())

    duplicate_rows = monthly_panel.duplicated(subset=["permno", "signal_month"]).sum()
    print(f"\nDuplicate permno-signal_month rows: {duplicate_rows:,}")

    permnos_per_month = monthly_panel.groupby("signal_month")["permno"].nunique()
    print("\nNumber of permnos per signal_month summary:")
    print(permnos_per_month.describe().to_string())

    summary_columns = [
        "iv_spread",
        "iv_skew",
        "vrp",
        "composite_signal",
        "ret_fwd_1m",
        "mktcap",
    ]
    print("\nMonthly panel summary stats:")
    print(monthly_panel[summary_columns].describe().to_string())

    corr_columns = ["iv_spread_adj", "iv_skew_adj", "vrp_adj", "composite_signal", "ret_fwd_1m"]
    print("\nMonthly signal correlation matrix:")
    print(monthly_panel[corr_columns].corr().to_string())

    print("\nSample rows:")
    print(monthly_panel.head(10).to_string(index=False))


def save_monthly_panel(monthly_panel, output_path):
    """Save the monthly signal panel to parquet."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    monthly_panel.to_parquet(output_path, index=False)

    print(f"\nSaved monthly signal panel to {output_path}")
    print(f"Saved shape: {monthly_panel.shape}")
