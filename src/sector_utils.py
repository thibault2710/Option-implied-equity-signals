"""Sector classification helpers for IV spread diagnostics."""

import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


MPLCONFIGDIR = Path(tempfile.gettempdir()) / "options_implied_signals_matplotlib"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick


DESIRED_SECURITY_COLUMNS = [
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

UNIVERSE_CUTOFFS = {
    "all": None,
    "mktcap_100m": 100,
    "mktcap_500m": 500,
    "mktcap_1b": 1000,
}


def sic_to_sector(sic):
    """Map an SIC code to a broad sector label."""
    if pd.isna(sic):
        return "Unknown"

    try:
        text = str(sic).strip()
    except Exception:
        return "Unknown"

    if text == "" or text.lower() in {"nan", "none", "<na>"}:
        return "Unknown"

    try:
        sic_int = int(float(text))
    except Exception:
        return "Unknown"

    if 100 <= sic_int <= 999:
        return "Agriculture / Primary"
    if 1000 <= sic_int <= 1499:
        return "Mining"
    if 1500 <= sic_int <= 1799:
        return "Construction"
    if 2000 <= sic_int <= 3999:
        return "Manufacturing"
    if 4000 <= sic_int <= 4999:
        return "Transportation / Utilities"
    if 5000 <= sic_int <= 5199:
        return "Wholesale"
    if 5200 <= sic_int <= 5999:
        return "Retail"
    if 6000 <= sic_int <= 6799:
        return "Finance / Real Estate"
    if 7000 <= sic_int <= 8999:
        return "Services"
    if 9000 <= sic_int <= 9999:
        return "Public / Other"
    return "Unknown"


def _get_wrds_table_columns(db, schema, table):
    """Return available column names for a WRDS table."""
    query = f"""
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = '{schema}'
          AND table_name = '{table}'
        ORDER BY ordinal_position
    """
    columns = db.raw_sql(query)
    return [column.lower() for column in columns["column_name"].tolist()]


def load_or_pull_security_master_full(raw_data_dir):
    """Load full security master if present, otherwise pull from WRDS."""
    raw_data_dir = Path(raw_data_dir)
    raw_data_dir.mkdir(parents=True, exist_ok=True)
    output_path = raw_data_dir / "security_master_full.parquet"

    if output_path.exists():
        security_master_full = pd.read_parquet(output_path)
        print(f"Loaded existing full security master: {output_path}")
        print(f"Shape: {security_master_full.shape}")
        print(f"Columns: {list(security_master_full.columns)}")
        return security_master_full

    print("Full security master not found locally.")
    print("Pulling lightweight OptionMetrics metadata from optionm.securd.")

    db = None
    try:
        from src.data_pull import connect_wrds

        try:
            db = connect_wrds()
        except EOFError as exc:
            raise RuntimeError(
                "WRDS credentials are required to pull security_master_full.parquet. "
                "Run `python scripts/16_add_sector_data.py` from an interactive terminal "
                "where you can enter WRDS credentials, or configure WRDS credentials "
                "with `.pgpass` before rerunning."
            ) from exc

        available_columns = _get_wrds_table_columns(db, "optionm", "securd")
        selected_columns = [
            column for column in DESIRED_SECURITY_COLUMNS if column.lower() in available_columns
        ]
        if "secid" not in selected_columns:
            raise ValueError("optionm.securd does not expose required column: secid")

        column_sql = ",\n                ".join(selected_columns)
        query = f"""
            SELECT
                {column_sql}
            FROM optionm.securd
        """
        security_master_full = db.raw_sql(query)
        security_master_full.to_parquet(output_path, index=False)

        print(f"Saved full security master: {output_path}")
        print(f"Shape: {security_master_full.shape}")
        print(f"Columns: {list(security_master_full.columns)}")
        if "sic" in security_master_full.columns:
            print(f"Missing SIC count: {security_master_full['sic'].isna().sum():,}")
        else:
            print("Warning: pulled security master does not include SIC.")
        return security_master_full
    finally:
        if db is not None:
            print("Closing WRDS connection.")
            db.close()


def prepare_security_sector_table(security_master_full):
    """Clean security metadata and create a secid-to-sector table."""
    available_columns = [
        column
        for column in ["secid", "cusip", "ticker", "sic", "industry_group"]
        if column in security_master_full.columns
    ]
    if "secid" not in available_columns:
        raise ValueError("security_master_full must contain secid.")

    security = security_master_full[available_columns].copy()
    security.loc[:, "secid"] = pd.to_numeric(security["secid"], errors="coerce").astype("Int64")

    if "sic" in security.columns:
        security.loc[:, "sic_clean"] = pd.to_numeric(security["sic"], errors="coerce")
    else:
        security.loc[:, "sic"] = pd.NA
        security.loc[:, "sic_clean"] = pd.NA

    if "cusip" not in security.columns:
        security.loc[:, "cusip"] = pd.NA
    if "ticker" not in security.columns:
        security.loc[:, "ticker"] = pd.NA
    if "industry_group" not in security.columns:
        security.loc[:, "industry_group"] = pd.NA

    security.loc[:, "sector"] = security["sic_clean"].apply(sic_to_sector)
    security.loc[:, "has_sic"] = security["sic_clean"].notna()
    security = security.sort_values(["secid", "has_sic"], ascending=[True, False])
    security = security.dropna(subset=["secid"]).drop_duplicates(subset=["secid"], keep="first")

    sector_table = security[
        ["secid", "cusip", "ticker", "sic", "sic_clean", "sector", "industry_group"]
    ].reset_index(drop=True)

    print("\nPrepared security sector table.")
    print(f"Shape: {sector_table.shape}")
    print("Sector distribution:")
    print(sector_table["sector"].value_counts(dropna=False).to_string())
    return sector_table


def enrich_monthly_panel_with_sector(monthly_panel, security_sector_table):
    """Add SIC and sector to the existing monthly signal panel."""
    panel = monthly_panel.copy()
    original_shape = panel.shape
    panel.loc[:, "secid"] = pd.to_numeric(panel["secid"], errors="coerce").astype("Int64")

    sector_table = security_sector_table.copy()
    sector_table.loc[:, "secid"] = pd.to_numeric(sector_table["secid"], errors="coerce").astype("Int64")

    enriched = panel.merge(
        sector_table,
        on="secid",
        how="left",
        suffixes=("", "_security"),
    )
    if len(enriched) != len(panel):
        raise ValueError(
            f"Row count changed during sector merge: {len(panel):,} -> {len(enriched):,}"
        )

    enriched.loc[:, "sector"] = enriched["sector"].fillna("Unknown")
    missing_sector = int((enriched["sector"] == "Unknown").sum())
    unknown_pct = missing_sector / len(enriched) if len(enriched) else np.nan

    print("\nEnriched monthly panel with sector.")
    print(f"Original shape: {original_shape}")
    print(f"Enriched shape: {enriched.shape}")
    print(f"Missing/Unknown sector rows: {missing_sector:,}")
    print(f"Unknown sector share: {unknown_pct:.2%}")
    print("Sector distribution:")
    print(enriched["sector"].value_counts(dropna=False).to_string())

    return enriched


def save_sector_mapping_summary(enriched_panel, tables_dir):
    """Save a sector mapping summary table."""
    tables_dir = Path(tables_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)
    output_path = tables_dir / "sector_mapping_summary.csv"

    summary = (
        enriched_panel.groupby("sector", as_index=False)
        .agg(
            n_rows=("permno", "size"),
            unique_permnos=("permno", "nunique"),
            unique_secids=("secid", "nunique"),
            average_mktcap=("mktcap", "mean"),
            average_iv_spread_adj=("iv_spread_adj", "mean"),
            average_ret_fwd_1m=("ret_fwd_1m", "mean"),
        )
        .sort_values("n_rows", ascending=False)
    )
    summary.loc[:, "row_share"] = summary["n_rows"] / len(enriched_panel)

    ordered_columns = [
        "sector",
        "n_rows",
        "unique_permnos",
        "unique_secids",
        "row_share",
        "average_mktcap",
        "average_iv_spread_adj",
        "average_ret_fwd_1m",
    ]
    summary = summary[ordered_columns]
    summary.to_csv(output_path, index=False)
    print(f"Saved sector mapping summary: {output_path}")
    return summary


def _prepare_panel(panel):
    """Normalize panel dtypes needed for sector diagnostics."""
    data = panel.copy()
    data = data.assign(
        permno=pd.to_numeric(data["permno"], errors="coerce").astype("Int64"),
        secid=pd.to_numeric(data["secid"], errors="coerce").astype("Int64"),
        signal_month=pd.PeriodIndex(data["signal_month"].astype(str), freq="M"),
        return_month=pd.PeriodIndex(data["return_month"].astype(str), freq="M"),
        iv_spread_adj=pd.to_numeric(data["iv_spread_adj"], errors="coerce"),
        ret_fwd_1m=pd.to_numeric(data["ret_fwd_1m"], errors="coerce"),
        mktcap=pd.to_numeric(data["mktcap"], errors="coerce"),
    )
    if "sector" not in data.columns:
        data.loc[:, "sector"] = "Unknown"
    data.loc[:, "sector"] = data["sector"].fillna("Unknown")
    return data.dropna(subset=["permno", "signal_month", "iv_spread_adj", "ret_fwd_1m"])


def _apply_universe_filter(panel, universe):
    """Apply market-cap universe filter."""
    if universe not in UNIVERSE_CUTOFFS:
        raise ValueError(f"Unknown universe: {universe}")
    data = panel.copy()
    cutoff = UNIVERSE_CUTOFFS[universe]
    if cutoff is not None:
        data = data.loc[data["mktcap"] >= cutoff].copy()
    return data


def assign_bottom_decile_by_month(
    panel,
    signal_col="iv_spread_adj",
    universe="all",
    industry_neutral=False,
):
    """Assign bottom-decile indicator by month, optionally within sector."""
    data = _prepare_panel(panel)
    data = _apply_universe_filter(data, universe)
    data.loc[:, "bottom_decile"] = False
    data.loc[:, "decile_assigned"] = False

    if industry_neutral:
        group_cols = ["signal_month", "sector"]
    else:
        group_cols = ["signal_month"]

    for _, group in data.groupby(group_cols, sort=True):
        if industry_neutral and len(group) < 30:
            continue
        valid = group.dropna(subset=[signal_col])
        if valid[signal_col].nunique() < 10:
            continue
        try:
            deciles = pd.qcut(valid[signal_col], q=10, labels=False, duplicates="drop")
        except ValueError:
            continue
        if deciles.nunique(dropna=True) < 10:
            continue
        bottom_mask = (deciles + 1) == 1
        data.loc[valid.index, "decile_assigned"] = True
        data.loc[valid.index, "bottom_decile"] = bottom_mask.to_numpy()

    print(
        f"Assigned bottom decile: universe={universe}, "
        f"industry_neutral={industry_neutral}, rows={len(data):,}, "
        f"bottom rows={data['bottom_decile'].sum():,}"
    )
    return data


def compute_sector_exposure(panel_with_bottom):
    """Compute bottom-decile sector over/underrepresentation."""
    data = _prepare_panel(panel_with_bottom)
    if "bottom_decile" not in data.columns:
        raise ValueError("panel_with_bottom must contain bottom_decile.")

    rows = []
    for signal_month, month_data in data.groupby("signal_month", sort=True):
        bottom = month_data.loc[month_data["bottom_decile"]]
        universe_counts = month_data["sector"].value_counts()
        bottom_counts = bottom["sector"].value_counts()
        n_universe = universe_counts.sum()
        n_bottom = bottom_counts.sum()
        sectors = sorted(set(universe_counts.index) | set(bottom_counts.index))

        for sector in sectors:
            sector_universe = month_data.loc[month_data["sector"] == sector]
            sector_bottom = bottom.loc[bottom["sector"] == sector]
            universe_count = int(universe_counts.get(sector, 0))
            bottom_count = int(bottom_counts.get(sector, 0))
            rows.append(
                {
                    "signal_month": signal_month,
                    "sector": sector,
                    "universe_count": universe_count,
                    "bottom_count": bottom_count,
                    "universe_share": universe_count / n_universe if n_universe else np.nan,
                    "bottom_share": bottom_count / n_bottom if n_bottom else np.nan,
                    "share_difference": (
                        bottom_count / n_bottom if n_bottom else np.nan
                    )
                    - (universe_count / n_universe if n_universe else np.nan),
                    "bottom_tail_return": sector_bottom["ret_fwd_1m"].mean(),
                    "universe_return": sector_universe["ret_fwd_1m"].mean(),
                    "mktcap_bottom": sector_bottom["mktcap"].mean(),
                    "iv_spread_bottom": sector_bottom["iv_spread_adj"].mean(),
                }
            )

    monthly = pd.DataFrame(rows)
    exposure = (
        monthly.groupby("sector", as_index=False)
        .agg(
            avg_universe_share=("universe_share", "mean"),
            avg_bottom_share=("bottom_share", "mean"),
            avg_share_difference=("share_difference", "mean"),
            avg_bottom_count=("bottom_count", "mean"),
            avg_universe_count=("universe_count", "mean"),
            avg_bottom_tail_return=("bottom_tail_return", "mean"),
            avg_universe_return=("universe_return", "mean"),
            avg_mktcap_bottom=("mktcap_bottom", "mean"),
            avg_iv_spread_bottom=("iv_spread_bottom", "mean"),
        )
        .sort_values("avg_share_difference", ascending=False)
        .reset_index(drop=True)
    )
    return exposure


def _t_stat(series):
    """Compute t-statistic of a mean."""
    series = pd.to_numeric(series, errors="coerce").dropna()
    if len(series) <= 1:
        return np.nan
    std = series.std()
    if pd.isna(std) or std == 0:
        return np.nan
    return series.mean() / (std / np.sqrt(len(series)))


def _summarize_returns(monthly_returns, strategy_name, universe, industry_neutral):
    """Summarize one monthly return dataframe."""
    returns = pd.to_numeric(monthly_returns["universe_minus_bottom"], errors="coerce").dropna()
    monthly_vol = returns.std()
    annualized_vol = monthly_vol * np.sqrt(12)
    annualized_return = returns.mean() * 12
    return {
        "strategy": strategy_name,
        "universe": universe,
        "industry_neutral": industry_neutral,
        "annualized_return": annualized_return,
        "t_stat": _t_stat(returns),
        "sharpe_ratio": annualized_return / annualized_vol
        if pd.notna(annualized_vol) and annualized_vol != 0
        else np.nan,
        "positive_month_pct": (returns > 0).mean(),
        "n_months": len(returns),
        "avg_n_bottom": monthly_returns["n_bottom"].mean(),
        "avg_n_universe": monthly_returns["n_universe"].mean(),
        "avg_sector_count_used": monthly_returns["sector_count_used"].mean(),
    }


def _compute_universe_minus_bottom(panel_with_bottom, universe, industry_neutral):
    """Compute monthly universe-minus-bottom EW returns."""
    rows = []
    for signal_month, month_data in panel_with_bottom.groupby("signal_month", sort=True):
        bottom = month_data.loc[month_data["bottom_decile"]]
        if bottom.empty:
            continue
        sector_count_used = (
            bottom["sector"].nunique() if industry_neutral and "sector" in bottom.columns else np.nan
        )
        rows.append(
            {
                "signal_month": signal_month,
                "return_month": month_data["return_month"].iloc[0],
                "universe": universe,
                "industry_neutral": industry_neutral,
                "universe_return": month_data["ret_fwd_1m"].mean(),
                "bottom_return": bottom["ret_fwd_1m"].mean(),
                "universe_minus_bottom": month_data["ret_fwd_1m"].mean()
                - bottom["ret_fwd_1m"].mean(),
                "n_universe": len(month_data),
                "n_bottom": len(bottom),
                "sector_count_used": sector_count_used,
            }
        )
    return pd.DataFrame(rows)


def run_sector_neutral_bottom_tail_test(enriched_panel):
    """Compare regular and sector-neutral bottom-decile strategies."""
    summary_rows = []
    monthly_frames = []

    specs = [
        ("regular all EW", "all", False),
        ("sector-neutral all EW", "all", True),
        ("regular $100M+ EW", "mktcap_100m", False),
        ("sector-neutral $100M+ EW", "mktcap_100m", True),
    ]

    for strategy, universe, industry_neutral in specs:
        assigned = assign_bottom_decile_by_month(
            enriched_panel,
            signal_col="iv_spread_adj",
            universe=universe,
            industry_neutral=industry_neutral,
        )
        monthly_returns = _compute_universe_minus_bottom(
            assigned,
            universe=universe,
            industry_neutral=industry_neutral,
        )
        monthly_returns.loc[:, "strategy"] = strategy
        monthly_frames.append(monthly_returns)
        summary_rows.append(
            _summarize_returns(
                monthly_returns,
                strategy_name=strategy,
                universe=universe,
                industry_neutral=industry_neutral,
            )
        )

    summary = pd.DataFrame(summary_rows)
    monthly = pd.concat(monthly_frames, ignore_index=True)
    return summary, monthly


def plot_sector_exposure(sector_exposure, charts_dir):
    """Plot bottom-decile sector overrepresentation."""
    charts_dir = Path(charts_dir)
    charts_dir.mkdir(parents=True, exist_ok=True)
    data = sector_exposure.sort_values("avg_share_difference", ascending=False).copy()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(data["sector"], data["avg_share_difference"])
    ax.set_title("Bottom-Decile Sector Overrepresentation")
    ax.set_xlabel("")
    ax.set_ylabel("Bottom share - universe share")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    output_path = charts_dir / "iv_spread_sector_exposure_enriched.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved chart: {output_path}")


def plot_sector_neutral_comparison(sector_neutral_summary, charts_dir):
    """Plot regular versus sector-neutral bottom-tail results."""
    charts_dir = Path(charts_dir)
    charts_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(sector_neutral_summary["strategy"], sector_neutral_summary["annualized_return"])
    ax.set_title("Does Low IV-Spread Underperformance Survive Sector Neutralization?")
    ax.set_xlabel("")
    ax.set_ylabel("Annualized return")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.tick_params(axis="x", rotation=25)

    y_min, y_max = ax.get_ylim()
    offset = (y_max - y_min) * 0.02
    for patch, t_stat in zip(ax.patches, sector_neutral_summary["t_stat"]):
        height = patch.get_height()
        ax.text(
            patch.get_x() + patch.get_width() / 2,
            height + offset if height >= 0 else height - offset,
            f"t={t_stat:.2f}",
            ha="center",
            va="bottom" if height >= 0 else "top",
            fontsize=8,
        )

    fig.tight_layout()
    output_path = charts_dir / "iv_spread_industry_neutral_comparison.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved chart: {output_path}")
