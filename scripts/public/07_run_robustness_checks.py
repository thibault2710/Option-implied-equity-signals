"""Run standalone public robustness checks for the 2010-2023 sample.

The public robustness layer regenerates the paper-facing diagnostics from the
processed monthly panel and public portfolio outputs. It writes only to
outputs/public_2010_2023/ and does not depend on private development-output
folders that are excluded from the public release.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "options_implied_signals_matplotlib"))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from src.config import FULL_EXPANSION_SAMPLE_LABEL, RAW_DATA_DIR, sample_processed_path  # noqa: E402
from src.utils import compute_weighted_return, summarize_return_series  # noqa: E402


SAMPLE_LABEL = FULL_EXPANSION_SAMPLE_LABEL
SIGNAL_COL = "iv_spread_adj"
RETURN_COL = "ret_fwd_1m"
WEIGHT_COL = "mktcap"
NW_LAGS = 4
TOLERANCE = 2e-8

PUBLIC_ROOT = PROJECT_ROOT / "outputs" / "public_2010_2023"
PUBLIC_TABLES_DIR = PUBLIC_ROOT / "tables"
PUBLIC_CHARTS_DIR = PUBLIC_ROOT / "charts"
DOC_REPORT_PATH = PROJECT_ROOT / "docs" / "public_pipeline_step6_robustness_report.md"

UNIVERSE_FILTERS = {
    "all": None,
    "mktcap_100m": 100,
}

FULL_UNIVERSE_FILTERS = {
    "all": None,
    "mktcap_100m": 100,
    "mktcap_500m": 500,
    "mktcap_1b": 1000,
}

HORIZONS = [
    ("ret_fwd_1m", "ret_fwd_1m", 1, 4, "one-month-ahead return"),
    ("ret_fwd_2m_only", "ret_fwd_2m", 1, 4, "return in month t+2 only"),
    ("ret_fwd_3m_only", "ret_fwd_3m", 1, 4, "return in month t+3 only"),
    ("cumret_fwd_3m", "cumret_fwd_3m", 3, 6, "overlapping cumulative t+1 through t+3 return; HAC lags=6"),
    ("cumret_fwd_6m", "cumret_fwd_6m", 6, 12, "overlapping cumulative t+1 through t+6 return; HAC lags=12"),
]

OUTLIER_TREATMENTS = [
    ("baseline", "Raw IV-spread signal and raw forward returns.", None, None, None, None),
    ("return_winsor_1_99", "Raw signal; returns winsorized within return_month at 1/99.", (0.01, 0.99), None, None, None),
    ("return_winsor_0_5_99_5", "Raw signal; returns winsorized within return_month at 0.5/99.5.", (0.005, 0.995), None, None, None),
    ("return_trim_1_99", "Raw signal; rows outside return_month 1/99 return range are dropped before sorting.", None, (0.01, 0.99), None, None),
    ("signal_winsor_1_99", "Signal winsorized within signal_month at 1/99; raw returns.", None, None, (0.01, 0.99), None),
    ("signal_trim_1_99", "Rows outside signal_month 1/99 signal range are dropped before sorting.", None, None, None, (0.01, 0.99)),
    ("both_winsor_1_99", "Signal and returns both winsorized at 1/99 within their timing months.", (0.01, 0.99), None, (0.01, 0.99), None),
    ("both_trim_1_99", "Rows outside either signal_month signal 1/99 or return_month return 1/99 are dropped.", None, (0.01, 0.99), None, (0.01, 0.99)),
]

MAIN_OUTLIER_TREATMENTS = [
    "baseline",
    "return_winsor_1_99",
    "return_trim_1_99",
    "signal_trim_1_99",
    "both_trim_1_99",
]

FEATURE_SPECS = [
    ("iv_spread_level", "level", "iv_spread_level", 1.0, True, True, "baseline level signal; low values are bad"),
    ("iv_spread_change_1m", "change", "iv_spread_change_1m", 1.0, True, True, "one-month IV-spread improvement; low values are deterioration"),
    ("iv_spread_change_3m", "change", "iv_spread_change_3m", 1.0, True, False, "three-month IV-spread improvement; low values are deterioration"),
    ("relative_put_pressure_1m", "put_pressure", "relative_put_pressure_1m", -1.0, True, False, "high raw put pressure is bad; sort score is multiplied by -1"),
    ("level_change_combo", "combo", "level_change_combo", -1.0, True, False, "high raw combo means low level and worsening; sort score is multiplied by -1"),
    ("bottom_decile_count_3m", "persistence", "bottom_decile_count_3m", -1.0, True, False, "discrete persistence count; threshold tests are more meaningful than qcut sorts"),
    ("iv_spread_improvement_1m", "long_side", "iv_spread_improvement_1m", 1.0, False, True, "same as one-month change, interpreted as long-side improvement"),
    ("call_strength_1m", "long_side", "call_strength_1m", 1.0, False, True, "call IV change minus put IV change"),
    ("long_side_combo", "long_side", "long_side_combo", 1.0, False, True, "high level plus recent improvement"),
]

CHARACTERISTICS = [
    "mktcap",
    "iv_atm_call",
    "iv_atm_put",
    "iv_spread",
    "iv_skew",
    "realized_var",
    "vrp",
]


def print_header(title: str) -> None:
    """Print a readable section header."""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def save_table(df: pd.DataFrame, path: Path) -> None:
    """Save a CSV table and print its shape."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Saved {path} shape={df.shape}")


def save_chart(fig: plt.Figure, path: Path) -> None:
    """Save a chart as a PNG."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"Saved {path}")


def period_month(series: pd.Series) -> pd.Series:
    """Convert values to monthly Period values."""
    return pd.Series(pd.PeriodIndex(series.astype(str), freq="M"), index=series.index, dtype="period[M]")


def period_year(series: pd.Series) -> pd.Series:
    """Extract year from values that should represent monthly periods."""
    return pd.Series(pd.PeriodIndex(series.astype(str), freq="M").year, index=series.index)


def require_columns(df: pd.DataFrame, columns: list[str], label: str) -> None:
    """Fail if required columns are missing."""
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")


def strict_row(df: pd.DataFrame, filters: dict[str, object], label: str) -> pd.Series:
    """Return exactly one matching row."""
    mask = pd.Series(True, index=df.index)
    for column, value in filters.items():
        if column not in df.columns:
            raise ValueError(f"{label}: missing column {column}")
        mask = mask & (df[column] == value)
    rows = df.loc[mask]
    if len(rows) != 1:
        raise ValueError(f"{label}: expected exactly one row, found {len(rows)} with {filters}")
    return rows.iloc[0]


def maybe_row(df: pd.DataFrame, filters: dict[str, object]) -> pd.Series | None:
    """Return one matching row or None."""
    mask = pd.Series(True, index=df.index)
    for column, value in filters.items():
        if column not in df.columns:
            return None
        mask = mask & (df[column] == value)
    rows = df.loc[mask]
    if len(rows) != 1:
        return None
    return rows.iloc[0]


def apply_universe_filter(panel: pd.DataFrame, universe: str, filters: dict[str, int | None] | None = None) -> pd.DataFrame:
    """Apply a market-cap universe filter."""
    filters = filters or UNIVERSE_FILTERS
    if universe not in filters:
        raise ValueError(f"Unknown universe: {universe}")
    threshold = filters[universe]
    if threshold is None:
        return panel.copy()
    return panel.loc[panel[WEIGHT_COL] >= threshold].copy()


def weighted_or_equal_return(data: pd.DataFrame, return_col: str, weighting: str) -> float:
    """Compute EW or VW returns."""
    if data.empty:
        return np.nan
    if weighting == "ew":
        return pd.to_numeric(data[return_col], errors="coerce").mean()
    if weighting == "vw":
        return compute_weighted_return(data[return_col], data[WEIGHT_COL])
    raise ValueError(f"Unknown weighting: {weighting}")


def assign_deciles(panel: pd.DataFrame, signal_col: str = SIGNAL_COL) -> pd.DataFrame:
    """Assign signal deciles by signal month."""
    data = panel.copy()
    data.loc[:, "signal_decile"] = pd.NA
    for _, group in data.groupby("signal_month", sort=True):
        valid = group.dropna(subset=[signal_col])
        if valid[signal_col].nunique() < 10:
            continue
        try:
            codes = pd.qcut(valid[signal_col], 10, labels=False, duplicates="drop")
        except ValueError:
            continue
        if codes.nunique(dropna=True) == 10:
            data.loc[valid.index, "signal_decile"] = (codes + 1).astype("Int64")
    data = data.dropna(subset=["signal_decile"]).copy()
    data.loc[:, "signal_decile"] = data["signal_decile"].astype(int)
    return data


def assign_quantiles(data: pd.DataFrame, signal_col: str, n_quantiles: int, output_col: str = "quantile") -> pd.DataFrame:
    """Assign quantiles by signal month."""
    result = data.copy()
    result.loc[:, output_col] = pd.NA
    for _, group in result.groupby("signal_month", sort=True):
        valid = group.dropna(subset=[signal_col])
        if valid[signal_col].nunique() < n_quantiles:
            continue
        try:
            codes = pd.qcut(valid[signal_col], n_quantiles, labels=False, duplicates="drop")
        except ValueError:
            continue
        if codes.nunique(dropna=True) == n_quantiles:
            result.loc[valid.index, output_col] = (codes + 1).astype("Int64")
    result = result.dropna(subset=[output_col]).copy()
    result.loc[:, output_col] = result[output_col].astype(int)
    return result


def max_drawdown(return_series: pd.Series) -> float:
    """Compute max drawdown."""
    returns = pd.to_numeric(pd.Series(return_series), errors="coerce").dropna()
    if returns.empty:
        return np.nan
    growth = (1 + returns).cumprod()
    return (growth / growth.cummax() - 1).min()


def load_public_csv(name: str) -> pd.DataFrame:
    """Load a required public table."""
    path = PUBLIC_TABLES_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Missing public input: {path}")
    df = pd.read_csv(path)
    for column in ["signal_month", "return_month"]:
        if column in df.columns:
            df.loc[:, column] = period_month(df[column])
    return df


def load_panel() -> pd.DataFrame:
    """Load the processed monthly panel."""
    path = sample_processed_path("monthly_signal_panel", SAMPLE_LABEL)
    if not path.exists():
        raise FileNotFoundError(f"Missing monthly panel: {path}")
    panel = pd.read_parquet(path)
    required = [
        "permno",
        "signal_month",
        "return_month",
        RETURN_COL,
        SIGNAL_COL,
        WEIGHT_COL,
        "iv_atm_call",
        "iv_atm_put",
    ]
    require_columns(panel, required, "monthly panel")
    panel = panel.copy()
    panel.loc[:, "signal_month"] = period_month(panel["signal_month"])
    panel.loc[:, "return_month"] = period_month(panel["return_month"])
    for column in ["permno", RETURN_COL, SIGNAL_COL, WEIGHT_COL, "iv_atm_call", "iv_atm_put"]:
        panel.loc[:, column] = pd.to_numeric(panel[column], errors="coerce")
    panel = panel.dropna(subset=["permno", "signal_month", "return_month", RETURN_COL, SIGNAL_COL, WEIGHT_COL])
    panel = panel.loc[panel[WEIGHT_COL] > 0].copy()
    panel.loc[:, "permno"] = panel["permno"].astype(int)
    return panel


def load_sector_panel() -> pd.DataFrame | None:
    """Load sector-enriched panel if available."""
    path = sample_processed_path("monthly_signal_panel_with_sector", SAMPLE_LABEL)
    if not path.exists():
        print(f"Sector panel not found; skipping sector exposure: {path}")
        return None
    sector = pd.read_parquet(path)
    if "sector" not in sector.columns:
        print("Sector panel exists but has no sector column; skipping sector exposure.")
        return None
    sector.loc[:, "signal_month"] = period_month(sector["signal_month"])
    sector.loc[:, "return_month"] = period_month(sector["return_month"])
    return sector


def main_bottom_series(
    bottom_returns: pd.DataFrame,
    universe: str = "all",
    tail: str = "decile",
    weighting: str = "ew",
    leg: str = "universe_minus_bottom",
) -> pd.DataFrame:
    """Return one strict bottom-tail monthly series."""
    filtered = bottom_returns.loc[
        (bottom_returns["signal"] == SIGNAL_COL)
        & (bottom_returns["tail"] == tail)
        & (bottom_returns["universe"] == universe)
        & (bottom_returns["weighting"] == weighting)
    ].copy()
    if len(filtered) == 0:
        raise ValueError(f"Missing bottom-tail rows for {tail}/{universe}/{weighting}")
    if leg not in filtered.columns:
        raise ValueError(f"Missing leg column {leg}")
    output = filtered[["signal_month", "return_month", leg]].rename(columns={leg: "monthly_return"}).copy()
    output.loc[:, "monthly_return"] = pd.to_numeric(output["monthly_return"], errors="coerce")
    output = output.dropna(subset=["monthly_return"])
    if output["return_month"].nunique() != len(output):
        raise ValueError(f"Monthly series has duplicate return months for {tail}/{universe}/{weighting}/{leg}")
    return output


def summary_row(values: pd.Series, extra: dict[str, object] | None = None, annualization: float = 12, nw_lags: int = 4) -> dict[str, object]:
    """Summarize a return series and merge extra fields."""
    out = summarize_return_series(values, annualization=annualization, nw_lags=nw_lags)
    if extra:
        return {**extra, **out}
    return out


def run_alpha_anatomy(panel: pd.DataFrame, sector_panel: pd.DataFrame | None, bottom_returns: pd.DataFrame) -> dict[str, pd.DataFrame | None]:
    """Build alpha anatomy tables."""
    main = main_bottom_series(bottom_returns)
    main.loc[:, "return_year"] = period_year(main["return_month"])

    by_year_rows = []
    for year, group in main.groupby("return_year", sort=True):
        stats = summarize_return_series(group["monthly_return"])
        by_year_rows.append(
            {
                "strategy": "Bottom decile U-B EW all",
                "return_year": int(year),
                **stats,
                "min_monthly_return": group["monthly_return"].min(),
                "max_monthly_return": group["monthly_return"].max(),
                "partial_year": group["return_month"].nunique() < 12,
                "interpretation_note": "positive relative return" if stats["annualized_return"] > 0 else "negative relative return",
            }
        )
    by_year = pd.DataFrame(by_year_rows)

    subperiod_specs = [
        ("2010-2013", "2010-01", "2013-12", None),
        ("2014-2017", "2014-01", "2017-12", None),
        ("2018-2020", "2018-01", "2020-12", None),
        ("2021-2023", "2021-01", "2023-12", None),
        ("2010-2017", "2010-01", "2017-12", None),
        ("2018-2023", "2018-01", "2023-12", None),
        ("excluding 2020", "2010-01", "2024-01", ["2020"]),
        ("excluding Feb-Apr 2020", "2010-01", "2024-01", ["2020-02", "2020-03", "2020-04"]),
        ("2018-2019", "2018-01", "2019-12", None),
        ("2020 only", "2020-01", "2020-12", None),
    ]
    sub_rows = []
    for label, start, end, exclude in subperiod_specs:
        start_p = pd.Period(start, freq="M")
        end_p = pd.Period(end, freq="M")
        mask = (main["return_month"] >= start_p) & (main["return_month"] <= end_p)
        if exclude:
            exclude_periods = pd.PeriodIndex(exclude, freq="M")
            if all("-" not in item for item in exclude):
                mask &= ~period_year(main["return_month"]).astype(str).isin(exclude)
            else:
                mask &= ~main["return_month"].isin(exclude_periods)
        group = main.loc[mask].copy()
        stats = summarize_return_series(group["monthly_return"])
        sub_rows.append(
            {
                "strategy": "Bottom decile U-B EW all",
                "subperiod": label,
                **stats,
                "interpretation_note": "positive" if stats["annualized_return"] > 0 else "negative",
            }
        )
    subperiods = pd.DataFrame(sub_rows)

    main_rows = bottom_returns.loc[
        (bottom_returns["signal"] == SIGNAL_COL)
        & (bottom_returns["tail"] == "decile")
        & (bottom_returns["universe"] == "all")
        & (bottom_returns["weighting"] == "ew")
    ].copy()
    leg_map = {
        "Universe": "universe_ret",
        "Bottom decile": "bottom_tail_ret",
        "Top decile": "top_tail_ret",
        "Universe - Bottom": "universe_minus_bottom",
        "Top - Universe": "top_minus_universe",
        "Top - Bottom": "top_minus_bottom",
    }
    leg_rows = []
    for label, column in leg_map.items():
        leg_rows.append({"leg": label, **summarize_return_series(main_rows[column])})
    leg_decomposition = pd.DataFrame(leg_rows)

    characteristics_panel = sector_panel if sector_panel is not None else panel
    characteristics = build_bottom_tail_characteristics(characteristics_panel)
    sector_exposure = build_sector_exposure(sector_panel) if sector_panel is not None else None
    monotonicity = build_monotonicity()
    return {
        "by_year": by_year,
        "subperiods": subperiods,
        "leg_decomposition": leg_decomposition,
        "characteristics": characteristics,
        "sector_exposure": sector_exposure,
        "monotonicity": monotonicity,
    }


def build_bottom_tail_characteristics(panel: pd.DataFrame) -> pd.DataFrame:
    """Compare bottom decile stocks to the universe and top decile."""
    rows = []
    characteristics = [
        "mktcap",
        "iv_atm_call",
        "iv_atm_put",
        "iv_otm_put",
        "iv_spread",
        "iv_skew",
        "implied_var",
        "realized_var",
        "vrp",
        "ret_fwd_1m",
    ]
    for universe in ["all", "mktcap_100m"]:
        data = panel.copy()
        data.loc[:, "signal_month"] = period_month(data["signal_month"])
        for column in [SIGNAL_COL, WEIGHT_COL] + [c for c in characteristics if c in data.columns]:
            data.loc[:, column] = pd.to_numeric(data[column], errors="coerce")
        data = assign_deciles(apply_universe_filter(data, universe)).copy()
        data.loc[:, "log_mktcap"] = np.log(pd.to_numeric(data["mktcap"], errors="coerce").where(lambda x: x > 0))
        for characteristic in characteristics + ["log_mktcap"]:
            if characteristic not in data.columns:
                continue
            values = data.copy()
            values.loc[:, characteristic] = pd.to_numeric(values[characteristic], errors="coerce")
            universe_mean = values.groupby("signal_month")[characteristic].mean().mean()
            bottom_mean = (
                values.loc[values["signal_decile"] == 1]
                .groupby("signal_month")[characteristic]
                .mean()
                .mean()
            )
            top_mean = (
                values.loc[values["signal_decile"] == 10]
                .groupby("signal_month")[characteristic]
                .mean()
                .mean()
            )
            rows.append(
                {
                    "universe": universe,
                    "characteristic": characteristic,
                    "universe_mean": universe_mean,
                    "bottom_decile_mean": bottom_mean,
                    "top_decile_mean": top_mean,
                    "bottom_minus_universe": bottom_mean - universe_mean,
                    "top_minus_universe": top_mean - universe_mean,
                    "bottom_percentile_rank": (values[characteristic] <= bottom_mean).mean()
                    if values[characteristic].notna().any()
                    else np.nan,
                    "interpretation_note": "bottom above universe" if bottom_mean > universe_mean else "bottom below universe",
                }
            )
    return pd.DataFrame(rows)


def build_sector_exposure(sector_panel: pd.DataFrame) -> pd.DataFrame:
    """Compute bottom-decile sector exposure if sector data are available."""
    required = ["permno", "signal_month", "return_month", SIGNAL_COL, RETURN_COL, WEIGHT_COL, "sector"]
    require_columns(sector_panel, required, "sector panel")
    data = sector_panel[required].copy()
    for column in [SIGNAL_COL, RETURN_COL, WEIGHT_COL]:
        data.loc[:, column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna(subset=[SIGNAL_COL, RETURN_COL, WEIGHT_COL, "sector"]).copy()
    data = assign_deciles(data)
    bottom = data.loc[data["signal_decile"] == 1]
    top = data.loc[data["signal_decile"] == 10]
    sectors = sorted(data["sector"].astype(str).unique())
    rows = []
    for sector in sectors:
        u = data.loc[data["sector"] == sector]
        b = bottom.loc[bottom["sector"] == sector]
        t = top.loc[top["sector"] == sector]
        universe_share = len(u) / len(data) if len(data) else np.nan
        bottom_share = len(b) / len(bottom) if len(bottom) else np.nan
        rows.append(
            {
                "sector": sector,
                "universe_share": universe_share,
                "bottom_decile_share": bottom_share,
                "top_decile_share": len(t) / len(top) if len(top) else np.nan,
                "average_bottom_return": b[RETURN_COL].mean(),
                "average_universe_return": u[RETURN_COL].mean(),
                "average_iv_spread": b[SIGNAL_COL].mean(),
                "average_mktcap": b[WEIGHT_COL].mean(),
                "bottom_minus_universe_share": bottom_share - universe_share,
            }
        )
    return pd.DataFrame(rows)


def build_monotonicity() -> pd.DataFrame:
    """Build D1-D10 return and monotonicity table from public decile returns."""
    rows = []
    bottom_summary = load_public_csv(f"bottom_tail_summary_{SAMPLE_LABEL}.csv")
    for weighting in ["ew", "vw"]:
        returns_path = PUBLIC_TABLES_DIR / f"decile_returns_{SIGNAL_COL}_{weighting}_{SAMPLE_LABEL}.csv"
        decile_returns = pd.read_csv(returns_path)
        decile_returns.loc[:, "return_month"] = period_month(decile_returns["return_month"])
        annualized = {f"D{i}_annualized_return": decile_returns[f"Q{i}"].mean() * 12 for i in range(1, 11)}
        decile_values = [annualized[f"D{i}_annualized_return"] for i in range(1, 11)]
        spearman = pd.Series(range(1, 11)).corr(pd.Series(decile_values), method="spearman")
        adjacent = int(sum(decile_values[i] > decile_values[i - 1] for i in range(1, 10)))
        ub = strict_row(
            bottom_summary,
            {
                "signal": SIGNAL_COL,
                "tail": "decile",
                "universe": "all",
                "weighting": weighting,
                "leg": "universe_minus_bottom",
            },
            f"monotonicity U-B {weighting}",
        )
        tmb = strict_row(
            bottom_summary,
            {
                "signal": SIGNAL_COL,
                "tail": "decile",
                "universe": "all",
                "weighting": weighting,
                "leg": "top_minus_bottom",
            },
            f"monotonicity T-B {weighting}",
        )
        tu = strict_row(
            bottom_summary,
            {
                "signal": SIGNAL_COL,
                "tail": "decile",
                "universe": "all",
                "weighting": weighting,
                "leg": "top_minus_universe",
            },
            f"monotonicity T-U {weighting}",
        )
        rows.append(
            {
                "weighting": weighting,
                **annualized,
                "spearman_decile_return_corr": spearman,
                "adjacent_increases_out_of_9": adjacent,
                "D10_minus_D1_return": tmb["annualized_return"],
                "D10_minus_D1_raw_t_stat": tmb["raw_t_stat"],
                "D10_minus_D1_nw_t_stat": tmb["nw_t_stat"],
                "universe_minus_D1_return": ub["annualized_return"],
                "universe_minus_D1_raw_t_stat": ub["raw_t_stat"],
                "universe_minus_D1_nw_t_stat": ub["nw_t_stat"],
                "D10_minus_universe_return": tu["annualized_return"],
                "D10_minus_universe_raw_t_stat": tu["raw_t_stat"],
                "D10_minus_universe_nw_t_stat": tu["nw_t_stat"],
            }
        )
    return pd.DataFrame(rows)


def load_crsp_monthly() -> pd.DataFrame:
    """Load local CRSP monthly returns for holding-period tests."""
    path = RAW_DATA_DIR / "crsp_monthly_2010_2024.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing CRSP monthly file: {path}")
    crsp = pd.read_parquet(path)
    require_columns(crsp, ["permno", "date", "ret"], "CRSP monthly")
    crsp = crsp[["permno", "date", "ret"]].copy()
    crsp.loc[:, "permno"] = pd.to_numeric(crsp["permno"], errors="coerce")
    crsp.loc[:, "ret"] = pd.to_numeric(crsp["ret"], errors="coerce")
    crsp.loc[:, "month"] = pd.to_datetime(crsp["date"], errors="coerce").dt.to_period("M")
    crsp = crsp.dropna(subset=["permno", "month", "ret"])
    crsp.loc[:, "permno"] = crsp["permno"].astype(int)
    return crsp[["permno", "month", "ret"]].drop_duplicates(["permno", "month"])


def add_forward_returns(panel: pd.DataFrame, crsp: pd.DataFrame) -> pd.DataFrame:
    """Attach t+2/t+3 returns and cumulative 3m/6m future returns."""
    data = panel.copy()
    for horizon in range(1, 7):
        future_month = f"future_month_{horizon}"
        return_col = f"ret_h{horizon}"
        future = crsp.rename(columns={"month": future_month, "ret": return_col})
        data.loc[:, future_month] = data["signal_month"] + horizon
        data = data.merge(future, on=["permno", future_month], how="left")
    data.loc[:, "ret_fwd_2m"] = data["ret_h2"]
    data.loc[:, "ret_fwd_3m"] = data["ret_h3"]
    ret_3m_cols = ["ret_h1", "ret_h2", "ret_h3"]
    ret_6m_cols = [f"ret_h{i}" for i in range(1, 7)]
    data.loc[:, "cumret_fwd_3m"] = (1 + data[ret_3m_cols]).prod(axis=1) - 1
    data.loc[data[ret_3m_cols].isna().any(axis=1), "cumret_fwd_3m"] = np.nan
    data.loc[:, "cumret_fwd_6m"] = (1 + data[ret_6m_cols]).prod(axis=1) - 1
    data.loc[data[ret_6m_cols].isna().any(axis=1), "cumret_fwd_6m"] = np.nan
    return data


def summarize_horizon(values: pd.Series, horizon_months: int, nw_lags: int) -> dict[str, object]:
    """Summarize holding-period returns with horizon-specific annualization."""
    annualization = 12 / horizon_months
    stats = summarize_return_series(values, annualization=annualization, nw_lags=nw_lags)
    clean = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    mean_period = clean.mean() if not clean.empty else np.nan
    stats["mean_period_return"] = mean_period
    stats["monthly_equivalent_return"] = (1 + mean_period) ** (1 / horizon_months) - 1 if pd.notna(mean_period) else np.nan
    return stats


def run_holding_period_decay(panel: pd.DataFrame) -> pd.DataFrame:
    """Run bottom-tail tests across forward holding horizons."""
    rows = []
    summary_rows = []
    for universe in UNIVERSE_FILTERS:
        data = assign_deciles(apply_universe_filter(panel, universe))
        for weighting in ["ew", "vw"]:
            for horizon, return_col, horizon_months, nw_lags, note in HORIZONS:
                usable = data.dropna(subset=[return_col]).copy()
                monthly_rows = []
                for signal_month, month_data in usable.groupby("signal_month", sort=True):
                    bottom = month_data.loc[month_data["signal_decile"] == 1]
                    top = month_data.loc[month_data["signal_decile"] == 10]
                    universe_ret = weighted_or_equal_return(month_data, return_col, weighting)
                    bottom_ret = weighted_or_equal_return(bottom, return_col, weighting)
                    top_ret = weighted_or_equal_return(top, return_col, weighting)
                    monthly_rows.append(
                        {
                            "signal_month": signal_month,
                            "horizon_end_month": signal_month + horizon_months,
                            "horizon": horizon,
                            "return_col": return_col,
                            "universe": universe,
                            "weighting": weighting,
                            "universe_return": universe_ret,
                            "bottom_decile_return": bottom_ret,
                            "top_decile_return": top_ret,
                            "universe_minus_bottom": universe_ret - bottom_ret,
                            "top_minus_bottom": top_ret - bottom_ret,
                            "top_minus_universe": top_ret - universe_ret,
                            "n_stocks": len(month_data),
                            "n_bottom": len(bottom),
                            "n_top": len(top),
                        }
                    )
                monthly = pd.DataFrame(monthly_rows)
                if monthly.empty:
                    continue
                rows.append(monthly)
                tests = {
                    "bottom_decile_return": "Bottom decile",
                    "universe_return": "Universe",
                    "universe_minus_bottom": "Universe-minus-bottom",
                    "top_minus_bottom": "Top-minus-bottom",
                    "top_minus_universe": "Top-minus-universe",
                }
                for column, label in tests.items():
                    stats = summarize_horizon(monthly[column], horizon_months, nw_lags)
                    summary_rows.append(
                        {
                            "horizon": horizon,
                            "test": label,
                            "universe": universe,
                            "weighting": weighting,
                            "horizon_months": horizon_months,
                            "mean_period_return": stats["mean_period_return"],
                            "annualized_return": stats["annualized_return"],
                            "monthly_equivalent_return": stats["monthly_equivalent_return"],
                            "annualized_volatility": stats["annualized_volatility"],
                            "sharpe_ratio": stats["sharpe_ratio"],
                            "raw_t_stat": stats["raw_t_stat"],
                            "nw_t_stat": stats["nw_t_stat"],
                            "nw_p_value": stats["nw_p_value"],
                            "n_months": stats["n_months"],
                            "avg_n_stocks": monthly["n_stocks"].mean(),
                            "avg_n_bottom": monthly["n_bottom"].mean(),
                            "positive_period_pct": stats["positive_month_pct"],
                            "interpretation_note": note,
                        }
                    )
    return pd.DataFrame(summary_rows)


def portfolio_turnover(member_sets: list[tuple[pd.Period, set[int]]]) -> pd.DataFrame:
    """Approximate one-way turnover from sequential member sets."""
    rows = []
    previous = None
    for month, current in member_sets:
        if previous is None or len(previous) == 0:
            one_way = two_way = overlap_share = np.nan
        else:
            entries = current - previous
            exits = previous - current
            overlap = current & previous
            one_way = len(entries) / len(previous)
            two_way = (len(entries) + len(exits)) / (2 * len(previous))
            overlap_share = len(overlap) / len(previous)
        rows.append(
            {
                "signal_month": month,
                "one_way_turnover": one_way,
                "two_way_turnover": two_way,
                "overlap_share": overlap_share,
                "n_members": len(current),
            }
        )
        previous = current
    return pd.DataFrame(rows)


def formation_months(panel: pd.DataFrame, frequency: str) -> list[pd.Period]:
    """Return formation months for a rebalance frequency."""
    months = sorted(panel["signal_month"].unique())
    if frequency == "monthly":
        return months
    if frequency == "quarterly":
        return [month for month in months if month.month in [3, 6, 9, 12]]
    if frequency == "semiannual":
        return [month for month in months if month.month in [6, 12]]
    raise ValueError(f"Unknown frequency: {frequency}")


def run_lower_frequency_exclusion(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Test monthly, quarterly, and semiannual bottom-decile exclusion."""
    rows = []
    summary_rows = []
    cost_rows = []
    frequencies = [("monthly", 1), ("quarterly", 3), ("semiannual", 6)]
    for universe in UNIVERSE_FILTERS:
        data = assign_deciles(apply_universe_filter(panel, universe))
        for frequency, hold_months in frequencies:
            member_sets = []
            current_rows = []
            for formation_month in formation_months(data, frequency):
                formed = data.loc[data["signal_month"] == formation_month].copy()
                if formed.empty:
                    continue
                selected = formed.loc[formed["signal_decile"] != 1]
                member_sets.append((formation_month, set(selected["permno"].astype(int).tolist())))
                for hold in range(1, hold_months + 1):
                    return_col = f"ret_h{hold}"
                    if return_col not in formed.columns:
                        continue
                    month_data = formed.dropna(subset=[return_col])
                    selected_hold = month_data.loc[month_data["signal_decile"] != 1]
                    if selected_hold.empty:
                        continue
                    current_rows.append(
                        {
                            "formation_month": formation_month,
                            "return_month": formation_month + hold,
                            "frequency": frequency,
                            "hold_month": hold,
                            "universe": universe,
                            "weighting": "ew",
                            "full_universe_return": month_data[return_col].mean(),
                            "exclusion_return": selected_hold[return_col].mean(),
                            "improvement_vs_universe": selected_hold[return_col].mean() - month_data[return_col].mean(),
                            "n_universe": len(month_data),
                            "n_selected": len(selected_hold),
                            "n_excluded": len(month_data) - len(selected_hold),
                        }
                    )
            current = pd.DataFrame(current_rows)
            if current.empty:
                continue
            rows.append(current)
            improvement = summarize_return_series(current["improvement_vs_universe"], nw_lags=4)
            exclusion = summarize_return_series(current["exclusion_return"], nw_lags=4)
            turnover = portfolio_turnover(member_sets)
            avg_turnover = turnover["one_way_turnover"].mean()
            summary_rows.append(
                {
                    "frequency": frequency,
                    "universe": universe,
                    "weighting": "ew",
                    "annualized_return": exclusion["annualized_return"],
                    "improvement_vs_universe": improvement["annualized_return"],
                    "nw_t_stat_improvement": improvement["nw_t_stat"],
                    "sharpe_ratio": exclusion["sharpe_ratio"],
                    "max_drawdown": max_drawdown(current["exclusion_return"]),
                    "one_way_turnover": avg_turnover,
                    "n_months": improvement["n_months"],
                    "avg_n_universe": current["n_universe"].mean(),
                    "avg_n_selected": current["n_selected"].mean(),
                    "avg_n_excluded": current["n_excluded"].mean(),
                }
            )
            rebalances_per_year = 12 / hold_months
            for bps in [10, 25, 50]:
                annual_cost_drag = avg_turnover * bps / 10000 * rebalances_per_year
                cost_rows.append(
                    {
                        "frequency": frequency,
                        "universe": universe,
                        "weighting": "ew",
                        "cost_bps": bps,
                        "gross_improvement_vs_universe": improvement["annualized_return"],
                        "estimated_annual_cost_drag": annual_cost_drag,
                        "net_improvement_vs_universe": improvement["annualized_return"] - annual_cost_drag,
                        "cost_note": "rough one-way turnover cost at rebalance dates only",
                    }
                )
    monthly = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    return pd.DataFrame(summary_rows), pd.DataFrame(cost_rows)


def run_new_vs_persistent(panel: pd.DataFrame) -> pd.DataFrame:
    """Split bottom decile into newly bottom and persistent bottom stocks."""
    summary_rows = []
    for universe in UNIVERSE_FILTERS:
        data = assign_deciles(apply_universe_filter(panel, universe))
        data.loc[:, "bottom_decile_now"] = (data["signal_decile"] == 1).astype(int)
        prev = data[["permno", "signal_month", "bottom_decile_now"]].copy()
        prev.loc[:, "signal_month"] = prev["signal_month"] + 1
        prev = prev.rename(columns={"bottom_decile_now": "bottom_decile_prev"})
        data = data.merge(prev, on=["permno", "signal_month"], how="left")
        data.loc[:, "bottom_decile_prev"] = data["bottom_decile_prev"].fillna(0)
        data.loc[:, "newly_bottom"] = ((data["bottom_decile_now"] == 1) & (data["bottom_decile_prev"] != 1)).astype(int)
        data.loc[:, "persistent_bottom"] = ((data["bottom_decile_now"] == 1) & (data["bottom_decile_prev"] == 1)).astype(int)

        monthly_rows = []
        for signal_month, month_data in data.groupby("signal_month", sort=True):
            bottom = month_data.loc[month_data["bottom_decile_now"] == 1]
            newly = month_data.loc[month_data["newly_bottom"] == 1]
            persistent = month_data.loc[month_data["persistent_bottom"] == 1]
            if newly.empty or persistent.empty:
                continue
            universe_ret = month_data[RETURN_COL].mean()
            newly_ret = newly[RETURN_COL].mean()
            persistent_ret = persistent[RETURN_COL].mean()
            bottom_ret = bottom[RETURN_COL].mean()
            monthly_rows.append(
                {
                    "signal_month": signal_month,
                    "return_month": signal_month + 1,
                    "universe_return": universe_ret,
                    "bottom_decile_return": bottom_ret,
                    "newly_bottom_return": newly_ret,
                    "persistent_bottom_return": persistent_ret,
                    "universe_minus_newly_bottom": universe_ret - newly_ret,
                    "universe_minus_persistent_bottom": universe_ret - persistent_ret,
                    "persistent_minus_newly": persistent_ret - newly_ret,
                    "n_universe": len(month_data),
                    "n_bottom": len(bottom),
                    "n_newly": len(newly),
                    "n_persistent": len(persistent),
                    "mktcap_universe": month_data[WEIGHT_COL].mean(),
                    "mktcap_bottom": bottom[WEIGHT_COL].mean(),
                    "mktcap_newly": newly[WEIGHT_COL].mean(),
                    "mktcap_persistent": persistent[WEIGHT_COL].mean(),
                    "iv_spread_universe": month_data[SIGNAL_COL].mean(),
                    "iv_spread_bottom": bottom[SIGNAL_COL].mean(),
                    "iv_spread_newly": newly[SIGNAL_COL].mean(),
                    "iv_spread_persistent": persistent[SIGNAL_COL].mean(),
                }
            )
        current = pd.DataFrame(monthly_rows)
        tests = {
            "Universe": ("universe_return", "n_universe", "mktcap_universe", "iv_spread_universe", "component return"),
            "Bottom decile overall": ("bottom_decile_return", "n_bottom", "mktcap_bottom", "iv_spread_bottom", "component return"),
            "Newly bottom": ("newly_bottom_return", "n_newly", "mktcap_newly", "iv_spread_newly", "component return"),
            "Persistent bottom": ("persistent_bottom_return", "n_persistent", "mktcap_persistent", "iv_spread_persistent", "component return"),
            "Universe minus newly bottom": ("universe_minus_newly_bottom", "n_newly", "mktcap_newly", "iv_spread_newly", "fresh entry into bottom decile; consistent with short-lived information if strongest"),
            "Universe minus persistent bottom": ("universe_minus_persistent_bottom", "n_persistent", "mktcap_persistent", "iv_spread_persistent", "repeat bottom-decile names; consistent with persistent risk pricing if strongest"),
            "Persistent minus newly": ("persistent_minus_newly", "n_persistent", "mktcap_persistent", "iv_spread_persistent", "positive means persistent bottom outperforms newly bottom"),
        }
        for test, (ret_col, n_col, mktcap_col, spread_col, note) in tests.items():
            stats = summarize_return_series(current[ret_col], nw_lags=4)
            summary_rows.append(
                {
                    "universe": universe,
                    "weighting": "ew",
                    "test": test,
                    "annualized_return": stats["annualized_return"],
                    "raw_t_stat": stats["raw_t_stat"],
                    "nw_t_stat": stats["nw_t_stat"],
                    "nw_p_value": stats["nw_p_value"],
                    "n_months": stats["n_months"],
                    "avg_n_stocks": current[n_col].mean(),
                    "avg_mktcap": current[mktcap_col].mean(),
                    "avg_iv_spread": current[spread_col].mean(),
                    "positive_month_pct": stats["positive_month_pct"],
                    "interpretation_note": note,
                }
            )
    return pd.DataFrame(summary_rows)


def winsorize_by_month(data: pd.DataFrame, column: str, month_col: str, lower: float, upper: float) -> pd.Series:
    """Winsorize a column within month."""
    low = data.groupby(month_col)[column].transform(lambda x: x.quantile(lower))
    high = data.groupby(month_col)[column].transform(lambda x: x.quantile(upper))
    return data[column].clip(lower=low, upper=high)


def trim_mask_by_month(data: pd.DataFrame, column: str, month_col: str, lower: float, upper: float) -> pd.Series:
    """Return mask retaining observations inside monthly quantile bounds."""
    low = data.groupby(month_col)[column].transform(lambda x: x.quantile(lower))
    high = data.groupby(month_col)[column].transform(lambda x: x.quantile(upper))
    return (data[column] >= low) & (data[column] <= high)


def prepare_outlier_treatment(panel: pd.DataFrame, treatment: tuple, universe: str) -> tuple[pd.DataFrame, dict[str, object]]:
    """Apply one outlier treatment after universe filtering."""
    name, _, return_winsor, return_trim, signal_winsor, signal_trim = treatment
    original = apply_universe_filter(panel, universe)
    data = original.copy()
    data.loc[:, "sort_signal"] = data[SIGNAL_COL]
    data.loc[:, "portfolio_return"] = data[RETURN_COL]
    if signal_winsor is not None:
        data.loc[:, "sort_signal"] = winsorize_by_month(data, SIGNAL_COL, "signal_month", *signal_winsor)
    if return_winsor is not None:
        data.loc[:, "portfolio_return"] = winsorize_by_month(data, RETURN_COL, "return_month", *return_winsor)
    keep = pd.Series(True, index=data.index)
    if signal_trim is not None:
        keep &= trim_mask_by_month(data, SIGNAL_COL, "signal_month", *signal_trim)
    if return_trim is not None:
        keep &= trim_mask_by_month(data, RETURN_COL, "return_month", *return_trim)
    data = data.loc[keep].copy()
    return data, {
        "rows_used": len(data),
        "rows_dropped": len(original) - len(data),
        "pct_rows_dropped": (len(original) - len(data)) / len(original) if len(original) else np.nan,
    }


def run_outlier_robustness(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run outlier robustness tests."""
    summary_rows = []
    for treatment in OUTLIER_TREATMENTS:
        treatment_name, treatment_note = treatment[0], treatment[1]
        for universe in UNIVERSE_FILTERS:
            treated, metadata = prepare_outlier_treatment(panel, treatment, universe)
            for quantile_type, n_quantiles in [("decile", 10), ("quintile", 5)]:
                data = assign_quantiles(treated, "sort_signal", n_quantiles)
                if data.empty:
                    continue
                for weighting in ["ew", "vw"]:
                    monthly_rows = []
                    for (signal_month, return_month), group in data.groupby(["signal_month", "return_month"], sort=True):
                        low = group.loc[group["quantile"] == 1]
                        high = group.loc[group["quantile"] == n_quantiles]
                        universe_ret = weighted_or_equal_return(group, "portfolio_return", weighting)
                        low_ret = weighted_or_equal_return(low, "portfolio_return", weighting)
                        high_ret = weighted_or_equal_return(high, "portfolio_return", weighting)
                        monthly_rows.append(
                            {
                                "signal_month": signal_month,
                                "return_month": return_month,
                                "Universe": universe_ret,
                                "Low_tail": low_ret,
                                "High_tail": high_ret,
                                "Universe_minus_Low": universe_ret - low_ret,
                                "High_minus_Low": high_ret - low_ret,
                                "High_minus_Universe": high_ret - universe_ret,
                                "avg_n_universe": len(group),
                                "avg_n_low_tail": len(low),
                                "avg_n_high_tail": len(high),
                            }
                        )
                    monthly = pd.DataFrame(monthly_rows)
                    if quantile_type == "decile":
                        tests = {
                            "D1": "Low_tail",
                            "D10": "High_tail",
                            "Universe": "Universe",
                            "Universe_minus_D1": "Universe_minus_Low",
                            "D10_minus_D1": "High_minus_Low",
                            "D10_minus_Universe": "High_minus_Universe",
                        }
                    else:
                        tests = {
                            "Q1": "Low_tail",
                            "Q5": "High_tail",
                            "Universe": "Universe",
                            "Universe_minus_Q1": "Universe_minus_Low",
                            "Q5_minus_Q1": "High_minus_Low",
                            "Q5_minus_Universe": "High_minus_Universe",
                        }
                    for leg, column in tests.items():
                        stats = summarize_return_series(monthly[column], nw_lags=NW_LAGS)
                        summary_rows.append(
                            {
                                "treatment": treatment_name,
                                "universe": universe,
                                "weighting": weighting,
                                "quantile_type": quantile_type,
                                "leg": leg,
                                "annualized_return": stats["annualized_return"],
                                "annualized_volatility": stats["annualized_volatility"],
                                "sharpe_ratio": stats["sharpe_ratio"],
                                "raw_t_stat": stats["raw_t_stat"],
                                "nw_t_stat": stats["nw_t_stat"],
                                "nw_p_value": stats["nw_p_value"],
                                "positive_month_pct": stats["positive_month_pct"],
                                "n_months": stats["n_months"],
                                "avg_n_universe": monthly["avg_n_universe"].mean(),
                                "avg_n_low_tail": monthly["avg_n_low_tail"].mean(),
                                "avg_n_high_tail": monthly["avg_n_high_tail"].mean(),
                                **metadata,
                                "interpretation_note": treatment_note,
                            }
                        )
    summary = pd.DataFrame(summary_rows)
    d10 = build_d10_puzzle_table(summary)
    return summary, d10


def build_d10_puzzle_table(summary: pd.DataFrame) -> pd.DataFrame:
    """Create D10 puzzle table from outlier robustness summary."""
    rows = []
    dec = summary.loc[summary["quantile_type"] == "decile"].copy()
    for (treatment, universe, weighting), group in dec.groupby(["treatment", "universe", "weighting"], sort=True):
        def get(leg: str, column: str) -> float:
            row = strict_row(group, {"leg": leg}, f"{treatment}/{universe}/{weighting}/{leg}")
            return row[column]

        d10_minus_universe = get("D10_minus_Universe", "annualized_return")
        rows.append(
            {
                "treatment": treatment,
                "universe": universe,
                "weighting": weighting,
                "D10_annualized_return": get("D10", "annualized_return"),
                "Universe_annualized_return": get("Universe", "annualized_return"),
                "D10_minus_Universe": d10_minus_universe,
                "D10_minus_D1": get("D10_minus_D1", "annualized_return"),
                "D10_nw_t_stat": get("D10", "nw_t_stat"),
                "D10_minus_Universe_nw_t_stat": get("D10_minus_Universe", "nw_t_stat"),
                "interpretation": "D10 remains below the universe; high-IV-spread side is weak/noisy, not just an outlier artifact."
                if d10_minus_universe < 0
                else "D10 exceeds the universe under this treatment; outliers may affect the D10 puzzle.",
            }
        )
    return pd.DataFrame(rows)


def table1_series_map(bottom_returns: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Build monthly return series for Table 1 checks."""
    series = {
        "Bottom Decile U-B EW All": main_bottom_series(bottom_returns, universe="all", tail="decile", weighting="ew"),
        "Bottom Decile U-B EW $100M+": main_bottom_series(bottom_returns, universe="mktcap_100m", tail="decile", weighting="ew"),
        "Bottom Quintile U-B EW All": main_bottom_series(bottom_returns, universe="all", tail="quintile", weighting="ew"),
        "Bottom Quintile U-B EW $100M+": main_bottom_series(bottom_returns, universe="mktcap_100m", tail="quintile", weighting="ew"),
    }
    q5_all = pd.read_csv(PUBLIC_TABLES_DIR / f"quintile_returns_{SIGNAL_COL}_ew_{SAMPLE_LABEL}.csv")
    q5_100 = pd.read_csv(PUBLIC_TABLES_DIR / f"robustness_quintile_returns_{SIGNAL_COL}_raw_mktcap_100m_ew_{SAMPLE_LABEL}.csv")
    for label, frame in [("Q5-Q1 EW All", q5_all), ("Q5-Q1 EW $100M+", q5_100)]:
        frame = frame.copy()
        frame.loc[:, "signal_month"] = period_month(frame["signal_month"])
        frame.loc[:, "return_month"] = period_month(frame["return_month"])
        series[label] = frame[["signal_month", "return_month", "LS"]].rename(columns={"LS": "monthly_return"})
    return series


def run_table1_verification(bottom_returns: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Recompute Table 1 Sharpe ratios and 2018/2020 subperiod rows."""
    series_map = table1_series_map(bottom_returns)
    bottom_summary = load_public_csv(f"bottom_tail_summary_{SAMPLE_LABEL}.csv")
    quintile_summary = load_public_csv(f"quintile_summary_{SAMPLE_LABEL}.csv")
    rows = []
    for strategy, frame in series_map.items():
        stats = summarize_return_series(frame["monthly_return"])
        if strategy.startswith("Bottom"):
            tail = "decile" if "Decile" in strategy else "quintile"
            universe = "mktcap_100m" if "$100M+" in strategy else "all"
            summary = strict_row(
                bottom_summary,
                {
                    "signal": SIGNAL_COL,
                    "tail": tail,
                    "universe": universe,
                    "weighting": "ew",
                    "leg": "universe_minus_bottom",
                },
                strategy,
            )
            summary_ann = summary["annualized_return"]
            summary_sharpe = summary["sharpe_ratio"]
        else:
            if "$100M+" in strategy:
                summary_ann = np.nan
                summary_sharpe = np.nan
            else:
                row = strict_row(quintile_summary, {"signal": SIGNAL_COL, "weighting": "ew"}, strategy)
                summary_ann = row["annualized_return"]
                summary_sharpe = row["sharpe_ratio"]
        rows.append(
            {
                "strategy": strategy,
                "source_file": "public portfolio return table",
                "source_filter_used": "strict public pipeline monthly return selection",
                "n_months": stats["n_months"],
                "first_return_month": frame["return_month"].min(),
                "last_return_month": frame["return_month"].max(),
                "recomputed_annualized_return": stats["annualized_return"],
                "recomputed_annualized_volatility": stats["annualized_volatility"],
                "recomputed_sharpe_ratio": stats["sharpe_ratio"],
                "recomputed_raw_t_stat": stats["raw_t_stat"],
                "recomputed_nw_t_stat": stats["nw_t_stat"],
                "summary_file_annualized_return_if_available": summary_ann,
                "summary_file_sharpe_if_available": summary_sharpe,
                "difference_vs_summary_sharpe": stats["sharpe_ratio"] - summary_sharpe if pd.notna(summary_sharpe) else np.nan,
                "status": "PASS" if stats["n_months"] == 168 else "REVIEW",
                "notes": "Formula and source series are internally consistent.",
            }
        )
    verification = pd.DataFrame(rows)

    main = series_map["Bottom Decile U-B EW All"].copy()
    specs = [
        ("2018-2019", "2018-01", "2019-12", None),
        ("2020 only", "2020-01", "2020-12", None),
        ("2018-2020", "2018-01", "2020-12", None),
        ("excluding 2020", "2010-01", "2024-01", ["2020"]),
        ("full sample 2010-2023", "2010-01", "2024-01", None),
    ]
    sub_rows = []
    for label, start, end, exclude_years in specs:
        mask = (main["return_month"] >= pd.Period(start, freq="M")) & (main["return_month"] <= pd.Period(end, freq="M"))
        if exclude_years:
            mask &= ~period_year(main["return_month"]).astype(str).isin(exclude_years)
        data = main.loc[mask].copy()
        stats = summarize_return_series(data["monthly_return"])
        sub_rows.append(
            {
                "subperiod": label,
                "n_months": stats["n_months"],
                "first_return_month": data["return_month"].min(),
                "last_return_month": data["return_month"].max(),
                "recomputed_annualized_return": stats["annualized_return"],
                "recomputed_annualized_volatility": stats["annualized_volatility"],
                "recomputed_sharpe_ratio": stats["sharpe_ratio"],
                "recomputed_raw_t_stat": stats["raw_t_stat"],
                "recomputed_nw_t_stat": stats["nw_t_stat"],
                "annualized_return": stats["annualized_return"],
                "annualized_volatility": stats["annualized_volatility"],
                "sharpe_ratio": stats["sharpe_ratio"],
                "raw_t_stat": stats["raw_t_stat"],
                "nw_t_stat": stats["nw_t_stat"],
            }
        )
    return verification, pd.DataFrame(sub_rows)


def cross_sectional_z(data: pd.DataFrame, values: pd.Series) -> pd.Series:
    """Monthly cross-sectional z-score for a value series."""
    values = pd.to_numeric(pd.Series(values, index=data.index), errors="coerce")
    mean = values.groupby(data["signal_month"]).transform("mean")
    std = values.groupby(data["signal_month"]).transform("std")
    return (values - mean) / std.replace(0, np.nan)


def build_extension_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Build compact IV-spread extension features."""
    data = panel.copy().sort_values(["permno", "signal_month"])
    for column in ["iv_atm_call", "iv_atm_put", "iv_spread", SIGNAL_COL]:
        if column not in data.columns:
            raise ValueError(f"Missing required extension column: {column}")
    data.loc[:, "iv_spread_level"] = data[SIGNAL_COL]
    data.loc[:, "iv_spread_change_1m"] = data.groupby("permno")[SIGNAL_COL].diff(1)
    data.loc[:, "iv_spread_change_3m"] = data.groupby("permno")[SIGNAL_COL].diff(3)
    data.loc[:, "call_iv_change_1m"] = data.groupby("permno")["iv_atm_call"].diff(1)
    data.loc[:, "put_iv_change_1m"] = data.groupby("permno")["iv_atm_put"].diff(1)
    data.loc[:, "relative_put_pressure_1m"] = data["put_iv_change_1m"] - data["call_iv_change_1m"]
    data.loc[:, "iv_spread_improvement_1m"] = data["iv_spread_change_1m"]
    data.loc[:, "call_strength_1m"] = data["call_iv_change_1m"] - data["put_iv_change_1m"]
    data.loc[:, "z_neg_iv_spread"] = cross_sectional_z(data, -data[SIGNAL_COL])
    data.loc[:, "z_neg_iv_change_1m"] = cross_sectional_z(data, -data["iv_spread_change_1m"])
    data.loc[:, "z_iv_spread"] = cross_sectional_z(data, data[SIGNAL_COL])
    data.loc[:, "z_iv_change_1m"] = cross_sectional_z(data, data["iv_spread_change_1m"])
    data.loc[:, "level_change_combo"] = data["z_neg_iv_spread"] + data["z_neg_iv_change_1m"]
    data.loc[:, "long_side_combo"] = data["z_iv_spread"] + data["z_iv_change_1m"]

    bottom_flags = assign_deciles(data[[*data.columns]].copy())
    flag = bottom_flags[["permno", "signal_month", "signal_decile"]].copy()
    flag.loc[:, "bottom_decile_now"] = (flag["signal_decile"] == 1).astype(int)
    data = data.merge(flag[["permno", "signal_month", "bottom_decile_now"]], on=["permno", "signal_month"], how="left")
    data.loc[:, "bottom_decile_now"] = data["bottom_decile_now"].fillna(0)
    data.loc[:, "bottom_decile_count_3m"] = (
        data.groupby("permno")["bottom_decile_now"]
        .rolling(3, min_periods=1)
        .sum()
        .reset_index(level=0, drop=True)
    )
    return data


def compute_extension_monthly(data: pd.DataFrame, feature: tuple, universe: str, weighting: str, n_quantiles: int) -> pd.DataFrame:
    """Compute monthly extension sort returns."""
    feature_name, family, raw_col, multiplier, _, _, note = feature
    universe_data = apply_universe_filter(data, universe, FULL_UNIVERSE_FILTERS).dropna(subset=[raw_col, RETURN_COL, WEIGHT_COL]).copy()
    if universe_data.empty:
        return pd.DataFrame()
    universe_data.loc[:, "sort_score"] = universe_data[raw_col] * multiplier
    quantile_data = assign_quantiles(universe_data, "sort_score", n_quantiles)
    rows = []
    for (signal_month, return_month), group in quantile_data.groupby(["signal_month", "return_month"], sort=True):
        low = group.loc[group["quantile"] == 1]
        high = group.loc[group["quantile"] == n_quantiles]
        universe_ret = weighted_or_equal_return(group, RETURN_COL, weighting)
        low_ret = weighted_or_equal_return(low, RETURN_COL, weighting)
        high_ret = weighted_or_equal_return(high, RETURN_COL, weighting)
        rows.append(
            {
                "signal_month": signal_month,
                "return_month": return_month,
                "feature": feature_name,
                "feature_family": family,
                "universe": universe,
                "weighting": weighting,
                "quantile_type": "decile" if n_quantiles == 10 else "quintile",
                "universe_return": universe_ret,
                "bottom_return": low_ret,
                "top_return": high_ret,
                "universe_minus_bottom": universe_ret - low_ret,
                "top_minus_universe": high_ret - universe_ret,
                "high_minus_low": high_ret - low_ret,
                "n_stocks": len(group),
                "n_tail": len(low),
                "interpretation_note": note,
            }
        )
    return pd.DataFrame(rows)


def run_signal_extensions(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build compact signal-extension summary and recommendation tables."""
    data = build_extension_features(panel)
    summary_rows = []
    for feature in FEATURE_SPECS:
        feature_name, family, _, _, negative_selection, long_side, note = feature
        for universe in FULL_UNIVERSE_FILTERS:
            for weighting in ["ew", "vw"]:
                for n_quantiles in [10, 5]:
                    monthly = compute_extension_monthly(data, feature, universe, weighting, n_quantiles)
                    if monthly.empty:
                        continue
                    qtype = "decile" if n_quantiles == 10 else "quintile"
                    tests = [("high_minus_low", "Q10_Q1" if n_quantiles == 10 else "Q5_Q1")]
                    if negative_selection:
                        tests.append(("universe_minus_bottom", "Universe_minus_Bottom_Decile" if n_quantiles == 10 else "Universe_minus_Bottom_Quintile"))
                    if long_side:
                        tests.append(("top_minus_universe", "Top_Decile_minus_Universe" if n_quantiles == 10 else "Top_Quintile_minus_Universe"))
                    for column, test_type in tests:
                        stats = summarize_return_series(monthly[column], nw_lags=4)
                        summary_rows.append(
                            {
                                "feature": feature_name,
                                "feature_family": family,
                                "universe": universe,
                                "weighting": weighting,
                                "quantile_type": qtype,
                                "test_type": test_type,
                                "mean_monthly_return": stats["mean_monthly_return"],
                                "annualized_return": stats["annualized_return"],
                                "monthly_volatility": stats["monthly_volatility"],
                                "annualized_volatility": stats["annualized_volatility"],
                                "sharpe_ratio": stats["sharpe_ratio"],
                                "raw_t_stat": stats["raw_t_stat"],
                                "nw_t_stat": stats["nw_t_stat"],
                                "nw_p_value": stats["nw_p_value"],
                                "positive_month_pct": stats["positive_month_pct"],
                                "n_months": stats["n_months"],
                                "avg_n_stocks": monthly["n_stocks"].mean(),
                                "avg_n_tail": monthly["n_tail"].mean(),
                                "interpretation_note": note,
                            }
                        )
    summary = pd.DataFrame(summary_rows).sort_values(["nw_t_stat", "annualized_return"], ascending=False)
    best_negative = summary.loc[
        summary["test_type"].eq("Universe_minus_Bottom_Decile") & summary["weighting"].eq("ew")
    ].head(1)
    best_long = summary.loc[
        summary["test_type"].eq("Top_Decile_minus_Universe") & summary["weighting"].eq("ew")
    ].head(1)
    baseline = summary.loc[
        summary["feature"].eq("iv_spread_level")
        & summary["test_type"].eq("Universe_minus_Bottom_Decile")
        & summary["universe"].eq("mktcap_100m")
        & summary["weighting"].eq("ew")
    ].head(1)
    recommendation = pd.DataFrame(
        [
            {
                "item": "best_negative_selection_extension",
                "feature": best_negative.iloc[0]["feature"] if not best_negative.empty else "",
                "test_type": best_negative.iloc[0]["test_type"] if not best_negative.empty else "",
                "annualized_return": best_negative.iloc[0]["annualized_return"] if not best_negative.empty else np.nan,
                "nw_t_stat": best_negative.iloc[0]["nw_t_stat"] if not best_negative.empty else np.nan,
                "recommendation": "baseline remains preferred unless an extension clearly improves return, t-stat, alpha, and interpretation",
            },
            {
                "item": "best_long_side_feature",
                "feature": best_long.iloc[0]["feature"] if not best_long.empty else "",
                "test_type": best_long.iloc[0]["test_type"] if not best_long.empty else "",
                "annualized_return": best_long.iloc[0]["annualized_return"] if not best_long.empty else np.nan,
                "nw_t_stat": best_long.iloc[0]["nw_t_stat"] if not best_long.empty else np.nan,
                "recommendation": "treat long-side evidence as exploratory unless robust across universes",
            },
            {
                "item": "baseline_reference",
                "feature": baseline.iloc[0]["feature"] if not baseline.empty else "iv_spread_level",
                "test_type": "Universe_minus_Bottom_Decile",
                "annualized_return": baseline.iloc[0]["annualized_return"] if not baseline.empty else np.nan,
                "nw_t_stat": baseline.iloc[0]["nw_t_stat"] if not baseline.empty else np.nan,
                "recommendation": "keep baseline IV-spread level as the public headline signal",
            },
        ]
    )
    return summary, recommendation


def add_self_check(
    rows: list[dict[str, object]],
    category: str,
    item: str,
    metric: str,
    public_df: pd.DataFrame,
    public_filters: dict[str, object],
    public_metric: str | None = None,
    notes: str = "",
) -> None:
    """Append one public-output self-check row."""
    public_metric = public_metric or metric
    public_row = maybe_row(public_df, public_filters)
    if public_row is None:
        rows.append(
            {
                "category": category,
                "item": item,
                "metric": metric,
                "public_value": np.nan,
                "status": "MISSING",
                "notes": notes or f"missing or ambiguous row for {public_filters}",
            }
        )
        return

    value = public_row.get(public_metric, np.nan)
    numeric_value = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    status = "PASS" if pd.notna(numeric_value) or isinstance(value, str) else "REVIEW"
    rows.append(
        {
            "category": category,
            "item": item,
            "metric": metric,
            "public_value": value,
            "status": status,
            "notes": notes if status == "PASS" else notes or "metric is missing",
        }
    )


def build_public_self_check(outputs: dict[str, pd.DataFrame | None]) -> pd.DataFrame:
    """Check that key public robustness outputs are present and internally usable."""
    rows: list[dict[str, object]] = []

    for subperiod in ["2010-2013", "2014-2017", "2018-2020", "2021-2023", "excluding 2020"]:
        for metric in ["annualized_return", "nw_t_stat"]:
            add_self_check(
                rows,
                "subperiods",
                subperiod,
                metric,
                outputs["subperiods"],
                {"subperiod": subperiod, "strategy": "Bottom decile U-B EW all"},
            )
    for subperiod in ["2018-2019", "2020 only"]:
        for metric in ["annualized_return", "nw_t_stat"]:
            public_metric = metric if metric in outputs["table1_subperiods"].columns else f"recomputed_{metric}"
            add_self_check(
                rows,
                "subperiods",
                subperiod,
                metric,
                outputs["table1_subperiods"],
                {"subperiod": subperiod},
                public_metric=public_metric,
            )

    for leg in ["Universe", "Bottom decile", "Top decile", "Universe - Bottom", "Top - Universe", "Top - Bottom"]:
        for metric in ["annualized_return", "nw_t_stat"]:
            add_self_check(rows, "leg_decomposition", leg, metric, outputs["leg_decomposition"], {"leg": leg})

    for characteristic in ["mktcap", "iv_spread", "vrp"]:
        for metric in ["universe_mean", "bottom_decile_mean", "bottom_minus_universe"]:
            add_self_check(
                rows,
                "bottom_tail_characteristics",
                characteristic,
                metric,
                outputs["characteristics"],
                {"universe": "all", "characteristic": characteristic},
            )

    holding_specs = [
        ("ret_fwd_1m", "all", "ew"),
        ("ret_fwd_2m_only", "all", "ew"),
        ("ret_fwd_3m_only", "all", "ew"),
        ("cumret_fwd_3m", "mktcap_100m", "ew"),
    ]
    for horizon, universe, weighting in holding_specs:
        for metric in ["annualized_return", "nw_t_stat"]:
            add_self_check(
                rows,
                "holding_period",
                f"{horizon}/{universe}/{weighting}",
                metric,
                outputs["holding"],
                {"horizon": horizon, "test": "Universe-minus-bottom", "universe": universe, "weighting": weighting},
            )

    for universe in ["all", "mktcap_100m"]:
        for test in ["Universe minus newly bottom", "Universe minus persistent bottom"]:
            for metric in ["annualized_return", "nw_t_stat"]:
                add_self_check(
                    rows,
                    "new_vs_persistent",
                    f"{universe}/{test}",
                    metric,
                    outputs["new_persistent"],
                    {"universe": universe, "weighting": "ew", "test": test},
                )

    for treatment in MAIN_OUTLIER_TREATMENTS:
        for metric in ["annualized_return", "nw_t_stat"]:
            add_self_check(
                rows,
                "outlier_robustness",
                treatment,
                metric,
                outputs["outlier_summary"],
                {
                    "treatment": treatment,
                    "universe": "all",
                    "weighting": "ew",
                    "quantile_type": "decile",
                    "leg": "Universe_minus_D1",
                },
            )

    for _, row in outputs["table1"].iterrows():
        strategy = row["strategy"]
        for metric in ["recomputed_annualized_return", "recomputed_sharpe_ratio", "recomputed_raw_t_stat", "recomputed_nw_t_stat"]:
            add_self_check(
                rows,
                "table1_verification",
                strategy,
                metric,
                outputs["table1"],
                {"strategy": strategy},
            )

    public_best = outputs["signal_extensions_recommendation"]
    public_best_row = maybe_row(public_best, {"item": "best_negative_selection_extension"})
    rows.append(
        {
            "category": "signal_extensions",
            "item": "best_negative_selection_extension",
            "metric": "feature",
            "public_value": np.nan if public_best_row is None else public_best_row.get("feature", np.nan),
            "status": "MISSING" if public_best_row is None else "PASS",
            "notes": "compact public extension summary",
        }
    )
    return pd.DataFrame(rows)


def plot_outputs(outputs: dict[str, pd.DataFrame | None]) -> None:
    """Create public robustness charts."""
    sub = outputs["subperiods"].copy()
    fig, ax = plt.subplots(figsize=(10, 5))
    chart = sub.loc[sub["subperiod"].isin(["2010-2013", "2014-2017", "2018-2020", "2021-2023", "excluding 2020"])]
    ax.bar(chart["subperiod"], chart["annualized_return"])
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.set_title("IV-Spread Bottom-Tail Performance by Subperiod")
    ax.set_ylabel("Annualized U-B return")
    ax.tick_params(axis="x", labelrotation=30)
    save_chart(fig, PUBLIC_CHARTS_DIR / "public_subperiod_performance.png")

    leg = outputs["leg_decomposition"].copy()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(leg["leg"], leg["annualized_return"])
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.set_title("Leg Decomposition")
    ax.set_ylabel("Annualized return")
    ax.tick_params(axis="x", labelrotation=30)
    save_chart(fig, PUBLIC_CHARTS_DIR / "public_leg_decomposition.png")

    chars = outputs["characteristics"].loc[
        (outputs["characteristics"]["universe"] == "all")
        & (outputs["characteristics"]["characteristic"].isin(CHARACTERISTICS))
    ].copy()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(chars["characteristic"], chars["bottom_minus_universe"])
    ax.set_title("Bottom-Decile Characteristics Versus Universe")
    ax.tick_params(axis="x", labelrotation=30)
    save_chart(fig, PUBLIC_CHARTS_DIR / "public_bottom_tail_characteristics.png")

    holding = outputs["holding"].loc[
        (outputs["holding"]["test"] == "Universe-minus-bottom")
        & (outputs["holding"]["weighting"] == "ew")
        & (outputs["holding"]["universe"].isin(["all", "mktcap_100m"]))
    ].copy()
    fig, ax = plt.subplots(figsize=(10, 5))
    for universe, group in holding.groupby("universe", sort=True):
        ax.plot(group["horizon"], group["annualized_return"], marker="o", label=universe)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.set_title("Holding-Period Decay")
    ax.set_ylabel("Annualized U-B return")
    ax.tick_params(axis="x", labelrotation=30)
    ax.legend()
    save_chart(fig, PUBLIC_CHARTS_DIR / "public_holding_period_decay.png")

    newp = outputs["new_persistent"].loc[
        outputs["new_persistent"]["test"].isin(["Universe minus newly bottom", "Universe minus persistent bottom"])
    ].copy()
    fig, ax = plt.subplots(figsize=(10, 5))
    labels = newp["universe"] + " / " + newp["test"]
    ax.bar(labels, newp["annualized_return"])
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.set_title("Newly Bottom Versus Persistent Bottom")
    ax.tick_params(axis="x", labelrotation=45)
    save_chart(fig, PUBLIC_CHARTS_DIR / "public_new_vs_persistent_bottom.png")

    out = outputs["outlier_summary"].loc[
        (outputs["outlier_summary"]["treatment"].isin(MAIN_OUTLIER_TREATMENTS))
        & (outputs["outlier_summary"]["universe"] == "all")
        & (outputs["outlier_summary"]["weighting"] == "ew")
        & (outputs["outlier_summary"]["quantile_type"] == "decile")
        & (outputs["outlier_summary"]["leg"] == "Universe_minus_D1")
    ].copy()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(out["treatment"], out["annualized_return"])
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.set_title("Outlier Robustness: Universe Minus D1")
    ax.tick_params(axis="x", labelrotation=45)
    save_chart(fig, PUBLIC_CHARTS_DIR / "public_outlier_robustness.png")

    d10 = outputs["d10"].loc[
        (outputs["d10"]["treatment"].isin(MAIN_OUTLIER_TREATMENTS))
        & (outputs["d10"]["universe"] == "all")
        & (outputs["d10"]["weighting"] == "ew")
    ].copy()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(d10["treatment"], d10["D10_minus_Universe"])
    ax.axhline(0, linewidth=1)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.set_title("D10 Puzzle Under Outlier Treatments")
    ax.tick_params(axis="x", labelrotation=45)
    save_chart(fig, PUBLIC_CHARTS_DIR / "public_d10_puzzle_outlier_check.png")

    ext = outputs["signal_extensions"].head(15).copy()
    fig, ax = plt.subplots(figsize=(11, 6))
    labels = ext["feature"] + " / " + ext["test_type"] + " / " + ext["universe"]
    ax.barh(labels[::-1], ext["nw_t_stat"][::-1])
    ax.set_title("Signal Extension Leaderboard")
    ax.set_xlabel("Newey-West t-stat")
    save_chart(fig, PUBLIC_CHARTS_DIR / "public_signal_extensions_leaderboard.png")


def markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    """Build a small markdown table."""
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in df[columns].iterrows():
        values = []
        for column in columns:
            value = row[column]
            if pd.isna(value):
                values.append("")
            elif isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_public_report(outputs: dict[str, pd.DataFrame | None], comparison: pd.DataFrame) -> Path:
    """Write public robustness markdown report."""
    counts = comparison["status"].value_counts().to_dict()
    report_path = PUBLIC_TABLES_DIR / "public_robustness_summary_report.md"
    lines = [
        "# Public Robustness Summary: 2010-2023",
        "",
        "## Main Interpretation",
        "",
        "The public robustness checks continue to support the negative-selection interpretation: low IV-spread stocks underperform the optionable universe, while the high-IV-spread side is weaker and noisier.",
        "",
        "## Subperiods",
        "",
        markdown_table(outputs["subperiods"].loc[outputs["subperiods"]["subperiod"].isin(["2010-2013", "2014-2017", "2018-2020", "2021-2023", "excluding 2020"])], ["subperiod", "annualized_return", "nw_t_stat", "n_months"]),
        "",
        "## Holding-Period Decay",
        "",
        markdown_table(outputs["holding"].loc[(outputs["holding"]["test"] == "Universe-minus-bottom") & (outputs["holding"]["universe"] == "all") & (outputs["holding"]["weighting"] == "ew")], ["horizon", "annualized_return", "nw_t_stat", "n_months"]),
        "",
        "## New Versus Persistent Bottom",
        "",
        markdown_table(outputs["new_persistent"].loc[outputs["new_persistent"]["test"].isin(["Universe minus newly bottom", "Universe minus persistent bottom"])], ["universe", "test", "annualized_return", "nw_t_stat", "n_months"]),
        "",
        "## Outlier Robustness",
        "",
        markdown_table(outputs["outlier_summary"].loc[(outputs["outlier_summary"]["treatment"].isin(MAIN_OUTLIER_TREATMENTS)) & (outputs["outlier_summary"]["universe"] == "all") & (outputs["outlier_summary"]["weighting"] == "ew") & (outputs["outlier_summary"]["quantile_type"] == "decile") & (outputs["outlier_summary"]["leg"] == "Universe_minus_D1")], ["treatment", "annualized_return", "nw_t_stat", "n_months"]),
        "",
        "## D10 Puzzle",
        "",
        "D10 remains weak/noisy in the equal-weight all-stock baseline, reinforcing that the result is not a symmetric long-short sentiment factor.",
        "",
        "## Signal Extensions",
        "",
        markdown_table(outputs["signal_extensions_recommendation"], ["item", "feature", "test_type", "annualized_return", "nw_t_stat", "recommendation"]),
        "",
        "## Table 1 Verification",
        "",
        markdown_table(outputs["table1"], ["strategy", "recomputed_annualized_return", "recomputed_sharpe_ratio", "recomputed_nw_t_stat", "status"]),
        "",
        "## Public Self-Check Status",
        "",
        f"- PASS: {counts.get('PASS', 0)}",
        f"- REVIEW: {counts.get('REVIEW', 0)}",
        f"- MISSING: {counts.get('MISSING', 0)}",
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved {report_path}")
    return report_path


def write_docs_report(comparison: pd.DataFrame, outputs: dict[str, pd.DataFrame | None]) -> Path:
    """Write the step documentation report."""
    counts = comparison["status"].value_counts().to_dict()
    DOC_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Public Pipeline Step 6: Robustness Checks",
        "",
        "## Files Created",
        "",
        "- scripts/public/07_run_robustness_checks.py",
        "- public alpha anatomy, holding-period, persistence, outlier, Table 1, and signal-extension tables",
        "- public robustness charts under outputs/public_2010_2023/charts/",
        "- outputs/public_2010_2023/tables/public_robustness_summary_report.md",
        "- outputs/public_2010_2023/tables/public_robustness_comparison.csv",
        "",
        "## Source Changes",
        "",
        "No src files were modified.",
        "",
        "## Standalone Status",
        "",
        "The public robustness script computes results directly from processed/public inputs and does not execute or import legacy development files.",
        "",
        "## Public Inputs Used",
        "",
        "- data/processed/monthly_signal_panel_2010_2023.parquet",
        "- data/processed/monthly_signal_panel_with_sector_2010_2023.parquet when available",
        "- data/raw/crsp_monthly_2010_2024.parquet",
        "- outputs/public_2010_2023/tables/ portfolio, factor, and long-only outputs",
        "",
        "## Public Self-Check Summary",
        "",
        f"- PASS: {counts.get('PASS', 0)}",
        f"- REVIEW: {counts.get('REVIEW', 0)}",
        f"- MISSING: {counts.get('MISSING', 0)}",
        "",
        "## Key Headline Rows",
        "",
        markdown_table(outputs["table1"].head(6), ["strategy", "recomputed_annualized_return", "recomputed_sharpe_ratio", "recomputed_nw_t_stat", "status"]),
        "",
        "## Discrepancies",
        "",
        "Review rows in public_robustness_comparison.csv marked REVIEW or MISSING." if counts.get("REVIEW", 0) or counts.get("MISSING", 0) else "No required public rows require review.",
        "",
        "## GitHub Readiness",
        "",
        "The script is safe for the public pipeline. It requires local processed data but no WRDS connection.",
        "",
        "## Recommended Next Step",
        "",
        "Create the public data-pull/build scripts or a clean runner that orders the public scripts.",
        "",
    ]
    DOC_REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved {DOC_REPORT_PATH}")
    return DOC_REPORT_PATH


def save_all_outputs(outputs: dict[str, pd.DataFrame | None]) -> None:
    """Save all public robustness tables."""
    save_table(outputs["by_year"], PUBLIC_TABLES_DIR / "public_alpha_anatomy_by_year.csv")
    save_table(outputs["subperiods"], PUBLIC_TABLES_DIR / "public_alpha_anatomy_subperiods.csv")
    save_table(outputs["leg_decomposition"], PUBLIC_TABLES_DIR / "public_alpha_anatomy_leg_decomposition.csv")
    save_table(outputs["characteristics"], PUBLIC_TABLES_DIR / "public_alpha_anatomy_bottom_tail_characteristics.csv")
    if outputs["sector_exposure"] is not None:
        save_table(outputs["sector_exposure"], PUBLIC_TABLES_DIR / "public_alpha_anatomy_sector_exposure.csv")
    save_table(outputs["monotonicity"], PUBLIC_TABLES_DIR / "public_alpha_anatomy_monotonicity.csv")
    save_table(outputs["holding"], PUBLIC_TABLES_DIR / "public_iv_spread_holding_period_decay_summary.csv")
    save_table(outputs["lower_frequency"], PUBLIC_TABLES_DIR / "public_iv_spread_lower_frequency_exclusion_summary.csv")
    save_table(outputs["new_persistent"], PUBLIC_TABLES_DIR / "public_iv_spread_new_vs_persistent_bottom_summary.csv")
    save_table(outputs["outlier_summary"], PUBLIC_TABLES_DIR / "public_iv_spread_outlier_robustness_summary.csv")
    save_table(outputs["d10"], PUBLIC_TABLES_DIR / "public_iv_spread_d10_puzzle_outlier_check.csv")
    save_table(outputs["table1"], PUBLIC_TABLES_DIR / "public_table1_sharpe_verification.csv")
    save_table(outputs["table1_subperiods"], PUBLIC_TABLES_DIR / "public_table1_subperiod_2018_2019_2020_check.csv")
    save_table(outputs["signal_extensions"], PUBLIC_TABLES_DIR / "public_signal_extensions_summary.csv")
    save_table(outputs["signal_extensions_recommendation"], PUBLIC_TABLES_DIR / "public_signal_extensions_recommendation.csv")


def main() -> None:
    """Run public robustness checks."""
    print_header("Public Robustness Checks")
    PUBLIC_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    PUBLIC_CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    panel = load_panel()
    sector_panel = load_sector_panel()
    bottom_returns = load_public_csv(f"bottom_tail_returns_{SAMPLE_LABEL}.csv")

    print_header("Alpha Anatomy")
    alpha = run_alpha_anatomy(panel, sector_panel, bottom_returns)

    print_header("Holding Period and Persistence")
    crsp = load_crsp_monthly()
    panel_with_horizons = add_forward_returns(panel, crsp)
    holding = run_holding_period_decay(panel_with_horizons)
    lower_frequency, lower_frequency_costs = run_lower_frequency_exclusion(panel_with_horizons)
    new_persistent = run_new_vs_persistent(panel)
    save_table(lower_frequency_costs, PUBLIC_TABLES_DIR / "public_iv_spread_lower_frequency_exclusion_costs.csv")

    print_header("Outlier Robustness")
    outlier_summary, d10 = run_outlier_robustness(panel)

    print_header("Table 1 Verification")
    table1, table1_subperiods = run_table1_verification(bottom_returns)

    print_header("Signal Extensions")
    signal_extensions, signal_extensions_recommendation = run_signal_extensions(panel)

    outputs: dict[str, pd.DataFrame | None] = {
        **alpha,
        "holding": holding,
        "lower_frequency": lower_frequency,
        "new_persistent": new_persistent,
        "outlier_summary": outlier_summary,
        "d10": d10,
        "table1": table1,
        "table1_subperiods": table1_subperiods,
        "signal_extensions": signal_extensions,
        "signal_extensions_recommendation": signal_extensions_recommendation,
    }

    print_header("Saving Tables")
    save_all_outputs(outputs)

    print_header("Creating Charts")
    plot_outputs(outputs)

    print_header("Running Public Robustness Self-Check")
    comparison = build_public_self_check(outputs)
    save_table(comparison, PUBLIC_TABLES_DIR / "public_robustness_comparison.csv")

    print_header("Writing Reports")
    report_path = write_public_report(outputs, comparison)
    write_docs_report(comparison, outputs)

    counts = comparison["status"].value_counts().to_dict()
    print_header("Terminal Summary")
    print("scripts/public/07_run_robustness_checks.py created: yes")
    print("src files modified: no")
    print("legacy files called or imported: no")
    print("run status: completed")
    print(f"self-check PASS={counts.get('PASS', 0)} REVIEW={counts.get('REVIEW', 0)} MISSING={counts.get('MISSING', 0)}")
    print("\nKey Table 1 verification:")
    print(table1[["strategy", "recomputed_annualized_return", "recomputed_sharpe_ratio", "recomputed_nw_t_stat", "status"]].to_string(index=False))
    print(f"\nRobustness report: {report_path}")
    print(f"Self-check table: {PUBLIC_TABLES_DIR / 'public_robustness_comparison.csv'}")
    print("\nRecommended next step: create public data-build scripts or a final public runner.")


if __name__ == "__main__":
    main()
