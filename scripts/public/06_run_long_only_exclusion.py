"""Run the standalone public long-only IV-spread exclusion analysis.

This script reads the processed 2010-2023 monthly panel and tests whether the
lowest IV-spread stocks are useful as a long-only exclusion screen. It writes
all public outputs to outputs/public_2010_2023/ and does not depend on private
development-output folders that are excluded from the public release.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


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
from src.regressions import FACTOR_MODELS, load_factor_data  # noqa: E402
from src.utils import compute_weighted_return, summarize_return_series  # noqa: E402


SAMPLE_LABEL = FULL_EXPANSION_SAMPLE_LABEL
SIGNAL_COL = "iv_spread_adj"
RETURN_COL = "ret_fwd_1m"
WEIGHT_COL = "mktcap"
NW_LAGS = 4
TOLERANCE = 2e-8

PUBLIC_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "public_2010_2023"
PUBLIC_TABLES_DIR = PUBLIC_OUTPUT_DIR / "tables"
PUBLIC_CHARTS_DIR = PUBLIC_OUTPUT_DIR / "charts"
DOC_REPORT_PATH = PROJECT_ROOT / "docs" / "public_pipeline_step5_long_only_report.md"

UNIVERSE_FILTERS = {
    "all": None,
    "mktcap_100m": 100,
    "mktcap_500m": 500,
    "mktcap_1b": 1000,
}

PORTFOLIO_DEFINITIONS = [
    {
        "portfolio": "full_universe",
        "description": "Full optionable universe",
        "exclude": "none",
    },
    {
        "portfolio": "exclude_bottom_decile",
        "description": "Exclude bottom IV-spread decile",
        "exclude": "bottom_decile",
    },
    {
        "portfolio": "exclude_bottom_quintile",
        "description": "Exclude bottom IV-spread quintile",
        "exclude": "bottom_quintile",
    },
    {
        "portfolio": "top_90_only",
        "description": "Top 90% only, equivalent to excluding bottom decile",
        "exclude": "bottom_decile",
    },
    {
        "portfolio": "top_80_only",
        "description": "Top 80% only, equivalent to excluding bottom quintile",
        "exclude": "bottom_quintile",
    },
    {
        "portfolio": "exclude_bottom_top_decile",
        "description": "Exclude bottom and top IV-spread deciles",
        "exclude": "bottom_top_decile",
    },
    {
        "portfolio": "exclude_bottom_top_quintile",
        "description": "Exclude bottom and top IV-spread quintiles",
        "exclude": "bottom_top_quintile",
    },
]

SELECTED_FACTOR_PORTFOLIOS = [
    ("all", "ew", "full_universe"),
    ("all", "ew", "exclude_bottom_decile"),
    ("all", "ew", "exclude_bottom_quintile"),
    ("mktcap_100m", "ew", "full_universe"),
    ("mktcap_100m", "ew", "exclude_bottom_decile"),
    ("mktcap_100m", "ew", "exclude_bottom_quintile"),
    ("all", "vw", "full_universe"),
    ("all", "vw", "exclude_bottom_decile"),
    ("mktcap_100m", "vw", "full_universe"),
    ("mktcap_100m", "vw", "exclude_bottom_decile"),
]

DIFFERENCE_REGRESSION_SPECS = [
    ("all", "ew", "exclude_bottom_decile"),
    ("all", "ew", "exclude_bottom_quintile"),
    ("mktcap_100m", "ew", "exclude_bottom_decile"),
    ("mktcap_100m", "ew", "exclude_bottom_quintile"),
    ("all", "vw", "exclude_bottom_decile"),
    ("mktcap_100m", "vw", "exclude_bottom_decile"),
]


def print_header(title: str) -> None:
    """Print a readable section header."""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def save_table(df: pd.DataFrame, output_path: Path) -> None:
    """Save a dataframe and print its shape."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved {output_path} shape={df.shape}")


def save_chart(fig: plt.Figure, output_path: Path) -> None:
    """Save a matplotlib figure."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved {output_path}")


def period_month(series: pd.Series) -> pd.Series:
    """Convert a series to monthly Period values."""
    return pd.Series(
        pd.PeriodIndex(series.astype(str), freq="M"),
        index=series.index,
        dtype="period[M]",
    )


def require_columns(df: pd.DataFrame, columns: list[str], label: str) -> None:
    """Fail loudly if required columns are missing."""
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def load_panel() -> pd.DataFrame:
    """Load and clean the processed monthly panel."""
    panel_path = sample_processed_path("monthly_signal_panel", SAMPLE_LABEL)
    if not panel_path.exists():
        raise FileNotFoundError(f"Missing monthly panel: {panel_path}")

    panel = pd.read_parquet(panel_path)
    required = [
        "permno",
        "signal_month",
        "return_month",
        SIGNAL_COL,
        RETURN_COL,
        WEIGHT_COL,
    ]
    require_columns(panel, required, "monthly_signal_panel_2010_2023")

    panel = panel[required].copy()
    panel.loc[:, "signal_month"] = period_month(panel["signal_month"])
    panel.loc[:, "return_month"] = period_month(panel["return_month"])
    for column in ["permno", SIGNAL_COL, RETURN_COL, WEIGHT_COL]:
        panel.loc[:, column] = pd.to_numeric(panel[column], errors="coerce")

    panel = panel.dropna(subset=required).copy()
    panel = panel.loc[panel[WEIGHT_COL] > 0].copy()
    panel.loc[:, "permno"] = panel["permno"].astype(int)

    print(f"Loaded panel: {panel_path}")
    print(f"Shape after cleanup: {panel.shape}")
    print(f"signal_month range: {panel['signal_month'].min()} to {panel['signal_month'].max()}")
    print(f"return_month range: {panel['return_month'].min()} to {panel['return_month'].max()}")
    print(f"Unique permnos: {panel['permno'].nunique():,}")
    return panel


def apply_universe_filter(panel: pd.DataFrame, universe: str) -> pd.DataFrame:
    """Apply one market-cap universe filter."""
    if universe not in UNIVERSE_FILTERS:
        raise ValueError(f"Unknown universe filter: {universe}")
    cutoff = UNIVERSE_FILTERS[universe]
    if cutoff is None:
        return panel.copy()
    return panel.loc[panel[WEIGHT_COL] >= cutoff].copy()


def assign_tail_groups(panel: pd.DataFrame) -> pd.DataFrame:
    """Assign IV-spread deciles and quintiles within each signal month."""
    data = panel.copy()
    data.loc[:, "iv_spread_decile"] = pd.NA
    data.loc[:, "iv_spread_quintile"] = pd.NA

    for _, group in data.groupby("signal_month", sort=True):
        valid = group.dropna(subset=[SIGNAL_COL])

        if valid[SIGNAL_COL].nunique() >= 10:
            try:
                deciles = pd.qcut(valid[SIGNAL_COL], 10, labels=False, duplicates="drop")
                if deciles.nunique(dropna=True) == 10:
                    data.loc[valid.index, "iv_spread_decile"] = (deciles + 1).astype("Int64")
            except ValueError:
                pass

        if valid[SIGNAL_COL].nunique() >= 5:
            try:
                quintiles = pd.qcut(valid[SIGNAL_COL], 5, labels=False, duplicates="drop")
                if quintiles.nunique(dropna=True) == 5:
                    data.loc[valid.index, "iv_spread_quintile"] = (quintiles + 1).astype("Int64")
            except ValueError:
                pass

    data = data.dropna(subset=["iv_spread_decile", "iv_spread_quintile"]).copy()
    data.loc[:, "iv_spread_decile"] = data["iv_spread_decile"].astype(int)
    data.loc[:, "iv_spread_quintile"] = data["iv_spread_quintile"].astype(int)
    return data


def membership_mask(group: pd.DataFrame, portfolio: dict[str, str]) -> pd.Series:
    """Return a stock-selection mask for one portfolio."""
    exclude = portfolio["exclude"]
    if exclude == "none":
        return pd.Series(True, index=group.index)
    if exclude == "bottom_decile":
        return group["iv_spread_decile"] > 1
    if exclude == "bottom_quintile":
        return group["iv_spread_quintile"] > 1
    if exclude == "bottom_top_decile":
        return (group["iv_spread_decile"] > 1) & (group["iv_spread_decile"] < 10)
    if exclude == "bottom_top_quintile":
        return (group["iv_spread_quintile"] > 1) & (group["iv_spread_quintile"] < 5)
    raise ValueError(f"Unknown exclusion rule: {exclude}")


def portfolio_return(group: pd.DataFrame, selected_mask: pd.Series, weighting: str) -> float:
    """Compute an equal-weighted or value-weighted return."""
    selected = group.loc[selected_mask].copy()
    if selected.empty:
        return np.nan
    if weighting == "ew":
        return selected[RETURN_COL].mean()
    if weighting == "vw":
        return compute_weighted_return(selected[RETURN_COL], selected[WEIGHT_COL])
    raise ValueError(f"Unknown weighting: {weighting}")


def build_long_only_returns(panel: pd.DataFrame) -> pd.DataFrame:
    """Build monthly returns for long-only exclusion portfolios."""
    rows = []
    for universe in UNIVERSE_FILTERS:
        universe_panel = assign_tail_groups(apply_universe_filter(panel, universe))
        if universe_panel.empty:
            raise ValueError(f"No rows after assigning groups for universe={universe}")

        print(
            f"Universe {universe}: rows={len(universe_panel):,}, "
            f"months={universe_panel['signal_month'].nunique()}, "
            f"avg stocks/month={universe_panel.groupby('signal_month')['permno'].nunique().mean():.1f}"
        )

        for (signal_month, return_month), month_data in universe_panel.groupby(
            ["signal_month", "return_month"], sort=True
        ):
            for portfolio in PORTFOLIO_DEFINITIONS:
                selected_mask = membership_mask(month_data, portfolio)
                n_selected = int(selected_mask.sum())
                n_excluded = 0 if portfolio["exclude"] == "none" else int((~selected_mask).sum())
                excluded_share = n_excluded / len(month_data) if len(month_data) else np.nan
                selected = month_data.loc[selected_mask]

                for weighting in ["ew", "vw"]:
                    rows.append(
                        {
                            "signal_month": signal_month,
                            "return_month": return_month,
                            "universe": universe,
                            "weighting": weighting,
                            "portfolio": portfolio["portfolio"],
                            "portfolio_description": portfolio["description"],
                            "monthly_return": portfolio_return(month_data, selected_mask, weighting),
                            "n_stocks": n_selected,
                            "average_mktcap": selected[WEIGHT_COL].mean() if n_selected else np.nan,
                            "average_iv_spread": selected[SIGNAL_COL].mean() if n_selected else np.nan,
                            "excluded_share": excluded_share,
                            "n_excluded": n_excluded,
                        }
                    )

    returns = pd.DataFrame(rows)
    returns = returns.dropna(subset=["monthly_return"]).reset_index(drop=True)
    return returns


def max_drawdown(return_series: pd.Series) -> float:
    """Compute max drawdown from a return series."""
    returns = pd.to_numeric(pd.Series(return_series), errors="coerce").dropna()
    if returns.empty:
        return np.nan
    growth = (1 + returns).cumprod()
    drawdown = growth / growth.cummax() - 1
    return drawdown.min()


def drawdown_details(return_series: pd.Series, month_series: pd.Series) -> dict[str, object]:
    """Compute drawdown and best/worst month details."""
    returns = pd.to_numeric(pd.Series(return_series), errors="coerce")
    months = pd.Series(month_series).astype(str)
    valid = returns.notna()
    returns = returns.loc[valid]
    months = months.loc[valid]
    if returns.empty:
        return {
            "max_drawdown": np.nan,
            "worst_month": "",
            "worst_month_return": np.nan,
            "best_month": "",
            "best_month_return": np.nan,
            "downside_volatility": np.nan,
        }

    downside = returns.where(returns < 0).dropna()
    downside_vol = downside.std(ddof=1) * np.sqrt(12) if len(downside) > 1 else np.nan
    return {
        "max_drawdown": max_drawdown(returns),
        "worst_month": months.loc[returns.idxmin()],
        "worst_month_return": returns.min(),
        "best_month": months.loc[returns.idxmax()],
        "best_month_return": returns.max(),
        "downside_volatility": downside_vol,
    }


def get_pair_returns(
    returns: pd.DataFrame,
    universe: str,
    weighting: str,
    left_portfolio: str,
    right_portfolio: str,
) -> pd.DataFrame:
    """Strictly merge two monthly return series."""
    left = returns.loc[
        (returns["universe"] == universe)
        & (returns["weighting"] == weighting)
        & (returns["portfolio"] == left_portfolio)
    ].copy()
    right = returns.loc[
        (returns["universe"] == universe)
        & (returns["weighting"] == weighting)
        & (returns["portfolio"] == right_portfolio)
    ].copy()
    if left.empty or right.empty:
        raise ValueError(
            f"Missing pair returns for {left_portfolio} and {right_portfolio}, "
            f"universe={universe}, weighting={weighting}"
        )

    merged = left[["signal_month", "return_month", "monthly_return"]].merge(
        right[["signal_month", "return_month", "monthly_return"]],
        on=["signal_month", "return_month"],
        how="inner",
        suffixes=("_left", "_right"),
    )
    if len(merged) != min(len(left), len(right)):
        raise ValueError(
            f"Pair merge lost months for {left_portfolio} and {right_portfolio}, "
            f"universe={universe}, weighting={weighting}"
        )
    return merged.rename(
        columns={
            "monthly_return_left": "left_return",
            "monthly_return_right": "right_return",
        }
    )


def summarize_long_only_returns(returns: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize long-only portfolios and improvements versus the full universe."""
    rows = []
    for (universe, weighting, portfolio), group in returns.groupby(
        ["universe", "weighting", "portfolio"], sort=True
    ):
        summary = summarize_return_series(group["monthly_return"])
        rows.append(
            {
                "universe": universe,
                "weighting": weighting,
                "portfolio": portfolio,
                **summary,
                "max_drawdown": max_drawdown(group["monthly_return"]),
                "average_n_stocks": group["n_stocks"].mean(),
                "average_n_excluded": group["n_excluded"].mean(),
                "average_excluded_share": group["excluded_share"].mean(),
            }
        )

    summary_df = pd.DataFrame(rows)
    full_lookup = summary_df.loc[summary_df["portfolio"] == "full_universe"].set_index(
        ["universe", "weighting"]
    )

    improvement_rows = []
    for idx, row in summary_df.iterrows():
        full = full_lookup.loc[(row["universe"], row["weighting"])]
        summary_df.loc[idx, "return_improvement"] = row["annualized_return"] - full["annualized_return"]
        summary_df.loc[idx, "volatility_change"] = row["annualized_volatility"] - full["annualized_volatility"]
        summary_df.loc[idx, "sharpe_improvement"] = row["sharpe_ratio"] - full["sharpe_ratio"]
        summary_df.loc[idx, "max_drawdown_improvement"] = row["max_drawdown"] - full["max_drawdown"]

        merged = get_pair_returns(
            returns,
            row["universe"],
            row["weighting"],
            row["portfolio"],
            "full_universe",
        )
        diff = merged["left_return"] - merged["right_return"]
        diff_summary = summarize_return_series(diff)
        summary_df.loc[idx, "nw_tstat_of_difference"] = diff_summary["nw_t_stat"]

        if row["portfolio"] in ["exclude_bottom_decile", "exclude_bottom_quintile"]:
            tracking_error = diff.std(ddof=1) * np.sqrt(12)
            improvement_rows.append(
                {
                    "portfolio": f"{row['portfolio']} minus full_universe",
                    "universe": row["universe"],
                    "weighting": row["weighting"],
                    "annualized_difference": diff_summary["annualized_return"],
                    "nw_t_stat": diff_summary["nw_t_stat"],
                    "nw_p_value": diff_summary["nw_p_value"],
                    "average_monthly_difference": diff_summary["mean_monthly_return"],
                    "positive_month_pct": diff_summary["positive_month_pct"],
                    "tracking_error": tracking_error,
                    "information_ratio": diff_summary["annualized_return"] / tracking_error
                    if pd.notna(tracking_error) and tracking_error != 0
                    else np.nan,
                    "max_relative_drawdown": max_drawdown(diff),
                    "n_months": diff_summary["n_months"],
                }
            )

    return summary_df, pd.DataFrame(improvement_rows)


def portfolio_label(universe: str, weighting: str, portfolio: str) -> str:
    """Readable portfolio label for reports and regressions."""
    universe_label = {
        "all": "All",
        "mktcap_100m": "$100M+",
        "mktcap_500m": "$500M+",
        "mktcap_1b": "$1B+",
    }.get(universe, universe)
    portfolio_label_map = {
        "full_universe": "Full Universe",
        "exclude_bottom_decile": "Exclude Bottom Decile",
        "exclude_bottom_quintile": "Exclude Bottom Quintile",
        "top_90_only": "Top 90% Only",
        "top_80_only": "Top 80% Only",
        "exclude_bottom_top_decile": "Exclude Bottom and Top Deciles",
        "exclude_bottom_top_quintile": "Exclude Bottom and Top Quintiles",
    }
    return f"{portfolio_label_map.get(portfolio, portfolio)} {weighting.upper()} {universe_label}"


def run_single_factor_regression(
    regression_df: pd.DataFrame,
    factor_df: pd.DataFrame,
    portfolio: str,
    universe: str,
    weighting: str,
    model: str,
    factor_cols: list[str],
    excess: bool,
) -> dict[str, object]:
    """Run one factor regression using HAC standard errors."""
    data = regression_df.copy()
    data.loc[:, "return_month"] = period_month(data["return_month"])

    factors = factor_df.copy()
    factors.loc[:, "return_month"] = period_month(factors["return_month"])

    merged = data.merge(factors, on="return_month", how="inner")
    merged = merged.dropna(subset=["strategy_return"] + factor_cols)
    if merged.empty:
        raise ValueError(f"No merged regression rows for {portfolio}, {model}")

    if excess:
        if "RF" not in merged.columns:
            raise ValueError("RF column is required for long-only excess returns")
        y = merged["strategy_return"].astype(float) - merged["RF"].astype(float)
    else:
        y = merged["strategy_return"].astype(float)

    x = sm.add_constant(merged[factor_cols].astype(float), has_constant="add")
    result = sm.OLS(y, x).fit(cov_type="HAC", cov_kwds={"maxlags": NW_LAGS})

    output = {
        "portfolio": portfolio_label(universe, weighting, portfolio)
        if not portfolio.startswith("difference_")
        else portfolio,
        "portfolio_key": portfolio,
        "universe": universe,
        "weighting": weighting,
        "model": model,
        "alpha_monthly": result.params["const"],
        "alpha_annualized": result.params["const"] * 12,
        "alpha_tstat": result.tvalues["const"],
        "alpha_pvalue": result.pvalues["const"],
        "r_squared": result.rsquared,
        "n_months": int(result.nobs),
        "first_return_month": merged["return_month"].min(),
        "last_return_month": merged["return_month"].max(),
        "dependent_variable": "portfolio_return_minus_RF" if excess else "return_spread_no_RF_subtraction",
    }
    for factor in factor_cols:
        output[f"beta_{factor}"] = result.params.get(factor, np.nan)
        output[f"tstat_{factor}"] = result.tvalues.get(factor, np.nan)
    return output


def build_regression_inputs(returns: pd.DataFrame) -> pd.DataFrame:
    """Create long-only and difference return series for factor regressions."""
    rows = []
    for universe, weighting, portfolio in SELECTED_FACTOR_PORTFOLIOS:
        selected = returns.loc[
            (returns["universe"] == universe)
            & (returns["weighting"] == weighting)
            & (returns["portfolio"] == portfolio)
        ].copy()
        if selected.empty:
            raise ValueError(f"Missing selected regression portfolio: {universe}, {weighting}, {portfolio}")
        for _, row in selected.iterrows():
            rows.append(
                {
                    "signal_month": row["signal_month"],
                    "return_month": row["return_month"],
                    "universe": universe,
                    "weighting": weighting,
                    "portfolio": portfolio,
                    "strategy_return": row["monthly_return"],
                    "excess": True,
                }
            )

    for universe, weighting, exclusion in DIFFERENCE_REGRESSION_SPECS:
        merged = get_pair_returns(returns, universe, weighting, exclusion, "full_universe")
        key = f"difference_{exclusion}_minus_full_universe"
        for _, row in merged.iterrows():
            rows.append(
                {
                    "signal_month": row["signal_month"],
                    "return_month": row["return_month"],
                    "universe": universe,
                    "weighting": weighting,
                    "portfolio": key,
                    "strategy_return": row["left_return"] - row["right_return"],
                    "excess": False,
                }
            )
    return pd.DataFrame(rows)


def run_factor_regressions(returns: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run CAPM, FF3, FF5, and FF5+MOM regressions."""
    factor_df = load_factor_data(RAW_DATA_DIR)
    regression_inputs = build_regression_inputs(returns)
    rows = []
    for (portfolio, universe, weighting, excess), group in regression_inputs.groupby(
        ["portfolio", "universe", "weighting", "excess"], sort=True
    ):
        for model, factor_cols in FACTOR_MODELS.items():
            rows.append(
                run_single_factor_regression(
                    group,
                    factor_df,
                    portfolio,
                    universe,
                    weighting,
                    model,
                    factor_cols,
                    excess=bool(excess),
                )
            )

    regressions = pd.DataFrame(rows)
    alpha_summary = regressions.loc[regressions["model"] == "FF5_MOM"].copy()
    alpha_summary = alpha_summary[
        [
            "portfolio",
            "portfolio_key",
            "universe",
            "weighting",
            "alpha_annualized",
            "alpha_tstat",
            "alpha_pvalue",
            "r_squared",
            "n_months",
            "dependent_variable",
        ]
    ].copy()
    alpha_summary = alpha_summary.sort_values(["weighting", "universe", "portfolio_key"]).reset_index(drop=True)
    return regressions, alpha_summary


def membership_sets_by_month(panel: pd.DataFrame, universe: str, portfolio: dict[str, str]) -> list[dict[str, object]]:
    """Return selected permno sets by signal month for one portfolio."""
    data = assign_tail_groups(apply_universe_filter(panel, universe))
    rows = []
    for month, group in data.groupby("signal_month", sort=True):
        selected_mask = membership_mask(group, portfolio)
        rows.append(
            {
                "signal_month": month,
                "members": set(group.loc[selected_mask, "permno"].astype(int).tolist()),
            }
        )
    return rows


def turnover_from_sets(monthly_sets: list[dict[str, object]]) -> pd.DataFrame:
    """Compute approximate membership turnover from monthly member sets."""
    rows = []
    previous = None
    for item in monthly_sets:
        current = item["members"]
        if previous is None or len(previous) == 0:
            entries = exits = overlap = np.nan
            one_way = two_way = overlap_share = np.nan
        else:
            entries_set = current - previous
            exits_set = previous - current
            overlap_set = current & previous
            entries = len(entries_set)
            exits = len(exits_set)
            overlap = len(overlap_set)
            one_way = entries / len(previous)
            two_way = (entries + exits) / (2 * len(previous))
            overlap_share = overlap / len(previous)
        rows.append(
            {
                "signal_month": item["signal_month"],
                "n_members": len(current),
                "entries": entries,
                "exits": exits,
                "one_way_turnover": one_way,
                "two_way_turnover": two_way,
                "overlap_share": overlap_share,
            }
        )
        previous = current
    return pd.DataFrame(rows)


def compute_turnover(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute membership-turnover summaries for long-only portfolios."""
    rows = []
    monthly_rows = []
    for universe in UNIVERSE_FILTERS:
        for weighting in ["ew", "vw"]:
            full_turnover = None
            for portfolio in PORTFOLIO_DEFINITIONS:
                sets = membership_sets_by_month(panel, universe, portfolio)
                turnover = turnover_from_sets(sets)
                turnover.loc[:, "universe"] = universe
                turnover.loc[:, "weighting"] = weighting
                turnover.loc[:, "portfolio"] = portfolio["portfolio"]
                monthly_rows.append(turnover)

                if portfolio["portfolio"] == "full_universe":
                    full_turnover = turnover

                incremental = np.nan
                if full_turnover is not None and portfolio["portfolio"] != "full_universe":
                    incremental = turnover["one_way_turnover"].mean() - full_turnover["one_way_turnover"].mean()

                rows.append(
                    {
                        "universe": universe,
                        "weighting": weighting,
                        "portfolio": portfolio["portfolio"],
                        "average_one_way_turnover": turnover["one_way_turnover"].mean(),
                        "average_two_way_turnover": turnover["two_way_turnover"].mean(),
                        "average_overlap_share": turnover["overlap_share"].mean(),
                        "average_n_stocks": turnover["n_members"].mean(),
                        "incremental_one_way_turnover_vs_full": incremental,
                        "turnover_note": "membership-change approximation; value-weight drift/rebalancing not fully modeled",
                    }
                )

    return pd.DataFrame(rows), pd.concat(monthly_rows, ignore_index=True)


def build_cost_sensitivity(summary: pd.DataFrame, turnover_summary: pd.DataFrame) -> pd.DataFrame:
    """Estimate rough net returns under simple membership-turnover cost assumptions."""
    rows = []
    cost_bps_values = [0, 5, 10, 25, 50]
    exclusion_portfolios = ["exclude_bottom_decile", "exclude_bottom_quintile"]
    summary_key = summary.set_index(["universe", "weighting", "portfolio"])
    turnover_key = turnover_summary.set_index(["universe", "weighting", "portfolio"])

    for universe in UNIVERSE_FILTERS:
        for weighting in ["ew", "vw"]:
            full_ann = summary_key.loc[(universe, weighting, "full_universe"), "annualized_return"]
            for portfolio in exclusion_portfolios:
                if (universe, weighting, portfolio) not in summary_key.index:
                    continue
                gross_ann = summary_key.loc[(universe, weighting, portfolio), "annualized_return"]
                gross_improvement = gross_ann - full_ann
                turnover = turnover_key.loc[(universe, weighting, portfolio), "average_two_way_turnover"]
                for cost_bps in cost_bps_values:
                    annual_cost_drag = turnover * cost_bps / 10000 * 12 if pd.notna(turnover) else np.nan
                    net_ann = gross_ann - annual_cost_drag
                    rows.append(
                        {
                            "universe": universe,
                            "weighting": weighting,
                            "portfolio": portfolio,
                            "cost_bps": cost_bps,
                            "gross_annualized_return": gross_ann,
                            "gross_annualized_improvement_vs_universe": gross_improvement,
                            "estimated_annual_cost_drag": annual_cost_drag,
                            "net_annualized_return": net_ann,
                            "net_improvement_vs_universe": net_ann - full_ann,
                            "cost_model_note": "rough two-sided turnover cost; excludes normal benchmark rebalancing and market impact",
                        }
                    )
    return pd.DataFrame(rows)


def build_drawdown_table(returns: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    """Create drawdown/risk table for full and key exclusion portfolios."""
    key_portfolios = ["full_universe", "exclude_bottom_decile", "exclude_bottom_quintile"]
    rows = []
    for (universe, weighting, portfolio), group in returns.groupby(
        ["universe", "weighting", "portfolio"], sort=True
    ):
        if portfolio not in key_portfolios:
            continue
        risk = drawdown_details(group["monthly_return"], group["return_month"])
        summary_row = strict_row(
            summary,
            {
                "universe": universe,
                "weighting": weighting,
                "portfolio": portfolio,
            },
            f"summary {universe}/{weighting}/{portfolio}",
        )
        rows.append(
            {
                "universe": universe,
                "weighting": weighting,
                "portfolio": portfolio,
                **risk,
                "annualized_volatility": summary_row["annualized_volatility"],
                "sharpe_ratio": summary_row["sharpe_ratio"],
            }
        )
    return pd.DataFrame(rows)


def strict_row(df: pd.DataFrame, filters: dict[str, object], label: str) -> pd.Series:
    """Select exactly one row from a dataframe."""
    mask = pd.Series(True, index=df.index)
    for column, value in filters.items():
        if column not in df.columns:
            raise ValueError(f"{label}: missing column {column}")
        mask = mask & (df[column] == value)
    rows = df.loc[mask]
    if len(rows) != 1:
        raise ValueError(f"{label}: expected exactly one row, found {len(rows)} with {filters}")
    return rows.iloc[0]


def cumulative_return_series(group: pd.DataFrame) -> pd.DataFrame:
    """Return cumulative growth of one dollar for a monthly return series."""
    data = group.sort_values("return_month").copy()
    data.loc[:, "growth"] = (1 + data["monthly_return"]).cumprod()
    return data


def plot_cumulative_returns(returns: pd.DataFrame) -> None:
    """Plot cumulative long-only returns for key portfolios."""
    specs = [
        ("all", "ew", "full_universe", "EW all full universe"),
        ("all", "ew", "exclude_bottom_decile", "EW all exclude bottom decile"),
        ("all", "ew", "exclude_bottom_quintile", "EW all exclude bottom quintile"),
        ("mktcap_100m", "ew", "full_universe", "EW $100M+ full universe"),
        ("mktcap_100m", "ew", "exclude_bottom_decile", "EW $100M+ exclude bottom decile"),
    ]
    fig, ax = plt.subplots(figsize=(11, 6))
    for universe, weighting, portfolio, label in specs:
        group = returns.loc[
            (returns["universe"] == universe)
            & (returns["weighting"] == weighting)
            & (returns["portfolio"] == portfolio)
        ].copy()
        group = cumulative_return_series(group)
        ax.plot(group["return_month"].astype(str), group["growth"], label=label)
    ax.set_title("Long-Only IV-Spread Exclusion Screen: Cumulative Returns")
    ax.set_ylabel("Growth of $1")
    ax.set_xlabel("Return month")
    ax.tick_params(axis="x", labelrotation=45)
    ax.xaxis.set_major_locator(plt.MaxNLocator(10))
    ax.legend()
    save_chart(fig, PUBLIC_CHARTS_DIR / "long_only_exclusion_cumulative_returns.png")


def plot_improvement_cumulative(returns: pd.DataFrame) -> None:
    """Plot cumulative return difference versus the full universe."""
    specs = [
        ("all", "ew", "exclude_bottom_decile", "All: exclude bottom decile - universe"),
        ("all", "ew", "exclude_bottom_quintile", "All: exclude bottom quintile - universe"),
        ("mktcap_100m", "ew", "exclude_bottom_decile", "$100M+: exclude bottom decile - universe"),
        ("mktcap_100m", "ew", "exclude_bottom_quintile", "$100M+: exclude bottom quintile - universe"),
    ]
    fig, ax = plt.subplots(figsize=(11, 6))
    for universe, weighting, portfolio, label in specs:
        pair = get_pair_returns(returns, universe, weighting, portfolio, "full_universe")
        pair.loc[:, "cumulative_difference"] = (pair["left_return"] - pair["right_return"]).cumsum()
        ax.plot(pair["return_month"].astype(str), pair["cumulative_difference"], label=label)
    ax.axhline(0, linewidth=1)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.set_title("Cumulative Return Improvement Versus Full Universe")
    ax.set_ylabel("Cumulative return difference")
    ax.set_xlabel("Return month")
    ax.tick_params(axis="x", labelrotation=45)
    ax.xaxis.set_major_locator(plt.MaxNLocator(10))
    ax.legend()
    save_chart(fig, PUBLIC_CHARTS_DIR / "long_only_exclusion_improvement_cumulative.png")


def plot_summary_bars(summary: pd.DataFrame) -> None:
    """Plot annualized return and Sharpe ratio for key long-only portfolios."""
    chart_data = summary.loc[
        (summary["weighting"] == "ew")
        & (summary["universe"].isin(["all", "mktcap_100m"]))
        & (summary["portfolio"].isin(["full_universe", "exclude_bottom_decile", "exclude_bottom_quintile"]))
    ].copy()
    chart_data.loc[:, "label"] = chart_data["universe"] + " / " + chart_data["portfolio"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].bar(chart_data["label"], chart_data["annualized_return"])
    axes[0].yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    axes[0].set_title("Annualized Return")
    axes[0].tick_params(axis="x", labelrotation=45)

    axes[1].bar(chart_data["label"], chart_data["sharpe_ratio"])
    axes[1].set_title("Sharpe Ratio")
    axes[1].tick_params(axis="x", labelrotation=45)
    save_chart(fig, PUBLIC_CHARTS_DIR / "long_only_exclusion_summary_bars.png")


def plot_cost_sensitivity(costs: pd.DataFrame) -> None:
    """Plot net improvement versus the full universe under cost assumptions."""
    chart_data = costs.loc[
        (costs["weighting"] == "ew")
        & (costs["universe"].isin(["all", "mktcap_100m"]))
        & (costs["portfolio"].isin(["exclude_bottom_decile", "exclude_bottom_quintile"]))
    ].copy()
    fig, ax = plt.subplots(figsize=(10, 6))
    for (universe, portfolio), group in chart_data.groupby(["universe", "portfolio"], sort=True):
        label = f"{universe} / {portfolio}"
        ax.plot(group["cost_bps"], group["net_improvement_vs_universe"], marker="o", label=label)
    ax.axhline(0, linewidth=1)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.set_title("Long-Only Exclusion Cost Sensitivity")
    ax.set_xlabel("Membership-turnover cost assumption, bps")
    ax.set_ylabel("Net annualized improvement vs universe")
    ax.legend()
    save_chart(fig, PUBLIC_CHARTS_DIR / "long_only_exclusion_cost_sensitivity.png")


def lookup_row(df: pd.DataFrame, filters: dict[str, object]) -> pd.Series | None:
    """Return one matching row, or None if the row is missing or ambiguous."""
    mask = pd.Series(True, index=df.index)
    for column, value in filters.items():
        if column not in df.columns:
            return None
        mask = mask & (df[column] == value)
    rows = df.loc[mask]
    if len(rows) != 1:
        return None
    return rows.iloc[0]


def create_public_long_only_self_check(
    summary: pd.DataFrame,
    improvement: pd.DataFrame,
    alpha_summary: pd.DataFrame,
    turnover: pd.DataFrame,
    costs: pd.DataFrame,
) -> pd.DataFrame:
    """Check that the public long-only outputs include required rows and metrics."""
    rows: list[dict[str, object]] = []

    def add_self_check_rows(
        category: str,
        public_df: pd.DataFrame,
        filters: dict[str, object],
        metrics: list[str],
        label: str,
    ) -> None:
        public_row = lookup_row(public_df, filters)
        if public_row is None:
            for metric in metrics:
                rows.append(
                    {
                        "category": category,
                        "item": label,
                        "metric": metric,
                        "public_value": np.nan,
                        "status": "MISSING",
                        "message": f"missing or ambiguous row for {filters}",
                    }
                )
            return

        for metric in metrics:
            value = public_row.get(metric, np.nan)
            numeric_value = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
            status = "PASS" if pd.notna(numeric_value) else "REVIEW"
            rows.append(
                {
                    "category": category,
                    "item": label,
                    "metric": metric,
                    "public_value": value,
                    "status": status,
                    "message": "" if status == "PASS" else "metric is missing or nonnumeric",
                }
            )

    performance_specs = [
        ("all", "ew", "full_universe"),
        ("all", "ew", "exclude_bottom_decile"),
        ("all", "ew", "exclude_bottom_quintile"),
        ("mktcap_100m", "ew", "full_universe"),
        ("mktcap_100m", "ew", "exclude_bottom_decile"),
        ("mktcap_100m", "ew", "exclude_bottom_quintile"),
    ]
    for universe, weighting, portfolio in performance_specs:
        filters = {"universe": universe, "weighting": weighting, "portfolio": portfolio}
        add_self_check_rows(
            "performance_summary",
            summary,
            filters,
            ["annualized_return", "sharpe_ratio", "nw_t_stat", "max_drawdown"],
            f"{universe}/{weighting}/{portfolio}",
        )

    improvement_specs = [
        ("all", "ew", "exclude_bottom_decile minus full_universe"),
        ("all", "ew", "exclude_bottom_quintile minus full_universe"),
        ("mktcap_100m", "ew", "exclude_bottom_decile minus full_universe"),
        ("mktcap_100m", "ew", "exclude_bottom_quintile minus full_universe"),
    ]
    for universe, weighting, portfolio in improvement_specs:
        filters = {"universe": universe, "weighting": weighting, "portfolio": portfolio}
        add_self_check_rows(
            "improvement",
            improvement,
            filters,
            ["annualized_difference", "nw_t_stat", "information_ratio"],
            f"{universe}/{weighting}/{portfolio}",
        )

    alpha_specs = [
        ("all", "ew", "difference_exclude_bottom_decile_minus_full_universe"),
        ("all", "ew", "difference_exclude_bottom_quintile_minus_full_universe"),
        ("mktcap_100m", "ew", "difference_exclude_bottom_decile_minus_full_universe"),
        ("mktcap_100m", "ew", "difference_exclude_bottom_quintile_minus_full_universe"),
    ]
    for universe, weighting, portfolio_key in alpha_specs:
        filters = {"universe": universe, "weighting": weighting, "portfolio_key": portfolio_key}
        add_self_check_rows(
            "alpha_summary",
            alpha_summary,
            filters,
            ["alpha_annualized", "alpha_tstat", "r_squared"],
            f"{universe}/{weighting}/{portfolio_key}",
        )

    turnover_specs = [
        ("all", "ew", "exclude_bottom_decile"),
        ("mktcap_100m", "ew", "exclude_bottom_decile"),
    ]
    for universe, weighting, portfolio in turnover_specs:
        filters = {"universe": universe, "weighting": weighting, "portfolio": portfolio}
        add_self_check_rows(
            "turnover",
            turnover,
            filters,
            ["average_one_way_turnover", "average_two_way_turnover", "average_overlap_share"],
            f"{universe}/{weighting}/{portfolio}",
        )

    cost_specs = [
        ("all", "ew", "exclude_bottom_decile", 25),
        ("all", "ew", "exclude_bottom_decile", 50),
        ("mktcap_100m", "ew", "exclude_bottom_decile", 25),
        ("mktcap_100m", "ew", "exclude_bottom_decile", 50),
    ]
    for universe, weighting, portfolio, cost_bps in cost_specs:
        filters = {
            "universe": universe,
            "weighting": weighting,
            "portfolio": portfolio,
            "cost_bps": cost_bps,
        }
        add_self_check_rows(
            "cost_sensitivity",
            costs,
            filters,
            ["net_improvement_vs_universe", "estimated_annual_cost_drag"],
            f"{universe}/{weighting}/{portfolio}/{cost_bps}bps",
        )

    comparison = pd.DataFrame(rows)
    save_table(comparison, PUBLIC_TABLES_DIR / "public_long_only_exclusion_comparison.csv")
    return comparison


def build_headline_table(
    summary: pd.DataFrame,
    improvement: pd.DataFrame,
    alpha_summary: pd.DataFrame,
    turnover: pd.DataFrame,
    costs: pd.DataFrame,
) -> pd.DataFrame:
    """Create a compact headline table for the public report."""
    rows = []
    for universe in ["all", "mktcap_100m"]:
        full = strict_row(summary, {"universe": universe, "weighting": "ew", "portfolio": "full_universe"}, universe)
        decile = strict_row(
            summary,
            {"universe": universe, "weighting": "ew", "portfolio": "exclude_bottom_decile"},
            f"{universe} decile",
        )
        improvement_row = strict_row(
            improvement,
            {
                "universe": universe,
                "weighting": "ew",
                "portfolio": "exclude_bottom_decile minus full_universe",
            },
            f"{universe} improvement",
        )
        alpha = strict_row(
            alpha_summary,
            {
                "universe": universe,
                "weighting": "ew",
                "portfolio_key": "difference_exclude_bottom_decile_minus_full_universe",
            },
            f"{universe} alpha",
        )
        turnover_row = strict_row(
            turnover,
            {"universe": universe, "weighting": "ew", "portfolio": "exclude_bottom_decile"},
            f"{universe} turnover",
        )
        cost_25 = strict_row(
            costs,
            {
                "universe": universe,
                "weighting": "ew",
                "portfolio": "exclude_bottom_decile",
                "cost_bps": 25,
            },
            f"{universe} 25bps cost",
        )
        rows.append(
            {
                "universe": universe,
                "weighting": "ew",
                "full_universe_annualized_return": full["annualized_return"],
                "full_universe_sharpe": full["sharpe_ratio"],
                "exclude_bottom_decile_annualized_return": decile["annualized_return"],
                "exclude_bottom_decile_sharpe": decile["sharpe_ratio"],
                "improvement_annualized_return": improvement_row["annualized_difference"],
                "improvement_nw_t_stat": improvement_row["nw_t_stat"],
                "ff5_mom_difference_alpha": alpha["alpha_annualized"],
                "ff5_mom_difference_alpha_tstat": alpha["alpha_tstat"],
                "average_one_way_turnover": turnover_row["average_one_way_turnover"],
                "net_improvement_25bps": cost_25["net_improvement_vs_universe"],
            }
        )

    headline = pd.DataFrame(rows)
    save_table(headline, PUBLIC_TABLES_DIR / "public_long_only_exclusion_headline.csv")
    return headline


def pct(value: object) -> str:
    """Format a decimal return as a percentage string."""
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(number):
        return "NA"
    return f"{number:.2%}"


def num(value: object) -> str:
    """Format a numeric value."""
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(number):
        return "NA"
    return f"{number:.2f}"


def markdown_table(df: pd.DataFrame) -> str:
    """Create a small markdown table without optional pandas dependencies."""
    columns = list(df.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in df.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if pd.isna(value):
                values.append("")
            elif isinstance(value, float):
                values.append(f"{value:.6g}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(
    summary: pd.DataFrame,
    improvement: pd.DataFrame,
    alpha_summary: pd.DataFrame,
    turnover: pd.DataFrame,
    costs: pd.DataFrame,
    drawdowns: pd.DataFrame,
) -> Path:
    """Write the public markdown interpretation report."""
    full_all = strict_row(summary, {"universe": "all", "weighting": "ew", "portfolio": "full_universe"}, "full all")
    dec_all = strict_row(
        summary,
        {"universe": "all", "weighting": "ew", "portfolio": "exclude_bottom_decile"},
        "decile all",
    )
    quin_all = strict_row(
        summary,
        {"universe": "all", "weighting": "ew", "portfolio": "exclude_bottom_quintile"},
        "quintile all",
    )
    imp_dec_all = strict_row(
        improvement,
        {
            "universe": "all",
            "weighting": "ew",
            "portfolio": "exclude_bottom_decile minus full_universe",
        },
        "decile all improvement",
    )
    imp_dec_100m = strict_row(
        improvement,
        {
            "universe": "mktcap_100m",
            "weighting": "ew",
            "portfolio": "exclude_bottom_decile minus full_universe",
        },
        "decile 100m improvement",
    )
    turnover_all = strict_row(
        turnover,
        {"universe": "all", "weighting": "ew", "portfolio": "exclude_bottom_decile"},
        "turnover all",
    )
    turnover_100m = strict_row(
        turnover,
        {"universe": "mktcap_100m", "weighting": "ew", "portfolio": "exclude_bottom_decile"},
        "turnover 100m",
    )
    diff_alpha_all = strict_row(
        alpha_summary,
        {
            "universe": "all",
            "weighting": "ew",
            "portfolio_key": "difference_exclude_bottom_decile_minus_full_universe",
        },
        "alpha all",
    )
    diff_alpha_100m = strict_row(
        alpha_summary,
        {
            "universe": "mktcap_100m",
            "weighting": "ew",
            "portfolio_key": "difference_exclude_bottom_decile_minus_full_universe",
        },
        "alpha 100m",
    )
    cost_all_25 = strict_row(
        costs,
        {"universe": "all", "weighting": "ew", "portfolio": "exclude_bottom_decile", "cost_bps": 25},
        "cost all 25",
    )
    cost_all_50 = strict_row(
        costs,
        {"universe": "all", "weighting": "ew", "portfolio": "exclude_bottom_decile", "cost_bps": 50},
        "cost all 50",
    )
    drawdown_full = strict_row(
        drawdowns,
        {"universe": "all", "weighting": "ew", "portfolio": "full_universe"},
        "drawdown full",
    )
    drawdown_dec = strict_row(
        drawdowns,
        {"universe": "all", "weighting": "ew", "portfolio": "exclude_bottom_decile"},
        "drawdown decile",
    )

    lines = [
        "# Long-Only IV-Spread Exclusion Screen: Public 2010-2023 Pipeline",
        "",
        "## Executive Summary",
        "",
        (
            "This analysis tests whether the IV-spread signal is useful as a long-only "
            "negative-selection screen. The screen keeps the optionable universe but removes "
            "the lowest call-minus-put implied-volatility names before forming the long-only portfolio."
        ),
        "",
        "## Main Performance",
        "",
        (
            f"The EW all-stock full universe earns {pct(full_all['annualized_return'])} annualized. "
            f"Excluding the bottom decile earns {pct(dec_all['annualized_return'])}, an annualized "
            f"improvement of {pct(imp_dec_all['annualized_difference'])} with Newey-West t-stat "
            f"{num(imp_dec_all['nw_t_stat'])}. Excluding the bottom quintile earns "
            f"{pct(quin_all['annualized_return'])}."
        ),
        "",
        "## Size-Filtered Robustness",
        "",
        (
            f"For the $100M+ universe, excluding the bottom decile improves returns by "
            f"{pct(imp_dec_100m['annualized_difference'])} with Newey-West t-stat "
            f"{num(imp_dec_100m['nw_t_stat'])}."
        ),
        "",
        "## Sharpe, Drawdowns, and Factor Alpha",
        "",
        (
            f"The EW all full-universe Sharpe is {num(full_all['sharpe_ratio'])}; the bottom-decile "
            f"exclusion Sharpe is {num(dec_all['sharpe_ratio'])}. Max drawdown changes from "
            f"{pct(drawdown_full['max_drawdown'])} to {pct(drawdown_dec['max_drawdown'])}."
        ),
        "",
        (
            f"The exclusion-minus-universe difference has FF5+Momentum alpha of "
            f"{pct(diff_alpha_all['alpha_annualized'])} with t-stat {num(diff_alpha_all['alpha_tstat'])}. "
            f"For the $100M+ universe, the corresponding alpha is "
            f"{pct(diff_alpha_100m['alpha_annualized'])} with t-stat {num(diff_alpha_100m['alpha_tstat'])}."
        ),
        "",
        "## Turnover and Cost Sensitivity",
        "",
        (
            f"The EW all bottom-decile exclusion portfolio has approximate one-way membership "
            f"turnover of {pct(turnover_all['average_one_way_turnover'])}. The $100M+ version has "
            f"one-way turnover of {pct(turnover_100m['average_one_way_turnover'])}."
        ),
        "",
        (
            f"At a 25 bps membership-turnover cost assumption, the all-stock net improvement is "
            f"{pct(cost_all_25['net_improvement_vs_universe'])}. At 50 bps it is "
            f"{pct(cost_all_50['net_improvement_vs_universe'])}. This is a rough sensitivity check, "
            "not a full implementation-cost model."
        ),
        "",
        "## Interpretation",
        "",
        (
            "The signal is more useful as a negative-selection screen than as a standalone long-only "
            "alpha portfolio. The long-short bottom-tail spread remains the cleanest statistical test; "
            "the long-only exclusion version is a more practical implementation framing."
        ),
        "",
    ]

    path = PUBLIC_TABLES_DIR / "long_only_exclusion_summary_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved {path}")
    return path


def write_docs_report(comparison: pd.DataFrame, headline: pd.DataFrame) -> Path:
    """Write the public pipeline step report."""
    DOC_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    counts = comparison["status"].value_counts().to_dict()
    src_modified = "No src files were modified in this step."
    lines = [
        "# Public Pipeline Step 5: Long-Only Exclusion",
        "",
        "## Files Created",
        "",
        "- scripts/public/06_run_long_only_exclusion.py",
        "- outputs/public_2010_2023/tables/long_only_exclusion_returns_2010_2023.csv",
        "- outputs/public_2010_2023/tables/long_only_exclusion_summary_2010_2023.csv",
        "- outputs/public_2010_2023/tables/long_only_exclusion_improvement_2010_2023.csv",
        "- outputs/public_2010_2023/tables/long_only_exclusion_factor_regressions_2010_2023.csv",
        "- outputs/public_2010_2023/tables/long_only_exclusion_alpha_summary_2010_2023.csv",
        "- outputs/public_2010_2023/tables/long_only_exclusion_turnover_2010_2023.csv",
        "- outputs/public_2010_2023/tables/long_only_exclusion_turnover_monthly_2010_2023.csv",
        "- outputs/public_2010_2023/tables/long_only_exclusion_cost_sensitivity_2010_2023.csv",
        "- outputs/public_2010_2023/tables/long_only_exclusion_drawdown_2010_2023.csv",
        "- outputs/public_2010_2023/tables/public_long_only_exclusion_comparison.csv",
        "- outputs/public_2010_2023/tables/public_long_only_exclusion_headline.csv",
        "- outputs/public_2010_2023/tables/long_only_exclusion_summary_report.md",
        "- outputs/public_2010_2023/charts/long_only_exclusion_cumulative_returns.png",
        "- outputs/public_2010_2023/charts/long_only_exclusion_improvement_cumulative.png",
        "- outputs/public_2010_2023/charts/long_only_exclusion_summary_bars.png",
        "- outputs/public_2010_2023/charts/long_only_exclusion_cost_sensitivity.png",
        "",
        "## Source Changes",
        "",
        src_modified,
        "",
        "## Standalone Status",
        "",
        "The public script implements the analysis directly and uses reusable src utilities. It does not execute or import legacy development files.",
        "",
        "## Public Inputs Used",
        "",
        "- data/processed/monthly_signal_panel_2010_2023.parquet",
        "- data/raw/factors/ local Fama-French factor files",
        "- public output tables generated by this script",
        "",
        "## Public Self-Check Summary",
        "",
        f"- PASS: {counts.get('PASS', 0)}",
        f"- REVIEW: {counts.get('REVIEW', 0)}",
        f"- MISSING: {counts.get('MISSING', 0)}",
        "",
        "## Headline Rows",
        "",
        markdown_table(headline),
        "",
        "## Discrepancies",
        "",
        "No required public rows require review." if counts.get("REVIEW", 0) == 0 and counts.get("MISSING", 0) == 0 else "Review self-check rows marked REVIEW or MISSING.",
        "",
        "## GitHub Readiness",
        "",
        "This script is safe for the public pipeline. It depends on processed panel and local factor files, but it does not require WRDS access.",
        "",
        "## Recommended Next Step",
        "",
        "Build the standalone public robustness script that covers alpha anatomy, signal extensions, holding-period tests, outlier robustness, and Table 1 verification.",
        "",
    ]
    DOC_REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved {DOC_REPORT_PATH}")
    return DOC_REPORT_PATH


def print_headlines(
    summary: pd.DataFrame,
    improvement: pd.DataFrame,
    alpha_summary: pd.DataFrame,
    turnover: pd.DataFrame,
    costs: pd.DataFrame,
    comparison: pd.DataFrame,
    report_path: Path,
) -> None:
    """Print concise terminal output."""
    headline = summary.loc[
        (summary["weighting"] == "ew")
        & (summary["universe"].isin(["all", "mktcap_100m"]))
        & (summary["portfolio"].isin(["full_universe", "exclude_bottom_decile", "exclude_bottom_quintile"]))
    ][
        [
            "universe",
            "portfolio",
            "annualized_return",
            "annualized_volatility",
            "sharpe_ratio",
            "nw_t_stat",
            "max_drawdown",
            "n_months",
        ]
    ]
    print("\nHeadline performance comparison:")
    print(headline.to_string(index=False))

    imp = improvement.loc[
        (improvement["weighting"] == "ew")
        & (improvement["universe"].isin(["all", "mktcap_100m", "mktcap_500m", "mktcap_1b"]))
    ][
        [
            "universe",
            "portfolio",
            "annualized_difference",
            "nw_t_stat",
            "positive_month_pct",
            "information_ratio",
            "n_months",
        ]
    ]
    print("\nImprovement versus full universe:")
    print(imp.to_string(index=False))

    alpha = alpha_summary.loc[
        (alpha_summary["weighting"] == "ew") & (alpha_summary["universe"].isin(["all", "mktcap_100m"]))
    ][["portfolio", "portfolio_key", "universe", "alpha_annualized", "alpha_tstat", "r_squared", "n_months"]]
    print("\nFF5+MOM alpha summary:")
    print(alpha.to_string(index=False))

    turn = turnover.loc[
        (turnover["weighting"] == "ew")
        & (turnover["universe"].isin(["all", "mktcap_100m"]))
        & (turnover["portfolio"].isin(["full_universe", "exclude_bottom_decile", "exclude_bottom_quintile"]))
    ][
        [
            "universe",
            "portfolio",
            "average_one_way_turnover",
            "average_two_way_turnover",
            "average_overlap_share",
            "average_n_stocks",
        ]
    ]
    print("\nTurnover comparison:")
    print(turn.to_string(index=False))

    cost_headline = costs.loc[
        (costs["weighting"] == "ew")
        & (costs["universe"].isin(["all", "mktcap_100m"]))
        & (costs["portfolio"] == "exclude_bottom_decile")
        & (costs["cost_bps"].isin([0, 10, 25, 50]))
    ][["universe", "portfolio", "cost_bps", "net_improvement_vs_universe", "estimated_annual_cost_drag"]]
    print("\nTransaction-cost sensitivity headline:")
    print(cost_headline.to_string(index=False))

    counts = comparison["status"].value_counts().to_dict()
    print("\nPublic self-check totals:")
    print(f"PASS={counts.get('PASS', 0)} REVIEW={counts.get('REVIEW', 0)} MISSING={counts.get('MISSING', 0)}")

    print("\nSuggested final interpretation:")
    print(
        "The IV-spread signal is most useful as a negative-selection screen. "
        "The strongest statistical evidence comes from universe-minus-bottom-tail portfolios, "
        "while the most practical use may be to avoid the lowest IV-spread names in a long-only portfolio."
    )
    print(f"\nMarkdown report: {report_path}")


def main() -> None:
    """Run the public long-only exclusion-screen analysis."""
    print_header("Public Long-Only Exclusion Analysis")
    PUBLIC_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    PUBLIC_CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    panel = load_panel()

    print_header("Building Long-Only Returns")
    returns = build_long_only_returns(panel)
    summary, improvement = summarize_long_only_returns(returns)

    print_header("Running Factor Regressions")
    regressions, alpha_summary = run_factor_regressions(returns)

    print_header("Computing Turnover, Costs, and Drawdowns")
    turnover, monthly_turnover = compute_turnover(panel)
    costs = build_cost_sensitivity(summary, turnover)
    drawdowns = build_drawdown_table(returns, summary)

    print_header("Saving Public Outputs")
    save_table(returns, PUBLIC_TABLES_DIR / "long_only_exclusion_returns_2010_2023.csv")
    save_table(summary, PUBLIC_TABLES_DIR / "long_only_exclusion_summary_2010_2023.csv")
    save_table(improvement, PUBLIC_TABLES_DIR / "long_only_exclusion_improvement_2010_2023.csv")
    save_table(regressions, PUBLIC_TABLES_DIR / "long_only_exclusion_factor_regressions_2010_2023.csv")
    save_table(alpha_summary, PUBLIC_TABLES_DIR / "long_only_exclusion_alpha_summary_2010_2023.csv")
    save_table(turnover, PUBLIC_TABLES_DIR / "long_only_exclusion_turnover_2010_2023.csv")
    save_table(monthly_turnover, PUBLIC_TABLES_DIR / "long_only_exclusion_turnover_monthly_2010_2023.csv")
    save_table(costs, PUBLIC_TABLES_DIR / "long_only_exclusion_cost_sensitivity_2010_2023.csv")
    save_table(drawdowns, PUBLIC_TABLES_DIR / "long_only_exclusion_drawdown_2010_2023.csv")

    print_header("Creating Charts")
    plot_cumulative_returns(returns)
    plot_improvement_cumulative(returns)
    plot_summary_bars(summary)
    plot_cost_sensitivity(costs)

    print_header("Running Public Self-Check")
    comparison = create_public_long_only_self_check(summary, improvement, alpha_summary, turnover, costs)
    headline = build_headline_table(summary, improvement, alpha_summary, turnover, costs)

    print_header("Writing Reports")
    report_path = write_report(summary, improvement, alpha_summary, turnover, costs, drawdowns)
    write_docs_report(comparison, headline)

    print_header("Terminal Summary")
    print("scripts/public/06_run_long_only_exclusion.py created: yes")
    print("src files modified: no")
    print("legacy files called or imported: no")
    print("run status: completed")
    print(f"long-only output directory: {PUBLIC_TABLES_DIR}")
    print_headlines(summary, improvement, alpha_summary, turnover, costs, comparison, report_path)
    print("\nRecommended next step: create scripts/public/07_run_robustness_checks.py.")


if __name__ == "__main__":
    main()
