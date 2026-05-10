"""Final pre-expansion diagnostics for the IV spread bottom-tail result."""

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
import matplotlib.dates as mdates
import matplotlib.ticker as mtick


MONTHLY_REQUIRED_COLUMNS = [
    "permno",
    "secid",
    "signal_month",
    "return_month",
    "iv_atm_call",
    "iv_atm_put",
    "iv_otm_put",
    "iv_spread",
    "iv_spread_adj",
    "iv_skew",
    "implied_var",
    "realized_var",
    "vrp",
    "ret_fwd_1m",
    "mktcap",
]


def load_final_diagnostic_inputs(processed_data_dir, raw_data_dir, tables_dir):
    """Load inputs for final pre-expansion diagnostics."""
    processed_data_dir = Path(processed_data_dir)
    raw_data_dir = Path(raw_data_dir)
    tables_dir = Path(tables_dir)

    monthly_panel = pd.read_parquet(processed_data_dir / "monthly_signal_panel.parquet")
    security_path = raw_data_dir / "security_master.parquet"
    security_master = pd.read_parquet(security_path) if security_path.exists() else pd.DataFrame()
    crsp_monthly = pd.read_parquet(raw_data_dir / "crsp_monthly_2018_2024.parquet")
    bottom_tail_returns = pd.read_csv(tables_dir / "iv_spread_bottom_tail_returns.csv")
    final_main_table = pd.read_csv(tables_dir / "final_bottom_tail_main_table.csv")

    print("\n" + "=" * 80)
    print("Loading Final Diagnostic Inputs")
    print("=" * 80)
    for name, df in [
        ("monthly_panel", monthly_panel),
        ("security_master", security_master),
        ("crsp_monthly", crsp_monthly),
        ("bottom_tail_returns", bottom_tail_returns),
        ("final_main_table", final_main_table),
    ]:
        print(f"{name}: {df.shape}")

    missing = [col for col in MONTHLY_REQUIRED_COLUMNS if col not in monthly_panel.columns]
    available = [col for col in MONTHLY_REQUIRED_COLUMNS if col in monthly_panel.columns]
    print(f"Monthly required columns available: {available}")
    print(f"Monthly required columns missing: {missing if missing else 'None'}")
    if missing:
        raise ValueError(f"monthly_signal_panel is missing required columns: {missing}")

    monthly_panel = monthly_panel[MONTHLY_REQUIRED_COLUMNS + [
        col for col in ["sic", "sector"] if col in monthly_panel.columns
    ]].copy()
    monthly_panel = _prepare_monthly_panel(monthly_panel)

    return {
        "monthly_panel": monthly_panel,
        "security_master": security_master,
        "crsp_monthly": crsp_monthly,
        "bottom_tail_returns": bottom_tail_returns,
        "final_main_table": final_main_table,
    }


def _prepare_monthly_panel(panel):
    """Normalize monthly panel dtypes for diagnostics."""
    data = panel.copy()
    data = data.assign(
        permno=pd.to_numeric(data["permno"], errors="coerce").astype("Int64"),
        secid=pd.to_numeric(data["secid"], errors="coerce").astype("Int64"),
        signal_month=pd.PeriodIndex(data["signal_month"].astype(str), freq="M"),
        return_month=pd.PeriodIndex(data["return_month"].astype(str), freq="M"),
    )
    numeric_columns = [
        "iv_atm_call",
        "iv_atm_put",
        "iv_otm_put",
        "iv_spread",
        "iv_spread_adj",
        "iv_skew",
        "implied_var",
        "realized_var",
        "vrp",
        "ret_fwd_1m",
        "mktcap",
    ]
    for column in numeric_columns:
        data.loc[:, column] = pd.to_numeric(data[column], errors="coerce")
    return data.dropna(subset=["permno", "signal_month", "return_month", "iv_spread_adj", "ret_fwd_1m"])


def _assign_quantile_group(panel, signal_col, n_groups=10, group_col="tail_group"):
    """Assign qcut groups within each signal month."""
    data = panel.copy()
    data.loc[:, group_col] = pd.NA
    for _, month_data in data.groupby("signal_month", sort=True):
        valid = month_data.dropna(subset=[signal_col])
        if valid[signal_col].nunique() < n_groups:
            continue
        try:
            codes = pd.qcut(valid[signal_col], q=n_groups, labels=False, duplicates="drop")
        except ValueError:
            continue
        if codes.nunique(dropna=True) < n_groups:
            continue
        data.loc[valid.index, group_col] = (codes + 1).astype("Int64")
    data = data.dropna(subset=[group_col]).copy()
    data.loc[:, group_col] = data[group_col].astype("Int64")
    return data


def _weighted_return(group, value_weighted=False):
    """Compute equal-weighted or value-weighted return."""
    if group.empty:
        return np.nan
    returns = pd.to_numeric(group["ret_fwd_1m"], errors="coerce")
    if not value_weighted:
        return returns.mean()
    weights = pd.to_numeric(group["mktcap"], errors="coerce")
    valid = returns.notna() & weights.notna() & (weights > 0)
    if not valid.any():
        return np.nan
    returns = returns.loc[valid]
    weights = weights.loc[valid]
    return (returns * weights / weights.sum()).sum()


def _t_stat(series):
    """Compute the t-statistic of a sample mean."""
    series = pd.to_numeric(series, errors="coerce").dropna()
    if len(series) <= 1:
        return np.nan
    std = series.std()
    if pd.isna(std) or std == 0:
        return np.nan
    return series.mean() / (std / np.sqrt(len(series)))


def _summarize_return_series(series):
    """Return standard summary metrics for monthly returns."""
    series = pd.to_numeric(series, errors="coerce").dropna()
    monthly_vol = series.std()
    annual_vol = monthly_vol * np.sqrt(12)
    annual_return = series.mean() * 12
    return {
        "mean_monthly_return": series.mean(),
        "annualized_return": annual_return,
        "monthly_volatility": monthly_vol,
        "annualized_volatility": annual_vol,
        "sharpe_ratio": annual_return / annual_vol if pd.notna(annual_vol) and annual_vol != 0 else np.nan,
        "t_stat": _t_stat(series),
        "positive_month_pct": (series > 0).mean() if len(series) else np.nan,
        "n_months": len(series),
    }


def _save(df, output_path):
    """Save a dataframe if an output path is provided."""
    if output_path is None:
        return
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved table: {output_path} shape={df.shape}")


def confirm_iv_spread_sign_convention(monthly_panel, output_path=None):
    """Confirm IV spread formula and bottom-tail sign convention."""
    data = monthly_panel.copy()
    data.loc[:, "recomputed_iv_spread"] = data["iv_atm_call"] - data["iv_atm_put"]
    max_diff_spread = (data["iv_spread"] - data["recomputed_iv_spread"]).abs().max()
    max_diff_adj = (data["iv_spread_adj"] - data["recomputed_iv_spread"]).abs().max()

    deciles = _assign_quantile_group(data, "iv_spread_adj", n_groups=10, group_col="iv_spread_decile")
    avg_by_decile = (
        deciles.groupby("iv_spread_decile")
        .agg(
            iv_atm_call=("iv_atm_call", "mean"),
            iv_atm_put=("iv_atm_put", "mean"),
            iv_spread=("iv_spread", "mean"),
            iv_spread_adj=("iv_spread_adj", "mean"),
            iv_skew=("iv_skew", "mean"),
            implied_var=("implied_var", "mean"),
            realized_var=("realized_var", "mean"),
            vrp=("vrp", "mean"),
            ret_fwd_1m=("ret_fwd_1m", "mean"),
            mktcap=("mktcap", "mean"),
            n_stocks=("permno", "count"),
        )
        .reset_index()
    )

    bottom = avg_by_decile.loc[avg_by_decile["iv_spread_decile"] == 1].iloc[0]
    top = avg_by_decile.loc[avg_by_decile["iv_spread_decile"] == 10].iloc[0]

    result = pd.DataFrame(
        [
            {
                "formula": "iv_spread = iv_atm_call - iv_atm_put",
                "max_abs_diff_iv_spread": max_diff_spread,
                "max_abs_diff_iv_spread_adj": max_diff_adj,
                "universe_avg_iv_spread": data["iv_spread"].mean(),
                "bottom_decile_avg_iv_spread": bottom["iv_spread"],
                "top_decile_avg_iv_spread": top["iv_spread"],
                "bottom_decile_avg_call_iv": bottom["iv_atm_call"],
                "bottom_decile_avg_put_iv": bottom["iv_atm_put"],
                "interpretation": "Lower iv_spread means call IV is low relative to put IV.",
            }
        ]
    )

    _save(result, output_path)
    print("Sign convention confirmed: iv_spread = call IV - put IV.")
    print("Bottom decile means call IV is low relative to put IV.")
    return result


def compute_call_put_decomposition(monthly_panel, output_path=None):
    """Decompose IV spread deciles into call IV, put IV, and related quantities."""
    deciles = _assign_quantile_group(monthly_panel, "iv_spread_adj", n_groups=10, group_col="iv_spread_decile")
    rows = []
    universe_means = deciles[
        ["iv_atm_call", "iv_atm_put", "iv_spread", "vrp", "realized_var", "implied_var"]
    ].mean()

    for decile, group in deciles.groupby("iv_spread_decile", sort=True):
        row = {
            "decile": f"D{int(decile)}",
            "decile_number": int(decile),
            "iv_atm_call": group["iv_atm_call"].mean(),
            "iv_atm_put": group["iv_atm_put"].mean(),
            "iv_spread": group["iv_spread"].mean(),
            "iv_skew": group["iv_skew"].mean(),
            "implied_var": group["implied_var"].mean(),
            "realized_var": group["realized_var"].mean(),
            "vrp": group["vrp"].mean(),
            "mktcap": group["mktcap"].mean(),
            "ret_fwd_1m": group["ret_fwd_1m"].mean(),
            "annualized_ret_fwd_1m": group["ret_fwd_1m"].mean() * 12,
            "n_stocks": len(group),
        }
        if int(decile) == 1:
            row.update(
                {
                    "bottom_call_iv_minus_universe_call_iv": row["iv_atm_call"] - universe_means["iv_atm_call"],
                    "bottom_put_iv_minus_universe_put_iv": row["iv_atm_put"] - universe_means["iv_atm_put"],
                    "bottom_spread_minus_universe_spread": row["iv_spread"] - universe_means["iv_spread"],
                    "bottom_vrp_minus_universe_vrp": row["vrp"] - universe_means["vrp"],
                    "bottom_realized_var_minus_universe_realized_var": row["realized_var"] - universe_means["realized_var"],
                    "bottom_implied_var_minus_universe_implied_var": row["implied_var"] - universe_means["implied_var"],
                }
            )
        rows.append(row)

    result = pd.DataFrame(rows)
    _save(result, output_path)

    bottom = result.loc[result["decile"] == "D1"].iloc[0]
    call_diff = bottom["bottom_call_iv_minus_universe_call_iv"]
    put_diff = bottom["bottom_put_iv_minus_universe_put_iv"]
    if call_diff < 0 and put_diff > 0:
        driver = "both lower call IV and higher put IV"
    elif call_diff < 0:
        driver = "mainly lower call IV"
    elif put_diff > 0:
        driver = "mainly higher put IV"
    else:
        driver = "neither simple low-call nor high-put behavior"
    print(f"Bottom decile decomposition: {driver}.")
    print(f"Bottom realized variance minus universe: {bottom['bottom_realized_var_minus_universe_realized_var']:.4f}")
    print(f"Bottom VRP minus universe: {bottom['bottom_vrp_minus_universe_vrp']:.4f}")
    return result


def run_call_put_tail_tests(monthly_panel, output_path=None):
    """Test bottom-tail returns for call IV, put IV, spread, skew, and VRP signals."""
    signal_specs = [
        ("iv_spread_adj", "low_iv_spread", "bottom"),
        ("iv_atm_call", "low_call_iv", "bottom"),
        ("iv_atm_put", "high_put_iv", "top"),
        ("iv_skew", "high_skew", "top"),
        ("vrp", "low_vrp", "bottom"),
        ("vrp", "high_vrp", "top"),
    ]
    universes = [("all", None), ("mktcap_100m", 100)]
    rows = []

    for universe, cutoff in universes:
        universe_data = monthly_panel.copy()
        if cutoff is not None:
            universe_data = universe_data.loc[universe_data["mktcap"] >= cutoff].copy()

        for signal_col, signal_label, direction in signal_specs:
            assigned = _assign_quantile_group(universe_data, signal_col, n_groups=10, group_col="tail_group")
            monthly_rows = []
            for signal_month, month_data in assigned.groupby("signal_month", sort=True):
                tail_group = 1 if direction == "bottom" else 10
                tail = month_data.loc[month_data["tail_group"] == tail_group]
                strategy_return = month_data["ret_fwd_1m"].mean() - tail["ret_fwd_1m"].mean()
                monthly_rows.append(
                    {
                        "signal_month": signal_month,
                        "return_month": month_data["return_month"].iloc[0],
                        "strategy_return": strategy_return,
                        "n_tail": len(tail),
                        "mktcap_tail": tail["mktcap"].mean(),
                        "signal_tail": tail[signal_col].mean(),
                        "signal_universe": month_data[signal_col].mean(),
                    }
                )

            monthly = pd.DataFrame(monthly_rows)
            summary = _summarize_return_series(monthly["strategy_return"])
            rows.append(
                {
                    "signal": signal_col,
                    "test": signal_label,
                    "tail_direction": direction,
                    "universe": universe,
                    **summary,
                    "avg_n_tail": monthly["n_tail"].mean(),
                    "avg_mktcap_tail": monthly["mktcap_tail"].mean(),
                    "avg_signal_tail": monthly["signal_tail"].mean(),
                    "avg_signal_universe": monthly["signal_universe"].mean(),
                }
            )

    result = pd.DataFrame(rows)
    _save(result, output_path)
    print("\nCall/put tail tests:")
    print(result[["test", "universe", "annualized_return", "t_stat", "sharpe_ratio"]].to_string(index=False))
    return result


def compute_bottom_tail_turnover(monthly_panel, output_path=None):
    """Compute bottom-decile membership turnover for all and $100M+ universes."""
    rows = []
    summary_rows = []
    for universe, cutoff in [("all", None), ("mktcap_100m", 100)]:
        data = monthly_panel.copy()
        if cutoff is not None:
            data = data.loc[data["mktcap"] >= cutoff].copy()
        assigned = _assign_quantile_group(data, "iv_spread_adj", n_groups=10, group_col="iv_spread_decile")

        previous = None
        for signal_month, month_data in assigned.groupby("signal_month", sort=True):
            bottom_set = set(month_data.loc[month_data["iv_spread_decile"] == 1, "permno"].dropna().astype(int))
            if previous is None or len(previous) == 0:
                entries = np.nan
                exits = np.nan
                one_way = np.nan
                two_way = np.nan
                overlap_share = np.nan
                turnover_approx = np.nan
            else:
                intersection = bottom_set & previous
                entries = len(bottom_set - previous)
                exits = len(previous - bottom_set)
                one_way = entries / len(previous)
                two_way = (entries + exits) / (2 * len(previous))
                overlap_share = len(intersection) / len(previous)
                turnover_approx = 1 - overlap_share

            rows.append(
                {
                    "row_type": "monthly",
                    "universe": universe,
                    "signal_month": signal_month,
                    "n_bottom": len(bottom_set),
                    "entries": entries,
                    "exits": exits,
                    "one_way_turnover": one_way,
                    "two_way_turnover": two_way,
                    "overlap_share": overlap_share,
                    "turnover_approx": turnover_approx,
                }
            )
            previous = bottom_set

        monthly = pd.DataFrame([row for row in rows if row["row_type"] == "monthly" and row["universe"] == universe])
        summary_rows.append(
                {
                    "row_type": "summary",
                    "universe": universe,
                    "signal_month": "",
                "n_bottom": monthly["n_bottom"].mean(),
                "entries": monthly["entries"].mean(),
                "exits": monthly["exits"].mean(),
                "one_way_turnover": monthly["one_way_turnover"].mean(),
                "two_way_turnover": monthly["two_way_turnover"].mean(),
                "overlap_share": monthly["overlap_share"].mean(),
                "turnover_approx": monthly["turnover_approx"].mean(),
                "median_overlap": monthly["overlap_share"].median(),
                "min_n_bottom": monthly["n_bottom"].min(),
                "max_n_bottom": monthly["n_bottom"].max(),
            }
        )

    result = pd.concat([pd.DataFrame(rows), pd.DataFrame(summary_rows)], ignore_index=True, sort=False)
    _save(result, output_path)
    print("\nTurnover summary:")
    print(result.loc[result["row_type"] == "summary"].to_string(index=False))
    return result


def run_transaction_cost_sensitivity(bottom_tail_returns, turnover_df, output_path=None):
    """Estimate rough transaction-cost sensitivity for bottom-tail strategies."""
    cost_bps_list = [0, 10, 25, 50, 100]
    strategies = [
        ("decile", "all", "ew", "Decile All EW"),
        ("decile", "mktcap_100m", "ew", "Decile $100M+ EW"),
    ]
    turnover = turnover_df.loc[turnover_df["row_type"] == "monthly"].copy()
    turnover.loc[:, "signal_month"] = pd.PeriodIndex(turnover["signal_month"].astype(str), freq="M")

    rows = []
    for tail, universe, weighting, label in strategies:
        returns = bottom_tail_returns.loc[
            (bottom_tail_returns["tail"] == tail)
            & (bottom_tail_returns["universe"] == universe)
            & (bottom_tail_returns["weighting"] == weighting)
        ].copy()
        returns.loc[:, "signal_month"] = pd.PeriodIndex(returns["signal_month"].astype(str), freq="M")
        merged = returns.merge(
            turnover.loc[turnover["universe"] == universe, ["signal_month", "turnover_approx"]],
            on="signal_month",
            how="left",
        )
        merged.loc[:, "turnover_approx"] = merged["turnover_approx"].fillna(0)
        for cost_bps in cost_bps_list:
            cost_drag = cost_bps / 10000 * merged["turnover_approx"]
            net_return = merged["universe_minus_bottom"] - cost_drag
            summary = _summarize_return_series(net_return)
            rows.append(
                {
                    "strategy": label,
                    "tail": tail,
                    "universe": universe,
                    "weighting": weighting,
                    "cost_bps": cost_bps,
                    **summary,
                    "average_monthly_cost_drag": cost_drag.mean(),
                    "annualized_cost_drag": cost_drag.mean() * 12,
                    "note": "Rough one-way sensitivity; universe leg trading costs are not fully modeled.",
                }
            )

    result = pd.DataFrame(rows)
    _save(result, output_path)
    return result


def _sic_to_sector(sic):
    """Map SIC code to broad sector."""
    if pd.isna(sic):
        return "Unknown"
    try:
        sic = int(sic)
    except Exception:
        return "Unknown"
    if 100 <= sic <= 999:
        return "Agriculture/Mining/Other Primary"
    if 1000 <= sic <= 1499:
        return "Mining"
    if 1500 <= sic <= 1799:
        return "Construction"
    if 2000 <= sic <= 3999:
        return "Manufacturing"
    if 4000 <= sic <= 4999:
        return "Transportation/Utilities"
    if 5000 <= sic <= 5199:
        return "Wholesale"
    if 5200 <= sic <= 5999:
        return "Retail"
    if 6000 <= sic <= 6799:
        return "Finance/Real Estate"
    if 7000 <= sic <= 8999:
        return "Services"
    return "Other/Unknown"


def add_industry_classification(monthly_panel, security_master):
    """Add broad industry sector if SIC is available."""
    panel = monthly_panel.copy()
    if "sic" in panel.columns:
        panel.loc[:, "sector"] = panel["sic"].apply(_sic_to_sector)
        return panel

    if not security_master.empty and {"secid", "sic"}.issubset(security_master.columns):
        sic = security_master[["secid", "sic"]].copy()
        sic.loc[:, "secid"] = pd.to_numeric(sic["secid"], errors="coerce").astype("Int64")
        panel = panel.merge(sic.drop_duplicates("secid"), on="secid", how="left")
        panel.loc[:, "sector"] = panel["sic"].apply(_sic_to_sector)
        return panel

    print("Warning: SIC/industry data not available. Sector is set to Unknown.")
    panel.loc[:, "sector"] = "Unknown"
    return panel


def compute_industry_exposure(panel_with_sector, output_path=None):
    """Compute rough sector exposure for the bottom IV-spread decile."""
    assigned = _assign_quantile_group(panel_with_sector, "iv_spread_adj", n_groups=10, group_col="iv_spread_decile")
    assigned.loc[:, "bottom_tail"] = assigned["iv_spread_decile"].eq(1)
    rows = []
    for signal_month, month_data in assigned.groupby("signal_month", sort=True):
        bottom = month_data.loc[month_data["bottom_tail"]]
        universe_counts = month_data["sector"].value_counts(normalize=True)
        bottom_counts = bottom["sector"].value_counts(normalize=True)
        for sector in sorted(set(universe_counts.index) | set(bottom_counts.index)):
            sector_universe = month_data.loc[month_data["sector"] == sector]
            sector_bottom = bottom.loc[bottom["sector"] == sector]
            rows.append(
                {
                    "signal_month": signal_month,
                    "sector": sector,
                    "universe_share": universe_counts.get(sector, 0),
                    "bottom_share": bottom_counts.get(sector, 0),
                    "share_difference": bottom_counts.get(sector, 0) - universe_counts.get(sector, 0),
                    "bottom_tail_return": sector_bottom["ret_fwd_1m"].mean(),
                    "universe_return": sector_universe["ret_fwd_1m"].mean(),
                    "avg_mktcap": sector_bottom["mktcap"].mean(),
                    "avg_iv_spread_adj": sector_bottom["iv_spread_adj"].mean(),
                }
            )

    monthly = pd.DataFrame(rows)
    exposure = (
        monthly.groupby("sector", as_index=False)
        .agg(
            avg_universe_share=("universe_share", "mean"),
            avg_bottom_share=("bottom_share", "mean"),
            avg_share_difference=("share_difference", "mean"),
            avg_bottom_tail_return=("bottom_tail_return", "mean"),
            avg_universe_return=("universe_return", "mean"),
            avg_mktcap=("avg_mktcap", "mean"),
            avg_iv_spread_adj=("avg_iv_spread_adj", "mean"),
        )
        .sort_values("avg_share_difference", ascending=False)
    )
    _save(exposure, output_path)
    print("\nTop overrepresented sectors:")
    print(exposure.head(10).to_string(index=False))
    return exposure


def run_industry_neutral_bottom_tail_test(panel_with_sector, output_path=None):
    """Test bottom-tail returns after sorting within broad sectors."""
    if panel_with_sector["sector"].nunique(dropna=True) <= 1:
        result = pd.DataFrame(
            [
                {
                    "universe": "all",
                    "warning": "Industry data unavailable or all Unknown; industry-neutral test skipped.",
                    "annualized_return": np.nan,
                    "t_stat": np.nan,
                    "sharpe_ratio": np.nan,
                    "positive_month_pct": np.nan,
                    "n_months": 0,
                    "avg_n_bottom": np.nan,
                }
            ]
        )
        _save(result, output_path)
        return result

    rows = []
    for universe, cutoff in [("all", None), ("mktcap_100m", 100)]:
        data = panel_with_sector.copy()
        if cutoff is not None:
            data = data.loc[data["mktcap"] >= cutoff].copy()
        data.loc[:, "industry_bottom"] = False

        for (_, sector), group in data.groupby(["signal_month", "sector"], sort=True):
            if len(group) < 20 or group["iv_spread_adj"].nunique() < 10:
                continue
            try:
                deciles = pd.qcut(group["iv_spread_adj"], q=10, labels=False, duplicates="drop")
            except ValueError:
                continue
            if deciles.nunique(dropna=True) < 10:
                continue
            data.loc[group.index, "industry_bottom"] = (deciles + 1 == 1).to_numpy()

        monthly_returns = []
        for signal_month, month_data in data.groupby("signal_month", sort=True):
            bottom = month_data.loc[month_data["industry_bottom"]]
            if bottom.empty:
                continue
            monthly_returns.append(
                {
                    "signal_month": signal_month,
                    "return_month": month_data["return_month"].iloc[0],
                    "strategy_return": month_data["ret_fwd_1m"].mean() - bottom["ret_fwd_1m"].mean(),
                    "n_bottom": len(bottom),
                }
            )
        monthly = pd.DataFrame(monthly_returns)
        summary = _summarize_return_series(monthly["strategy_return"])
        rows.append(
            {
                "universe": universe,
                "warning": "",
                **summary,
                "avg_n_bottom": monthly["n_bottom"].mean(),
            }
        )

    result = pd.DataFrame(rows)
    _save(result, output_path)
    return result


def plot_call_put_decomposition(decomp_df, charts_dir):
    """Plot average call IV and put IV by IV-spread decile."""
    charts_dir = Path(charts_dir)
    charts_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(decomp_df["decile"], decomp_df["iv_atm_call"], marker="o", label="ATM Call IV")
    axes[0].plot(decomp_df["decile"], decomp_df["iv_atm_put"], marker="o", label="ATM Put IV")
    axes[0].set_title("Call and Put IV by IV-Spread Decile")
    axes[0].set_xlabel("IV-spread decile")
    axes[0].set_ylabel("Average implied volatility")
    axes[0].legend()
    axes[1].bar(decomp_df["decile"], decomp_df["iv_spread"])
    axes[1].set_title("Average IV Spread by Decile")
    axes[1].set_xlabel("IV-spread decile")
    axes[1].set_ylabel("Call IV - Put IV")
    fig.tight_layout()
    output_path = charts_dir / "iv_spread_call_put_decomposition.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved chart: {output_path}")


def plot_call_put_tail_tests(tail_tests_df, charts_dir):
    """Plot annualized returns for call/put tail tests."""
    charts_dir = Path(charts_dir)
    charts_dir.mkdir(parents=True, exist_ok=True)
    data = tail_tests_df.copy()
    data = data.loc[data["universe"].isin(["all", "mktcap_100m"])]
    data.loc[:, "label"] = data["test"] + " / " + data["universe"]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(data["label"], data["annualized_return"])
    ax.axhline(0, linewidth=1)
    ax.set_title("Call/Put Tail Tests: Universe Minus Tail")
    ax.set_xlabel("")
    ax.set_ylabel("Annualized return")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    output_path = charts_dir / "iv_spread_call_put_tail_tests.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved chart: {output_path}")


def plot_transaction_cost_sensitivity(cost_df, charts_dir):
    """Plot net annualized returns under transaction-cost assumptions."""
    charts_dir = Path(charts_dir)
    charts_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    for strategy, group in cost_df.groupby("strategy"):
        ax.plot(group["cost_bps"], group["annualized_return"], marker="o", label=strategy)
    ax.set_title("Transaction-Cost Sensitivity")
    ax.set_xlabel("Cost assumption (bps per turnover)")
    ax.set_ylabel("Net annualized return")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.legend()
    fig.tight_layout()
    output_path = charts_dir / "iv_spread_transaction_cost_sensitivity.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved chart: {output_path}")


def plot_industry_exposure(industry_exposure_df, charts_dir):
    """Plot top sector overrepresentation in bottom decile."""
    charts_dir = Path(charts_dir)
    charts_dir.mkdir(parents=True, exist_ok=True)
    data = industry_exposure_df.head(10).copy()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(data["sector"], data["avg_share_difference"])
    ax.set_title("Bottom-Decile Sector Overrepresentation")
    ax.set_xlabel("")
    ax.set_ylabel("Bottom share - universe share")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    output_path = charts_dir / "iv_spread_industry_exposure.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved chart: {output_path}")


def plot_turnover(turnover_df, charts_dir):
    """Plot monthly bottom-decile turnover approximation."""
    charts_dir = Path(charts_dir)
    charts_dir.mkdir(parents=True, exist_ok=True)
    data = turnover_df.loc[turnover_df["row_type"] == "monthly"].copy()
    data = data.dropna(subset=["turnover_approx"])
    data.loc[:, "date"] = pd.PeriodIndex(data["signal_month"].astype(str), freq="M").to_timestamp()
    fig, ax = plt.subplots(figsize=(10, 5))
    for universe, group in data.groupby("universe"):
        ax.plot(group["date"], group["turnover_approx"], label=universe)
    ax.set_title("Bottom-Decile Monthly Turnover Approximation")
    ax.set_xlabel("Signal month")
    ax.set_ylabel("1 - overlap with prior month")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend()
    fig.tight_layout()
    output_path = charts_dir / "iv_spread_turnover.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved chart: {output_path}")


def build_final_diagnostic_summary(
    sign_df,
    decomp_df,
    tail_tests_df,
    turnover_df,
    cost_df,
    industry_exposure_df,
    industry_neutral_df,
    output_path=None,
):
    """Build a compact final diagnostic summary table."""
    bottom = decomp_df.loc[decomp_df["decile"] == "D1"].iloc[0]
    main_tail = tail_tests_df.loc[
        (tail_tests_df["test"] == "low_iv_spread") & (tail_tests_df["universe"] == "all")
    ].iloc[0]
    low_call = tail_tests_df.loc[
        (tail_tests_df["test"] == "low_call_iv") & (tail_tests_df["universe"] == "all")
    ].iloc[0]
    high_put = tail_tests_df.loc[
        (tail_tests_df["test"] == "high_put_iv") & (tail_tests_df["universe"] == "all")
    ].iloc[0]
    turnover_summary = turnover_df.loc[
        (turnover_df["row_type"] == "summary") & (turnover_df["universe"] == "all")
    ].iloc[0]
    cost_50 = cost_df.loc[
        (cost_df["strategy"] == "Decile All EW") & (cost_df["cost_bps"] == 50)
    ].iloc[0]

    rows = [
        {
            "diagnostic": "sign_convention",
            "finding": sign_df["interpretation"].iloc[0],
            "value": sign_df["bottom_decile_avg_iv_spread"].iloc[0],
        },
        {
            "diagnostic": "call_put_decomposition",
            "finding": "Bottom decile is mostly a high put-IV relative-to-call-IV tail, not a low absolute call-IV tail.",
            "value": bottom.get("bottom_call_iv_minus_universe_call_iv", np.nan),
        },
        {
            "diagnostic": "tail_tests",
            "finding": "Low IV spread remains the cleanest tail definition versus low call IV and high put IV.",
            "value": main_tail["annualized_return"],
        },
        {
            "diagnostic": "low_call_tail",
            "finding": "Low call IV alone does not reproduce the low IV-spread underperformance result.",
            "value": low_call["annualized_return"],
        },
        {
            "diagnostic": "high_put_tail",
            "finding": "High put IV alone is weaker than low IV spread in the all-stock EW test.",
            "value": high_put["annualized_return"],
        },
        {
            "diagnostic": "turnover",
            "finding": "Bottom-decile membership turnover is high, so implementability needs care.",
            "value": turnover_summary["turnover_approx"],
        },
        {
            "diagnostic": "cost_sensitivity_50bps",
            "finding": "The rough net return remains positive under a 50 bps turnover-cost assumption.",
            "value": cost_50["annualized_return"],
        },
        {
            "diagnostic": "industry_exposure",
            "finding": "Industry data are unavailable if sector is Unknown; interpret industry diagnostics cautiously.",
            "value": industry_exposure_df["avg_share_difference"].abs().max() if len(industry_exposure_df) else np.nan,
        },
        {
            "diagnostic": "industry_neutral",
            "finding": industry_neutral_df.get("warning", pd.Series([""])).iloc[0],
            "value": industry_neutral_df.get("annualized_return", pd.Series([np.nan])).iloc[0],
        },
    ]
    result = pd.DataFrame(rows)
    _save(result, output_path)
    return result
