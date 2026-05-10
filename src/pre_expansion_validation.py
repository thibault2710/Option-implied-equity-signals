"""Pre-expansion validation checks for the IV spread bottom-tail result."""

import ast
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils import compute_weighted_return, safe_mean, summarize_return_series


MPLCONFIGDIR = Path(tempfile.gettempdir()) / "options_implied_signals_matplotlib"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick


REQUIRED_MONTHLY_COLUMNS = [
    "permno",
    "secid",
    "signal_month",
    "return_month",
    "iv_spread_adj",
    "ret_fwd_1m",
    "mktcap",
    "shrcd",
    "exchcd",
    "shrcd_return",
    "exchcd_return",
]

COMMON_SHARE_CODES = [10, 11]
MAJOR_EXCHANGE_CODES = [1, 2, 3]


def load_pre_expansion_inputs(processed_data_dir, tables_dir, project_root):
    """Load the existing files needed for pre-expansion validation."""
    processed_data_dir = Path(processed_data_dir)
    tables_dir = Path(tables_dir)
    project_root = Path(project_root)

    monthly_panel_path = processed_data_dir / "monthly_signal_panel.parquet"
    monthly_sector_path = processed_data_dir / "monthly_signal_panel_with_sector.parquet"
    bottom_tail_returns_path = tables_dir / "iv_spread_bottom_tail_returns.csv"
    bottom_tail_summary_path = tables_dir / "iv_spread_bottom_tail_summary.csv"
    factor_alpha_path = tables_dir / "iv_spread_bottom_tail_factor_alpha.csv"

    monthly_panel = pd.read_parquet(monthly_panel_path)
    monthly_panel_with_sector = (
        pd.read_parquet(monthly_sector_path) if monthly_sector_path.exists() else None
    )
    bottom_tail_returns = pd.read_csv(bottom_tail_returns_path)
    bottom_tail_summary = pd.read_csv(bottom_tail_summary_path)
    bottom_tail_factor_alpha = pd.read_csv(factor_alpha_path)

    print("\n" + "=" * 80)
    print("Loading Pre-Expansion Validation Inputs")
    print("=" * 80)
    print(f"monthly_signal_panel: {monthly_panel.shape} from {monthly_panel_path}")
    if monthly_panel_with_sector is not None:
        print(f"monthly_signal_panel_with_sector: {monthly_panel_with_sector.shape}")
    else:
        print("monthly_signal_panel_with_sector: not found, continuing without it")
    print(f"bottom_tail_returns: {bottom_tail_returns.shape}")
    print(f"bottom_tail_summary: {bottom_tail_summary.shape}")
    print(f"bottom_tail_factor_alpha: {bottom_tail_factor_alpha.shape}")

    missing = [col for col in REQUIRED_MONTHLY_COLUMNS if col not in monthly_panel.columns]
    print(f"Required monthly columns missing: {missing if missing else 'None'}")
    if missing:
        raise ValueError(f"monthly_signal_panel is missing required columns: {missing}")

    return {
        "monthly_panel": _prepare_monthly_panel(monthly_panel),
        "monthly_panel_with_sector": monthly_panel_with_sector,
        "bottom_tail_returns": bottom_tail_returns,
        "bottom_tail_summary": bottom_tail_summary,
        "bottom_tail_factor_alpha": bottom_tail_factor_alpha,
        "paths": {
            "project_root": project_root,
            "monthly_panel": monthly_panel_path,
            "monthly_panel_with_sector": monthly_sector_path,
            "bottom_tail_returns": bottom_tail_returns_path,
            "bottom_tail_summary": bottom_tail_summary_path,
            "bottom_tail_factor_alpha": factor_alpha_path,
        },
    }


def _prepare_monthly_panel(panel):
    """Normalize monthly panel dtypes used by validation checks."""
    columns = REQUIRED_MONTHLY_COLUMNS.copy()
    data = panel[columns].copy()

    data.loc[:, "permno"] = pd.to_numeric(data["permno"], errors="coerce").astype("Int64")
    data.loc[:, "secid"] = pd.to_numeric(data["secid"], errors="coerce").astype("Int64")
    data.loc[:, "signal_month"] = pd.PeriodIndex(data["signal_month"].astype(str), freq="M")
    data.loc[:, "return_month"] = pd.PeriodIndex(data["return_month"].astype(str), freq="M")

    numeric_columns = [
        "iv_spread_adj",
        "ret_fwd_1m",
        "mktcap",
        "shrcd",
        "exchcd",
        "shrcd_return",
        "exchcd_return",
    ]
    for column in numeric_columns:
        data.loc[:, column] = pd.to_numeric(data[column], errors="coerce")

    return data.dropna(subset=["permno", "signal_month", "return_month", "iv_spread_adj", "ret_fwd_1m"])


def _save_csv(df, output_path):
    """Save a dataframe and print a consistent message."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved table: {output_path} shape={df.shape}")


def _assign_deciles_by_month(panel, signal_col="iv_spread_adj", decile_col="iv_spread_decile"):
    """Assign signal deciles inside each signal month."""
    data = panel.copy()
    data.loc[:, decile_col] = pd.NA

    for _, month_data in data.groupby("signal_month", sort=True):
        valid = month_data.dropna(subset=[signal_col])
        if valid[signal_col].nunique() < 10:
            continue
        try:
            deciles = pd.qcut(valid[signal_col], q=10, labels=False, duplicates="drop")
        except ValueError:
            continue
        if deciles.nunique(dropna=True) < 10:
            continue
        data.loc[valid.index, decile_col] = (deciles + 1).astype("Int64")

    data = data.dropna(subset=[decile_col]).copy()
    data.loc[:, decile_col] = data[decile_col].astype("Int64")
    return data


def _common_major_mask(data):
    """Return a common-share and major-exchange mask using signal-month fields."""
    return data["shrcd"].isin(COMMON_SHARE_CODES) & data["exchcd"].isin(MAJOR_EXCHANGE_CODES)


def _composition_summary_for_group(group, group_name):
    """Create share-code and exchange-code composition rows for one group."""
    total_rows = len(group)
    unique_months = group["signal_month"].nunique()
    rows = []

    common_major_share = _common_major_mask(group).mean() if total_rows else np.nan
    base = {
        "group": group_name,
        "total_rows": total_rows,
        "unique_permnos": group["permno"].nunique(),
        "unique_months": unique_months,
        "avg_stocks_per_month": total_rows / unique_months if unique_months else np.nan,
        "share_common_shrcd": group["shrcd"].isin(COMMON_SHARE_CODES).mean() if total_rows else np.nan,
        "share_major_exchange": group["exchcd"].isin(MAJOR_EXCHANGE_CODES).mean() if total_rows else np.nan,
        "share_common_major_exchange": common_major_share,
    }

    rows.append({**base, "metric": "summary", "code": "all", "count": total_rows, "share": 1.0})

    for column in ["shrcd", "exchcd", "shrcd_return", "exchcd_return"]:
        counts = group[column].value_counts(dropna=False).sort_index()
        for code, count in counts.items():
            code_value = "missing" if pd.isna(code) else str(int(code))
            rows.append(
                {
                    **base,
                    "metric": column,
                    "code": code_value,
                    "count": int(count),
                    "share": count / total_rows if total_rows else np.nan,
                }
            )

    return rows


def diagnose_universe_composition(monthly_panel, tables_dir):
    """Check share-code and exchange-code composition for key universes."""
    tables_dir = Path(tables_dir)

    all_deciles = _assign_deciles_by_month(monthly_panel)
    mktcap100 = monthly_panel.loc[monthly_panel["mktcap"] >= 100].copy()
    mktcap100_deciles = _assign_deciles_by_month(mktcap100)

    groups = {
        "full_universe": all_deciles,
        "bottom_decile": all_deciles.loc[all_deciles["iv_spread_decile"] == 1],
        "mktcap100_universe": mktcap100_deciles,
        "mktcap100_bottom_decile": mktcap100_deciles.loc[
            mktcap100_deciles["iv_spread_decile"] == 1
        ],
    }

    records = []
    for group_name, group in groups.items():
        records.extend(_composition_summary_for_group(group, group_name))

    composition = pd.DataFrame(records)
    output_path = tables_dir / "pre_expansion_universe_composition.csv"
    _save_csv(composition, output_path)

    summary = composition.loc[composition["metric"] == "summary"].copy()
    print("\n" + "=" * 80)
    print("Universe Composition")
    print("=" * 80)
    print(summary[["group", "total_rows", "share_common_major_exchange"]].to_string(index=False))

    for group_name in ["full_universe", "bottom_decile"]:
        shrcd_top = (
            composition.loc[
                (composition["group"] == group_name) & (composition["metric"] == "shrcd")
            ]
            .sort_values("share", ascending=False)
            .head(3)
        )
        exchcd_top = (
            composition.loc[
                (composition["group"] == group_name) & (composition["metric"] == "exchcd")
            ]
            .sort_values("share", ascending=False)
            .head(3)
        )
        print(f"\nTop share codes for {group_name}:")
        print(shrcd_top[["code", "share"]].to_string(index=False))
        print(f"Top exchange codes for {group_name}:")
        print(exchcd_top[["code", "share"]].to_string(index=False))

    return composition


def _filter_panel_for_name(panel, filter_name):
    """Apply one of the pre-expansion universe filters."""
    data = panel.copy()
    if filter_name == "baseline_all":
        return data

    data = data.loc[_common_major_mask(data)].copy()
    if filter_name == "common_major_exchange":
        return data
    if filter_name == "common_major_exchange_mktcap_100m":
        return data.loc[data["mktcap"] >= 100].copy()
    if filter_name == "common_major_exchange_mktcap_500m":
        return data.loc[data["mktcap"] >= 500].copy()
    if filter_name == "common_major_exchange_mktcap_1b":
        return data.loc[data["mktcap"] >= 1000].copy()

    raise ValueError(f"Unknown filter_name: {filter_name}")


def _build_monthly_bottom_tail_returns_for_filter(panel, filter_name):
    """Compute monthly EW universe-minus-bottom returns for one filter."""
    filtered = _filter_panel_for_name(panel, filter_name)
    deciles = _assign_deciles_by_month(filtered)

    rows = []
    for signal_month, month_data in deciles.groupby("signal_month", sort=True):
        bottom = month_data.loc[month_data["iv_spread_decile"] == 1]
        universe_return = safe_mean(month_data["ret_fwd_1m"])
        bottom_return = safe_mean(bottom["ret_fwd_1m"])
        rows.append(
            {
                "filter_name": filter_name,
                "signal_month": signal_month,
                "return_month": month_data["return_month"].iloc[0],
                "universe_return": universe_return,
                "bottom_decile_return": bottom_return,
                "universe_minus_bottom": universe_return - bottom_return,
                "n_universe": len(month_data),
                "n_bottom": len(bottom),
                "avg_mktcap_universe": month_data["mktcap"].mean(),
                "avg_mktcap_bottom": bottom["mktcap"].mean(),
                "avg_iv_spread_universe": month_data["iv_spread_adj"].mean(),
                "avg_iv_spread_bottom": bottom["iv_spread_adj"].mean(),
            }
        )

    return pd.DataFrame(rows)


def build_filtered_bottom_tail_returns(monthly_panel, tables_dir):
    """Rerun the bottom-tail strategy with common-share and exchange filters."""
    tables_dir = Path(tables_dir)
    filter_names = [
        "baseline_all",
        "common_major_exchange",
        "common_major_exchange_mktcap_100m",
        "common_major_exchange_mktcap_500m",
        "common_major_exchange_mktcap_1b",
    ]

    monthly_returns = pd.concat(
        [_build_monthly_bottom_tail_returns_for_filter(monthly_panel, name) for name in filter_names],
        ignore_index=True,
    )

    summary_rows = []
    for filter_name, group in monthly_returns.groupby("filter_name", sort=False):
        summary = summarize_return_series(group["universe_minus_bottom"])
        summary_rows.append(
            {
                "filter_name": filter_name,
                **summary,
                "avg_n_universe": group["n_universe"].mean(),
                "avg_n_bottom": group["n_bottom"].mean(),
                "avg_mktcap_universe": group["avg_mktcap_universe"].mean(),
                "avg_mktcap_bottom": group["avg_mktcap_bottom"].mean(),
                "avg_iv_spread_universe": group["avg_iv_spread_universe"].mean(),
                "avg_iv_spread_bottom": group["avg_iv_spread_bottom"].mean(),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    _save_csv(monthly_returns, tables_dir / "pre_expansion_filtered_bottom_tail_returns.csv")
    _save_csv(summary_df, tables_dir / "pre_expansion_filtered_bottom_tail_summary.csv")

    print("\n" + "=" * 80)
    print("Filtered Bottom-Tail Results")
    print("=" * 80)
    print(
        summary_df[
            ["filter_name", "annualized_return", "raw_t_stat", "nw_t_stat", "n_months"]
        ].to_string(index=False)
    )

    return monthly_returns, summary_df


def _month_in_periods(months, start, end=None):
    """Return a boolean mask for Period months in a closed interval."""
    months = pd.PeriodIndex(months.astype(str), freq="M")
    start = pd.Period(start, freq="M")
    if end is None:
        return months == start
    end = pd.Period(end, freq="M")
    return (months >= start) & (months <= end)


def run_covid_exclusion_tests(filtered_monthly_returns, tables_dir):
    """Test whether filtered bottom-tail results are dominated by COVID months."""
    tables_dir = Path(tables_dir)
    data = filtered_monthly_returns.copy()
    data.loc[:, "return_month"] = pd.PeriodIndex(data["return_month"].astype(str), freq="M")

    sample_rules = {
        "full_sample": ("No months excluded", lambda df: pd.Series(True, index=df.index)),
        "exclude_2020_full_year": (
            "Exclude return months 2020-01 through 2020-12",
            lambda df: ~_month_in_periods(df["return_month"], "2020-01", "2020-12"),
        ),
        "exclude_2020_q1_q2": (
            "Exclude return months 2020-01 through 2020-06",
            lambda df: ~_month_in_periods(df["return_month"], "2020-01", "2020-06"),
        ),
        "exclude_mar_2020": (
            "Exclude return month 2020-03",
            lambda df: ~_month_in_periods(df["return_month"], "2020-03"),
        ),
        "exclude_mar_apr_2020": (
            "Exclude return months 2020-03 through 2020-04",
            lambda df: ~_month_in_periods(df["return_month"], "2020-03", "2020-04"),
        ),
        "exclude_feb_to_apr_2020": (
            "Exclude return months 2020-02 through 2020-04",
            lambda df: ~_month_in_periods(df["return_month"], "2020-02", "2020-04"),
        ),
    }

    rows = []
    for filter_name, group in data.groupby("filter_name", sort=False):
        for sample, (description, mask_func) in sample_rules.items():
            sample_data = group.loc[mask_func(group)]
            summary = summarize_return_series(sample_data["universe_minus_bottom"])
            rows.append(
                {
                    "filter_name": filter_name,
                    "sample": sample,
                    **summary,
                    "excluded_months_description": description,
                }
            )

    covid_summary = pd.DataFrame(rows)
    _save_csv(covid_summary, tables_dir / "pre_expansion_covid_exclusion_summary.csv")

    print("\n" + "=" * 80)
    print("COVID Exclusion Checks")
    print("=" * 80)
    print(
        covid_summary.loc[
            covid_summary["filter_name"].isin(["baseline_all", "common_major_exchange_mktcap_100m"]),
            ["filter_name", "sample", "annualized_return", "nw_t_stat", "n_months"],
        ].to_string(index=False)
    )

    return covid_summary


def compute_newey_west_tstats_for_existing_results(bottom_tail_returns, tables_dir):
    """Add Newey-West t-stats for existing bottom-tail return series."""
    tables_dir = Path(tables_dir)
    data = bottom_tail_returns.copy()
    legs = ["universe_minus_bottom", "top_minus_bottom", "top_minus_universe"]

    rows = []
    for (tail, universe, weighting), group in data.groupby(["tail", "universe", "weighting"], sort=True):
        for leg in legs:
            summary = summarize_return_series(group[leg])
            rows.append(
                {
                    "tail": tail,
                    "universe": universe,
                    "weighting": weighting,
                    "leg": leg,
                    "annualized_return": summary["annualized_return"],
                    "raw_t_stat": summary["raw_t_stat"],
                    "nw_t_stat": summary["nw_t_stat"],
                    "nw_p_value": summary["nw_p_value"],
                    "sharpe_ratio": summary["sharpe_ratio"],
                    "n_months": summary["n_months"],
                }
            )

    nw_summary = pd.DataFrame(rows)
    _save_csv(nw_summary, tables_dir / "pre_expansion_newey_west_tstats.csv")

    print("\n" + "=" * 80)
    print("Existing Bottom-Tail Newey-West t-stats")
    print("=" * 80)
    main = nw_summary.loc[
        (nw_summary["tail"] == "decile")
        & (nw_summary["universe"].isin(["all", "mktcap_100m"]))
        & (nw_summary["weighting"] == "ew")
        & (nw_summary["leg"] == "universe_minus_bottom")
    ]
    print(main.to_string(index=False))

    return nw_summary


def check_linking_function_usage(project_root, tables_dir):
    """Determine which time-aware linking implementation script 03 imports."""
    project_root = Path(project_root)
    tables_dir = Path(tables_dir)

    script_path = project_root / "scripts" / "03_construct_daily_iv_signals.py"
    linking_path = project_root / "src" / "linking.py"
    signals_path = project_root / "src" / "signals.py"

    script_text = script_path.read_text()
    linking_text = linking_path.read_text() if linking_path.exists() else ""
    signals_text = signals_path.read_text() if signals_path.exists() else ""

    def imports_name_from_module(module_name, imported_name):
        try:
            tree = ast.parse(script_text)
        except SyntaxError:
            return False
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == module_name:
                if any(alias.name == imported_name for alias in node.names):
                    return True
        return False

    imports_from_linking = imports_name_from_module(
        "src.linking",
        "link_signals_to_permno_time_aware",
    )
    imports_from_signals = imports_name_from_module(
        "src.signals",
        "link_signals_to_permno_time_aware",
    )
    linking_defines = "def link_signals_to_permno_time_aware" in linking_text
    signals_defines = "def link_signals_to_permno_time_aware" in signals_text

    if imports_from_linking and signals_defines and linking_defines:
        recommendation = "Keep src.linking implementation and deprecate/remove duplicate in src.signals later."
    elif imports_from_signals:
        recommendation = "Review carefully; src.linking row-id implementation is preferred."
    else:
        recommendation = "Manual review required."

    rows = [
        {
            "file": "src/linking.py",
            "defines_function": linking_defines,
            "imported_by_script_03": imports_from_linking,
            "recommendation": recommendation,
        },
        {
            "file": "src/signals.py",
            "defines_function": signals_defines,
            "imported_by_script_03": imports_from_signals,
            "recommendation": recommendation,
        },
    ]
    result = pd.DataFrame(rows)
    _save_csv(result, tables_dir / "pre_expansion_linking_function_check.csv")

    print("\n" + "=" * 80)
    print("Linking Function Usage")
    print("=" * 80)
    print(result.to_string(index=False))

    return result


def create_pre_expansion_validation_summary(
    composition_df,
    filtered_summary_df,
    covid_df,
    nw_df,
    linking_df,
    project_root,
    tables_dir,
):
    """Create a compact final validation summary table."""
    project_root = Path(project_root)
    tables_dir = Path(tables_dir)
    rows = []

    comp_summary = composition_df.loc[composition_df["metric"] == "summary"].set_index("group")
    full_common_major = comp_summary.loc["full_universe", "share_common_major_exchange"]
    bottom_common_major = comp_summary.loc["bottom_decile", "share_common_major_exchange"]
    rows.append(
        {
            "check": "universe_composition",
            "status": "PASS" if full_common_major >= 0.95 else "WARN",
            "key_result": (
                f"Full universe common-major share {full_common_major:.1%}; "
                f"bottom decile {bottom_common_major:.1%}."
            ),
            "recommendation": "Use common-share and major-exchange filters as a reported robustness check.",
        }
    )

    filtered = filtered_summary_df.set_index("filter_name")
    filter_row = filtered.loc["common_major_exchange_mktcap_100m"]
    rows.append(
        {
            "check": "common_share_filter",
            "status": "PASS" if filter_row["annualized_return"] > 0 else "WARN",
            "key_result": (
                f"Common major exchange $100M+ annualized return "
                f"{filter_row['annualized_return']:.2%}, NW t-stat {filter_row['nw_t_stat']:.2f}."
            ),
            "recommendation": "Report this alongside the original all-stock bottom-tail result.",
        }
    )

    covid_check = covid_df.loc[
        (covid_df["filter_name"] == "baseline_all")
        & (covid_df["sample"] == "exclude_2020_full_year")
    ].iloc[0]
    rows.append(
        {
            "check": "covid_exclusion",
            "status": "PASS" if covid_check["annualized_return"] > 0 else "WARN",
            "key_result": (
                f"Excluding 2020 full year annualized return "
                f"{covid_check['annualized_return']:.2%}, NW t-stat {covid_check['nw_t_stat']:.2f}."
            ),
            "recommendation": "Keep COVID-exclusion results in the expansion checklist.",
        }
    )

    main_nw = nw_df.loc[
        (nw_df["tail"] == "decile")
        & (nw_df["universe"] == "all")
        & (nw_df["weighting"] == "ew")
        & (nw_df["leg"] == "universe_minus_bottom")
    ].iloc[0]
    rows.append(
        {
            "check": "newey_west_tstats",
            "status": "PASS" if main_nw["nw_t_stat"] > 2 else "WARN",
            "key_result": (
                f"Main decile/all/EW raw t-stat {main_nw['raw_t_stat']:.2f}; "
                f"NW t-stat {main_nw['nw_t_stat']:.2f}."
            ),
            "recommendation": "Use Newey-West t-stats for time-series return summaries.",
        }
    )

    linking_import = linking_df.loc[linking_df["imported_by_script_03"] == True]  # noqa: E712
    imported_file = linking_import["file"].iloc[0] if not linking_import.empty else "ambiguous"
    linking_status = "PASS" if imported_file == "src/linking.py" else "WARN"
    rows.append(
        {
            "check": "linking_function_usage",
            "status": linking_status,
            "key_result": f"scripts/03_construct_daily_iv_signals.py imports from {imported_file}.",
            "recommendation": linking_df["recommendation"].iloc[0],
        }
    )

    audit_text = (project_root / "src" / "audit.py").read_text()
    hardcoded_72 = "72" in audit_text
    rows.append(
        {
            "check": "audit_dynamic_months_needed",
            "status": "WARN" if hardcoded_72 else "PASS",
            "key_result": "Hardcoded '72' month checks found in src/audit.py."
            if hardcoded_72
            else "No hardcoded '72' month check found in src/audit.py.",
            "recommendation": "Make n_months checks dynamic before expanding the sample."
            if hardcoded_72
            else "No immediate audit month-count change needed.",
        }
    )

    summary = pd.DataFrame(rows)
    _save_csv(summary, tables_dir / "pre_expansion_validation_summary.csv")
    return summary


def plot_universe_composition(composition_df, charts_dir):
    """Plot common-share and major-exchange share by group."""
    charts_dir = Path(charts_dir)
    charts_dir.mkdir(parents=True, exist_ok=True)

    summary = composition_df.loc[composition_df["metric"] == "summary"].copy()
    order = ["full_universe", "bottom_decile", "mktcap100_universe", "mktcap100_bottom_decile"]
    summary.loc[:, "group"] = pd.Categorical(summary["group"], categories=order, ordered=True)
    summary = summary.sort_values("group")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(summary["group"].astype(str), summary["share_common_major_exchange"])
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.set_ylabel("Share of rows")
    ax.set_title("Common-Share and Major-Exchange Coverage")
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis="x", labelrotation=20)
    fig.tight_layout()

    output_path = charts_dir / "pre_expansion_universe_composition.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved chart: {output_path}")


def plot_filtered_results(filtered_summary_df, charts_dir):
    """Plot annualized returns for common-share and exchange filters."""
    charts_dir = Path(charts_dir)
    charts_dir.mkdir(parents=True, exist_ok=True)

    data = filtered_summary_df.copy()
    labels = data["filter_name"].str.replace("common_major_exchange", "common+major", regex=False)
    labels = labels.str.replace("_", " ", regex=False)

    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.bar(labels, data["annualized_return"])
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.set_ylabel("Annualized return")
    ax.set_title("Bottom-Tail Result Under Common-Share and Exchange Filters")
    ax.tick_params(axis="x", labelrotation=25)
    for bar, t_stat in zip(bars, data["nw_t_stat"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"NW t={t_stat:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    fig.tight_layout()

    output_path = charts_dir / "pre_expansion_filtered_results.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved chart: {output_path}")


def plot_covid_exclusion(covid_df, charts_dir):
    """Plot annualized returns across COVID exclusion samples."""
    charts_dir = Path(charts_dir)
    charts_dir.mkdir(parents=True, exist_ok=True)

    keep_filters = ["baseline_all", "common_major_exchange_mktcap_100m"]
    data = covid_df.loc[covid_df["filter_name"].isin(keep_filters)].copy()
    pivot = data.pivot(index="sample", columns="filter_name", values="annualized_return")
    pivot = pivot.loc[
        [
            "full_sample",
            "exclude_2020_full_year",
            "exclude_2020_q1_q2",
            "exclude_mar_2020",
            "exclude_mar_apr_2020",
            "exclude_feb_to_apr_2020",
        ]
    ]

    fig, ax = plt.subplots(figsize=(11, 5))
    pivot.plot(kind="bar", ax=ax)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.set_xlabel("")
    ax.set_ylabel("Annualized return")
    ax.set_title("COVID Exclusion Robustness")
    ax.tick_params(axis="x", labelrotation=25)
    fig.tight_layout()

    output_path = charts_dir / "pre_expansion_covid_exclusion.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved chart: {output_path}")
