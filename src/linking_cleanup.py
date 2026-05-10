"""Diagnostics and comparisons for the linking cleanup step."""

from pathlib import Path

import numpy as np
import pandas as pd


KEY_BACKUP_FILES = [
    "final_bottom_tail_main_table.csv",
    "pre_expansion_filtered_bottom_tail_summary.csv",
    "pre_expansion_covid_exclusion_summary.csv",
    "factor_regression_summary.csv",
    "research_audit_summary.csv",
]


def _read_text(path):
    """Read a text file if it exists."""
    path = Path(path)
    return path.read_text() if path.exists() else ""


def _script_import_source(script_text):
    """Identify where script 03 imports the time-aware linker from."""
    if "from src.linking import link_signals_to_permno_time_aware" in script_text:
        return "src.linking.py"
    if "from src.signals import" in script_text and "link_signals_to_permno_time_aware" in script_text:
        return "src.signals.py"
    return "ambiguous"


def create_linking_cleanup_diagnostic(project_root, tables_dir):
    """Save a diagnostic explaining the linking implementation cleanup."""
    project_root = Path(project_root)
    tables_dir = Path(tables_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)

    script_path = project_root / "scripts" / "03_construct_daily_iv_signals.py"
    linking_path = project_root / "src" / "linking.py"
    signals_path = project_root / "src" / "signals.py"
    previous_check_path = tables_dir / "pre_expansion_linking_function_check.csv"

    script_text = _read_text(script_path)
    linking_text = _read_text(linking_path)
    signals_text = _read_text(signals_path)

    previous_import = ""
    if previous_check_path.exists():
        previous_check = pd.read_csv(previous_check_path)
        previous_imported = previous_check.loc[previous_check["imported_by_script_03"] == True]  # noqa: E712
        if not previous_imported.empty:
            previous_import = previous_imported["file"].iloc[0]

    current_import = _script_import_source(script_text)
    preferred_file = "src.linking.py"
    impact_note = (
        "Changing the import is expected to leave results unchanged or nearly unchanged "
        "for the current pivoted signal panel. The preferred implementation is safer "
        "because it deduplicates by a unique _signal_row_id after the time-aware merge, "
        "rather than by secid/date."
    )

    rows = [
        {
            "file": "src/linking.py",
            "defines_function": "def link_signals_to_permno_time_aware" in linking_text,
            "uses_signal_row_id": "_signal_row_id" in linking_text,
            "currently_imported_by_script_03": current_import == "src.linking.py",
            "previous_pre_expansion_import": previous_import,
            "preferred_implementation": True,
            "recommendation": "Use this row-id implementation.",
            "could_affect_results": impact_note,
        },
        {
            "file": "src/signals.py",
            "defines_function": "def link_signals_to_permno_time_aware" in signals_text,
            "uses_signal_row_id": "_signal_row_id" in signals_text,
            "currently_imported_by_script_03": current_import == "src.signals.py",
            "previous_pre_expansion_import": previous_import,
            "preferred_implementation": False,
            "recommendation": "Keep only as a deprecated compatibility wrapper.",
            "could_affect_results": impact_note,
        },
        {
            "file": "scripts/03_construct_daily_iv_signals.py",
            "defines_function": False,
            "uses_signal_row_id": False,
            "currently_imported_by_script_03": current_import,
            "previous_pre_expansion_import": previous_import,
            "preferred_implementation": current_import == preferred_file,
            "recommendation": f"Current import source should be {preferred_file}.",
            "could_affect_results": impact_note,
        },
    ]

    diagnostic = pd.DataFrame(rows)
    output_path = tables_dir / "linking_cleanup_diagnostic.csv"
    diagnostic.to_csv(output_path, index=False)
    print(f"Saved linking cleanup diagnostic: {output_path} shape={diagnostic.shape}")
    return diagnostic


def create_audit_dynamic_month_check_note(tables_dir):
    """Save a short note documenting the dynamic audit month-count change."""
    tables_dir = Path(tables_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)

    note = pd.DataFrame(
        [
            {
                "old_behavior": (
                    "Audit expected robustness and factor-regression n_months to equal "
                    "the hardcoded 2018-2023 value of 72."
                ),
                "new_behavior": (
                    "Audit infers expected month counts from robustness return files "
                    "and from each factor-regression row's first/last return_month. "
                    "Monthly panel timing checks also validate contiguous observed ranges."
                ),
                "reason": (
                    "The audit should pass for the current 2018-2023 sample and for "
                    "future expanded samples without weakening checks for missing months."
                ),
            }
        ]
    )

    output_path = tables_dir / "audit_dynamic_month_check_note.csv"
    note.to_csv(output_path, index=False)
    print(f"Saved audit dynamic month note: {output_path} shape={note.shape}")
    return note


def _status_for_difference(before_value, after_value, metric):
    """Classify before/after differences."""
    if pd.isna(before_value) and pd.isna(after_value):
        return "PASS"

    try:
        before_float = float(before_value)
        after_float = float(after_value)
    except (TypeError, ValueError):
        return "PASS" if str(before_value) == str(after_value) else "REVIEW"

    diff = after_float - before_float
    if abs(diff) <= 1e-12:
        return "PASS"

    if metric == "FAIL count":
        return "REVIEW"
    if metric in {"PASS count", "WARN count"}:
        return "WARN"

    if abs(diff) <= 1e-6:
        return "WARN"
    return "REVIEW"


def _append_metric(rows, table_name, strategy_or_check, metric, before_value, after_value):
    """Append one comparison row."""
    try:
        difference = float(after_value) - float(before_value)
    except (TypeError, ValueError):
        difference = "" if str(before_value) == str(after_value) else "changed"

    rows.append(
        {
            "table_name": table_name,
            "strategy_or_check": strategy_or_check,
            "metric": metric,
            "before_value": before_value,
            "after_value": after_value,
            "difference": difference,
            "status": _status_for_difference(before_value, after_value, metric),
        }
    )


def _compare_by_key(rows, table_name, before, after, key_cols, metrics):
    """Compare common rows in two tables using key columns."""
    merged = before.merge(after, on=key_cols, how="inner", suffixes=("_before", "_after"))
    for _, row in merged.iterrows():
        key = " | ".join(str(row[col]) for col in key_cols)
        for metric in metrics:
            before_col = f"{metric}_before"
            after_col = f"{metric}_after"
            if before_col in merged.columns and after_col in merged.columns:
                _append_metric(rows, table_name, key, metric, row[before_col], row[after_col])


def _compare_audit_counts(rows, backup_dir, tables_dir):
    """Compare PASS/WARN/FAIL counts in research_audit_summary.csv."""
    before_path = backup_dir / "research_audit_summary.csv"
    after_path = tables_dir / "research_audit_summary.csv"
    if not before_path.exists() or not after_path.exists():
        return

    before = pd.read_csv(before_path)
    after = pd.read_csv(after_path)
    for status in ["PASS", "WARN", "FAIL"]:
        before_count = int((before["status"] == status).sum())
        after_count = int((after["status"] == status).sum())
        _append_metric(
            rows,
            "research_audit_summary.csv",
            "audit status counts",
            f"{status} count",
            before_count,
            after_count,
        )


def create_linking_cleanup_result_comparison(tables_dir):
    """Compare backed-up key outputs to current outputs after rerunning scripts."""
    tables_dir = Path(tables_dir)
    backup_dir = tables_dir / "pre_linking_cleanup_backup"
    rows = []

    final_before = backup_dir / "final_bottom_tail_main_table.csv"
    final_after = tables_dir / "final_bottom_tail_main_table.csv"
    if final_before.exists() and final_after.exists():
        _compare_by_key(
            rows,
            "final_bottom_tail_main_table.csv",
            pd.read_csv(final_before),
            pd.read_csv(final_after),
            ["Strategy"],
            [
                "Annualized return",
                "Return t-stat",
                "FF5+MOM alpha",
                "Alpha t-stat",
                "N months",
            ],
        )

    filtered_before = backup_dir / "pre_expansion_filtered_bottom_tail_summary.csv"
    filtered_after = tables_dir / "pre_expansion_filtered_bottom_tail_summary.csv"
    if filtered_before.exists() and filtered_after.exists():
        _compare_by_key(
            rows,
            "pre_expansion_filtered_bottom_tail_summary.csv",
            pd.read_csv(filtered_before),
            pd.read_csv(filtered_after),
            ["filter_name"],
            ["annualized_return", "raw_t_stat", "nw_t_stat", "n_months"],
        )

    covid_before = backup_dir / "pre_expansion_covid_exclusion_summary.csv"
    covid_after = tables_dir / "pre_expansion_covid_exclusion_summary.csv"
    if covid_before.exists() and covid_after.exists():
        _compare_by_key(
            rows,
            "pre_expansion_covid_exclusion_summary.csv",
            pd.read_csv(covid_before),
            pd.read_csv(covid_after),
            ["filter_name", "sample"],
            ["annualized_return", "raw_t_stat", "nw_t_stat", "n_months"],
        )

    factor_before = backup_dir / "factor_regression_summary.csv"
    factor_after = tables_dir / "factor_regression_summary.csv"
    if factor_before.exists() and factor_after.exists():
        _compare_by_key(
            rows,
            "factor_regression_summary.csv",
            pd.read_csv(factor_before),
            pd.read_csv(factor_after),
            ["portfolio", "model"],
            ["alpha_annualized", "alpha_tstat", "r_squared", "n_months"],
        )

    _compare_audit_counts(rows, backup_dir, tables_dir)

    comparison = pd.DataFrame(rows)
    output_path = tables_dir / "linking_cleanup_result_comparison.csv"
    comparison.to_csv(output_path, index=False)
    print(f"Saved cleanup result comparison: {output_path} shape={comparison.shape}")
    return comparison


def summarize_comparison(comparison):
    """Return a compact status summary for the comparison table."""
    if comparison.empty:
        return {"PASS": 0, "WARN": 0, "REVIEW": 0}
    return comparison["status"].value_counts().to_dict()
