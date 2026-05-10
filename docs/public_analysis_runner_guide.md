# Public Analysis Runner Guide

## Purpose

`scripts/public/run_public_analysis.py` runs the completed public analysis pipeline from the processed 2010-2023 monthly panel through final public outputs.

It is intended for users who already have the required local processed data and factor files. It does not pull WRDS data, does not rebuild raw datasets, and does not reconstruct the processed monthly panel.

## What It Runs

The runner executes these public scripts in order:

1. `00_check_environment.py`
2. `04_run_main_results.py`
3. `05_run_factor_regressions.py`
4. `06_run_long_only_exclusion.py`
5. `07_run_robustness_checks.py`
6. `08_create_final_outputs.py`
7. `09_audit_results.py`

## What It Does Not Run

The runner intentionally does not run the data-construction scripts that are still planned:

- `01_pull_data.py`
- `02_build_option_signals.py`
- `03_build_monthly_panel.py`

Those steps involve WRDS/OptionMetrics/CRSP access, cache management, and processed-data construction. They should be built and documented separately.

## Prerequisites

Before running the public analysis runner, these local inputs should exist:

- `data/processed/monthly_signal_panel_2010_2023.parquet`
- Local Fama-French factor CSVs under `data/raw/factors/`
- The public scripts under `scripts/public/`

Some robustness checks also use:

- `data/processed/monthly_signal_panel_with_sector_2010_2023.parquet`, if available
- `data/raw/crsp_monthly_2010_2024.parquet`

Raw WRDS, OptionMetrics, and CRSP data are not included in the public GitHub repository.

## Commands

Preview the public analysis steps without running them:

```bash
python scripts/public/run_public_analysis.py --check-only
```

Run the full public analysis pipeline from the processed panel onward:

```bash
python scripts/public/run_public_analysis.py
```

Start at factor regressions:

```bash
python scripts/public/run_public_analysis.py --start-at 05
```

Run only factor regressions through robustness checks:

```bash
python scripts/public/run_public_analysis.py --start-at run_factor_regressions --stop-after run_robustness_checks
```

Skip the environment check:

```bash
python scripts/public/run_public_analysis.py --skip-check
```

Use a specific Python executable:

```bash
python scripts/public/run_public_analysis.py --python .venv/bin/python
```

## Output Locations

The runner writes public outputs under:

- `outputs/public_2010_2023/tables/`
- `outputs/public_2010_2023/charts/`
- `outputs/public_2010_2023/final_tables/`
- `outputs/public_2010_2023/final_figures/`

Major final files include:

- `outputs/public_2010_2023/final_results_pack.md`
- `outputs/public_2010_2023/final_tables/`
- `outputs/public_2010_2023/final_figures/`
- `outputs/public_2010_2023/tables/public_research_audit_summary_2010_2023.csv`

## Data Access Note

This project uses licensed OptionMetrics, CRSP, and WRDS data. Those raw and processed data files should not be committed to GitHub. Public users with the appropriate data access can regenerate the local inputs once the public data-construction scripts are completed.
