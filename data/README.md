# Data Directory

This folder is intentionally empty or mostly empty in the public GitHub repository.

## Why Data Are Not Included

The empirical project uses licensed WRDS, OptionMetrics, and CRSP data. Raw data files, processed Parquet files, and public-processed intermediate panels are excluded from GitHub to respect licensing and keep the repository lightweight.

## Required Data Sources

Full reproduction requires access to:

- OptionMetrics IvyDB volatility surface data
- CRSP daily and monthly stock returns
- the WRDS CRSP-OptionMetrics historical link table
- Fama-French 5-factor and momentum factor CSV files

## Recreating Data With Access

Users with the required data access can recreate local caches and processed panels with:

```bash
python scripts/public/01_pull_data.py --allow-wrds
python scripts/public/02_build_option_signals.py
python scripts/public/03_build_monthly_panel.py
```

The cache-check command is safe and does not contact WRDS:

```bash
python scripts/public/01_pull_data.py --check-cache-only
```

In a fresh public clone, the cache check is expected to report missing data.

## Local Files Created During Reproduction

The full workflow creates local files such as:

- `data/raw/vol_surface_2010_2023.parquet`
- `data/raw/crsp_daily_2009_12_2023.parquet`
- `data/raw/crsp_monthly_2010_2024.parquet`
- `data/raw/security_master_full.parquet`
- `data/processed/secid_permno_bridge_wrdsapps.parquet`
- `data/processed/daily_iv_signals_2010_2023.parquet`
- `data/processed/daily_signals_with_vrp_2010_2023.parquet`
- `data/processed/monthly_signal_panel_2010_2023.parquet`
- `data/processed/monthly_signal_panel_with_sector_2010_2023.parquet`
- `data/raw/factors/F-F_Research_Data_5_Factors_2x3.csv`
- `data/raw/factors/F-F_Momentum_Factor.csv`

These files are ignored by Git.

## Without Licensed Data Access

Users without WRDS, OptionMetrics, or CRSP access can still review:

- the public code in `src/` and `scripts/public/`
- the curated final results pack
- final tables and figures under `outputs/public_2010_2023/`
- the research paper documents in `docs/`, if included

## Credentials

Credentials are handled by the standard `wrds` package. Do not commit `.pgpass`, `.env`, keys, tokens, or credential files.
