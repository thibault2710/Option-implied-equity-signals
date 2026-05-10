# Public Pipeline Step 10: Data Pull and Cache Check

## Summary

Created a cache-first public data preparation entrypoint for the 2010-2023 pipeline.

## Files Created

- `scripts/public/01_pull_data.py`
- `outputs/public_2010_2023/tables/public_data_pull_cache_status.csv`
- `outputs/public_2010_2023/tables/public_data_pull_summary.csv`
- `outputs/public_2010_2023/tables/public_data_pull_validation.csv`

## Source Changes

No `src/` files were modified.

## Legacy Development Files

The public script does not execute legacy development files.

## CLI Flags

- `--check-cache-only`: check local caches only and do not connect to WRDS.
- `--allow-wrds`: allow WRDS connection for missing selected files.
- `--force-pull`: repull selected WRDS-backed files; requires `--allow-wrds`.
- `--skip-optionmetrics`, `--skip-crsp`, `--skip-security-master`, `--skip-link-table`, `--skip-factors`: skip selected file groups.
- `--start-date`, `--crsp-daily-start-date`, `--end-date`, `--crsp-monthly-end-date`: override public sample date bounds.

## Cache Status

- Cache status totals: PASS=0, WARN=7, FAIL=0, INFO=0
- Missing files: vol_surface, crsp_daily, crsp_monthly, security_master, link_table, ff5_factors, momentum_factor

## Validation

- Validation totals: PASS=0, WARN=7, FAIL=0, INFO=0

## WRDS and File Writes

- WRDS contacted: False
- Raw or processed cache files written: 0

Validation rows needing attention:

| file_key | check | status | details |
| --- | --- | --- | --- |
| vol_surface | file_exists | WARN | Missing cache. |
| crsp_daily | file_exists | WARN | Missing cache. |
| crsp_monthly | file_exists | WARN | Missing cache. |
| security_master | file_exists | WARN | Missing cache. |
| link_table | file_exists | WARN | Missing cache. |
| ff5_factors | file_exists | WARN | Missing local factor CSV. |
| momentum_factor | file_exists | WARN | Missing local factor CSV. |

## GitHub Safety

The script is safe for GitHub because default/cache-check mode does not connect to WRDS, does not store credentials, and reports missing licensed data clearly.

## Current Run Summary

| run_mode | selected_files | missing_files | would_pull_with_wrds | wrds_contacted | force_pull | raw_or_processed_files_written | written_paths | validation_pass | validation_warn | validation_fail | validation_info |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| check_cache_only | vol_surface, crsp_daily, crsp_monthly, security_master, link_table, ff5_factors, momentum_factor | vol_surface, crsp_daily, crsp_monthly, security_master, link_table, ff5_factors, momentum_factor | vol_surface, crsp_daily, crsp_monthly, security_master, link_table | False | False | 0 |  | 0 | 7 | 0 | 0 |

## Recommended Next Step

The standalone public pipeline scripts are now complete. Next, update the public README to document the full workflow and licensed-data exclusions.
