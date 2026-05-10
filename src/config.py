"""Project paths used across the research pipeline.

Keep paths centralized here so scripts and notebooks can refer to the same
folders without hard-coding local machine paths.
"""

from pathlib import Path


# src/config.py lives inside the src/ folder, so the project root is one level up.
ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

OUTPUTS_DIR = ROOT_DIR / "outputs"
CHARTS_DIR = OUTPUTS_DIR / "charts"
TABLES_DIR = OUTPUTS_DIR / "tables"


# Current active research sample. Keep this at 2018-2023 until the expansion
# data pulls and audits are complete.
SAMPLE_START_YEAR = 2018
SAMPLE_END_YEAR = 2023
SIGNAL_START_YEAR = 2018
SIGNAL_END_YEAR = 2023
CURRENT_SAMPLE_LABEL = "2018_2023"
CURRENT_SIGNAL_START_YEAR = 2018
CURRENT_SIGNAL_END_YEAR = 2023

# CRSP daily starts before the signal sample so trailing realized-variance
# lookbacks are available at the beginning of the signal period.
CRSP_DAILY_START_DATE = "2017-12-01"
CRSP_MONTHLY_START_DATE = "2018-01-01"
CRSP_MONTHLY_END_DATE = "2024-01-31"

# Planned expansion settings. These are documented here but are not active yet.
# Suggested first expansion: 2015-2023. Full intended expansion: 2010-2023.
STAGED_EXPANSION_START_YEAR = 2015
STAGED_EXPANSION_END_YEAR = 2023
EXPANSION_START_YEAR = 2010
EXPANSION_END_YEAR = 2023
STAGED_SAMPLE_LABEL = "2015_2023"
STAGED_SIGNAL_START_YEAR = 2015
STAGED_SIGNAL_END_YEAR = 2023
FULL_EXPANSION_SAMPLE_LABEL = "2010_2023"
FULL_EXPANSION_SIGNAL_START_YEAR = 2010
FULL_EXPANSION_SIGNAL_END_YEAR = 2023
FULL_EXPANSION_CRSP_DAILY_START_DATE = "2009-12-01"
FULL_EXPANSION_CRSP_DAILY_END_DATE = "2023-12-31"
FULL_EXPANSION_CRSP_MONTHLY_START_DATE = "2010-01-01"
FULL_EXPANSION_CRSP_MONTHLY_END_DATE = "2024-01-31"


def sample_label(start_year, end_year):
    """Return the standard label for a sample period."""
    return f"{start_year}_{end_year}"


def raw_vol_surface_path(start_year, end_year):
    """Return the raw volatility surface path for a sample period."""
    return RAW_DATA_DIR / f"vol_surface_{start_year}_{end_year}.parquet"


def sample_raw_vol_surface_path(start_year, end_year):
    """Return the raw volatility surface path for a sample period."""
    return RAW_DATA_DIR / f"vol_surface_{start_year}_{end_year}.parquet"


def sample_crsp_daily_path(start_year, end_year):
    """Return the raw CRSP daily path with one prior December for lookbacks."""
    return RAW_DATA_DIR / f"crsp_daily_{start_year - 1}_12_{end_year}.parquet"


def sample_crsp_monthly_path(start_year, return_end_year):
    """Return the raw CRSP monthly path for a signal start and return end year."""
    return RAW_DATA_DIR / f"crsp_monthly_{start_year}_{return_end_year}.parquet"


def processed_panel_path(start_year, end_year):
    """Return the processed monthly panel path for a sample period."""
    return PROCESSED_DATA_DIR / f"monthly_signal_panel_{start_year}_{end_year}.parquet"


def sample_processed_path(filename, sample_label):
    """Return a sample-specific processed parquet path."""
    return PROCESSED_DATA_DIR / f"{filename}_{sample_label}.parquet"


def sample_outputs_dir(sample_label, end_year=None):
    """Return the sample-specific output directory for expansion work."""
    if end_year is not None:
        sample_label = f"{sample_label}_{end_year}"
    return OUTPUTS_DIR / f"sample_{sample_label}"


def sample_tables_dir(sample_label):
    """Return the sample-specific tables directory."""
    return sample_outputs_dir(sample_label) / "tables"


def sample_charts_dir(sample_label):
    """Return the sample-specific charts directory."""
    return sample_outputs_dir(sample_label) / "charts"
