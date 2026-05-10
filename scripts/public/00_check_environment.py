"""Check the local environment for the public 2010-2023 pipeline.

This script is public-release aware. It exits with a nonzero code only when
required Python packages or required public repository files/folders are
missing. Licensed raw data, processed Parquet files, and regenerated diagnostic
outputs are expected to be absent from a clean GitHub clone and are reported as
warnings only.
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "options_implied_signals_matplotlib"))
os.environ.setdefault("MPLBACKEND", "Agg")


REQUIRED_PACKAGES = ["pandas", "numpy", "matplotlib", "statsmodels", "pyarrow"]
OPTIONAL_PACKAGES = ["scipy", "wrds"]

PUBLIC_REPO_STRUCTURE = [
    "README.md",
    "requirements.txt",
    "src",
    "scripts/public",
    "data/README.md",
    "outputs/README.md",
]

CURATED_PUBLIC_OUTPUTS = [
    "outputs/public_2010_2023/final_results_pack.md",
    "outputs/public_2010_2023/final_tables",
    "outputs/public_2010_2023/final_figures",
]

LOCAL_REPRODUCTION_FILES = [
    "data/raw/vol_surface_2010_2023.parquet",
    "data/raw/crsp_daily_2009_12_2023.parquet",
    "data/raw/crsp_monthly_2010_2024.parquet",
    "data/raw/security_master_full.parquet",
    "data/processed/secid_permno_bridge_wrdsapps.parquet",
    "data/processed/daily_iv_signals_2010_2023.parquet",
    "data/processed/daily_signals_with_vrp_2010_2023.parquet",
    "data/processed/monthly_signal_panel_2010_2023.parquet",
    "data/processed/monthly_signal_panel_with_sector_2010_2023.parquet",
    "data/raw/factors/F-F_Research_Data_5_Factors_2x3.csv",
    "data/raw/factors/F-F_Momentum_Factor.csv",
]

PROCESSED_MONTHLY_PANEL = "data/processed/monthly_signal_panel_2010_2023.parquet"


def print_header(title: str) -> None:
    """Print a clean section header."""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def project_path(relative_path: str) -> Path:
    """Return a project-root-relative path."""
    return PROJECT_ROOT / relative_path


def check_packages(packages: list[str], required: bool) -> list[str]:
    """Check importability for a list of packages."""
    missing = []
    label = "required" if required else "optional"
    print_header(f"Checking {label} Python packages")
    for package in packages:
        try:
            importlib.import_module(package)
            print(f"PASS {package}")
        except Exception as exc:
            missing.append(package)
            status = "MISSING" if required else "WARN optional missing"
            print(f"{status} {package}: {exc}")
    return missing


def check_paths(title: str, relative_paths: list[str], missing_status: str) -> list[str]:
    """Check path existence and return missing paths."""
    missing = []
    print_header(title)
    for relative_path in relative_paths:
        path = project_path(relative_path)
        if path.exists():
            print(f"PASS {relative_path}")
        else:
            missing.append(relative_path)
            print(f"{missing_status} missing: {relative_path}")
    return missing


def check_wrds() -> bool:
    """Optionally check WRDS connectivity without pulling data."""
    print_header("Optional WRDS connectivity check")
    try:
        import wrds

        db = wrds.Connection()
        db.close()
        print("PASS WRDS connection opened and closed successfully.")
        return True
    except Exception as exc:
        print(f"FAIL WRDS connection check failed: {exc}")
        return False


def main() -> None:
    """Run public-release-aware environment checks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-wrds", action="store_true", help="Check WRDS connectivity without pulling data.")
    args = parser.parse_args()

    print_header("Public Pipeline Environment Check")
    print(f"Project root: {PROJECT_ROOT}")

    missing_required_packages = check_packages(REQUIRED_PACKAGES, required=True)
    missing_optional_packages = check_packages(OPTIONAL_PACKAGES, required=False)
    missing_public_structure = check_paths(
        "Checking required public repository structure",
        PUBLIC_REPO_STRUCTURE,
        missing_status="FAIL",
    )
    missing_curated_outputs = check_paths(
        "Checking curated public outputs",
        CURATED_PUBLIC_OUTPUTS,
        missing_status="WARN",
    )
    missing_local_data = check_paths(
        "Checking local reproduction data caches",
        LOCAL_REPRODUCTION_FILES,
        missing_status="WARN",
    )

    if missing_local_data:
        print(
            "\nLicensed raw and processed data are not included in the public repository. "
            "Users with WRDS/OptionMetrics/CRSP access can recreate local caches with "
            "`python scripts/public/01_pull_data.py --allow-wrds`."
        )

    wrds_checked = args.check_wrds
    wrds_ok = True
    if args.check_wrds:
        wrds_ok = check_wrds()
    else:
        print_header("WRDS check")
        print("SKIP WRDS was not checked. Pass --check-wrds to test connectivity.")

    processed_panel_available = project_path(PROCESSED_MONTHLY_PANEL).exists()
    public_repo_ready = not missing_required_packages and not missing_public_structure and wrds_ok
    local_data_present = not missing_local_data
    curated_outputs_available = not missing_curated_outputs

    print_header("Project status summary")
    print(f"Missing required packages: {missing_required_packages or 'none'}")
    print(f"Missing optional packages: {missing_optional_packages or 'none'}")
    print(f"Missing required public repo files/folders: {missing_public_structure or 'none'}")
    print(f"Missing curated public outputs: {missing_curated_outputs or 'none'}")
    print(f"Missing local data caches: {missing_local_data or 'none'}")
    print(f"Public repo structure ready: {public_repo_ready}")
    print(f"Local data caches present: {local_data_present}")
    print(f"Processed panel available: {processed_panel_available}")
    print(f"Ready to run analysis scripts 04-09: {processed_panel_available}")
    print(f"Curated final outputs available: {curated_outputs_available}")
    print(f"WRDS checked: {wrds_checked}")

    if missing_required_packages or missing_public_structure or not wrds_ok:
        raise SystemExit(1)

    print("\nPASS environment check completed. Missing licensed data or curated outputs are warnings only.")


if __name__ == "__main__":
    main()
