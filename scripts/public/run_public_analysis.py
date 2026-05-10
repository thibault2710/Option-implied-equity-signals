"""Run the public analysis pipeline from processed data to final outputs.

This runner executes only scripts in scripts/public/. It assumes the processed
2010-2023 monthly panel and local factor files already exist. It does not pull
WRDS data and does not rebuild raw or processed datasets.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PUBLIC_DIR = Path(__file__).resolve().parent
PUBLIC_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "public_2010_2023"
PROCESSED_MONTHLY_PANEL = PROJECT_ROOT / "data" / "processed" / "monthly_signal_panel_2010_2023.parquet"


@dataclass(frozen=True)
class PipelineStep:
    """One public analysis step."""

    number: str
    script_name: str
    title: str

    @property
    def path(self) -> Path:
        """Absolute path to this public script."""
        return PUBLIC_DIR / self.script_name

    @property
    def file_stem(self) -> str:
        """Script filename without .py."""
        return Path(self.script_name).stem

    @property
    def short_stem(self) -> str:
        """Script stem without leading numeric prefix."""
        prefix = f"{self.number}_"
        if self.file_stem.startswith(prefix):
            return self.file_stem[len(prefix) :]
        return self.file_stem

    @property
    def aliases(self) -> set[str]:
        """Valid CLI selectors for this step."""
        return {
            self.number,
            str(int(self.number)),
            self.file_stem,
            self.short_stem,
            self.script_name,
        }


STEPS = [
    PipelineStep("00", "00_check_environment.py", "Check public analysis environment"),
    PipelineStep("04", "04_run_main_results.py", "Run main portfolio results"),
    PipelineStep("05", "05_run_factor_regressions.py", "Run factor regressions"),
    PipelineStep("06", "06_run_long_only_exclusion.py", "Run long-only exclusion analysis"),
    PipelineStep("07", "07_run_robustness_checks.py", "Run robustness checks"),
    PipelineStep("08", "08_create_final_outputs.py", "Create final public outputs"),
    PipelineStep("09", "09_audit_results.py", "Audit public outputs"),
]


FINAL_OUTPUT_PATHS = [
    PUBLIC_OUTPUT_DIR / "final_results_pack.md",
    PUBLIC_OUTPUT_DIR / "final_tables",
    PUBLIC_OUTPUT_DIR / "final_figures",
    PUBLIC_OUTPUT_DIR / "final_tables" / "audit_status_table_2010_2023.csv",
]


def print_banner(title: str) -> None:
    """Print a readable section banner."""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Run the public processed-panel-to-results analysis pipeline. "
            "This does not pull WRDS data or rebuild processed datasets."
        )
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Print the selected steps without executing them.",
    )
    parser.add_argument(
        "--start-at",
        default=None,
        help="Start at a step number or script stem, for example 05 or run_factor_regressions.",
    )
    parser.add_argument(
        "--stop-after",
        default=None,
        help="Stop after a step number or script stem, for example 07 or run_robustness_checks.",
    )
    parser.add_argument(
        "--skip-check",
        action="store_true",
        help="Skip the environment-check step.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable to use. Defaults to the current interpreter.",
    )
    return parser.parse_args()


def normalize_selector(selector: str | None) -> str | None:
    """Normalize a user-provided step selector."""
    if selector is None:
        return None
    value = selector.strip()
    if value.endswith(".py"):
        value = value[:-3]
    return value


def find_step_index(selector: str | None, label: str) -> int | None:
    """Resolve a CLI selector to a step index."""
    selector = normalize_selector(selector)
    if selector is None:
        return None

    for index, step in enumerate(STEPS):
        if selector in step.aliases:
            return index
    valid = sorted(alias for step in STEPS for alias in step.aliases)
    raise ValueError(f"Unknown {label} selector: {selector}. Valid selectors include: {', '.join(valid)}")


def selected_steps(args: argparse.Namespace) -> list[PipelineStep]:
    """Return the requested step slice."""
    start_index = find_step_index(args.start_at, "--start-at")
    stop_index = find_step_index(args.stop_after, "--stop-after")

    if start_index is None:
        start_index = 0
    if stop_index is None:
        stop_index = len(STEPS) - 1
    if start_index > stop_index:
        raise ValueError("--start-at must be earlier than or equal to --stop-after")

    steps = STEPS[start_index : stop_index + 1]
    if args.skip_check:
        steps = [step for step in steps if step.number != "00"]
    return steps


def validate_steps(steps: list[PipelineStep]) -> None:
    """Verify selected public scripts exist."""
    missing = [step.path for step in steps if not step.path.exists()]
    if missing:
        missing_text = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing public script(s):\n{missing_text}")


def print_step_plan(steps: list[PipelineStep]) -> None:
    """Print the selected step plan."""
    print_banner("Selected Public Analysis Steps")
    if not steps:
        print("No steps selected.")
        return
    for step in steps:
        print(f"{step.number}: {step.title} ({step.script_name})")


def selected_steps_need_processed_panel(steps: list[PipelineStep]) -> bool:
    """Return True if the selected steps include analysis scripts 04-09."""
    return any(step.number != "00" for step in steps)


def print_missing_data_message() -> None:
    """Explain how to proceed from a clean public repository without data."""
    print_banner("Licensed Data Not Found")
    print(f"Missing processed monthly panel: {PROCESSED_MONTHLY_PANEL.relative_to(PROJECT_ROOT)}")
    print()
    print("The public GitHub repository does not include raw or processed licensed data.")
    print("To reproduce from scratch, users with WRDS/OptionMetrics/CRSP access should run:")
    print()
    print("  python scripts/public/01_pull_data.py --allow-wrds")
    print("  python scripts/public/02_build_option_signals.py")
    print("  python scripts/public/03_build_monthly_panel.py")
    print("  python scripts/public/run_public_analysis.py")
    print()
    print("If you do not have licensed data access, you can still inspect:")
    print("  outputs/public_2010_2023/final_results_pack.md")
    print("  outputs/public_2010_2023/final_tables/")
    print("  outputs/public_2010_2023/final_figures/")


def run_step(step: PipelineStep, python_executable: str) -> None:
    """Execute one public script and fail loudly on error."""
    print_banner(f"Step {step.number}: {step.title}")
    print(f"Running: {python_executable} {step.path.relative_to(PROJECT_ROOT)}")
    start = time.time()
    result = subprocess.run(
        [python_executable, str(step.path)],
        cwd=PROJECT_ROOT,
        check=False,
    )
    elapsed = time.time() - start
    if result.returncode != 0:
        raise RuntimeError(
            f"Step {step.number} failed with exit code {result.returncode} after {elapsed:.1f}s: {step.script_name}"
        )
    print(f"Completed step {step.number} in {elapsed:.1f}s")


def print_final_outputs() -> None:
    """Print major public output paths."""
    print_banner("Major Public Output Paths")
    for path in FINAL_OUTPUT_PATHS:
        status = "exists" if path.exists() else "missing"
        print(f"{path.relative_to(PROJECT_ROOT)} [{status}]")


def main() -> int:
    """Run selected public analysis steps."""
    try:
        args = parse_args()
        steps = selected_steps(args)
        validate_steps(steps)
        print_step_plan(steps)

        if args.check_only:
            print("\nCheck-only mode: no scripts were executed.")
            print_final_outputs()
            return 0

        if selected_steps_need_processed_panel(steps) and not PROCESSED_MONTHLY_PANEL.exists():
            print_missing_data_message()
            return 1

        start = time.time()
        for step in steps:
            run_step(step, args.python)
        elapsed = time.time() - start

        print_banner("Public Analysis Complete")
        print(f"Executed {len(steps)} step(s) in {elapsed:.1f}s")
        print_final_outputs()
        return 0
    except Exception as exc:
        print_banner("Public Analysis Failed")
        print(f"Error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
