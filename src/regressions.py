"""Functions for factor regressions and model output."""

import io
import ssl
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd
import statsmodels.api as sm


FACTOR_MODELS = {
    "CAPM": ["Mkt-RF"],
    "FF3": ["Mkt-RF", "SMB", "HML"],
    "FF5": ["Mkt-RF", "SMB", "HML", "RMW", "CMA"],
    "FF5_MOM": ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "MOM"],
}


def _find_local_factor_file(factors_dir, kind):
    """Find a likely local Ken French CSV for either FF5 or momentum."""
    csv_files = list(factors_dir.glob("*.csv")) + list(factors_dir.glob("*.CSV"))

    if kind == "ff5":
        candidates = [
            path
            for path in csv_files
            if "5" in path.name and "factor" in path.name.lower()
        ]
    else:
        candidates = [
            path
            for path in csv_files
            if "momentum" in path.name.lower() or "mom" in path.name.lower()
        ]

    return candidates[0] if candidates else None


def _read_ken_french_csv(path):
    """Read the monthly section of a Ken French CSV file."""
    with open(path, "r", encoding="latin1") as file:
        lines = file.readlines()

    header_index = None
    for index, line in enumerate(lines):
        lower_line = line.lower()
        line_parts = [item.strip().lower() for item in line.strip().split(",")]
        is_factor_header = len(line_parts) > 1 and (
            "mkt-rf" in line_parts or "mom" in line_parts
        )
        if is_factor_header:
            header_index = index
            break

    if header_index is None:
        raise ValueError(f"Could not find factor header in {path}")

    header = [item.strip() for item in lines[header_index].strip().split(",")]
    header[0] = "month"

    rows = []
    for line in lines[header_index + 1 :]:
        parts = [item.strip() for item in line.strip().split(",")]
        if not parts or not parts[0].isdigit() or len(parts[0]) != 6:
            break
        rows.append(parts[: len(header)])

    if not rows:
        raise ValueError(f"Could not find monthly factor rows in {path}")

    factors = pd.DataFrame(rows, columns=header)
    months = pd.Series(pd.PeriodIndex(factors["month"], freq="M"), name="month", dtype="period[M]")
    numeric_factors = factors.drop(columns=["month"]).apply(pd.to_numeric, errors="coerce")
    numeric_factors = numeric_factors.astype(float)
    factors = pd.concat([months, numeric_factors], axis=1)

    factors = factors.rename(columns={"Mom": "MOM", "mom": "MOM"})
    return factors


def _download_factor_zip(dataset_name, factors_dir):
    """Download one Ken French CSV ZIP and save the extracted CSV locally."""
    url = f"https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/{dataset_name}_CSV.zip"
    output_path = factors_dir / f"{dataset_name}.csv"

    print(f"Trying direct Ken French download: {url}")
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            zip_bytes = response.read()
    except urllib.error.URLError as exc:
        print(f"Default SSL download failed: {exc}")
        try:
            import certifi

            context = ssl.create_default_context(cafile=certifi.where())
            with urllib.request.urlopen(url, timeout=60, context=context) as response:
                zip_bytes = response.read()
        except Exception as certifi_exc:
            print(f"certifi SSL download failed: {certifi_exc}")
            print("Retrying with SSL verification disabled for the public Ken French CSV ZIP.")
            context = ssl._create_unverified_context()
            with urllib.request.urlopen(url, timeout=60, context=context) as response:
                zip_bytes = response.read()

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zip_file:
        csv_names = [name for name in zip_file.namelist() if name.lower().endswith(".csv")]
        if not csv_names:
            raise ValueError(f"No CSV found inside {dataset_name}_CSV.zip")
        output_path.write_bytes(zip_file.read(csv_names[0]))

    print(f"Saved factor CSV: {output_path}")
    return _read_ken_french_csv(output_path)


def _download_factor_data(factors_dir):
    """Download Ken French factors through pandas_datareader."""
    try:
        from pandas_datareader import data as pdr

        ff5 = pdr.DataReader("F-F_Research_Data_5_Factors_2x3", "famafrench", start="1900-01")[0]
        mom = pdr.DataReader("F-F_Momentum_Factor", "famafrench", start="1900-01")[0]

        ff5 = ff5.reset_index().rename(columns={"Date": "month"})
        mom = mom.reset_index().rename(columns={"Date": "month", "Mom": "MOM"})

        ff5.loc[:, "month"] = pd.PeriodIndex(ff5["month"], freq="M")
        mom.loc[:, "month"] = pd.PeriodIndex(mom["month"], freq="M")

        return ff5, mom
    except Exception as exc:
        print(f"pandas_datareader download failed: {exc}")
        print("Trying direct Ken French ZIP downloads...")

    ff5 = _download_factor_zip("F-F_Research_Data_5_Factors_2x3", factors_dir)
    mom = _download_factor_zip("F-F_Momentum_Factor", factors_dir)
    return ff5, mom


def load_factor_data(raw_data_dir):
    """Load monthly Fama-French 5 factors and momentum."""
    raw_data_dir = Path(raw_data_dir)
    factors_dir = raw_data_dir / "factors"
    factors_dir.mkdir(parents=True, exist_ok=True)

    ff5_path = _find_local_factor_file(factors_dir, kind="ff5")
    mom_path = _find_local_factor_file(factors_dir, kind="mom")

    if ff5_path is not None and mom_path is not None:
        print(f"Loading local Fama-French 5-factor CSV: {ff5_path}")
        print(f"Loading local momentum CSV: {mom_path}")
        ff5, mom = _read_ken_french_csv(ff5_path), _read_ken_french_csv(mom_path)
    else:
        print("Local factor CSVs not found in data/raw/factors/.")
        print("Trying to download Ken French factors with pandas_datareader...")
        try:
            ff5, mom = _download_factor_data(factors_dir)
        except Exception as exc:
            print("\nCould not download Ken French factors.")
            print(f"Error: {exc}")
            print("\nManual fallback:")
            print("1. Download these monthly CSV files from Ken French's data library:")
            print("   - F-F_Research_Data_5_Factors_2x3")
            print("   - F-F_Momentum_Factor")
            print(f"2. Place the CSV files in: {factors_dir}")
            print("3. Rerun: python scripts/08_run_factor_regressions.py")
            raise RuntimeError("Factor data are required for regressions.") from exc

    ff5 = ff5.rename(columns={"Mkt-RF": "Mkt-RF", "Mom": "MOM"})
    mom = mom.rename(columns={"Mom": "MOM", "mom": "MOM"})

    factor_df = ff5.merge(mom[["month", "MOM"]], on="month", how="inner")
    factor_columns = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF", "MOM"]
    factor_df = factor_df[["month"] + factor_columns].copy()
    factor_df.loc[:, "return_month"] = pd.PeriodIndex(factor_df["month"].astype(str), freq="M")

    decimal_factors = factor_df[factor_columns].apply(pd.to_numeric, errors="coerce")
    decimal_factors = decimal_factors.astype(float) / 100
    factor_df = pd.concat([factor_df[["month", "return_month"]], decimal_factors], axis=1)

    factor_df = factor_df.dropna(subset=factor_columns).reset_index(drop=True)

    print("\nLoaded factor data.")
    print(f"Shape: {factor_df.shape}")
    print(f"Date range: {factor_df['return_month'].min()} to {factor_df['return_month'].max()}")
    print(f"Columns: {list(factor_df.columns)}")
    print("\nFactor summary stats:")
    print(factor_df[factor_columns].describe().to_string())

    return factor_df


def load_long_short_returns(tables_dir, selected_portfolios):
    """Load selected long-short return CSV files."""
    tables_dir = Path(tables_dir)
    ls_returns = {}

    for portfolio in selected_portfolios:
        file_path = tables_dir / portfolio["file_name"]
        returns = pd.read_csv(file_path)

        if "signal_month" in returns.columns:
            signal_month_col = "signal_month"
        elif "month" in returns.columns:
            signal_month_col = "month"
        else:
            raise ValueError(f"No month column found in {file_path}")

        if "LS" not in returns.columns:
            raise ValueError(f"No LS column found in {file_path}")

        label = portfolio["label"]
        columns_to_keep = [signal_month_col, "LS"]
        if "return_month" in returns.columns:
            columns_to_keep.append("return_month")

        cleaned = returns[columns_to_keep].copy()
        cleaned = cleaned.rename(columns={signal_month_col: "signal_month"})
        cleaned = cleaned.assign(
            signal_month=pd.PeriodIndex(cleaned["signal_month"].astype(str), freq="M"),
            LS=pd.to_numeric(cleaned["LS"], errors="coerce"),
        )

        if "return_month" in cleaned.columns:
            cleaned = cleaned.assign(
                return_month=pd.PeriodIndex(cleaned["return_month"].astype(str), freq="M")
            )
        else:
            cleaned.loc[:, "return_month"] = cleaned["signal_month"] + 1

        cleaned = cleaned[["signal_month", "return_month", "LS"]]
        cleaned = cleaned.dropna(subset=["signal_month", "return_month", "LS"]).reset_index(drop=True)

        ls_returns[label] = cleaned

        print(f"Loaded {label}: {file_path}")
        print(f"Shape: {cleaned.shape}")
        print(f"signal_month range: {cleaned['signal_month'].min()} to {cleaned['signal_month'].max()}")
        print(f"return_month range: {cleaned['return_month'].min()} to {cleaned['return_month'].max()}")
        print("LS summary stats:")
        print(cleaned["LS"].describe().to_string())

    return ls_returns


def run_factor_regression(ls_df, factor_df, portfolio_label, model_name, factor_cols, nw_lags=4):
    """Run one time-series factor regression with Newey-West standard errors."""
    returns = ls_df.copy()
    factors = factor_df.copy()

    returns = returns.assign(
        signal_month=pd.PeriodIndex(returns["signal_month"].astype(str), freq="M"),
        return_month=pd.PeriodIndex(returns["return_month"].astype(str), freq="M"),
        LS=pd.to_numeric(returns["LS"], errors="coerce").astype(float),
    )

    if "return_month" not in factors.columns:
        factors = factors.assign(return_month=factors["month"])
    factors = factors.assign(
        return_month=pd.PeriodIndex(factors["return_month"].astype(str), freq="M")
    )

    merged = returns.merge(factors, on="return_month", how="inner")
    merged = merged.dropna(subset=["LS"] + factor_cols)

    # For long-short portfolios, LS is already a self-financing return, so we
    # do not subtract RF.
    y = merged["LS"].astype(float)

    x = sm.add_constant(merged[factor_cols].astype(float), has_constant="add")
    model = sm.OLS(y, x).fit(cov_type="HAC", cov_kwds={"maxlags": nw_lags})

    first_signal_month = merged["signal_month"].min()
    first_return_month = merged["return_month"].min()
    last_signal_month = merged["signal_month"].max()
    last_return_month = merged["return_month"].max()

    result = {
        "portfolio": portfolio_label,
        "model": model_name,
        "alpha_monthly": model.params["const"],
        "alpha_annualized": model.params["const"] * 12,
        "alpha_tstat": model.tvalues["const"],
        "alpha_pvalue": model.pvalues["const"],
        "r_squared": model.rsquared,
        "n_months": int(model.nobs),
        "first_signal_month": first_signal_month,
        "first_return_month": first_return_month,
        "last_signal_month": last_signal_month,
        "last_return_month": last_return_month,
    }

    for factor in factor_cols:
        result[f"beta_{factor}"] = model.params.get(factor, pd.NA)
        result[f"tstat_{factor}"] = model.tvalues.get(factor, pd.NA)

    return result


def run_all_factor_regressions(ls_returns_dict, factor_df):
    """Run all specified factor models for each selected portfolio."""
    rows = []

    for portfolio_label, ls_df in ls_returns_dict.items():
        for model_name, factor_cols in FACTOR_MODELS.items():
            print(f"Running {model_name} regression for {portfolio_label}")
            rows.append(
                run_factor_regression(
                    ls_df,
                    factor_df,
                    portfolio_label,
                    model_name,
                    factor_cols,
                )
            )

    summary_df = pd.DataFrame(rows)
    return summary_df


def save_regression_summary(summary_df, output_path):
    """Save the factor regression summary table."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output_path, index=False)
    print(f"\nSaved factor regression summary to {output_path}")
    print(f"Saved shape: {summary_df.shape}")


def print_regression_highlights(summary_df):
    """Print compact highlights from the factor regression output."""
    ff5_mom = summary_df.loc[summary_df["model"] == "FF5_MOM"].copy()

    print("\nFF5 + Momentum alpha highlights:")
    columns = [
        "portfolio",
        "alpha_annualized",
        "alpha_tstat",
        "alpha_pvalue",
        "r_squared",
        "n_months",
        "first_signal_month",
        "last_signal_month",
        "first_return_month",
        "last_return_month",
    ]
    print(ff5_mom[columns].to_string(index=False))

    print("\nTiming range by portfolio under FF5 + Momentum:")
    timing_columns = [
        "portfolio",
        "first_signal_month",
        "last_signal_month",
        "first_return_month",
        "last_return_month",
        "n_months",
    ]
    print(ff5_mom[timing_columns].to_string(index=False))

    print("\nTop portfolios by FF5 + Momentum alpha t-stat:")
    print(
        ff5_mom.sort_values("alpha_tstat", ascending=False)[columns]
        .head(10)
        .to_string(index=False)
    )

    significant = ff5_mom.loc[ff5_mom["alpha_tstat"] > 2, columns]
    print("\nPortfolios with FF5 + Momentum alpha t-stat > 2:")
    if significant.empty:
        print("None")
    else:
        print(significant.to_string(index=False))
