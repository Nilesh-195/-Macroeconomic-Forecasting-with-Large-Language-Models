"""
Stage 2, Section 2: Load FRED-MD, support full 120-series panel (paper specification)
with 60-series fallback, apply standard McCracken-Ng stationarity transforms,
and save clean parquet panels.
"""
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# The 60-series balanced representative subset (fallback)
FRED_60 = [
    "RPI", "INDPRO", "IPFPNSS", "IPFINAL", "IPMAT", "CUMFNS", "DPCERA3M086SBEA", "CMRMTSPLx",
    "CLF16OV", "CE16OV", "UNRATE", "UEMPMEAN", "UEMP15OV", "CLAIMSx", "PAYEMS",
    "USGOOD", "MANEMP", "SRVPRD", "USTRADE", "USGOVT", "AWHMAN", "CES0600000007",
    "HOUST", "HOUSTNE", "HOUSTS", "PERMIT", "PERMITS",
    "RETAILx", "ACOGNO", "AMDMNOx", "BUSINVx", "ISRATIOx",
    "M1SL", "M2SL", "M2REAL", "BOGMBASE", "TOTRESNS", "BUSLOANS",
    "FEDFUNDS", "TB3MS", "TB6MS", "GS1", "GS5", "GS10", "AAA", "BAA", "EXUSUKx", "EXJPUSx",
    "WPSFD49207", "PPICMM", "CPIAUCSL", "CPIAPPSL", "CPIMEDSL", "CPITRNSL", "CPIULFSL", "PCEPI", "OILPRICEx",
    "S&P 500", "S&P div yield", "VIXCLSx",
]


def apply_transform(s: pd.Series, code: int) -> pd.Series:
    """Apply McCracken-Ng (2016) stationarity transformation code."""
    code = int(code)
    if code == 1:
        return s
    if code == 2:
        return s.diff()
    if code == 3:
        return s.diff().diff()
    if code == 4:
        return np.log(s)
    if code == 5:
        return np.log(s).diff()
    if code == 6:
        return np.log(s).diff().diff()
    if code == 7:
        return s.pct_change().diff()
    raise ValueError(f"unknown transform code {code}")


def select_120_series(raw_df: pd.DataFrame) -> list[str]:
    """Select the 120 most complete series, matching Carriero et al. (2026)."""
    cols = [c for c in raw_df.columns if c != "sasdate"]
    data = raw_df.iloc[1:][cols].apply(pd.to_numeric, errors="coerce")
    nan_counts = data.isna().sum()
    
    # Sort by fewest NaNs to drop the ~6 series that started late in FRED-MD
    sorted_cols = nan_counts.sort_values().index.tolist()
    chosen_120 = sorted_cols[:120]
    return chosen_120


def process_dataset(src="data/fred_md.csv", series_count=120):
    raw = pd.read_csv(src)
    codes = raw.iloc[0]
    df = raw.iloc[1:].copy()
    df["sasdate"] = pd.to_datetime(df["sasdate"])
    df = df.set_index("sasdate")

    if series_count == 60:
        chosen_cols = [c for c in FRED_60 if c in df.columns]
    else:
        chosen_cols = select_120_series(raw)

    print(f"Processing panel with {len(chosen_cols)} series (target: {series_count})...")

    # Numeric conversion
    sub_df = df[chosen_cols].apply(pd.to_numeric, errors="coerce")

    # Impute small internal missing values before transformation
    sub_df = sub_df.ffill().bfill()

    # Apply McCracken-Ng transforms
    transformed = pd.DataFrame(
        {c: apply_transform(sub_df[c], codes[c]) for c in chosen_cols}, index=df.index
    )

    # Drop the first 2 months lost to 2nd-order differencing
    transformed = transformed.iloc[2:]
    # Limit sample to 2022-12 (matching evaluation standard)
    transformed = transformed.loc[:"2022-12-31"]

    # Final forward/back fill for any NaN created by differencing near borders
    transformed = transformed.ffill().bfill()

    # Destination paths
    dst_specific = ROOT / f"data/fredmd_clean_{len(chosen_cols)}.parquet"
    dst_main = ROOT / "data/fredmd_clean.parquet"

    transformed.to_parquet(dst_specific)
    print(f"Saved {transformed.shape[0]} months x {transformed.shape[1]} series -> {dst_specific}")

    # If 120 series, also update default fredmd_clean.parquet
    if len(chosen_cols) >= 120:
        transformed.to_parquet(dst_main)
        print(f"Updated default panel -> {dst_main}")

    return transformed


def main():
    parser = argparse.ArgumentParser(description="Clean and transform FRED-MD panel")
    parser.add_argument("--series", type=int, choices=[60, 120], default=120,
                        help="Number of series to select (default: 120)")
    parser.add_argument("--both", action="store_true", help="Generate both 60 and 120 series panels")
    args = parser.parse_args()

    if args.both:
        process_dataset(series_count=60)
        process_dataset(series_count=120)
    else:
        process_dataset(series_count=args.series)


if __name__ == "__main__":
    main()
