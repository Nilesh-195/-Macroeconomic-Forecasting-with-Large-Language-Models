"""Stage 1, Section 4: load FRED-MD, select 60-series subset, apply
standard McCracken-Ng stationarity transforms, save one clean panel
every downstream script reads."""
import numpy as np
import pandas as pd

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


def main(src="data/fred_md.csv", dst="data/fredmd_clean.parquet"):
    raw = pd.read_csv(src)
    codes = raw.iloc[0]
    df = raw.iloc[1:].copy()
    df["sasdate"] = pd.to_datetime(df["sasdate"])
    df = df.set_index("sasdate")

    missing = [c for c in FRED_60 if c not in df.columns]
    assert not missing, f"columns not found in csv: {missing}"

    df = df[FRED_60].apply(pd.to_numeric, errors="coerce")
    transformed = pd.DataFrame(
        {c: apply_transform(df[c], codes[c]) for c in FRED_60}, index=df.index
    )
    transformed = transformed.iloc[2:]  # drop rows lost to double-differencing
    transformed = transformed.loc[:"2022-12-31"]  # paper's sample ends Dec 2022; don't use data past it
    transformed.to_parquet(dst)
    print(f"saved {transformed.shape[0]} rows x {transformed.shape[1]} cols -> {dst}")
    print(f"date range: {transformed.index.min().date()} to {transformed.index.max().date()}")
    print(f"NaN cols (top 5):\n{transformed.isna().sum().sort_values(ascending=False).head()}")


def demo():
    """ponytail: minimal check that transform math is right, not a full suite."""
    s = pd.Series([100.0, 110.0, 121.0, 133.1])
    assert np.allclose(apply_transform(s, 1).values, s.values)
    assert np.allclose(apply_transform(s, 2).dropna().values, [10.0, 11.0, 12.1])
    got5 = apply_transform(s, 5).dropna().values
    assert np.allclose(got5, np.log(s).diff().dropna().values)
    print("demo ok")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        demo()
    else:
        main()
