"""Stage 1, Section 7: recursive out-of-sample forecasting loop and
RMSFE evaluation (paper Eq. 11), scaled per the project guide:
quarterly forecast origins, 2010-2022, horizons 1/3/6/12, AR(1)+BVAR+Factor-AR.
"""
import numpy as np
import pandas as pd

from econ_models import ar1_forecast, bvar_minnesota_forecast, factor_ar_forecast

HORIZONS = [1, 3, 6, 12]
ORIGIN_START = "2010-01-01"
ORIGIN_END = "2021-12-01"  # last origin s.t. origin+12m <= 2022-12


def run(src="data/fredmd_clean.parquet", dst="output/forecast_errors.parquet"):
    df = pd.read_parquet(src)
    origins = pd.date_range(ORIGIN_START, ORIGIN_END, freq="3MS")

    rows = []
    for t in origins:
        hist = df.loc[:t]
        actuals_future = df.loc[t:].iloc[1:max(HORIZONS) + 1]  # t+1 .. t+12

        bvar_fc = bvar_minnesota_forecast(hist, HORIZONS)
        factor_fc = factor_ar_forecast(hist, HORIZONS)

        for col in df.columns:
            ar_fc = ar1_forecast(hist[col], HORIZONS)
            for h in HORIZONS:
                if h > len(actuals_future):
                    continue
                actual = actuals_future[col].iloc[h - 1]
                if pd.isna(actual):
                    continue
                for model, fc in [
                    ("AR1", ar_fc.get(h)),
                    ("BVAR", bvar_fc[h].get(col)),
                    ("FactorAR", factor_fc[h].get(col)),
                ]:
                    if fc is None or not np.isfinite(fc):
                        continue
                    rows.append(dict(origin=t, series=col, horizon=h, model=model,
                                      sq_err=(actual - fc) ** 2))

    errs = pd.DataFrame(rows)
    errs.to_parquet(dst)
    print(f"saved {len(errs)} forecast-error rows -> {dst}")
    return errs


def rmsfe_table(errs: pd.DataFrame) -> pd.DataFrame:
    """Paper's Eq. 11: RMSFE_ijh = sqrt(sum e_i^2 / sum e_AR1^2), per series & horizon."""
    sums = errs.groupby(["model", "series", "horizon"])["sq_err"].sum().unstack("model")
    ratio = sums.div(sums["AR1"], axis=0).pow(0.5)
    return ratio.drop(columns="AR1")


def demo():
    """ponytail: sanity check on synthetic data, not the full pipeline."""
    rng = np.random.default_rng(0)
    idx = pd.date_range("2005-01-01", periods=220, freq="MS")
    df = pd.DataFrame(rng.normal(size=(220, 2)), index=idx, columns=["a", "b"])
    df.to_parquet("/tmp/_demo.parquet")
    errs = run(src="/tmp/_demo.parquet", dst="/tmp/_demo_errs.parquet")
    assert not errs.empty
    tbl = rmsfe_table(errs)
    assert {"BVAR", "FactorAR"}.issubset(tbl.columns)
    print("demo ok")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        demo()
    else:
        errs = run()
        tbl = rmsfe_table(errs)
        summary = tbl.groupby("horizon").agg(["median", "std", "min", "max"])
        print(summary)
        summary.to_csv("output/rmsfe_summary.csv")
