"""Merge a TSFM's raw forecasts into the shared RMSFE error table.
Usage: python scripts/merge_tsfm_errors.py output/forecasts_chronos.parquet
"""
import sys
import pandas as pd

from run_forecast_loop import rmsfe_table


def merge(tsfm_path, actuals_path="data/fredmd_clean.parquet",
          errors_path="output/forecast_errors.parquet"):
    actuals = pd.read_parquet(actuals_path)
    fc = pd.read_parquet(tsfm_path)

    fc["target_date"] = fc.apply(lambda r: r.origin + pd.DateOffset(months=int(r.horizon)), axis=1)
    fc["actual"] = fc.apply(lambda r: actuals.at[r.target_date, r.series]
                             if r.target_date in actuals.index else float("nan"), axis=1)
    fc = fc.dropna(subset=["actual"])
    fc["sq_err"] = (fc["actual"] - fc["value"]) ** 2

    errs = pd.read_parquet(errors_path)
    model_name = fc["model"].iloc[0]
    errs = errs[errs.model != model_name]  # replace if re-running
    merged = pd.concat([errs, fc[["origin", "series", "horizon", "model", "sq_err"]]], ignore_index=True)
    merged.to_parquet(errors_path)
    print(f"merged {len(fc)} {model_name} rows -> {errors_path} ({len(merged)} total)")
    return merged


if __name__ == "__main__":
    merged = merge(sys.argv[1])
    tbl = rmsfe_table(merged)
    summary = tbl.groupby("horizon").agg(["median", "std", "min", "max"])
    print(summary)
    summary.to_csv("output/rmsfe_summary.csv")
