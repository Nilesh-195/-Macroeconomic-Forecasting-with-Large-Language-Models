"""Stage 1, Section 6 (paper's Section 6): TSFM-augmented BVAR hybrid.
1. Fit BVAR, get its h-step forecast and its in-sample residuals (Eq. 12-13).
2. Forecast the residual series zero-shot with a TSFM (Eq. 14).
3. Hybrid forecast = BVAR forecast + TSFM's residual forecast (Eq. 15).
Run inside venv_core. Pass "chronos" (fast, default) or "timesfm" (our
best standalone TSFM, matches the paper's own choice of residual model)
as the residual-model argument.
"""
import sys
import numpy as np
import pandas as pd
import torch

from econ_models import bvar_minnesota_forecast, bvar_residuals

HORIZONS = [1, 3, 6, 12]
ORIGIN_START = "2010-01-01"
ORIGIN_END = "2021-12-01"


def _chronos_predictor():
    from chronos import BaseChronosPipeline
    pipe = BaseChronosPipeline.from_pretrained("amazon/chronos-bolt-small", device_map="cpu", torch_dtype=torch.float32)
    median_idx = pipe.quantiles.index(0.5)

    def predict(contexts):
        return pipe.predict(contexts, prediction_length=max(HORIZONS))[:, median_idx, :].numpy()
    return predict


def _timesfm_predictor():
    import timesfm
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
    model.compile(timesfm.ForecastConfig(max_context=1024, max_horizon=max(HORIZONS), per_core_batch_size=32))

    def predict(contexts):
        arrs = [c.numpy() for c in contexts]
        point_forecast, _ = model.forecast(horizon=max(HORIZONS), inputs=arrs)
        return point_forecast
    return predict


RESIDUAL_MODELS = {"chronos": _chronos_predictor, "timesfm": _timesfm_predictor}


def run(residual_model="chronos", src="data/fredmd_clean.parquet", dst=None):
    dst = dst or f"output/forecasts_hybrid_{residual_model}.parquet"
    model_tag = f"Hybrid_{residual_model}"
    predict = RESIDUAL_MODELS[residual_model]()

    df = pd.read_parquet(src)
    origins = pd.date_range(ORIGIN_START, ORIGIN_END, freq="3MS")

    rows = []
    for oi, t in enumerate(origins):
        hist = df.loc[:t]
        bvar_fc = bvar_minnesota_forecast(hist, HORIZONS)
        resid = bvar_residuals(hist)
        if resid.empty:
            continue

        contexts, valid_cols = [], []
        for c in resid.columns:
            ctx = resid[c].dropna().values
            if len(ctx) >= 10:
                contexts.append(torch.tensor(ctx, dtype=torch.float32))
                valid_cols.append(c)
        if not contexts:
            continue
        resid_fc = predict(contexts)

        for col, preds in zip(valid_cols, resid_fc):
            for h in HORIZONS:
                bvar_h = bvar_fc[h].get(col)
                if bvar_h is None or not np.isfinite(bvar_h):
                    continue
                rows.append(dict(origin=t, series=col, horizon=h, model=model_tag,
                                  value=bvar_h + preds[h - 1]))
        print(f"[{oi + 1}/{len(origins)}] origin {t.date()} done")

    out = pd.DataFrame(rows)
    out.to_parquet(dst)
    print(f"saved {len(out)} forecast rows -> {dst}")
    return out


def demo():
    rng = np.random.default_rng(0)
    idx = pd.date_range("2000-01-01", periods=200, freq="MS")
    df = pd.DataFrame(rng.normal(size=(200, 3)), index=idx, columns=["a", "b", "c"])
    df.to_parquet("/tmp/_hybrid_demo.parquet")
    out = run(residual_model="chronos", src="/tmp/_hybrid_demo.parquet", dst="/tmp/_hybrid_demo_out.parquet")
    assert not out.empty
    print("demo ok")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        demo()
    else:
        run(residual_model=sys.argv[1] if len(sys.argv) > 1 else "chronos")
