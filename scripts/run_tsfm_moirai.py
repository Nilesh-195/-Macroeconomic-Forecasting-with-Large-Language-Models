"""Stage 1, Section 6: Moirai (Salesforce), zero-shot, univariate instances.
Run inside venv_moirai (uni2ts + gluonts, Python 3.11 -- uni2ts's pinned
scipy has no 3.14 wheel, per the guide's per-model env isolation).
"""
import numpy as np
import pandas as pd
import torch
from uni2ts.model.moirai import MoiraiForecast, MoiraiModule

HORIZONS = [1, 3, 6, 12]
ORIGIN_START = "2010-01-01"
ORIGIN_END = "2021-12-01"
MODEL_NAME = "Salesforce/moirai-1.0-R-base"  # 91M params
MAX_CONTEXT = 200  # ponytail: cap context length for CPU speed, not full history


def load_predictor():
    model = MoiraiForecast(
        module=MoiraiModule.from_pretrained(MODEL_NAME),
        prediction_length=max(HORIZONS),
        context_length=MAX_CONTEXT,
        patch_size="auto",
        num_samples=20,
        target_dim=1,
        feat_dynamic_real_dim=0,
        past_feat_dynamic_real_dim=0,
    )
    return model.create_predictor(batch_size=32)


def run(src="data/fredmd_clean.parquet", dst="output/forecasts_moirai.parquet"):
    df = pd.read_parquet(src)
    origins = pd.date_range(ORIGIN_START, ORIGIN_END, freq="3MS")
    predictor = load_predictor()

    rows = []
    for oi, t in enumerate(origins):
        hist = df.loc[:t]
        dataset, valid_cols = [], []
        for c in df.columns:
            ctx = hist[c].dropna()
            if len(ctx) >= 10:
                ctx = ctx.iloc[-MAX_CONTEXT:]
                dataset.append({"start": pd.Period(ctx.index[0], freq="M"),
                                 "target": ctx.values.astype(np.float32)})
                valid_cols.append(c)
        if not dataset:
            continue
        forecasts = list(predictor.predict(dataset))
        for col, fc in zip(valid_cols, forecasts):
            median = np.median(fc.samples, axis=0)  # (H,)
            for h in HORIZONS:
                rows.append(dict(origin=t, series=col, horizon=h, model="Moirai", value=median[h - 1]))
        print(f"[{oi + 1}/{len(origins)}] origin {t.date()} done")

    out = pd.DataFrame(rows)
    out.to_parquet(dst)
    print(f"saved {len(out)} forecast rows -> {dst}")
    return out


def demo():
    idx = pd.date_range("2000-01-01", periods=60, freq="MS")
    ctx = np.sin(np.arange(60) / 3.0).astype(np.float32)
    predictor = load_predictor()
    ds = [{"start": pd.Period(idx[0], freq="M"), "target": ctx}]
    fc = list(predictor.predict(ds))[0]
    assert fc.samples.shape[1] == max(HORIZONS)
    print("demo ok")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        demo()
    else:
        run()
