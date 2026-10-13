"""Stage 1, Section 6: LagLlama, zero-shot, univariate.
Run inside venv_lagllama (git-cloned repo on sys.path + its pinned
gluonts/torch, own env per the guide -- most conflict-prone install).
"""
import sys
import warnings
import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from gluonts.dataset.pandas import PandasDataset

# harmless internal deprecation notice from gluonts' fancy-indexing helper,
# fires once per forecast call and drowns out real progress output
warnings.filterwarnings("ignore", message="Using a non-tuple sequence for multidimensional indexing")

sys.path.insert(0, "lag-llama")
from lag_llama.gluon.estimator import LagLlamaEstimator
from gluonts.torch.distributions.studentT import StudentTOutput
from gluonts.torch.modules.loss import NegativeLogLikelihood

# lag-llama's checkpoint pickles these gluonts classes; PyTorch >=2.6 defaults
# torch.load to weights_only=True, so allowlist them explicitly (official HF
# checkpoint from the model's own authors -- trusted source).
torch.serialization.add_safe_globals([StudentTOutput, NegativeLogLikelihood])

HORIZONS = [1, 3, 6, 12]
ORIGIN_START = "2010-01-01"
ORIGIN_END = "2021-12-01"
PRED_LEN = max(HORIZONS)


def load_predictor():
    ckpt_path = hf_hub_download(repo_id="time-series-foundation-models/Lag-Llama", filename="lag-llama.ckpt")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)  # official HF checkpoint, trusted source
    args = ckpt["hyper_parameters"]["model_kwargs"]

    estimator = LagLlamaEstimator(
        ckpt_path=ckpt_path,
        prediction_length=PRED_LEN,
        context_length=32 * PRED_LEN,
        input_size=args["input_size"],
        n_layer=args["n_layer"],
        n_embd_per_head=args["n_embd_per_head"],
        n_head=args["n_head"],
        scaling=args["scaling"],
        time_feat=args["time_feat"],
        batch_size=1,
        num_parallel_samples=20,
    )
    transformation = estimator.create_transformation()
    lightning_module = estimator.create_lightning_module()
    return estimator.create_predictor(transformation, lightning_module)


def run(src="data/fredmd_clean.parquet", dst="output/forecasts_lagllama.parquet"):
    df = pd.read_parquet(src)
    origins = pd.date_range(ORIGIN_START, ORIGIN_END, freq="3MS")
    predictor = load_predictor()

    rows = []
    for oi, t in enumerate(origins):
        hist = df.loc[:t]
        long_frames = []
        for c in df.columns:
            ctx = hist[c].dropna()
            if len(ctx) >= 10:
                long_frames.append(pd.DataFrame({"item_id": c, "timestamp": ctx.index,
                                                   "target": ctx.values.astype(np.float32)}))
        if not long_frames:
            continue
        long_df = pd.concat(long_frames).set_index("timestamp")
        dataset = PandasDataset.from_long_dataframe(long_df, item_id="item_id", target="target", freq="M")

        forecasts = list(predictor.predict(dataset, num_samples=20))
        for fc in forecasts:
            median = np.median(fc.samples, axis=0)
            for h in HORIZONS:
                rows.append(dict(origin=t, series=fc.item_id, horizon=h, model="LagLlama", value=median[h - 1]))
        print(f"[{oi + 1}/{len(origins)}] origin {t.date()} done")

    out = pd.DataFrame(rows)
    out.to_parquet(dst)
    print(f"saved {len(out)} forecast rows -> {dst}")
    return out


def demo():
    idx = pd.date_range("2000-01-01", periods=400, freq="M")
    ctx = np.sin(np.arange(400) / 3.0).astype(np.float32)
    long_df = pd.DataFrame({"item_id": "a", "timestamp": idx, "target": ctx}).set_index("timestamp")
    ds = PandasDataset.from_long_dataframe(long_df, item_id="item_id", target="target", freq="M")
    predictor = load_predictor()
    fc = list(predictor.predict(ds, num_samples=5))[0]
    assert fc.samples.shape[1] == PRED_LEN
    print("demo ok")


if __name__ == "__main__":
    import sys as _sys
    if len(_sys.argv) > 1 and _sys.argv[1] == "--test":
        demo()
    else:
        run()
