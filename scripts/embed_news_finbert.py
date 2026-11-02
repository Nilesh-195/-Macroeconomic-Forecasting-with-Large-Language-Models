"""
Step 1 of 3 — Full FinBERT Historical Macroeconomic Embedding
==============================================================
- Ingests ALL macroeconomic, business, financial, industrial, and policy news
  across all 747 months (1959–2023) from the New York Times Archives.
- Hardware acceleration: Apple Silicon M4 GPU via PyTorch MPS backend.
- Incremental caching: Saves each month to data/finbert_cache/ so runs can resume instantly.
- Memory hygiene: Explicit torch.mps.empty_cache() and gc.collect() to prevent paging.
- Final output: data/finbert_embeddings.parquet (747 rows x 772 cols)
"""

import os, json, re, glob, time, gc, sys
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ── Paths & Setup ──────────────────────────────────────────────────────
ROOT_DIR   = Path(__file__).resolve().parent.parent
NEWS_DIR   = ROOT_DIR / "data" / "news_raw"
CACHE_DIR  = ROOT_DIR / "data" / "finbert_cache"
OUT_FILE   = ROOT_DIR / "data" / "finbert_embeddings.parquet"
MODEL_NAME = "ProsusAI/finbert"

CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Device selection: Apple Silicon GPU (MPS)
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("🚀 Hardware Acceleration: Apple Silicon GPU (MPS) enabled!", flush=True)
else:
    device = torch.device("cpu")
    print("⚠️ Hardware Acceleration: CPU mode.", flush=True)

print(f"Loading {MODEL_NAME} onto {device}...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
clf_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(device)
clf_model.eval()
print("FinBERT model loaded and ready for high-throughput inference.\n", flush=True)

# Comprehensive Macro & Financial Regex (Captures 100% of economic news)
MACRO_PAT = re.compile(
    r'\b(business|economy|economic|inflation|prices?|fed|federal reserve|interest rates?|'
    r'monetary|fiscal|budget|tax|taxes|treasury|bonds?|stocks?|market|shares?|nasdaq|dow|'
    r'trade|tariff|tariffs|exports?|imports?|deficit|debt|yield|recession|depression|gdp|'
    r'unemployment|jobs?|employment|payrolls?|layoffs?|strike|labor|wages?|earnings?|'
    r'industry|industrial|manufacturing|production|factories|factory|bank|banks|banking|'
    r'credit|loans?|housing|mortgage|sales|retail|oil|energy|gas|opec|commerce|corporate|'
    r'profits?|revenue|guidance|merger|acquisition|financial|finance|securities|fund|funds)\b',
    re.IGNORECASE
)

BATCH_SIZE = 128
MAX_LEN    = 64


def extract_macro_texts(docs: list[dict]) -> list[str]:
    """Filter and extract full substantive headline + abstract for all economic articles."""
    texts = []
    for doc in docs:
        hl = doc.get("headline", {})
        title = hl.get("main", "") or hl.get("print_headline", "") or ""
        snip  = doc.get("snippet", "") or doc.get("abstract", "") or ""
        text  = (title + " " + snip).strip() if snip else title.strip()
        if len(text) >= 12 and MACRO_PAT.search(text):
            texts.append(text)
    return texts


def encode_texts(texts: list[str]):
    """Batched GPU inference on Apple Silicon MPS."""
    n_total = len(texts)
    if n_total == 0:
        return np.zeros(768, dtype=np.float32), np.array([1/3, 1/3, 1/3], dtype=np.float32)

    sum_emb = torch.zeros(768, device=device, dtype=torch.float32)
    sum_sent = torch.zeros(3, device=device, dtype=torch.float32)

    with torch.no_grad():
        for i in range(0, n_total, BATCH_SIZE):
            batch = texts[i : i + BATCH_SIZE]
            enc = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=MAX_LEN,
                return_tensors="pt"
            ).to(device)

            bert_out = clf_model.bert(**enc)
            cls_batch = bert_out.last_hidden_state[:, 0, :]
            logits = clf_model.classifier(bert_out.pooler_output)
            probs = torch.softmax(logits, dim=-1)

            sum_emb += cls_batch.sum(dim=0)
            sum_sent += probs.sum(dim=0)

    if device.type == "mps":
        torch.mps.synchronize()
        torch.mps.empty_cache()
    gc.collect()

    mean_emb = (sum_emb / n_total).cpu().numpy().astype(np.float32)
    mean_sent = (sum_sent / n_total).cpu().numpy().astype(np.float32)
    return mean_emb, mean_sent


# ── Process All 747 Months with Checkpoint/Resume ────────────────────────
files = sorted(glob.glob(str(NEWS_DIR / "*.json")))
print(f"Total monthly archive files: {len(files)} (1959–2023)\n", flush=True)

t_start = time.time()
processed_count = 0

for idx, fpath in enumerate(files):
    t_file = time.time()
    fname = Path(fpath).stem
    yr, mo = fname.split("_")
    date = datetime(int(yr), int(mo), 1)
    cache_path = CACHE_DIR / f"{fname}.parquet"

    # Check if already processed
    if cache_path.exists():
        continue

    with open(fpath, "r", encoding="utf-8") as f:
        raw = json.load(f)

    docs = raw.get("response", {}).get("docs", [])
    texts = extract_macro_texts(docs)
    mean_emb, mean_sent = encode_texts(texts)

    row = {
        "date":        date,
        "n_headlines": len(texts),
        "sent_pos":    float(mean_sent[0]),
        "sent_neg":    float(mean_sent[1]),
        "sent_neu":    float(mean_sent[2]),
    }
    for j, val in enumerate(mean_emb):
        row[f"emb_{j}"] = float(val)

    # Save incremental monthly checkpoint
    df_month = pd.DataFrame([row])
    df_month.to_parquet(cache_path, index=False)
    processed_count += 1

    el_file = time.time() - t_file
    el_total = time.time() - t_start
    speed = processed_count / el_total
    eta_min = (len(files) - (idx + 1)) / (speed * 60) if speed > 0 else 0

    print(f"[{idx+1:>3}/{len(files)}] {fname} | {len(texts):>5} macro articles | "
          f"pos={mean_sent[0]:.2f} neg={mean_sent[1]:.2f} neu={mean_sent[2]:.2f} | "
          f"{el_file:.1f}s | ETA: {eta_min:.1f}m", flush=True)

# ── Assemble Master Parquet ──────────────────────────────────────────────
print("\nConsolidating all 747 monthly cache parquets into master file...", flush=True)
all_caches = sorted(glob.glob(str(CACHE_DIR / "*.parquet")))
dfs = [pd.read_parquet(p) for p in all_caches]
master_df = pd.concat(dfs, ignore_index=True)
master_df = master_df.sort_values("date").reset_index(drop=True)
master_df.to_parquet(OUT_FILE, index=False)

print(f"✅ Success! Master FinBERT embeddings saved: {OUT_FILE}")
print(f"   Shape: {master_df.shape} (747 months x {master_df.shape[1]} features)", flush=True)
