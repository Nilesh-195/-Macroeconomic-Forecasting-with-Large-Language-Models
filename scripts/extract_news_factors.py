"""
Step 2 of 3 — PCA News Factor Extraction
==========================================
Reads data/finbert_embeddings.parquet
Compresses 768-dim FinBERT embeddings → 8 news factors via PCA
Also includes raw sentiment scores (pos, neg, neu)

Output → data/news_factors.parquet
Columns: date, sent_pos, sent_neg, sent_neu, nf_0 … nf_7
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from pathlib import Path

IN_FILE  = Path("data/finbert_embeddings.parquet")
OUT_FILE = Path("data/news_factors.parquet")
N_FACTORS = 8

print("Loading FinBERT embeddings …")
df = pd.read_parquet(IN_FILE)
print(f"  Shape: {df.shape}")

# ── Extract embedding matrix ─────────────────────────────────────────────
emb_cols = [c for c in df.columns if c.startswith("emb_")]
X = df[emb_cols].values.astype(np.float32)   # (747, 768)

# ── Standardize before PCA ───────────────────────────────────────────────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ── PCA → 8 factors ─────────────────────────────────────────────────────
print(f"\nRunning PCA: 768 dims → {N_FACTORS} factors …")
pca = PCA(n_components=N_FACTORS, random_state=42)
factors = pca.fit_transform(X_scaled)          # (747, 8)

explained = pca.explained_variance_ratio_
print(f"  Variance explained by each factor:")
for i, v in enumerate(explained):
    print(f"    Factor {i}: {v*100:.1f}%")
print(f"  Total: {explained.sum()*100:.1f}%")

# ── Also standardize sentiment scores ───────────────────────────────────
sent_cols = ["sent_pos", "sent_neg", "sent_neu"]
sent = df[sent_cols].values
sent_scaler = StandardScaler()
sent_scaled = sent_scaler.fit_transform(sent)

# ── Assemble final DataFrame ─────────────────────────────────────────────
out = pd.DataFrame({"date": df["date"]})
for i, col in enumerate(sent_cols):
    out[col] = sent_scaled[:, i]
for i in range(N_FACTORS):
    out[f"nf_{i}"] = factors[:, i]

out = out.sort_values("date").reset_index(drop=True)
out.to_parquet(OUT_FILE, index=False)

print(f"\nSaved → {OUT_FILE}  shape={out.shape}")
print(out.head(10).to_string())

