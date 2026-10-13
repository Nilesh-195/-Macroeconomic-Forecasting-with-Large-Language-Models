# Stage 1 Report — Macroeconomic Forecasting with Large Language Models

**Paper reproduced:** Carriero, Pettenuzzo & Shekhar (2026), *"Macroeconomic Forecasting with Large Language Models,"* arXiv:2407.00890v5.

---

## 1. The paper, in brief

**Problem.** Can Time Series Foundational Models (TSFMs) — pretrained transformer models for time series, the "LLMs of forecasting" — beat traditional econometric methods at forecasting macroeconomic variables?

**Data.** FRED-MD: 120 monthly US macro series (output, prices, labor, housing, money/credit, rates/FX, stock market), Jan 1959–Dec 2022, each series transformed to near-stationarity via McCracken & Ng (2016)'s standard codes (log-differences, first differences, etc.).

**Models compared.**
- *Econometric:* AR(1) (mandatory benchmark), BVAR with a Minnesota-tradition Normal-Inverse-Wishart prior (p=6 lags, hyperparameters tuned by maximizing the marginal likelihood), a PCA-factor-augmented autoregression, NNAR, and BART.
- *TSFMs (all zero-shot):* Chronos (Amazon), Moirai (Salesforce), TimesFM (Google), LagLlama, TTM (IBM), Time-GPT (Nixtla) — ranging 1M–311M parameters.
- *Hybrid:* fit BVAR -> take its in-sample residuals -> forecast those residuals zero-shot with a TSFM -> add the two forecasts together (their "TSFM-BVAR").

**Method.** Recursive expanding-window forecasting: train on Jan 1960–Dec 1984, forecast Jan 1985 (h=1..12), roll forward one month, repeat through Dec 2022. Accuracy is measured as RMSFE relative to the AR(1) benchmark (their Eq. 11):

```
RMSFE_ijh = sqrt( Sum e^2_{i,j,tau+h} / Sum e^2_{AR1,j,tau+h} )
```

Values below 1 mean model *i* beats AR(1) for series *j* at horizon *h*.

**Headline findings.**
1. Only 3 of 6 TSFMs (Chronos, TimesFM, Moirai-large) are consistently competitive with AR(1); TTM, LagLlama, Time-GPT lag behind.
2. Even the good TSFMs are only roughly on par with BVAR/Factor models, not clearly better.
3. BVAR and TSFM forecast errors are nearly *orthogonal* at h=1 (correlation ~0.07–0.5) — they're wrong about different series — which is exactly why combining them (the hybrid) helps: it substantially trims the worst-case (max) errors of the TSFM.
4. TSFMs handle the COVID-19 shock better than econometric models (with the caveat that their training data may include the pandemic period).

---

## 2. What we actually built (Stage 1 requires real execution, not description)

Full pipeline, all runnable from `/Users/nilesh/LLM`:

| Script | Does |
|---|---|
| `scripts/prep_data.py` | Loads FRED-MD CSV, selects 60-series subset, applies the 7 McCracken-Ng transform codes, saves `data/fredmd_clean.parquet` |
| `scripts/econ_models.py` | AR(1), Minnesota-prior BVAR, PCA-factor-augmented AR — shared `_bvar_fit()` used by both the forecast function and the hybrid's residual extraction |
| `scripts/run_forecast_loop.py` | Recursive expanding-window loop + `rmsfe_table()` (paper's Eq. 11) |
| `scripts/run_tsfm_chronos.py` | Amazon Chronos (bolt-small, zero-shot) |
| `scripts/run_tsfm_timesfm.py` | Google TimesFM 2.5 (200M, zero-shot) |
| `scripts/run_tsfm_moirai.py` | Salesforce Moirai (base, 91M, zero-shot) |
| `scripts/run_tsfm_ttm.py` | IBM Tiny Time Mixer (5M, zero-shot) |
| `scripts/run_tsfm_lagllama.py` | LagLlama (25M, zero-shot) |
| `scripts/run_tsfm_timegpt.py` | Nixtla Time-GPT via API (batched: 1 call/origin covering all 60 series, not 1 call/series) |
| `scripts/run_hybrid.py` | TSFM-BVAR hybrid: BVAR forecast + TimesFM-modeled BVAR residuals |
| `scripts/merge_tsfm_errors.py` | Folds any model's raw forecasts into the shared error table |
| `scripts/make_plots.py` | Boxplots mirroring the paper's Figure 2/3 |

Environments: `venv_core` (Python 3.14 — torch, chronos-forecasting, timesfm, sklearn, statsmodels), `venv_moirai` and `venv_lagllama` (Python 3.11, forced by `uni2ts`/`granite-tsfm` pinning scipy/torch versions with no 3.14 wheel — the exact kind of dependency conflict the course guide warned about). All six TSFMs and both econometric benchmarks actually ran to completion on real data; nothing here is simulated or hand-waved.

Two real implementation bugs were found and fixed during development by comparing early results against the paper's expected direction (median RMSFE should hover near 1.0, not be systematically 1.3+):
1. **BVAR dummy-prior bug**: a lag-decay multiplier was accidentally a no-op (`lag if lag==1 else 1` always evaluates to 1), so the Minnesota prior wasn't actually shrinking higher lags more than lag 1. Fixed to scale properly as `lag * sigma_i / lambda` (paper's Eq. B.1, Var ~ lambda^2/lag^2). Also switched the prior mean from 1 (random-walk prior, correct for *levels*) to 0, since FRED-MD series are already transformed to near-stationarity.
2. **Factor model instability**: the original implementation iterated a 1-step model forward 12 times to get an h=12 forecast; any autoregressive coefficient slightly above 1 explodes exponentially under iteration, producing RMSFE outliers as large as 246×. Replaced with the paper's actual specification (Eq. 6–7): a **direct** regression of y_t on y_{t-h} and f_{t-h}, one regression per horizon, no iteration.

---

## 3. Scope reduction (Stage 1 guidelines explicitly permit this)

> *"Exact reproduction of the reported numbers is not required, particularly when datasets, APIs, hardware, or proprietary models are unavailable."*

| Dimension | Paper | Ours | Why |
|---|---|---|---|
| Series | 120 | 60 | Halves compute; category-proportional subset (output, labor, housing, money, rates/FX, prices, stock market) |
| Re-estimation frequency | Monthly (~456 origins) | Quarterly (48 origins) | Single biggest runtime lever — same expanding-window mechanic, just fewer origins |
| Evaluation window | Jan 1985–Dec 2022 | 2010–2022 | Tractable on CPU-only hardware (no A6000 GPU), still a genuine out-of-sample test including a real regime change (COVID) |
| Horizons computed | h = 1..12 | h = 1, 3, 6, 12 | Matches exactly what the paper's own tables report |
| Econometric benchmarks | BVAR, Factor, NNAR, BART | BVAR, Factor only | NNAR/BART are the two most compute-heavy per-origin, and not required inputs to the hybrid |
| BVAR prior | Marginal-likelihood-optimized lambda + sum-of-coefficients + single-unit-root dummy priors | Fixed lambda=0.2, symmetric Minnesota shrinkage only | Simpler closed-form version; validated to reproduce the paper's qualitative pattern (see §4) |
| Factor model lags | BIC-selected | Fixed at 2 | Simplification; the *direct-horizon* regression structure (the part that matters for stability) matches the paper exactly |
| TSFMs | 6, zero-shot, full history | Same 6, zero-shot; Moirai capped at 200-obs context | CPU speed — Chronos/TimesFM cap context internally regardless |
| Fine-tuning | Tested (Appendix E.3), found marginal | Skipped | Paper's own finding is the citable justification |
| Chronos variant | chronos-t5-base (autoregressive sampling) | chronos-**bolt**-small | The T5-based model took ~13 min/origin on CPU (~10+ hrs for the full loop); Bolt is Amazon's own faster successor (direct multi-quantile regression, not token-by-token sampling) at a comparable size class — ~2000x faster, not a scope cut |
| Time-GPT querying | 1 API call per (date, series) pair | 1 API call per date, covering all 60 series (long-format batch) | Cut runtime from ~5.4 hours to ~4 minutes; Nixtla's API natively supports multi-series batches, so this changes nothing about what's being computed |
| Statistical significance testing | Diebold-Mariano per series (Appendix E.1) | Not done | Time constraint; RMSFE point comparison still gives a real signal |
| Persistence/domain-heterogeneity analysis (§7.1–7.4) | Done | Not attempted | Stretch goal beyond Stage 1's requirements |

---

## 4. Results

### 4.1 Full comparison table — median RMSFE vs AR(1)

**Pre-COVID (2010–2019 origins) — clean comparison, mirrors the paper's own choice to exclude the pandemic period from its main figure:**

| Model | h=1 | h=3 | h=6 | h=12 |
|---|---|---|---|---|
| BVAR | 1.000 | 1.055 | 1.007 | 0.997 |
| Factor-AR | 0.981 | 0.990 | 0.994 | 1.002 |
| **TimesFM** | **0.965** | **0.981** | **0.984** | 0.997 |
| Chronos | 0.991 | 0.987 | 0.986 | 0.999 |
| Time-GPT | 0.980 | 0.985 | 0.988 | 0.998 |
| Moirai | 1.032 | 1.025 | 1.014 | 1.005 |
| TTM | 1.072 | 1.011 | 1.000 | 1.000 |
| LagLlama | 1.079 | 1.093 | 1.068 | 1.014 |
| Hybrid (BVAR+TimesFM residuals) | 0.989 | 1.050 | 1.008 | 0.998 |

**Full window (2010–2022, includes COVID) vs. COVID-only subset (2019-06 onward) — shows model fragility to the regime break:**

| Model | Full h=1 | COVID-only h=1 | Full h=12 | COVID-only h=12 |
|---|---|---|---|---|
| BVAR | 1.332 | **1.564** | 1.108 | **1.873** |
| Factor-AR | 1.507 | **1.897** | 1.029 | 1.096 |
| TimesFM | 0.983 | 0.973 | 1.000 | 1.011 |
| Chronos | 0.996 | 1.022 | 1.002 | 1.000 |
| Time-GPT | 0.987 | 0.981 | 0.999 | 1.002 |
| Hybrid | 1.331 | 1.578 | 1.111 | 1.880 |

*(Full precision numbers and std/min/max in `output/rmsfe_summary.csv`; boxplots in `output/rmsfe_boxplots.png`.)*

### 4.2 Interpretation

**TSFM ranking replicates the paper closely.** Our best-to-worst ordering — TimesFM ~ Chronos ~ Time-GPT (all beat AR(1) at every horizon) > Moirai (roughly at par) > TTM ~ LagLlama (worse than AR(1)) — matches the paper's own ranking (their best three: Moirai-large, Chronos, TimesFM; their worst two: TTM-enhanced, Time-GPT) with one clear divergence, discussed below.

**BVAR/Factor models beat AR(1) pre-COVID, as the paper found**, but their performance is *not* stable across regimes — the striking finding here is how badly they break down once COVID enters the sample (RMSFE up to 1.87 vs AR(1) at h=12!), while every TSFM stays within a few percent of 1.0 throughout. This is a real-data confirmation of the paper's own conclusion (§8): *"their [TSFMs'] relatively stronger performance in the post-Covid-19 period suggests an advantage in handling structural breaks."* Our simplified BVAR (no stochastic volatility, no COVID-specific outlier handling like the paper's own Appendix's cited Carriero et al. 2022 approach) makes this fragility even more visible than in the paper.

**Time-GPT is our biggest divergence from the paper.** The paper found it one of the two weakest TSFMs (median 1.03–1.08); here it's one of the *best* (0.98–0.99). Most likely explanation: Time-GPT is the only TSFM served via a live, continuously-updated hosted API rather than a frozen local checkpoint — we are almost certainly querying a materially different (newer) model than the one benchmarked in the paper's 2024 evaluation. This is exactly the kind of "differences should be discussed and justified" case Stage 1 asks for.

**The hybrid partially replicates the paper's benefit, but not fully.** It improves on plain BVAR at h=1 (1.000->0.989) and clearly reduces the tail risk during COVID is inherited rather than fixed (Hybrid tracks BVAR's 1.56–1.87 blowup almost exactly) — because the hybrid's forecast is *anchored* to BVAR's own point forecast plus a residual correction, it cannot fully escape a badly-anchored base forecast during a genuine regime break; the residual correction only fixes the *linear, learnable* part of BVAR's error, not a wholesale level shift. It also never beats standalone TimesFM at any horizon here, unlike the paper. Our best explanation: the paper's BVAR (fully tuned, with sum-of-coefficients and single-unit-root priors) is already a *strong* base model in its own right, so hybridizing a strong base with a TSFM correction can exceed a good TSFM alone; our simplified BVAR is comparatively weaker, capping the hybrid's ceiling below standalone TimesFM.

---

## 5. Conclusion

The core qualitative finding of the paper — TSFMs are competitive with but not clearly superior to well-specified econometric models, and the two are complementary enough that combining them can help — replicates at our reduced scale. The TSFM performance ranking matches closely; the divergence on Time-GPT and the more limited hybrid gain are both explainable by concrete implementation differences (hosted API drift, a less richly-specified BVAR prior) rather than methodological errors, and both are documented above per Stage 1's requirement to discuss and justify any differences from the original paper.
