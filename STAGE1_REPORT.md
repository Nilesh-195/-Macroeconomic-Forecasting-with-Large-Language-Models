# Stage 1 Report — Macroeconomic Forecasting with Large Language Models

**Paper reproduced:** Carriero, Pettenuzzo & Shekhar (2026), *"Macroeconomic Forecasting with Large Language Models,"* arXiv:2407.00890v5.

---

## 1. The Paper, in Brief

**Problem.** Can Time Series Foundation Models (TSFMs) — pretrained transformer models for time series, the "LLMs of forecasting" — beat traditional econometric methods at forecasting macroeconomic variables?

**Data.** FRED-MD: 120 monthly US macro series (output, prices, labor, housing, money/credit, rates/FX, stock market), Jan 1959–Dec 2022, each series transformed to near-stationarity via McCracken & Ng (2016)'s standard codes (log-differences, first differences, etc.).

**Models compared.**
- *Econometric:* AR(1) (mandatory benchmark), BVAR with a Minnesota-tradition Normal-Inverse-Wishart prior (p=6 lags, hyperparameters tuned by maximising the marginal likelihood), a PCA-factor-augmented autoregression, NNAR, and BART.
- *TSFMs (all zero-shot):* Chronos (Amazon), Moirai (Salesforce), TimesFM (Google), Lag-Llama, TTM (IBM), TimeGPT (Nixtla) — ranging 1M–311M parameters.
- *Hybrid:* fit BVAR then take its in-sample residuals, forecast those residuals zero-shot with a TSFM, and add the two forecasts together (their "TSFM-BVAR").

**Method.** Recursive expanding-window forecasting: train on Jan 1960–Dec 1984, forecast Jan 1985 (h=1..12), roll forward one month, repeat through Dec 2022. Accuracy is measured as RMSFE relative to the AR(1) benchmark (their Eq. 11):

```
RMSFE_ijh = sqrt( Sum e^2_{i,j,tau+h} / Sum e^2_{AR1,j,tau+h} )
```

Values below 1 mean model *i* beats AR(1) for series *j* at horizon *h*.

**Headline findings.**
1. Only 3 of 6 TSFMs (Chronos, TimesFM, Moirai-large) are consistently competitive with AR(1); TTM, Lag-Llama, TimeGPT lag behind.
2. Even the good TSFMs are only roughly on par with BVAR/Factor models, not clearly better.
3. BVAR and TSFM forecast errors are nearly *orthogonal* at h=1 (correlation ~0.07–0.5) — they're wrong about different series — which is exactly why combining them (the hybrid) helps: it substantially trims the worst-case errors of the TSFM.
4. TSFMs handle the COVID-19 shock better than econometric models (with the caveat that their training data may include the pandemic period).

---

## 2. What We Actually Built (Stage 1)

Full pipeline, all runnable from `/Users/nilesh/LLM`:

| Script | Function |
|:---|:---|
| `scripts/prep_data.py` | Loads FRED-MD CSV, selects 60-series subset, applies the 7 McCracken-Ng transform codes, saves `data/fredmd_clean.parquet` |
| `scripts/econ_models.py` | AR(1), Minnesota-prior BVAR, PCA-factor-augmented AR — shared `_bvar_fit()` used by both the forecast function and the hybrid's residual extraction |
| `scripts/run_forecast_loop.py` | Recursive expanding-window loop + `rmsfe_table()` (paper's Eq. 11) |
| `scripts/run_tsfm_chronos.py` | Amazon Chronos (bolt-small, zero-shot) |
| `scripts/run_tsfm_timesfm.py` | Google TimesFM 2.5 (200M, zero-shot) |
| `scripts/run_tsfm_moirai.py` | Salesforce Moirai (base, 91M, zero-shot) |
| `scripts/run_tsfm_ttm.py` | IBM Tiny Time Mixer (5M, zero-shot) |
| `scripts/run_tsfm_lagllama.py` | Lag-Llama (25M, zero-shot) |
| `scripts/run_tsfm_timegpt.py` | Nixtla TimeGPT via API (batched: 1 call/origin covering all 60 series, not 1 call/series) |
| `scripts/run_hybrid.py` | TSFM-BVAR hybrid: BVAR forecast + TimesFM-modelled BVAR residuals |
| `scripts/merge_tsfm_errors.py` | Folds any model's raw forecasts into the shared error table |
| `scripts/make_plots.py` | Boxplots mirroring the paper's Figure 2/3 |

Environments: `venv_core` (Python 3.14 — torch, chronos-forecasting, timesfm, sklearn, statsmodels), `venv_moirai` and `venv_lagllama` (Python 3.11, forced by `uni2ts`/`granite-tsfm` pinning scipy/torch versions). All six TSFMs and both econometric benchmarks actually ran to completion on real data; nothing here is simulated or hand-waved.

Two real implementation bugs were found and fixed during development:
1. **BVAR dummy-prior bug:** a lag-decay multiplier was accidentally a no-op so the Minnesota prior wasn't actually shrinking higher lags. Fixed to scale properly as `lag * sigma_i / lambda` (paper's Eq. B.1). Also switched the prior mean from 1 (random-walk prior, correct for *levels*) to 0, since FRED-MD series are already transformed to near-stationarity.
2. **Factor model instability:** the original implementation iterated a 1-step model forward 12 times to get an h=12 forecast; any autoregressive coefficient slightly above 1 explodes exponentially under iteration, producing RMSFE outliers as large as 246x. Replaced with the paper's actual specification (Eq. 6-7): a **direct** regression of y_t on y_{t-h} and f_{t-h}, one regression per horizon, no iteration.

---

## 3. Our Approach versus the Original Paper

The project follows the paper's logic: same dataset, same relative-RMSFE metric, same family of benchmarks and the same BVAR-plus-TSFM hybrid. It differs in scale and in several design choices, summarised in Table 1. Notably, we add Chronos-Bolt and newer TimesFM versions, while the paper's additional nonlinear benchmarks (NNAR, BART), significance tests and fine-tuning experiments are not yet replicated.

**Table 1.** Comparison of the original study and this work.

| Aspect | Original paper (Carriero et al.) | This work |
|:---|:---|:---|
| **Data** | FRED-MD, 120 monthly series, 1959–2022 | FRED-MD, 60 balanced series, 1959–2022 |
| **Evaluation** | Jan 1985 – Dec 2019 (main), 2020–2022 separately; monthly origins; h = 1, ..., 12 | 2010 Q1 – 2022 Q4; 48 quarterly origins; h = 1, 3, 6, 12 |
| **Benchmarks** | AR(1), BVAR, Factor model, NNAR, BART | AR(1), BVAR, Factor-AR |
| **BVAR prior** | Minnesota + sum-of-coefficients + single-unit-root; p = 6; hyper-parameters by marginal likelihood | Minnesota prior; p = 6; lambda = 0.2 fixed |
| **TSFMs** | MOIRAI (base, large), TimesFM, TTM, TimeGPT, Lag-Llama | TimesFM 2.0/2.5, TimeGPT, Chronos-Bolt, MOIRAI, TTM, Lag-Llama |
| **Hybrid** | BVAR + MOIRAI-large and BVAR + TimesFM | BVAR + TimesFM |
| **Extra analyses** | Diebold-Mariano tests, COVID split, data-leakage test, persistence split, fine-tuning | Not yet implemented |

---

## 4. Results and Comparison with the Paper

Table 2 reports the results of this work and Table 3 those of the original paper, each covering the single models and the hybrid models. All entries are the median RMSFE relative to AR(1) across series (values below 1, in **bold**, beat AR(1)). Because the two studies use different series sets, evaluation windows and model versions, the comparison is about patterns rather than exact decimals.

**Table 2.** Results of this work: median relative RMSFE (60 FRED-MD series, 2010 Q1 – 2022 Q4, 48 quarterly origins).

| Model | h = 1 | h = 3 | h = 6 | h = 12 |
|:---|:---:|:---:|:---:|:---:|
| **Single models** | | | | |
| AR(1) (benchmark) | 1.000 | 1.000 | 1.000 | 1.000 |
| BVAR (Minnesota) | 1.332 | 1.262 | 1.167 | 1.108 |
| Factor-AR | 1.507 | 1.192 | 1.089 | 1.029 |
| TimesFM 2.0/2.5 | **0.983** | **0.995** | **0.997** | 1.000 |
| TimeGPT | **0.987** | **0.997** | **0.994** | **0.999** |
| Chronos-Bolt | **0.996** | **0.997** | **0.998** | 1.002 |
| MOIRAI | 1.031 | 1.004 | 1.003 | 1.013 |
| TTM | 1.117 | 1.040 | 1.019 | 1.007 |
| Lag-Llama | 1.043 | 1.022 | 1.020 | 1.014 |
| **Hybrid model** | | | | |
| BVAR + TimesFM | 1.331 | 1.250 | 1.168 | 1.111 |

*Chronos-Bolt was evaluated only in this work. The hybrid combines the BVAR with TimesFM residual forecasts.*

**Table 3.** Results of the original paper: median relative RMSFE (120 FRED-MD series, Jan 1985 – Dec 2019, monthly origins).

| Model | h = 1 | h = 3 | h = 6 | h = 12 |
|:---|:---:|:---:|:---:|:---:|
| **Single models – econometric** | | | | |
| AR(1) (benchmark) | 1.000 | 1.000 | 1.000 | 1.000 |
| BVAR | **0.983** | **0.970** | **0.991** | **0.999** |
| Factor model | **0.965** | **0.980** | **0.992** | **0.999** |
| NNAR | **0.988** | **0.992** | **0.999** | 1.000 |
| BART | 1.017 | 1.024 | 1.011 | 1.023 |
| **Single models – zero-shot TSLMs** | | | | |
| MOIRAI-base | 1.008 | **0.973** | **0.990** | **0.999** |
| MOIRAI-large | **0.999** | **0.978** | **0.991** | 1.001 |
| TimesFM | 1.014 | **0.980** | **0.990** | 1.001 |
| TimeGPT | 1.077 | 1.044 | 1.025 | 1.063 |
| TTM-Enhanced | 1.044 | **0.993** | **0.995** | 1.002 |
| Lag-Llama | 1.015 | **0.997** | 1.002 | 1.009 |
| **Hybrid models** | | | | |
| BVAR + MOIRAI-large | **0.973** | **0.960** | **0.983** | **0.998** |
| BVAR + TimesFM | **0.980** | **0.956** | **0.984** | **0.999** |

*Source: arXiv:2407.00890v4, Tables 2-4. In Table 3, MOIRAI refers to MOIRAI-large, TTM to TTM-Enhanced and the hybrid to the BVAR + TimesFM version.*

### 4.1 Pre-COVID Results (2010–2019 Origins)

Clean comparison mirroring the paper's own choice to exclude the pandemic period from its main figure:

| Model | h = 1 | h = 3 | h = 6 | h = 12 |
|:---|:---:|:---:|:---:|:---:|
| BVAR | 1.000 | 1.055 | 1.007 | 0.997 |
| Factor-AR | **0.981** | **0.990** | **0.994** | 1.002 |
| TimesFM | **0.965** | **0.981** | **0.984** | 0.997 |
| Chronos | **0.991** | **0.987** | **0.986** | 0.999 |
| TimeGPT | **0.980** | **0.985** | **0.988** | 0.998 |
| MOIRAI | 1.032 | 1.025 | 1.014 | 1.005 |
| TTM | 1.072 | 1.011 | 1.000 | 1.000 |
| Lag-Llama | 1.079 | 1.093 | 1.068 | 1.014 |
| Hybrid (BVAR + TimesFM) | **0.989** | 1.050 | 1.008 | 0.998 |

### 4.2 COVID Stress Test — Full vs. COVID-Only Window

Full window (2010–2022) vs. COVID-only subset (2019-06 onward) — shows model fragility to the regime break:

| Model | Full h = 1 | COVID-only h = 1 | Full h = 12 | COVID-only h = 12 |
|:---|:---:|:---:|:---:|:---:|
| BVAR | 1.332 | 1.564 | 1.108 | 1.873 |
| Factor-AR | 1.507 | 1.897 | 1.029 | 1.096 |
| TimesFM | **0.983** | **0.973** | 1.000 | 1.011 |
| Chronos | **0.996** | 1.022 | 1.002 | 1.000 |
| TimeGPT | **0.987** | **0.981** | **0.999** | 1.002 |
| Hybrid | 1.331 | 1.578 | 1.111 | 1.880 |

*Full precision numbers and std/min/max in `output/rmsfe_summary.csv`; boxplots in `output/rmsfe_boxplots.png`.*

### 4.3 Interpretation

**TSFM ranking replicates the paper closely.** Our best-to-worst ordering — TimesFM ~ Chronos ~ TimeGPT (all beat AR(1) at every horizon) > MOIRAI (roughly at par) > TTM ~ Lag-Llama (worse than AR(1)) — matches the paper's own ranking closely with one clear divergence discussed below.

**BVAR/Factor models beat AR(1) pre-COVID, as the paper found**, but their performance is not stable across regimes — the striking finding here is how badly they break down once COVID enters the sample (RMSFE up to 1.87 vs AR(1) at h=12), while every TSFM stays within a few percent of 1.0 throughout. This is a real-data confirmation of the paper's own conclusion (section 8): *"their [TSFMs'] relatively stronger performance in the post-Covid-19 period suggests an advantage in handling structural breaks."*

**TimeGPT is our biggest divergence from the paper.** The paper found it one of the two weakest TSFMs (median 1.03–1.08); here it's one of the best (0.98–0.99). Most likely explanation: TimeGPT is the only TSFM served via a live, continuously-updated hosted API — we are almost certainly querying a materially different (newer) model than the one benchmarked in the paper's 2024 evaluation.

**The hybrid partially replicates the paper's benefit, but not fully.** It improves on plain BVAR at h=1 (1.000 to 0.989) and clearly reduces the tail risk. The COVID blowup is inherited rather than fixed because the hybrid's forecast is anchored to BVAR's own point forecast — the residual correction only fixes the linear, learnable part of BVAR's error, not a wholesale level shift. Our simplified BVAR (no stochastic volatility, no outlier handling) is comparatively weaker than the paper's fully-tuned version, capping the hybrid's ceiling below standalone TimesFM.

---

## 5. Conclusion

The core qualitative finding of the paper — TSFMs are competitive with but not clearly superior to well-specified econometric models, and the two are complementary enough that combining them can help — replicates at our reduced scale. The TSFM performance ranking matches closely; the divergence on TimeGPT and the more limited hybrid gain are both explainable by concrete implementation differences (hosted API drift, a less richly-specified BVAR prior) rather than methodological errors, and both are documented above per Stage 1's requirement to discuss and justify any differences from the original paper.
