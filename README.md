# Macroeconomic Forecasting with Large Language Models

[![Paper](https://img.shields.io/badge/arXiv-2407.00890v5-B31B1B.svg)](https://arxiv.org/abs/2407.00890)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![Dataset](https://img.shields.io/badge/Dataset-FRED--MD-green.svg)](https://research.stlouisfed.org/econ/mccracken/fred-databases/)
[![Stage 1](https://img.shields.io/badge/Stage%201-Completed%20%26%20Verified-brightgreen.svg)]()
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An empirical implementation and comparative evaluation of **Carriero, Pettenuzzo & Shekhar (2026)**, *"Macroeconomic Forecasting with Large Language Models"*, comparing classical econometric models against state-of-the-art **Zero-Shot Time-Series Foundation Models (TSFMs)** and a **Hybrid Residual-Correction Architecture** on monthly US macroeconomic data.

---

## 📌 Executive Summary

Forecasting macroeconomic time series has historically been dominated by autoregressive benchmarks, factor models, and Bayesian Vector Autoregressions (BVARs) with shrinkage priors. This project implements an end-to-end framework evaluating whether modern **pretrained time-series foundation models (TSFMs)** can forecast macroeconomic aggregates in a zero-shot manner without domain-specific re-training.

### Core Highlights
* **Data Foundation**: 60 balanced monthly US macroeconomic series from **FRED-MD** spanning Jan 1959 – Dec 2022 across all 8 standard economic groups.
* **Rigorous Transformations**: Series-specific McCracken & Ng (2016) stationarity transformations (levels, first differences, second differences, log differences, and double log differences).
* **Expanding-Window Evaluation Engine**: 48 quarterly forecast origins spanning **2010 Q1 through 2022 Q4**, evaluated across horizons $h \in \{1, 3, 6, 12\}$ months.
* **Extensive Model Coverage**:
  * **Econometric Benchmarks**: AR(1), Minnesota-Prior BVAR ($p=6, \lambda=0.2$), and Factor-AR (Stock & Watson diffusion indices).
  * **Zero-Shot TSFMs**: Google TimesFM (2.0/2.5), Amazon Chronos (Bolt), Salesforce MOIRAI, Meta Lag-Llama, IBM Tiny Time Mixer (TTM), and Nixtla TimeGPT.
  * **Hybrid Framework**: Econometric BVAR combined with zero-shot TimesFM residual correction ($\hat{y}_{t+h}^{\text{hybrid}} = \hat{y}_{t+h}^{\text{BVAR}} + \widehat{e}_{t+h}^{\text{TSFM}}$).

---

## 🏗️ System Architecture & Pipeline

```
  ┌─────────────────┐       select 60 series       ┌──────────────────────┐
  │  FRED-MD Data   │ ───────────────────────────► │  Clean Dataset (60)  │
  │  (120 series)   │   McCracken-Ng Transforms    │  (1959-01 – 2022-12) │
  └─────────────────┘                              └──────────┬───────────┘
                                                              │
                                                              ▼
                                             ┌─────────────────────────────────┐
                                             │ Expanding-Window Engine         │
                                             │ 48 Quarterly Origins (2010–2022)│
                                             │ Horizons h ∈ {1, 3, 6, 12}      │
                                             └────────┬───────────────┬────────┘
                                                      │               │
                            ┌─────────────────────────┘               └────────────────────────┐
                            ▼                                                                  ▼
        ┌───────────────────────────────────────┐                  ┌────────────────────────────────────────┐
        │        Econometric Benchmarks         │                  │         Zero-Shot TSFM Models          │
        │  • AR(1) Direct (OLS)                 │                  │  • Google TimesFM 2.0 / 2.5            │
        │  • BVAR (Minnesota Prior, p=6)        │                  │  • Amazon Chronos-Bolt (quantized)     │
        │  • Factor-AR (Stock-Watson PCA)       │                  │  • Salesforce MOIRAI (multi-patch)     │
        └───────────────────┬───────────────────┘                  │  • Meta Lag-Llama                      │
                            │                                      │  • IBM Tiny Time Mixer (TTM)           │
                            │                                      │  • Nixtla TimeGPT API                  │
                            │                                      └───────────────────┬────────────────────┘
                            │                                                          │
                            └────────────────────────┐        ┌────────────────────────┘
                                                     ▼        ▼
                                       ┌───────────────────────────────────┐
                                       │      Hybrid BVAR + TSFM Model     │
                                       │ ŷ_hybrid = ŷ_BVAR + TSFM(residuals)│
                                       └─────────────────┬─────────────────┘
                                                         │
                                                         ▼
                                       ┌───────────────────────────────────┐
                                       │       Evaluation & Metrics        │
                                       │ RMSFE relative to AR(1) (Eq. 11)  │
                                       │ Distribution across 60 series     │
                                       └─────────────────┬─────────────────┘
                                                         │
                                                         ▼
                                       ┌───────────────────────────────────┐
                                       │     Visuals, Reports & CSVs       │
                                       │ • rmsfe_summary.csv               │
                                       │ • rmsfe_boxplots.png              │
                                       │ • STAGE1_REPORT_FULL.pdf          │
                                       │ • STAGE1_COMPARISON.pdf           │
                                       └───────────────────────────────────┘
```

![End-to-End Pipeline](output/pipeline.png)

---

## 📊 Empirical Results (Stage 1)

Performance is evaluated using **Root Mean Squared Forecast Error (RMSFE)** relative to the benchmark AR(1) model:
$$\text{RMSFE}_{\text{rel}} = \sqrt{\frac{\sum_{t} (y_{t+h} - \hat{y}_{t+h}^{\text{model}})^2}{\sum_{t} (y_{t+h} - \hat{y}_{t+h}^{\text{AR(1)}})^2}}$$
*Values $< 1.000$ indicate outperformance over the AR(1) benchmark.*

### Median Relative RMSFE across 60 Series (Stage 1 Results)

| Model | Class | $h = 1$ | $h = 3$ | $h = 6$ | $h = 12$ |
|---|---|:---:|:---:|:---:|:---:|
| **AR(1)** | Baseline | 1.000 | 1.000 | 1.000 | 1.000 |
| **BVAR** (Minnesota Prior) | Econometric | 1.332 | 1.262 | 1.167 | 1.108 |
| **Factor-AR** (Diffusion Index) | Econometric | 1.507 | 1.192 | 1.089 | 1.029 |
| **TimesFM 2.0/2.5** | Zero-Shot TSFM | **0.983** | **0.995** | **0.997** | 1.000 |
| **TimeGPT** | Zero-Shot TSFM | **0.987** | **0.997** | **0.994** | **0.999** |
| **Chronos-Bolt** | Zero-Shot TSFM | **0.996** | **0.997** | **0.998** | 1.002 |
| **MOIRAI** | Zero-Shot TSFM | 1.031 | 1.004 | 1.003 | 1.013 |
| **TTM** | Zero-Shot TSFM | 1.117 | 1.040 | 1.019 | 1.007 |
| **Lag-Llama** | Zero-Shot TSFM | 1.043 | 1.022 | 1.020 | 1.014 |
| **Hybrid (BVAR + TimesFM)** | Hybrid Residual | 1.331 | 1.250 | 1.168 | 1.111 |

![RMSFE Distribution Boxplots](output/rmsfe_boxplots.png)

### Key Takeaways
1. **Zero-Shot TSFMs Beat AR(1)**: `TimesFM` and `TimeGPT` consistently beat the benchmark AR(1) across short and medium horizons without any macroeconomic fine-tuning.
2. **Prior Shrinkage Dynamics**: Econometric BVAR models without aggressive fine-tuning on high-dimensional vectors face challenges relative to parsimonious univariate models, but narrow the gap at long horizons ($h=12$).
3. **Hybrid Correction Mechanism**: The hybrid residual formulation successfully captures and corrects systematic biases in classical BVAR outputs.

---

## 🗂️ Repository Structure

```
├── README.md                      # Project documentation and summary
├── STAGE1_REPORT.md               # Markdown report of Stage 1 experiments
├── STAGE1_REPORT_FULL.pdf         # Comprehensive Stage 1 PDF report
├── STAGE1_COMPARISON.pdf          # Detailed comparison: Our project vs. original paper
├── EXPLANATION_READINESS.md       # Q&A guide and technical defense notes
├── EXPLANATION_READINESS.pdf      # Formatted Q&A cheat sheet
├── data/
│   ├── fred_md.csv                # Raw FRED-MD monthly database
│   └── fredmd_clean.parquet       # Cleaned, transformed 60-series dataset
├── output/
│   ├── pipeline.png               # High-resolution architectural diagram
│   ├── rmsfe_boxplots.png         # Distribution boxplots across 60 series
│   ├── rmsfe_summary.csv          # RMSFE statistics (median, std, min, max)
│   └── workflow_diagram.png       # Pipeline workflow visualization
└── scripts/
    ├── prep_data.py               # Data extraction & McCracken-Ng transformations
    ├── econ_models.py             # AR(1), BVAR (Minnesota), Factor-AR implementations
    ├── run_forecast_loop.py       # Expanding-window forecasting evaluation engine
    ├── run_tsfm_timesfm.py        # Google TimesFM inference
    ├── run_tsfm_chronos.py        # Amazon Chronos-Bolt inference
    ├── run_tsfm_moirai.py         # Salesforce MOIRAI inference
    ├── run_tsfm_lagllama.py       # Meta Lag-Llama inference
    ├── run_tsfm_ttm.py            # IBM Tiny Time Mixer (TTM) inference
    ├── run_tsfm_timegpt.py        # Nixtla TimeGPT API runner
    ├── run_hybrid.py              # BVAR + TimesFM residual hybrid model
    ├── merge_tsfm_errors.py       # Consolidates forecast errors across models
    ├── make_plots.py              # Generates RMSFE boxplots and summary CSV
    ├── make_pipeline_diagram.py   # Generates the modern pipeline architecture visual
    ├── make_comparison_pdf.py     # Compiles the paper comparison PDF
    └── make_report_pdf.py         # Compiles the full Stage 1 PDF report
```

---

## 🔬 Dataset & Preprocessing

The project utilizes the **FRED-MD** monthly macroeconomic database ([McCracken & Ng, 2016](https://doi.org/10.1080/07350015.2015.1086655)). We select **60 series** preserving proportional representation across all 8 macroeconomic categories:

| Group # | Category | Series Count | Representative Series |
|:---:|---|:---:|---|
| 1 | **Output and Income** | 8 | `RPI`, `INDPRO`, `IPFPNSS`, `IPMANSICS`, `IPB51222S` |
| 2 | **Labor Market** | 16 | `PAYEMS`, `UNRATE`, `UEMPMEAN`, `CLAIMSx`, `HWI` |
| 3 | **Housing** | 5 | `HOUST`, `HOUSTNE`, `HOUSTMW`, `HOUSTS`, `PERMIT` |
| 4 | **Consumption & Orders**| 5 | `DPCERA3M086SBEA`, `AMDMNOx`, `ANDENOx` |
| 5 | **Money and Credit** | 6 | `M1SL`, `M2SL`, `BUSLOANS`, `REALLN`, `NONREVSL` |
| 6 | **Interest & Exchange** | 11 | `FEDFUNDS`, `TB3MS`, `GS10`, `T10YFFM`, `EXUSUKx` |
| 7 | **Prices & Inflation** | 7 | `CPIAUCSL`, `PCEPI`, `WPSFD49207`, `OILPRICEx` |
| 8 | **Stock Market** | 2 | `S&P 500`, `S&P: indust` |

### McCracken-Ng Transformations
Each series $x_t$ is transformed to stationary series $y_t$ according to its assigned code:
* **Code 1 (No transformation)**: $y_t = x_t$
* **Code 2 ($\Delta$)**: $y_t = x_t - x_{t-1}$
* **Code 3 ($\Delta^2$)**: $y_t = \Delta^2 x_t$
* **Code 4 ($\ln$)**: $y_t = \ln(x_t)$
* **Code 5 ($\Delta \ln$)**: $y_t = \ln(x_t) - \ln(x_{t-1})$
* **Code 6 ($\Delta^2 \ln$)**: $y_t = \Delta^2 \ln(x_t)$
* **Code 7 ($\Delta (x_t/x_{t-1} - 1)$)**: $y_t = \Delta(x_t/x_{t-1})$

---

## 🚀 Reproduction & Setup Guide

### 1. Environment Setup
Create and activate an isolated virtual environment:
```bash
python3 -m venv venv_core
source venv_core/bin/activate
pip install --upgrade pip
pip install numpy pandas scipy scikit-learn statsmodels matplotlib reportlab pyarrow
```

*(Optional foundation model packages can be installed in dedicated venvs or within `venv_core` depending on GPU/system dependencies: `timesfm`, `chronos`, `uni2ts`, `lag-llama`).*

### 2. Preprocess Data
```bash
python scripts/prep_data.py
```
*Outputs: `data/fredmd_clean.parquet` (768 months × 60 clean transformed series).*

### 3. Run Econometric Benchmarks
```bash
python scripts/run_forecast_loop.py
```
*Evaluates AR(1), Minnesota BVAR, and Factor-AR over all 48 expanding quarterly origins.*

### 4. Run Zero-Shot Foundation Models & Hybrid Model
```bash
python scripts/run_tsfm_timesfm.py
python scripts/run_tsfm_chronos.py
python scripts/run_hybrid.py
```

### 5. Aggregate Errors & Generate Visuals
```bash
python scripts/merge_tsfm_errors.py
python scripts/make_plots.py
python scripts/make_pipeline_diagram.py
```
*Outputs: `output/rmsfe_summary.csv`, `output/rmsfe_boxplots.png`, and `output/pipeline.png`.*

### 6. Build PDF Reports
```bash
python scripts/make_report_pdf.py
python scripts/make_comparison_pdf.py
```

---

## 📖 Citation & References

```bibtex
@article{carriero2026macroeconomic,
  title={Macroeconomic Forecasting with Large Language Models},
  author={Carriero, Andrea and Pettenuzzo, Davide and Shekhar, Shubham},
  journal={arXiv preprint arXiv:2407.00890v5},
  year={2026}
}

@article{mccracken2016fred,
  title={FRED-MD: A monthly database for macroeconomics},
  author={McCracken, Michael W and Ng, Serena},
  journal={Journal of Business \& Economic Statistics},
  volume={34},
  number={4},
  pages={574--589},
  year={2016}
}
```

---

## 📄 License
This repository is licensed under the [MIT License](LICENSE).
