# AI-Based Early Detection & Surveillance of Cryptocurrency Pump-and-Dump Schemes

> **Capstone Project · Group-20 · BCSE206L Foundations of Data Science · VIT Vellore**  
> **Repository**: [sahushreyansh2-hash/Crypto-Pump-and-Dump](https://github.com/sahushreyansh2-hash/Crypto-Pump-and-Dump)  
> **Base Paper**: M. La Morgia, A. Mei, F. Sassi, and J. Stefa, *"Pump and Dumps in the Bitcoin Era: Real Time Detection of Cryptocurrency Market Manipulations,"* IEEE ICCCN 2020.

---

## 📌 Executive Summary

This repository delivers **ETASE-2S**, a state-of-the-art, two-stage machine learning system that provides **early-warning detection and surveillance** of illicit cryptocurrency pump-and-dump (P&D) schemes on centralized exchanges (Binance, YoBit, Bittrex). 

Unlike traditional reactive anomaly detectors that flag an event only *after* the market price and volume have already spiked, **ETASE-2S** operationalizes pre-announcement prediction:
1. **Stage A (Pre-Announcement Ranker - CATR)**: Ranks candidate coins 3 days to 1 hour *before* Telegram organizers broadcast target coin names, placing the true pump target within a 30-coin watch-list **64.7% of the time** (against 11.0% by chance).
2. **Stage B (Explainable Stacked Detector - ETASE)**: A stacked ensemble of a Temporal Neural Encoder (MLP), an unsupervised Isolation Forest, and an XGBoost gradient-boosted classifier. On 25-second granular trade chunks, it detects **93.1% of real pumps within 25 seconds of onset** with **0.932 F1 score**, accompanied by local **TreeSHAP feature attributions** for every alert.
3. **PUMP-WATCH Surveillance Terminal**: A Zerodha Kite-inspired, institutional dark-theme web dashboard providing real-time multi-horizon risk assessments (5m, 15m, 30m, 60m) and a 0–100 Asset Safety Index.

---

## 🗂 Repository Structure

```
Crypto-Pump-and-Dump/
├── app.py                                # Streamlit institutional surveillance terminal (9 tabs)
├── run_app.py                            # Resilient runner (auto port detection & config routing)
├── config.yaml                           # Master pipeline and model hyperparameter configuration
├── requirements.txt                      # Reproducible Python dependencies
├── DATASET_CATALOG.md                    # Master dataset documentation & schema index
│
├── papers/                               # Academic research paper pack & literature survey
│   ├── PROJECT_PAPER.md                  # Complete final project paper (Group-20, VIT Vellore)
│   ├── BASE_PAPER_ANALYSIS.md            # Replication & comparative study of La Morgia et al. (2020)
│   ├── REFERENCE_PAPERS_PACK.md          # 14 reference papers catalog & benchmark matrix
│   ├── PD_Logs_Dataset_1335_Records.pdf  # Verified Telegram session dispatch records (Hu et al.)
│   └── Pump_Telegram_Dataset_1110_Records.pdf # Historical pump event logs (La Morgia et al.)
│
├── data/                                 # Point-in-time market data & ground truth labels
│   ├── institutional_pump_benchmark_master.csv # Master consolidated benchmark dataset
│   ├── README.md                         # Detailed data provenance & zero-leakage guidelines
│   ├── raw/ohlcv.csv                     # Binance 1-minute OHLCV candles (BTC, ETH, STMX, BNT, APPC)
│   ├── labels/pump_events.csv            # 2,445 verified Telegram pump-and-dump ground-truth events
│   ├── labels/PD_logs_cleaned.csv        # 1,335 channel dispatch logs
│   ├── labels/telegram_session_records.tsv # Tab-separated multi-channel session records
│   └── processed/featured_data.csv       # 50-dimensional engineered multi-horizon matrix
│
├── demo_datasets/                        # Turnkey test datasets for instant dashboard evaluation
│   ├── README.md                         # Testing guide and expected scenario walkthroughs
│   ├── dataset_1_STMX_PUMP_IN_PROGRESS.csv     # Active pump attack (>99.9% risk probability)
│   ├── dataset_2_BTC_SAFE_INSTITUTIONAL.csv    # Deep liquidity institutional benchmark (<0.1% risk)
│   └── dataset_3_BNT_ACCUMULATION_WATCHLIST.csv # Pre-announcement stealth accumulation phase
│
├── models/                               # Pre-trained production model weights (2.3 MB)
│   ├── xgboost_model.joblib              # 400-tree gradient boosted classifier with temporal deltas
│   ├── isolation_forest_model.joblib     # Unsupervised anomaly detector on normal market regimes
│   ├── random_forest_model.joblib        # Replicated base-paper Random Forest
│   └── logistic_regression_model.joblib  # Calibrated stacking meta-learner
│
├── results/                              # Faculty evaluation evidence, figures, & metric CSVs
│   ├── model_comparison.csv              # Full model benchmark comparison table
│   ├── cascade_matched_far_comparison.csv# Cascade vs ETASE alone at matched false alarm rates
│   ├── roc_curve.png, pr_curve.png       # Comprehensive ROC and Precision-Recall curves
│   └── confusion_matrix_*.png            # Confusion matrices for all evaluated architectures
│
└── src/                                  # Modular source package (12 Python modules)
    ├── etase.py                          # Full ETASE two-stage early-warning framework
    ├── cascade.py                        # Stage A + Stage B adaptive threshold cascade
    ├── stage_a_ranker.py                 # CATR pairwise learning-to-rank engine
    ├── stage_b_detector.py               # Real-time chunk classifier & meta-learner
    ├── features.py                       # 50-feature microstructural engineering pipeline
    ├── explainability.py                 # Instance-level TreeSHAP feature attribution
    ├── preprocessing.py                  # Auto-repairing OHLCV ingestion & normalization
    └── evaluation.py                     # Leakage-safe GroupKFold evaluation protocols
```

---

## 🔬 Benchmark Results & Faculty Validation

### 1. Stage B Real-Time Detection at 25-Second Chunks (Leak-Safe GroupKFold)
Evaluated across all 317 verified pump onsets with zero data leakage (all chunks of a pump strictly confined to a single fold; threshold calibrated exclusively on training pumps):

| Architecture / Model | Precision | Recall | F1 Score | PR-AUC | Event Recall | False Alarms / Coin-Day |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **ETASE Stacked Ensemble (Ours)** | 0.936 | **0.927** | **0.932** | 0.961 | **93.1% (295/317)** | 0.115 |
| **XGBoost + Temporal Deltas** | **0.963** | 0.905 | **0.933** | **0.975** | 90.9% (288/317) | **0.065** |
| **Random Forest (Base Paper)** | 0.963 | 0.893 | 0.926 | 0.962 | 89.6% (284/317) | 0.072 |
| **Logistic Regression Baseline** | 0.886 | 0.861 | 0.874 | 0.903 | 86.1% (273/317) | 0.215 |
| **Temporal MLP (10-Chunk)** | 0.814 | 0.814 | 0.814 | 0.856 | 82.0% (260/317) | 0.244 |
| **Isolation Forest (Unsupervised)**| 0.450 | 0.704 | 0.549 | 0.492 | 70.7% (224/317) | 1.892 |

### 2. Stage A Pre-Announcement Target Ranking (272 Test Pumps)
Evaluated on strictly chronological held-out pumps occurring after March 2021:
- **Hit Ratio @ 10 (HR@10)**: **28.3%** (vs. 3.7% random chance)
- **Hit Ratio @ 30 (HR@30)**: **64.7%** (vs. 11.0% random chance)
- **Cascade Impact**: Reduces detection delay from -0.6s to **-2.9s prior to Telegram onset**, increasing pre-onset alerts by **600%** (from 2 to 12) on matched events.

---

## ⚡ Quick Start & Dashboard Installation

### Prerequisites
- Python 3.10, 3.11, or 3.12 (Windows / macOS / Linux)

### Installation
```bash
# 1. Clone repository
git clone https://github.com/sahushreyansh2-hash/Crypto-Pump-and-Dump.git
cd Crypto-Pump-and-Dump

# 2. Install dependencies
pip install -r requirements.txt
```

### Launch Institutional Surveillance Dashboard
```bash
# Launch the Streamlit terminal
python run_app.py
```
*The browser will automatically open to `http://localhost:8501` featuring the Zerodha Kite-inspired surveillance dashboard.*

---

## 🧪 Testing with Demonstration Datasets
To observe the machine learning inference live on the dashboard:
1. Open the dashboard at `http://localhost:8501`.
2. In the sidebar under **Market Ingestion**, click **Browse files** and select any file from `demo_datasets/`:
   - `dataset_1_STMX_PUMP_IN_PROGRESS.csv`: Triggers **CRITICAL PUMP ALERT** (99.93% risk at 12:35 UTC, 25 minutes before public Telegram broadcast).
   - `dataset_2_BTC_SAFE_INSTITUTIONAL.csv`: Triggers **NOMINAL SAFE** (<0.05% risk, low volume z-score).
   - `dataset_3_BNT_ACCUMULATION_WATCHLIST.csv`: Triggers **ELEVATED WATCHLIST** (Stage A pre-pump stealth accumulation).

---

## 📖 Research Papers & Documentation
- **Final Project Paper**: [`papers/PROJECT_PAPER.md`](papers/PROJECT_PAPER.md)
- **Base Paper Analysis**: [`papers/BASE_PAPER_ANALYSIS.md`](papers/BASE_PAPER_ANALYSIS.md)
- **Reference Papers Pack**: [`papers/REFERENCE_PAPERS_PACK.md`](papers/REFERENCE_PAPERS_PACK.md)
- **Dataset Catalog & Metadata**: [`DATASET_CATALOG.md`](DATASET_CATALOG.md)

---

## 👥 Authors & Acknowledgments
- **Project Team**: Group 20 (BCSE206L Foundations of Data Science, VIT Vellore)
- **Project Advisor & Faculty Reviewers**: School of Computer Science and Engineering (SCOPE), VIT Vellore.
- **Academic Inspirations**: La Morgia et al. (IEEE ICCCN 2020), Hu et al. (ACM SIGMOD 2023), Zerodha Nudge Risk Framework.
