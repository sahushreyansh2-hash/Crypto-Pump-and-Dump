# AI-Based Cryptocurrency Pump-and-Dump Early Detection

> **B.Tech CSE Final-Year Project**

A machine-learning research system that provides **early warning** of cryptocurrency pump-and-dump manipulation events — predicting whether a pump is likely to occur within a configurable future time horizon (5 / 15 / 30 / 60 minutes), **before** the price and volume actually spike.

---

## Table of Contents

1. [Project Objective](#project-objective)
2. [Research Problem](#research-problem)
3. [Early-Warning Concept](#early-warning-concept)
4. [Dataset Format](#dataset-format)
5. [Labeling Methodology](#labeling-methodology)
6. [Feature Engineering](#feature-engineering)
7. [Chronological Splitting](#chronological-splitting)
8. [Models](#models)
9. [ETASE Architecture](#etase-architecture)
10. [Evaluation Metrics](#evaluation-metrics)
11. [Ablation Study](#ablation-study)
12. [Installation](#installation)
13. [Training & Evaluation](#training--evaluation)
14. [Streamlit Dashboard](#streamlit-dashboard)
15. [Project Structure](#project-structure)
16. [Research Safety Rules](#research-safety-rules)

---

## Project Objective

Develop a system that can **predict** cryptocurrency pump-and-dump events **before** they happen, giving traders and exchanges early warning to take protective action.

Unlike post-hoc anomaly detectors that flag events after the price has already spiked, this system is designed to raise warnings within a **configurable prediction horizon** (e.g., 30 minutes before the pump).

---

## Research Problem

Pump-and-dump schemes in cryptocurrency markets are a form of market manipulation where:

1. A group artificially inflates the price of a low-liquidity cryptocurrency ("pump")
2. They sell at the inflated price ("dump"), leaving other traders with losses

These events are characterized by sudden volume surges, rapid price increases, and subsequent crashes. Detecting them **before** they reach their peak is critical.

---

## Early-Warning Concept

For each observation at time `t`, we ask:

> **"Will a pump event happen within the next H minutes?"**

Where `H` is the prediction horizon (configurable: 5, 15, 30, or 60 minutes).

This is fundamentally different from detecting a pump *after* it has already occurred.

---

## Dataset Format

### OHLCV Market Data (`data/raw/ohlcv.csv`)

| Column    | Type     | Description                  |
|-----------|----------|------------------------------|
| timestamp | datetime | ISO-8601 or Unix epoch       |
| symbol    | string   | e.g., `BTC/USDT`            |
| open      | float    | Opening price                |
| high      | float    | Highest price in the period  |
| low       | float    | Lowest price in the period   |
| close     | float    | Closing price                |
| volume    | float    | Trading volume               |

### Pump-Event Labels (`data/labels/pump_events.csv`)

| Column          | Type     | Description                         |
|-----------------|----------|-------------------------------------|
| symbol          | string   | e.g., `BTC/USDT`                   |
| event_timestamp | datetime | Timestamp of the known pump event   |

> **Note:** This project does NOT fabricate datasets. You must provide your own real historical data. See data sources below.

### Where to Get Data

- **OHLCV data**: Download from cryptocurrency exchanges (Binance, Coinbase) via their APIs, or use datasets from Kaggle.
- **Pump-event labels**: Can be sourced from academic pump-and-dump datasets, Telegram pump-group logs, or manually labeled from price/volume anomalies.

Place your files at:
```
data/raw/ohlcv.csv
data/labels/pump_events.csv
```

---

## Labeling Methodology

For each observation at time `t` for a given `symbol`:

```
target = 1   if there exists at least one known pump event
             in the interval (t, t + prediction_horizon]

target = 0   otherwise
```

### Example (horizon = 30 minutes, pump at 12:00):

| Observation Time | Time to Pump | Target | Reason                     |
|-----------------|-------------|--------|----------------------------|
| 11:25           | 35 min      | 0      | Outside horizon            |
| 11:30           | 30 min      | 1      | Within 30-min horizon      |
| 11:45           | 15 min      | 1      | Within 30-min horizon      |
| 11:59           | 1 min       | 1      | Within 30-min horizon      |
| 12:00           | 0 min       | 0      | Event itself (not prediction)|
| 12:05           | past        | 0      | Event already occurred     |

**No future data leakage**: Features at time `t` use only data from `t` and earlier.

---

## Feature Engineering

Six feature groups are computed, all using only historical data:

### 1. Price Features
- Percentage return, log return
- Rolling returns (5, 10, 20, 50 periods)
- Price change (absolute)
- Distance from moving averages (7, 14, 30)

### 2. Volume Features
- Volume change (percentage)
- Rolling volume mean and standard deviation
- Volume z-score
- Abnormal volume ratio

### 3. Volatility Features
- Rolling volatility
- Price range (high - low)
- High-Low percentage
- True Range and ATR (Average True Range)

### 4. Momentum Features
- RSI (Relative Strength Index)
- Momentum (5, 10, 20-period rate of change)
- Moving average cross ratio

### 5. Time Features
- Hour, minute, day of week
- Cyclical encodings (sin/cos for hour and day-of-week)

### 6. Price-Volume Relationship Features
- Return × volume interaction
- Volume acceleration
- Rolling price-volume correlation

---

## Chronological Splitting

**NEVER random split.** Data is split chronologically:

| Split      | Proportion | Description          |
|------------|-----------|----------------------|
| Training   | 70%       | Earliest data        |
| Validation | 15%       | Middle data          |
| Test       | 15%       | Latest (most recent) |

This ensures the test set always contains future data relative to training.

---

## Models

### A. Rule-Based Baseline
Simple threshold rules on volume z-score, price return, and volatility.

### B. Isolation Forest
Unsupervised anomaly detection. Anomaly scores are converted to probabilities.

### C. Logistic Regression
Linear model with balanced class weights.

### D. Random Forest
Ensemble of decision trees with balanced class weights.

### E. XGBoost
Gradient-boosted trees with scale_pos_weight for class imbalance.
Falls back to Random Forest if XGBoost is not installed.

### F. Temporal Model (LSTM/GRU)
Recurrent neural network that processes sequences of historical features.
Captures temporal patterns leading up to pump events.

### G. ETASE Framework
The proposed framework combining market signals and temporal signals.
See [ETASE Architecture](#etase-architecture) below.

---

## ETASE Architecture

**ETASE** is the name of the proposed framework. It combines:

### Components:

1. **Market Signal Encoder**: LSTM with attention mechanism processing price, volume, volatility, and price-volume features.

2. **Temporal Signal Encoder**: LSTM processing time and momentum features.

3. **Context Encoder** (extensible): Linear layer placeholder for future social-media/channel features. Currently outputs zeros when no context data is available.

4. **Fusion Module**: Multi-layer feedforward network that combines all encoder outputs for final prediction.

```
Market Features  →  [Market Encoder + Attention]  ─┐
                                                     ├→ [Fusion Module] → Probability
Temporal Features → [Temporal Encoder]             ─┤
                                                     │
Context Features  → [Context Encoder] (placeholder)─┘
```

> **Note**: Context/channel features (e.g., social media data) are not available in the current dataset. The architecture is extensible for when such data becomes available.

---

## Evaluation Metrics

Every model is evaluated on:

| Metric             | Description                                        |
|--------------------|----------------------------------------------------|
| Precision          | TP / (TP + FP)                                     |
| Recall             | TP / (TP + FN)                                     |
| F1-Score           | Harmonic mean of precision and recall              |
| PR-AUC             | Area under precision-recall curve                  |
| ROC-AUC            | Area under ROC curve                               |
| False Alarm Rate   | FP / (FP + TN)                                     |
| Detection Rate     | TP / (TP + FN) — same as recall                   |
| Confusion Matrix   | Full TP/FP/TN/FN breakdown                        |
| **Lead Time**      | Minutes before pump the warning was raised         |

### Lead-Time Analysis

For each true-positive prediction, we measure how many minutes before the actual pump event the model raised its warning. If a model produces no true positives, this is reported honestly as "unavailable."

---

## Ablation Study

Compares model performance using different feature-group subsets:

| Experiment            | Feature Groups                                    |
|-----------------------|---------------------------------------------------|
| Market Only           | price, volume, volatility, price_volume           |
| Temporal Only         | time, momentum                                    |
| Market + Temporal     | All market + temporal groups                      |
| Market + Temporal + Context | All above + context (if available)          |

If context data is unavailable, that experiment is marked as "Not available in current dataset" — never fabricated.

---

## Installation

### Prerequisites

- Python 3.9+
- pip

### Steps

```bash
# Clone or download the project
cd crypto-pump-detection

# Install dependencies
pip install -r requirements.txt

# Place your data files
# data/raw/ohlcv.csv
# data/labels/pump_events.csv
```

---

## Training & Evaluation

### Run the full pipeline:

```bash
python main.py
```

### Run a specific phase:

```bash
python main.py --phase 1   # Data loading
python main.py --phase 2   # Preprocessing & labeling
python main.py --phase 3   # Feature engineering
python main.py --phase 4   # Chronological split
python main.py --phase 5   # Baseline models
python main.py --phase 6   # Temporal model
python main.py --phase 7   # ETASE framework
python main.py --phase 8   # Evaluation
python main.py --phase 9   # Ablation study
python main.py --phase 10  # Visualization
```

### Use a custom config:

```bash
python main.py --config my_config.yaml
```

### Change prediction horizon:

Edit `config.yaml`:
```yaml
prediction_horizon_minutes: 15  # 5, 15, 30, or 60
```

---

## Streamlit Dashboard

```bash
streamlit run app.py
```

The dashboard shows:
- **Cryptocurrency selector** and **prediction horizon**
- **Current pump probability** from trained model
- **Risk level** (LOW / MEDIUM / HIGH)
- **Warning status**
- **Market signals** (volume anomaly, momentum, volatility, RSI)
- **Prediction probability chart** over time
- **Price & volume chart**
- **Historical pump events**
- **Model comparison** and **ablation results**

> All values come from trained models and real data. Nothing is hardcoded.

---

## Project Structure

```
crypto-pump-detection/
│
├── data/
│   ├── raw/               # Place ohlcv.csv here
│   ├── processed/         # Generated by pipeline
│   └── labels/            # Place pump_events.csv here
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py     # Load & validate CSV data
│   ├── preprocessing.py   # Clean OHLCV data
│   ├── labeling.py        # Create binary pump labels
│   ├── features.py        # Feature engineering (6 groups)
│   ├── split.py           # Chronological train/val/test split
│   ├── baselines.py       # Rule-based, LogReg, RF, XGBoost
│   ├── anomaly_detection.py # Isolation Forest
│   ├── temporal_model.py  # LSTM / GRU
│   ├── etase.py           # ETASE framework
│   ├── evaluation.py      # Metrics, lead-time, threshold tuning
│   ├── ablation.py        # Feature-group ablation study
│   └── visualization.py   # All plots
│
├── models/                # Saved trained models
├── results/               # Metrics, plots, CSVs
├── notebooks/             # Jupyter notebooks (optional)
│
├── app.py                 # Streamlit dashboard
├── main.py                # Pipeline orchestrator
├── config.yaml            # All configuration
├── requirements.txt       # Python dependencies
├── README.md              # This file
└── .gitignore
```

---

## Research Safety Rules

This project follows strict research integrity rules:

- **NEVER** fabricate datasets, pump labels, or model results
- **NEVER** use future information in features (no data leakage)
- **NEVER** use random train/test splitting (chronological only)
- **NEVER** fabricate accuracy, lead time, or channel features
- If data is unavailable, report: *"Not available in the current dataset"*
- All results come from actual model training on real data

---

## Configuration Reference

See `config.yaml` for all configurable parameters:

- Dataset paths
- Prediction horizon (5, 15, 30, 60 minutes)
- Sequence length for temporal models
- Train/val/test split ratios
- Random seed for reproducibility
- Model hyperparameters (for each model)
- Feature engineering settings
- Evaluation parameters

---

## License

This project is for academic/research purposes.
