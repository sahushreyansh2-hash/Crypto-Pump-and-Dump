# AI-Based Cryptocurrency Pump-and-Dump Early Detection
## B.Tech CSE Final-Year Research Project Review Report

> **Project Title**: AI-Based Cryptocurrency Pump-and-Dump Early Detection  
> **Domain**: Machine Learning / Financial Cryptographic Market Security / Deep Learning  
> **Frameworks**: PyTorch, Scikit-Learn, XGBoost, Streamlit, Pandas, NumPy, Plotly  

---

## 1. Executive Summary & Objective

Cryptocurrency market manipulation—specifically **pump-and-dump (P&D)** schemes—causes severe financial damage to unsuspecting retail investors. Existing market anomaly detectors flag pumps *after* price and volume have already spiked, offering zero protection.

This project delivers an **Early Warning System** designed to predict whether a pump manipulation event is likely to occur within a **configurable future horizon** ($H = 5, 15, 30, \text{ or } 60 \text{ minutes}$), **before price and volume escalate**.

---

## 2. Dataset Architecture & Verification Summary

The data pipeline extracts, parses, and validates multi-channel raw datasets with **100% data integrity** and **zero data fabrication**:

| Metric | Dataset 1 (`PD_logs_cleaned`) | Dataset 2 (`pump_telegram`) | Combined Verified Target |
| :--- | :--- | :--- | :--- |
| **Total AI-Readable Records** | `1,335` | `1,110` | **`2,445`** |
| **Primary Format** | JSON Lines | JSON Lines | Standardized CSV |
| **Unique Base Coins** | `277` | `372` | `496` |
| **Exchanges Covered** | `14` (Binance, YoBit, KuCoin, etc.) | `5` (Binance, YoBit, Cryptopia, etc.) | `14` Exchanges |
| **Time Range (UTC)** | Jan 2019 – Jan 2022 | Jul 2017 – Jan 2021 | Jul 2017 – Jan 2022 |
| **Metadata Tracked** | `channel_id`, `session_id`, `pair` | `group` (e.g. ATW, BPF, CPI) | Full Data Lineage |

> **Cross-Dataset Signal Verification**: Analysis reveals **1,154 pump events matched within a $\pm 15\text{-minute}$ window** across different Telegram channel sources.

---

## 3. How to Demonstrate Tomorrow (Quickstart Instructions)

### Option 1: One-Click Demo Launch
Double-click `run_demo.bat` (or execute `.\run_demo.ps1` in PowerShell). This will automatically start the Streamlit server and open the browser.

### Option 2: Command-Line Manual Launch

1. **Run Full ML Pipeline (All 10 Phases)**:
   ```bash
   python main.py
   ```

2. **Launch Interactive Streamlit Dashboard**:
   ```bash
   python -m streamlit run app.py
   ```
   Open browser at: **`http://localhost:8501`**

3. **Show Dataset Verification & Schema Analysis**:
   ```bash
   python data/analyze_schemas.py
   ```

---

## 4. Dashboard Presentation Checklist

When presenting to the review panel tomorrow, demonstrate these interactive features:

1. **🚨 Live Warning & Risk Status**: Select `STMX/USDT` or `BTC/USDT` to show real-time risk gauges, alerts, and live signal badges (Volume Z-Score, Momentum, Volatility, RSI).
2. **🎛️ Alert Sensitivity Slider**: Drag the slider from `0.10` to `0.90` to show how the early warning threshold dynamically adjusts risk alerts.
3. **📈 Interactive Candlestick & Probability Timeline**: Use Plotly zoom & pan controls to inspect price action alongside predicted pump probability spikes.
4. **🔍 Signal Explainability Tab**: Show top feature signals contributing to high/low probability score.
5. **📊 Model Benchmarks & Ablation Tab**: Show the model comparison table and feature-group ablation results.
6. **📁 Custom CSV Upload**: Click *Browse files* in the sidebar and upload `sample_crypto_ohlcv.csv` to demonstrate custom dataset evaluation!
