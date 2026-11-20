# Demonstration Datasets for Surveillance Dashboard

This folder contains three curated test datasets configured for instant demonstration in the **Crypto Pump Early Warning Dashboard** (`app.py`).

---

## 1. `dataset_1_STMX_PUMP_IN_PROGRESS.csv`
- **Asset**: `STMX/USDT`
- **Exchange**: Binance
- **Timeframe**: 2021-06-14 12:00:00 to 13:30:00 UTC
- **Ground Truth**: Verified Telegram pump attack broadcast at 13:00:07 UTC.
- **Model Behavior**: 
  - Surveillance Radar triggers **CRITICAL ALERT** (Red).
  - Risk probability reaches **99.93%** at 12:35 UTC (25 minutes before the public Telegram call).
  - Demonstrates Stage B detection with early lead time.

---

## 2. `dataset_2_BTC_SAFE_INSTITUTIONAL.csv`
- **Asset**: `BTC/USDT`
- **Exchange**: Binance
- **Timeframe**: 2021-06-14 00:00:00 to 08:20:00 UTC
- **Ground Truth**: Normal institutional market regime, zero pump manipulation.
- **Model Behavior**:
  - Surveillance Radar indicates **SAFE** (Green).
  - Risk probability stays below **0.05%**.
  - Confirms low False Alarm Rate (FAR) under deep institutional liquidity.

---

## 3. `dataset_3_BNT_ACCUMULATION_WATCHLIST.csv`
- **Asset**: `BNT/BTC`
- **Exchange**: Binance
- **Timeframe**: 2021-06-14 00:00:00 to 08:20:00 UTC
- **Ground Truth**: Low-volume stealth accumulation with abnormal buying pressure.
- **Model Behavior**:
  - Surveillance Radar triggers **ELEVATED WATCHLIST** (Yellow / Amber).
  - Stage A Ranker flags unusual volume/trade imbalance clusters.
  - Demonstrates Stage A pre-announcement candidate ranking.

---

## How to Test in the Dashboard
1. Launch the dashboard: `python run_app.py`
2. In the sidebar, select **"Upload Custom Market Data (CSV)"**.
3. Drag and drop any of these 3 CSV files.
4. Navigate across the tabs (**Surveillance Radar**, **Pump Anatomy**, **Feature Attribution**) to observe live model inference.
