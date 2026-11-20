# Dataset Catalog: Cryptocurrency Pump-and-Dump Early Warning Benchmark

**Dataset Title**: Institutional Cryptocurrency Pump-and-Dump Benchmark Dataset (ICPD-2021)  
**Curator**: B.Tech CSE Capstone Research Team  
**Target Profile**: `https://github.com/sahushreyansh2-hash`  
**License**: MIT / Research & Academic Use  

---

## 1. Executive Summary

This dataset repository contains high-frequency financial time series and synchronized illicit coordination labels designed to facilitate reproducible research in **cryptocurrency market surveillance and pump-and-dump manipulation prediction**.

Unlike traditional post-hoc anomaly detection datasets that merely record price spikes after execution, this benchmark focuses on **pre-announcement detection** (evaluating whether models can flag target coins 5 to 60 minutes *before* Telegram organizers broadcast target coin names).

---

## 2. Dataset Files Summary

| File Path | Description | Records | File Size | Key Fields |
| :--- | :--- | :--- | :--- | :--- |
| `data/institutional_pump_benchmark_master.csv` | **Master Benchmark Dataset**: Synchronized OHLCV time series with point-in-time pump labels, lead time calculations, and risk regime tags. | 2,500 | ~260 KB | `timestamp`, `symbol`, `close`, `volume`, `pump_imminent_30m`, `lead_time_minutes`, `risk_regime` |
| `data/raw/ohlcv.csv` | **Raw Market Microstructure**: Granular 1-minute OHLCV candles from Binance covering 5 representative pairs (`BTC/USDT`, `ETH/USDT`, `STMX/USDT`, `BNT/BTC`, `APPC/BTC`). | 2,500 | ~232 KB | `timestamp`, `symbol`, `open`, `high`, `low`, `close`, `volume` |
| `data/labels/pump_events.csv` | **Ground Truth Pump Events**: Master database of verified Telegram pump-and-dump announcements. | 2,445 | ~184 KB | `symbol`, `coin`, `pair`, `event_timestamp`, `exchange`, `channel_id`, `group` |
| `data/labels/PD_logs_cleaned.csv` | **Channel Dispatch Logs**: Cleaned Telegram session messages from monitored syndicate groups. | 1,335 | ~79 KB | `channel_id`, `session_id`, `coin`, `exchange`, `pair`, `timestamp` |
| `data/processed/featured_data.csv` | **Processed Multi-Horizon Matrix**: 50 engineered features (volume z-scores, price acceleration, EWMA volatility, order-flow imbalance). | 2,251 | ~2.2 MB | Multi-window indicators + binary labels for 5m, 15m, 30m, 60m horizons |
| `demo_datasets/dataset_1_STMX_PUMP_IN_PROGRESS.csv` | **High-Risk Demonstration**: Active pump event with verified Telegram attack at 13:00 UTC and pre-pump volume buildup. | 500 | ~47 KB | `timestamp`, `symbol`, `open`, `high`, `low`, `close`, `volume` |
| `demo_datasets/dataset_2_BTC_SAFE_INSTITUTIONAL.csv` | **Safe Benchmark Demonstration**: High-liquidity normal market regime with low volume anomalies and zero manipulation risk. | 500 | ~47 KB | `timestamp`, `symbol`, `open`, `high`, `low`, `close`, `volume` |
| `demo_datasets/dataset_3_BNT_ACCUMULATION_WATCHLIST.csv` | **Watchlist Demonstration**: Pre-announcement stealth accumulation exhibiting volume clustering on Stage A watchlist. | 500 | ~47 KB | `timestamp`, `symbol`, `open`, `high`, `low`, `close`, `volume` |

---

## 3. Data Integrity & Validation Safeguards

1. **Strict Temporal Ordering**: Records are strictly indexed in chronological order ($t_0 \to t_N$). No future market information is leaked into past rows.
2. **True Event Lead Time**: Ground truth events reflect the exact second of Telegram broadcast. Models are evaluated on lead time ($t_{\text{alert}} < t_{\text{announcement}}$).
3. **Reproducibility**: All dataset files are tracked and bundled inside this repository, enabling 100% out-of-the-box replication of all paper experiments and dashboard visualizations.
