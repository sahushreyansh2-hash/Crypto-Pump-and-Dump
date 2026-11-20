# Institutional Cryptocurrency Pump-and-Dump Benchmark Dataset (ICPD-2021)

## Overview & Dataset Description

This benchmark dataset provides point-in-time, high-frequency cryptocurrency market microstructure data and synchronized ground-truth manipulation events from organized Telegram pump-and-dump syndicates. 

The dataset is curated for machine learning research into **early-warning detection** of market manipulation schemes—specifically training models to predict pump events **prior to peak price-volume execution** without lookahead bias.

---

## Dataset Catalog

### 1. Raw Market OHLCV Microstructure (`data/raw/ohlcv.csv`)
- **Title**: High-Frequency 1-Minute Crypto OHLCV Market Time Series
- **Description**: 2,500 granular 1-minute time series intervals spanning five representative trading pairs on Binance, capturing normal institutional trading, low-liquidity illiquidity regimes, and active pump-and-dump attacks.
- **Coverage**: `2021-06-14T00:00:00+00:00` to `2021-06-15T17:35:00+00:00`
- **Pairs Included**:
  - `BTC/USDT` (Institutional Deep Benchmark)
  - `ETH/USDT` (Large-Cap Baseline)
  - `STMX/USDT` (Active Pump-and-Dump Attack target)
  - `BNT/BTC` (Mid-Cap Accumulation Watchlist)
  - `APPC/BTC` (Micro-Cap Historical Vulnerability)

#### Schema:
| Column | Type | Description |
| :--- | :--- | :--- |
| `timestamp` | ISO-8601 UTC | Minute timestamp of the trading bar (point-in-time) |
| `symbol` | String | Trading pair identifier (e.g., `STMX/USDT`, `BTC/USDT`) |
| `open` | Float64 | Opening price in quote currency |
| `high` | Float64 | Highest traded price during the 1-minute window |
| `low` | Float64 | Lowest traded price during the 1-minute window |
| `close` | Float64 | Closing price in quote currency |
| `volume` | Float64 | Base asset aggregate traded volume during the window |

---

### 2. Ground-Truth Pump Event Labels (`data/labels/pump_events.csv`)
- **Title**: Synchronized Telegram Syndicate Pump Announcements
- **Description**: 2,445 rigorously validated pump-and-dump signals broadcast across verified Telegram channels and historical crypto pump communities. Each record marks the exact second the target coin was announced to retail followers.
- **Schema**:
| Column | Type | Description |
| :--- | :--- | :--- |
| `symbol` | String | Standardized pair symbol (`COIN/PAIR`) |
| `coin` | String | Target cryptocurrency symbol (e.g., `STMX`, `APPC`, `BNT`) |
| `pair` | String | Base denomination currency (e.g., `USDT`, `BTC`, `ETH`) |
| `event_timestamp` | ISO-8601 UTC | Millisecond/second precision timestamp of the public announcement |
| `exchange` | String | Target trading venue (e.g., `binance`, `bittrex`, `cryptopia`) |
| `channel_id` | Integer | Telegram syndicate broadcast channel unique identifier |
| `group` | String | Syndicate channel codename / operator group |
| `source_dataset` | String | Provenance tracker (`PD_logs_cleaned` vs `pump_telegram`) |

---

### 3. Telegram Signal Logs (`data/labels/PD_logs_cleaned.csv` & `pump_telegram.csv`)
- **Title**: Cleaned Multi-Channel Pump Group Dispatch Records
- **Description**: 1,335 session-level logs recording Telegram message dispatch sessions, message timestamps, coin tickers, target exchanges, and session progression metadata.
- **Verification**: Cleaned to resolve OCR discrepancies, normalize UTC offsets, and strip forward-looking channel metadata.

---

### 4. Processed Multi-Horizon Feature Matrix (`data/processed/featured_data.csv`)
- **Title**: Engineered Surveillance Feature Matrix (50 Dimensions)
- **Description**: 2,251 rows of point-in-time feature vectors engineered across rolling temporal windows (5m, 15m, 30m, 60m). Incorporates volume rolling z-scores, price acceleration, EWMA volatility spreads, order flow imbalance approximations, and ETASE pre-announcement ranking scores.
- **Label Horizons**: `pump_in_5m`, `pump_in_15m`, `pump_in_30m`, `pump_in_60m`.

---

### 5. Institutional Demo Datasets (`demo_datasets/`)
Pre-formatted, turnkey datasets created for immediate testing and live institutional validation on the Streamlit dashboard:

1. **`dataset_1_STMX_PUMP_IN_PROGRESS.csv`**
   - **Scenario**: Active high-risk market manipulation in progress.
   - **Target**: `STMX/USDT` on Binance.
   - **Key Features**: Extreme volume z-score (>6.5σ), aggressive buy momentum, model pump probability >99.9%.
   
2. **`dataset_2_BTC_SAFE_INSTITUTIONAL.csv`**
   - **Scenario**: Deep liquidity institutional benchmark.
   - **Target**: `BTC/USDT` on Binance.
   - **Key Features**: Low volume volatility, steady TWAP/VWAP alignment, model pump probability <0.05%.

3. **`dataset_3_BNT_ACCUMULATION_WATCHLIST.csv`**
   - **Scenario**: Pre-announcement stealth accumulation phase.
   - **Target**: `BNT/BTC` on Binance.
   - **Key Features**: Subtle volume clustering, Stage A elevated rank score, Stage B probability ~40-60%.

---

## Research Compliance & Integrity
1. **Strict Chronological Ordering**: Time series are sorted strictly by `timestamp` ascending. No random cross-validation folds are permitted.
2. **Zero Lookahead Bias**: All features at timestamp $t$ are computed using exclusively data available at or before $t$.
3. **Detection Delay**: Measured from the moment market data becomes accessible relative to the Telegram event announcement.
