# Real-Time Pump-and-Dump Detection: The Base Paper's Approach, Updated

**Affiliation**: Group-20 · BCSE206L Foundations of Data Science · VIT Vellore · 28 October 2026  
**Primary Reference**: La Morgia, M., Mei, A., Sassi, F. and Stefa, J. (2020). *Pump and Dumps in the Bitcoin Era: Real Time Detection of Cryptocurrency Market Manipulations*. IEEE ICCCN 2020. [arXiv:2005.06610](https://arxiv.org/abs/2005.06610).

---

## 1. Problem and Core Idea

Pump-and-dump (P&D) groups on Telegram and Discord announce a coin at a fixed minute, and members buy it at once. The price jumps and the organisers sell into the rush. The base paper asks how quickly a third party could notice this from public trade data alone, so that other investors can stay out.

Its central observation is that pump participants buy with **market orders**, which take whatever is on offer immediately. Normal trading rarely produces a burst of these. The authors therefore watch for an abnormal burst of market buys, together with changes in volume and price, instead of waiting for a price spike. This is what lets their detector fire in seconds, where earlier rule-based tools needed up to an hour.

---

## 2. Data

- **Base Paper**: 104 pumps on Binance run by 12 groups, identified across >100 monitored groups between July 2017 and January 2019. Trade history was extracted for 7 days before and after each pump (~900 trading days).
- **Rush Orders**: Binance trade stream APIs do not disclose order type. The authors approximate market buy orders by grouping all fills occurring within the exact same millisecond into a single "rush order".
- **Our Updated Release**: We evaluate on the authors' extended release of labelled trade features: **317 labelled onsets across 327 pumps on 85 coins**, encompassing **482,157 chunks at 25 s**. We augment this with two Telegram event logs (1,110 and 1,335 events) and CoinGecko snapshots for pre-announcement ranking.

---

## 3. Features and Labelling

Trades are segmented into non-overlapping temporal chunks (5, 15, or 25 seconds). For each chunk, moving statistics are computed over recent history (50 minutes to 7 hours). 

### The 9 Core Features:
1. `std_rush_order`: Moving standard deviation / spread of rush-order volume
2. `avg_rush_order`: Moving average of rush-order volume
3. `std_trades`: Moving standard deviation of trade counts
4. `std_volume`: Moving standard deviation of traded volume
5. `avg_volume`: Moving average of traded volume
6. `std_price`: Moving standard deviation of closing price
7. `avg_price`: Moving average of closing price
8. `avg_price_max`: Moving average of high prices
9. `avg_price_min`: Moving average of low prices

### Feature Importance:
The authors' Random Forest importance ranks `std_rush_order` highest (0.401), followed by `std_trades` (0.202) and `avg_rush_order` (0.153). Price features contribute minimally, proving that order aggressiveness (rush orders), rather than price movement, carries early detection.

**Our Addition**: We retain all base features and append 12 strictly past-only temporal features per pump: first-order differences ($d_1$) and rolling 20-chunk z-scores ($z$). No forward-looking data is utilized.

---

## 4. Model and Evaluation: Base vs Replication vs Update

| Setting | Chunk | Precision | Recall | F1 Score | Source |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Kamps Rules (Strict)** | 1 h | 50.1% | 75.0% | 60.5% | Base Paper |
| **Random Forest (5-fold row CV)** | 5 s | 92.4% | 78.4% | 84.0% | Base Paper |
| **Random Forest (5-fold row CV)** | 15 s | 91.3% | 84.4% | 87.7% | Base Paper |
| **Random Forest (5-fold row CV)** | 25 s | 93.7% | 91.3% | 91.8% | Base Paper |
| **Random Forest (10-fold row CV)** | 25 s | 93.1% | 91.4% | 92.0% | Base Paper |
| **Random Forest (Rerun on Extended Data)** | 25 s | 96.6% | 89.0% | **0.926** | **Our Replication** |
| **XGBoost + Temporal (GroupKFold)** | 25 s | **96.3%** | 90.5% | **0.933** | **Our Work** |
| **ETASE Stacked Ensemble (GroupKFold)** | 25 s | 93.6% | **92.7%** | **0.932** | **Our Work** |

---

## 5. Summary of Architectural Upgrades

| Dimension | Base Paper Approach | ETASE-2S Updated Framework |
| :--- | :--- | :--- |
| **Detection Horizon** | Post-onset only (detects within 25s chunk after pump begins) | Two-stage: Stage A ranks candidate coins hours ahead; Stage B monitors live order flow |
| **Classification Engine** | Single isolated Random Forest | Stacked ensemble: Temporal MLP + Isolation Forest + XGBoost with Logistic meta-learner |
| **Interpretability** | Global feature importances only | Local instance-level TreeSHAP attributions attached to every alert |
| **Cross-Validation** | Row-level stratified folds (vulnerable to intra-pump leakage) | GroupKFold grouped strictly by pump; threshold calibrated on training folds |
| **Datasets** | 104 pumps from a single group set | 327 pumps in trade stream + 2,445 events across 108 Telegram channels |
| **Actionable Output** | Binary yes/no alert flag | Institutional dashboard with 5m, 15m, 30m, 60m risk horizons and 0-100 Asset Safety Scores |
