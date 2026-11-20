# AI-Based Early Detection of Crypto Pump-and-Dump Schemes

**Date**: 28 October 2026  
**Course**: BCSE206L Foundations of Data Science, VIT Vellore  
**Team**: Group 20  
**Base Paper**: M. La Morgia, A. Mei, F. Sassi and J. Stefa, *"Pump and Dumps in the Bitcoin Era: Real Time Detection of Cryptocurrency Market Manipulations,"* IEEE ICCCN 2020.

---

## Abstract

We built and tested a two-stage framework that warns about crypto pump-and-dump (P&D) schemes before and during a pump. It detects 93.1% of real pumps within 25 seconds of onset. Most published detectors only confirm a pump after the price has already jumped. Our framework, **ETASE-2S**, adds an earlier stage that ranks the coins a Telegram group is likely to pick for its next pump, using only information available 3 days and 1 hour before. 

The second stage, **ETASE**, is a stacked ensemble of a temporal neural encoder, an Isolation Forest and an XGBoost model, and every alert comes with a SHAP explanation. We used two public Telegram P&D datasets (1,110 events from La Morgia et al. and 1,335 events from Hu et al.) together with the base paper’s 482,157 labelled 25-second trade chunks from Binance. Under a leakage-safe protocol that keeps every pump in a single fold, ETASE reached precision 0.936, recall 0.927 and F1 0.932. The base paper’s Random Forest scored F1 0.926 on the same split, and the base paper itself reports 0.918. Of the six models we tested, ETASE had the highest recall and the highest event-level recall. 

The early stage put the real target inside a 30-coin watch-list 64.7% of the time, about six times better than chance, out of a median of 163 candidate coins. We also built **PUMP-WATCH**, a Streamlit dashboard that shows the outputs as an asset safety screener, a multi-horizon risk view and an evidence view of a real pump. The results have limits. Ranking before the announcement is still weak at the top-1 level, and the stacked model does not beat plain XGBoost on F1 at 5- and 25-second chunk sizes.

**Keywords**: pump-and-dump, cryptocurrency, market manipulation, early warning, ensemble learning, XGBoost, Isolation Forest, SHAP, learning to rank.

---

## 1. Introduction

Crypto pump-and-dump schemes happen often, move fast and are organised in plain sight, which makes them a natural target for an automated early-warning system. Cryptocurrencies trade around the clock on hundreds of exchanges, and many small coins have very thin order books. A group with enough buyers can push one of these coins up by tens of percent within a minute.

In a P&D scheme the organisers first buy the target coin quietly. At a fixed time they announce the coin name to thousands of Telegram or Discord members. Members rush to buy, the price spikes, and the organisers sell into that demand. Late buyers are left holding a coin that loses its gains within minutes. La Morgia et al. (2020) found that one group, Big Pump Signal, generated 5,176 BTC of trading volume in a single operation on Binance. Their later study followed about 900 P&D events over three years (La Morgia et al., 2023).

Timing is the main weakness of current detection. Rule-based and descriptive studies confirm a pump only after the price spike is already visible (Kamps and Kleinberg, 2018). Machine learning detectors such as the base paper react within seconds, but only after the pump has started, and by then the organisers have already bought at the low price.

A pump has two phases that a model can use:
1. **Before the announcement**, a group’s past choices and the market profile of a coin say something about which coin will be picked (Hu et al., 2023).
2. **After the announcement**, a burst of aggressive market buy orders, called *rush orders*, shows up in the trade stream within seconds (La Morgia et al., 2020).

This paper uses both phases:
- **Stage A** ranks candidate coins hours before a scheduled pump, so that an exchange or broker can put them on a watch-list.
- **Stage B** watches the live trade stream and raises an alert within one 25-second chunk of the onset, together with an explanation of why it fired.

### Key Contributions:
1. A replication of the base paper’s Random Forest detector on its own released features (F1 0.926 against 0.918 reported).
2. A leakage-safe evaluation that keeps all chunks of one pump in the same fold, with event-level metrics for lead time and false alarms per day.
3. **ETASE**, a stacked ensemble of a temporal encoder, an anomaly detector and a gradient-boosted classifier, with a SHAP explanation for each alert.
4. **CATR**, a channel-aware target-coin ranker that combines market data with history from both Telegram datasets.
5. A cascade that links the two stages, and an analysis of where it helps and where it does not.
6. **PUMP-WATCH**, an interactive dashboard that turns the framework into an asset safety screener, a multi-horizon risk radar and an evidence view for traders and surveillance analysts.

---

## 2. Reference Case Study: Zerodha

Zerodha shows that a warning given to the customer before a trade can reduce exposure to pump-and-dump stocks, and our framework applies the same idea to crypto. Zerodha is one of India’s largest retail stockbrokers. Its Kite platform shows “nudges”, which are warnings on the order window before a trade is placed.

Several of these nudges target exactly the conditions that P&D schemes exploit:
- **Illiquid stocks nudge**: Market cap under Rs 100 crore, vulnerable to pump-and-dump.
- **Social media tip nudge**: Warns when a stock is promoted by tips from unregistered entities on social channels.
- **Trade-to-Trade segment flag**: Exchanges restrict segment when manipulation is suspected.

In March 2023, SEBI barred 45 entities for pumping shares of Sadhna Broadcast and Sharpline Broadcast via misleading YouTube videos, impounding Rs 54 crore. Zerodha CEO Nithin Kamath stated that Zerodha's nudges cut scammy trading volumes by 90% as a share of total equity volume.

| Zerodha mechanism (equities) | Equivalent in our crypto framework |
| :--- | :--- |
| Illiquid stock nudge (market cap < Rs 100 crore) | Stage A market features: market cap, volume and turnover 3 days before |
| Social-media tip nudge | Stage A channel history: which coins a Telegram group pumped before |
| Trade-to-Trade segment flag set by exchange | Stage A watch-list of top-K likely targets |
| Exchange surveillance on live order flow | Stage B ETASE detector on 25-second trade chunks |
| Warning text on the order window | Alert with SHAP explanation of the features that fired |

---

## 3. Literature Survey

### 3.1 Characterising P&D Schemes
- **Kamps and Kleinberg (2018)**: First formal definition and rule-based detector on 1-hour OHLCV windows (F1 0.605).
- **Xu and Livshits (2019)**: Modeled target-coin likelihood using pre-announcement market variables.
- **Hamrick et al. (2021)**: Quantified the Telegram/Discord ecosystem; small coins generate highest peak returns.

### 3.2 Real-time Detection After Onset
- **La Morgia et al. (2020) [Base Paper]**: Introduced rush orders (fills sharing the exact millisecond). RF on 25s chunks reached F1 0.918.
- **La Morgia et al. (2023)**: Extended study over ~900 events, F1 0.945 within 25 seconds.
- **Victor and Hagemann (2019)**: XGBoost on Telegram + Binance, AUC ~0.99 at 1-hour resolution.
- **Chadalapaka et al. (2022)**: C-LSTM (F1 0.893 at 25s) and Anomaly Transformer (F1 0.936 at 15s).
- **Yu, Park and Chai (2025)**: SMOTE + 5 ensembles on 25s features; XGBoost reached 94.87% recall.
- **LLD (2023)**: LSTM autoencoder on BTC with Gaussian tail threshold on altcoins; avg detection lag 2.2 min.
- **HD Pump (Moura et al., 2024)**: Hybrid volatility anomaly + change-point volume detection (+6.7% precision, +9.3% recall).

### 3.3 Predicting the Target Before the Announcement
- **Nghiem et al. (2021)**: Market + social target prediction before announcement.
- **Hu et al. (2023)**: SNN with positional attention on 1,335 samples (HR@10 0.599, HR@30 0.824).
- **Bolz et al. (2025)**: PumpBERT (weighted F1 0.982) + Z-score ranking (Top-5 hit 55.8% at 20s before pump).
- **Mahrous and Di Pietro (2026)**: PumpSense Telegram message classifier (F1 0.83) + LLM target extraction (0.91 joint accuracy).

---

## 4. Research Gaps & Framework Solutions

| # | Gap in Literature | Evidence | Our Framework Solution |
| :--- | :--- | :--- | :--- |
| **G1** | Reactive post-onset detection | Base paper, Chadalapaka only detect after spike | Two-stage cascade: Stage A watch-list hours before, Stage B alert seconds after |
| **G2** | Single model family reliance | RF, XGBoost, or LSTM isolated | ETASE stacks temporal MLP, Isolation Forest, and XGBoost |
| **G3** | No lead-time or false-alarm metric | Only chunk P, R, F1 reported | Event recall, detection delay, pre-onset alerts, false alarms/day |
| **G4** | Black-box classifiers | No explanations per alert | TreeSHAP attribution attached to every alert |
| **G5** | Cross-chunk evaluation leakage | Row-level random folds leak pump context | GroupKFold by pump; threshold tuned strictly on training pumps |
| **G6** | Disjoint Telegram datasets | Papers use only their own logs | Unified cross-source history from both La Morgia and Hu datasets |
| **G7** | Lack of broker-side nudges | No retail protective intervention | Watch-lists and SHAP alerts mapped to broker nudges |
| **G8** | Single-horizon binary flags | Every coin scored identically | Multi-horizon risk (5, 15, 30, 60 min) + 0-100 Asset Safety Score |

---

## 5. Dataset Description & Usage

| Dataset | Source | Size | Period | Fields | Used For |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **D1** `pump_telegram.csv` | La Morgia et al. (2020) | 1,110 pumps, 20 groups, 372 coins | 2017 - 2021 | symbol, group, date, hour, exchange | EDA, cross-source history, labels |
| **D2** `PD_logs_cleaned.txt` | Hu et al. (2023) | 1,335 samples, 108 channels, 277 coins | 2019 - 2022 | channel_id, session_id, coin, exchange, pair, timestamp | EDA, Stage A queries, channel history |
| **D3** Labelled trade features | Base paper repo | 482,157 rows (25s), 317 onsets, 85 coins | 2018 - 2021 | 12 features + pump_index, symbol, gt | Stage B training and validation |
| **D4** CoinGecko snapshots | Hu et al. (2023) | 651 target snapshots, 65,759 coin-days | 2019 - 2022 | mcap, price, volume, Twitter, Reddit, Alexa | Stage A market profile features |

---

## 6. Proposed Methodology: ETASE-2S

### Stage A: Channel-Aware Target Ranker (CATR)
- **Market Profile (22 features)**: Log market cap, log price, log volume, turnover ratio, social metrics, percentile ranks among active day universe.
- **Channel History (9 features)**: Past pump frequency of the coin by channel, recency, 30-day velocity, affinity score.
- **Cross-Source History (2 features)**: Cumulative historical pump count from La Morgia log.
- **Model**: `XGBRanker` with `rank:pairwise` objective, depth 3, 400 trees, 3-seed ensemble.

### Stage B: Explainable Temporal-Anomaly Stacked Ensemble (ETASE)
- **Component A (Temporal MLP)**: 2-layer MLP (64, 32 units) over 10-chunk rolling window across 8 core features + cyclic hour encodings (82 dimensions).
- **Component B (Isolation Forest)**: 200 isolation trees fitted strictly on non-pump normal market chunks.
- **Component C (XGBoost Classifier)**: 400 trees, depth 5, learning rate 0.05, square-root class balancing on base 12 features + 12 past-only deltas and rolling z-scores.
- **Meta-Learner**: Logistic Regression combining $\text{logit}(A)$, $S_{IF}(B)$, and $\text{logit}(C)$ fitted on held-out 25% training pumps.

$$\text{ETASE Score: } p(x) = \sigma(w_0 + w_A \cdot \text{logit}(A(x)) + w_B \cdot S_{IF}(x) + w_C \cdot \text{logit}(C(x)))$$

---

## 7. Results and Performance

### 7.1 Replication of Base Paper (Random Forest)
| Chunk Size | Reported P | Reported R | Reported F1 | Our Replicated P | Our Replicated R | Our Replicated F1 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **5 s** | 0.924 | 0.784 | 0.840 | 0.943 | 0.735 | 0.826 |
| **15 s** | 0.913 | 0.844 | 0.877 | 0.968 | 0.855 | 0.908 |
| **25 s** | 0.937 | 0.913 | 0.918 | 0.966 | 0.890 | **0.926** |

### 7.2 Main Benchmark at 25s (Leakage-Safe GroupKFold)
| Model | Precision | Recall | F1 Score | PR-AUC | Event Recall | Pre-Onset Alerts | False Alarms / Coin-Day |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **ETASE (Proposed)** | 0.936 | **0.927** | **0.932** | 0.961 | **0.931 (295/317)** | 4 | 0.115 |
| **XGBoost + Temporal** | **0.963** | 0.905 | **0.933** | **0.975** | 0.909 (288/317) | 2 | **0.065** |
| **Random Forest (Base)** | 0.963 | 0.893 | 0.926 | 0.962 | 0.896 (284/317) | 1 | 0.072 |
| **Logistic Regression** | 0.886 | 0.861 | 0.874 | 0.903 | 0.861 (273/317) | 4 | 0.215 |
| **Temporal MLP** | 0.814 | 0.814 | 0.814 | 0.856 | 0.820 (260/317) | 8 | 0.244 |
| **Isolation Forest** | 0.450 | 0.704 | 0.549 | 0.492 | 0.707 (224/317) | 8 | 1.892 |

### 7.3 Stage A Target Ranking Performance (272 Test Pumps)
| Ranker | HR@1 | HR@5 | HR@10 | HR@20 | HR@30 | MRR |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Random Baseline** | 0.004 | 0.018 | 0.037 | 0.073 | 0.110 | 0.023 |
| **Popularity Benchmark** | 0.000 | 0.085 | 0.239 | 0.287 | 0.338 | 0.065 |
| **CATR (Market + Channel + Cross-Source)** | 0.000 | 0.110 | **0.283** | **0.493** | **0.647** | 0.090 |

### 7.4 Two-Stage Cascade Performance (131 Matched Pumps)
| Setting | Watch-list Share | Event Recall | Mean Delay (s) | Pre-Onset Alerts | Exchange False Alarms / Day |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **ETASE Alone** | N/A | 0.908 | -0.6 s | 2 | 11.8 |
| **Cascade (K=10, thr $\times 0.5$)** | 25% | 0.908 | -1.9 s | 8 | 25.4 |
| **Cascade (K=20, thr $\times 0.5$)** | 40% | **0.924** | -2.3 s | 10 | 38.9 |
| **Cascade (K=20, thr $\times 0.25$)** | 40% | **0.924** | **-2.9 s** | **12** | 48.1 |

---

## 8. Conclusion

We developed **ETASE-2S**, a unified two-stage early-warning framework for cryptocurrency pump-and-dump manipulation. ETASE achieved 93.2% F1 and 93.1% event recall under leakage-free pump-grouped evaluation, while the CATR ranker placed 64.7% of upcoming pump targets into a 30-coin watch-list hours in advance. Cascade integration increased pre-onset alerts six-fold. The system provides institutional transparency through SHAP explanations and a Zerodha-inspired surveillance terminal interface.
