# Reference Papers Pack & Literature Survey Benchmarks

**Aligned with Project Paper**: *AI-Based Early Detection of Crypto Pump-and-Dump Schemes* (Group-20, BCSE206L, VIT Vellore).

---

## 1. Master Reference Papers Catalog

| # | Paper Title | Authors | Year & Venue | Methodology | Data Utilized | Primary Role in Project | Key Result Quoted |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Base** | *Pump and Dumps in the Bitcoin Era: Real Time Detection of Cryptocurrency Market Manipulations* | La Morgia, Mei, Sassi, Stefa | 2020, IEEE ICCCN | Random Forest on rush-order chunk features | Telegram P&D list (D1) + labelled Binance trade chunks (D3) | Base model, replication target, comparative baseline | 25s RF: P 93.7%, R 91.3%, F1 91.8% |
| **R1** | *The Doge of Wall Street: Analysis and Detection of Pump and Dump Cryptocurrency Manipulations* | La Morgia, Mei, Sassi, Stefa | 2023, ACM TOPS | Extended Random Forest detector | ~900 events across 3 years | Longitudinal comparison, performance benchmark | F1 94.5% within 25s of onset |
| **R2** | *Sequence-Based Target Coin Prediction for Cryptocurrency Pump-and-Dump* | Hu, Zhang, Lu, He, Li | 2023, ACM SIGMOD (PACMMOD) | SNN with positional attention | 1,335 Telegram samples (709 pumps) (D2); CoinGecko snapshots (D4) | Stage A architecture design, target ranking baseline; source of D2 & D4 | SNN: HR@1 0.253, HR@10 0.599, HR@30 0.824 |
| **R3** | *Crypto Pump and Dump Detection via Deep Learning Techniques* | Chadalapaka, Chang, Mahajan, Vasil | 2022, arXiv | C-LSTM, Anomaly Transformer | Base-paper 25s/15s/5s features | Deep learning sequence comparison | C-LSTM 25s F1 89.3%; Anomaly Transformer 15s F1 93.6% |
| **R4** | *Improving Cryptocurrency Pump-and-Dump Detection through Ensemble-Based Models and Synthetic Oversampling Techniques* | Yu, Park, Chai | 2025, arXiv | SMOTE + 5 Ensembles (XGBoost, LightGBM, CatBoost) | Base-paper 25s features (482k rows) | Class imbalance mitigation, recall ceiling benchmark | XGBoost + SMOTE Recall 94.87%, LightGBM 93.59% |
| **R5** | *Real-Time Machine Learning Detection of Telegram-Based Pump-and-Dump Schemes* | Bolz et al. | 2025, ACM DeFi ’25 | PumpBERT + Z-score target ranking | 91,295 Telegram messages, 43 backtests | Pre-pump message parsing & ranking comparison | PumpBERT Weighted F1 0.982; Top-5 hit 55.8% at 20s prior |
| **R6** | *PumpSense: Real-Time Detection and Target Extraction of Crypto Pump-and-Dumps on Telegram* | Mahrous, Di Pietro | 2026, arXiv | LightGBM, BGE-M3 embeddings, LLM extraction | 283,017 messages | Announcement classification, NLP target extraction | BGE-M3 F1 0.83; Gemini 2.5 Pro Target Extraction 0.91 Acc |
| **R7** | *Detecting cryptocurrency pump-and-dump frauds using market and social signals* | Nghiem, Muric, Morstatter, Ferrara | 2021, Expert Systems with Applications | Multi-modal market + social target classifier | Telegram, Twitter, Binance market feeds | Stage A multi-modal motivation | Pre-announcement ranking feasibility |
| **R8** | *A Survey on Pump and Dump Detection in the Cryptocurrency Market Using Machine Learning* | Rajaei, Mahmoud | 2023, Future Internet 15(8) | Comprehensive literature survey | Cross-study empirical analysis | Identifying research gaps (lack of explainability & stacking) | Comprehensive methodology taxonomy |
| **R9** | *An examination of the cryptocurrency pump-and-dump ecosystem* | Hamrick et al. | 2021, Information Processing & Management | Empirical ecosystem characterization | Telegram + Discord channels | Introduction & economic market distortion analysis | Smaller cap coins yield higher peak returns |
| **R10** | *The Anatomy of a Cryptocurrency Pump-and-Dump Scheme* | Xu, Livshits | 2019, USENIX Security | Target-coin likelihood model | Telegram channels + historical market data | Target-coin likelihood formulation | Profitable simulated front-running strategy |
| **R11** | *To the moon: defining and detecting cryptocurrency pump-and-dumps* | Kamps, Kleinberg | 2018, Crime Science 7 | Heuristic threshold rules on volume/price | Exchange OHLCV (1-hour resolution) | Classical threshold baseline | Strict rules F1 60.5% (tested on base-paper data) |
| **R12** | *Cryptocurrency Pump and Dump Schemes: Quantification and Detection* | Victor, Hagemann | 2019, IEEE ICDMW | Gradient boosted decision trees (XGBoost) | Telegram signals + Binance 1h data | Baseline tabular modeling | Hour-level AUC ~0.99 |
| **R13** | *LLD: A Low Latency Detection Solution to Thwart Cryptocurrency Pump & Dumps* | Anonymous | 2023, IEEE ICBC | LSTM auto-encoder on BTC + Gaussian-tail band on altcoins | 1-minute altcoin OHLCV | Unsupervised latency baseline | Detection in <5 min (average 2.2 min) |
| **R14** | *HD Pump: A Hybrid Detection Approach for Pump-and-Dump Schemes in Cryptocurrency Exchanges* | Moura, Baroni, Ogasawara, Mendonça | 2024, SBBD | Volatility anomaly detection + change-point volume detection | Binance pump event data | Hybrid statistical baseline | Precision +6.7%, Recall +9.3% over anomaly baseline |

---

## 2. Side-by-Side Experimental Benchmark Comparison

| Experiment Setting | Method / Model | Chunk / Timeframe | Evaluation Protocol | Precision | Recall | F1 Score | PR-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Rule Baseline** | Kamps & Kleinberg (2018) | 1-hour OHLCV | Strict Heuristic Threshold | 0.501 | 0.750 | 0.605 | — |
| **Base Paper Reported** | Random Forest (La Morgia 2020) | 25 s | 5-Fold Stratified Row CV | 0.937 | 0.913 | 0.918 | — |
| **Base Paper Reported** | Random Forest (La Morgia 2020) | 15 s | 5-Fold Stratified Row CV | 0.913 | 0.844 | 0.877 | — |
| **Base Paper Reported** | Random Forest (La Morgia 2020) | 5 s | 5-Fold Stratified Row CV | 0.924 | 0.784 | 0.840 | — |
| **Our Exact Replication** | Random Forest (Base Code) | 25 s | 5-Fold Stratified Row CV | 0.966 | 0.890 | 0.926 | 0.962 |
| **Our Exact Replication** | Random Forest (Base Code) | 15 s | 5-Fold Stratified Row CV | 0.968 | 0.855 | 0.908 | 0.946 |
| **Our Exact Replication** | Random Forest (Base Code) | 5 s | 5-Fold Stratified Row CV | 0.943 | 0.735 | 0.826 | 0.902 |
| **Deep Learning Baseline** | C-LSTM (Chadalapaka 2022) | 25 s | 80/20 Temporal Split | 0.942 | 0.850 | 0.893 | — |
| **Deep Learning Baseline** | Anomaly Transformer (Chadalapaka 2022) | 15 s | 80/20 Temporal Split | 0.930 | 0.942 | 0.936 | — |
| **Component Model** | XGBoost + 12 Temporal Features | 25 s | **GroupKFold by Pump (Leak-Free)** | **0.963** | 0.905 | **0.933** | **0.975** |
| **Proposed Framework** | **ETASE Stacked Ensemble** | 25 s | **GroupKFold by Pump (Leak-Free)** | 0.936 | **0.927** | **0.932** | 0.961 |
| **Proposed Framework** | **ETASE Stacked Ensemble** | 15 s | **GroupKFold by Pump (Leak-Free)** | 0.932 | **0.871** | **0.901** | 0.941 |
| **Proposed Framework** | **ETASE Stacked Ensemble** | 5 s | **GroupKFold by Pump (Leak-Free)** | 0.908 | 0.746 | 0.819 | 0.884 |
