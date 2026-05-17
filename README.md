# Tesla Stock Direction Prediction (Stock Indicators + Tweet Sentiment)

This repository contains a Jupyter Notebook prototype that predicts whether **Tesla (TSLA)** will go **UP (1)** or **DOWN (0)** on the **next trading day** using engineered stock indicators, and then tests whether adding **tweet sentiment** improves performance.

Two Logistic Regression models are compared:
1. **Stock-only Logistic Regression** (technical indicators only)
2. **Stock + Tweets Logistic Regression** (technical indicators + lagged tweet sentiment features)

> **Note:** Datasets are **not included** in this repository.

---

## Project Summary

- **Goal:** Predict next-day TSLA direction (UP/DOWN)
- **Period:** 2018–2022
- **Technique:** Logistic Regression (baseline + combined model)
- **Sentiment method:** VADER (compound score)
- **Evaluation:** Time-ordered train/test split (80/20), Accuracy, Confusion Matrix, ROC/AUC

---

## Notebook Contents (Pipeline)

### 1) Inputs
- **TSLA stock dataset:** Date + OHLCV (Open, High, Low, Close, Volume)
- **Tweet dataset:** Date + Tweet text

### 2) Stock preprocessing & feature engineering
- Convert Date → datetime (daily), sort chronologically, filter 2018–2022
- Engineer technical indicators:
  - **Returns / momentum:** Ret1, Ret3, Ret5, Ret12
  - **Trend:** MA10, MA20, MA_Gap, MA_Cross
  - **Volatility:** rolling volatility (rolling std of returns)
  - **Volume:** VolChg, VolZ20 (20-day z-score)
- Create target label:
  - **Target = 1** if next-day return > 0, else **0**

### 3) Tweet preprocessing & feature engineering
- Convert Date → datetime (daily)
- Compute **VADER compound sentiment** for each tweet
- Aggregate per day:
  - **Tweet_Sent:** daily mean sentiment
  - **Tweet_Count:** daily tweet volume
- Align with full daily calendar (includes weekends), fill missing days with 0
- **Lag by 1 day** (leakage control + weekend routing to next trading day):
  - Tweet_Sent_lag1, Tweet_Count_lag1
- Log-transform tweet count:
  - Tweet_Count_lag1_log = log1p(Tweet_Count_lag1)

### 4) Models
- **Time-ordered split:** first 80% train, last 20% test (no shuffling)
- **Scaling:** StandardScaler (fit on train only)
- **Model 1:** Logistic Regression (stock-only)
- **Model 2:** Logistic Regression (stock + tweets)

### 5) Evaluation
- Accuracy
- Confusion matrices (Stock-only vs Stock+Tweets)
- ROC curves + AUC
- Logistic Regression coefficients (interpretability)

---

## Results (Example From This Prototype)

Your exact numbers may vary depending on dataset versions and time window, but the prototype produced results around:

- **Stock-only:** Accuracy ≈ 0.536, AUC ≈ 0.536  
- **Stock + Tweets:** Accuracy ≈ 0.540, AUC ≈ 0.530  

Interpretation: tweet sentiment adds only a small improvement in this setup, and next-day direction remains difficult (weak separability).

---

## How to Run

### 1) Install dependencies
```bash
pip install pandas numpy scikit-learn matplotlib vaderSentiment
