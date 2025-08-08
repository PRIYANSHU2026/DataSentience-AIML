# ⛽ OilSight — Oil Price Sensitivity Forecasting

OilSight predicts the **Oil Price (USD/barrel)** based on macroeconomic signals like interest rates, bond yields, and equity index values.
[!ui](assets/image.png)
[!ui](assets/Screenshot%202025-08-06%20202201.png)
## 🎯 Target
Predict `Oil_Price(USD_per_bbl)` using:
- USD Exchange Rate
- Policy Rate (%)
- 10Y Bond Yield (%)
- Yield Spread vs US
- Equity Index Level

## 🤖 Model
- Gradient Boosting Regressor
- Evaluation Metrics: R² Score, MSE

## 🚀 How to Use

python train_model.py
python predict.py
OilSight/
├── data/
├── models/
├── preprocess.py
├── train_model.py
├── predict.py
└── README.md

---

## 📦 `requirements.txt`

```txt
pandas
scikit-learn
joblib
numpy

Dataset used - https://www.kaggle.com/datasets/frtgnn/daily-macrofinancial-pulse-of-emerging-markets