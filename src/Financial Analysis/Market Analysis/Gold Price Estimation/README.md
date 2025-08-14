# 🟡 Gold Price Close Prediction

This project predicts the **closing price of gold** based on its opening, high, low prices and trading volume using a regression model.

---
sample_input = {
        "priceOpen": 0.015,
        "priceHigh": 0.016,
        "priceLow": 0.014,
        "volume": 350.0
    }
[!ui ss](assets/image.png)

## 📌 Problem Statement

The price of gold fluctuates daily due to various market dynamics. Traders, investors, and analysts can benefit from a model that predicts the **`priceClose`** based on recent market data. This project aims to build a **regression model** to forecast the closing price using:

- `priceOpen`
- `priceHigh`
- `priceLow`
- `volume`

---

## 📊 Dataset

- **Rows**: 664  
- **Columns**: 9  
- **Source**: [Uploaded CSV file]
- **Key Features**:
  - `priceOpen`, `priceHigh`, `priceLow`, `priceClose` – gold prices (formatted as strings with commas)
  - `volume` – trading volume
  - `timeOpen`, `timeClose`, `timeHigh`, `timeLow` – timestamps (dropped)

---

## 📁 Folder Structure

```
Gold Price Close Prediction/
│
├── data/
│   └── gold_price.csv
│
├── model/
│   └── gold_price_model.pkl
│
├── preprocess.py         # Cleans & formats data
├── train.py              # Trains XGBoost regression model
├── predict.py            # Predicts priceClose from input
└── README.md
```

---

## ⚙️ Model Details

- **Model Type**: Regression
- **Algorithm**: XGBoost Regressor
- **Hyperparameters**:  
  - `n_estimators=100`  
  - `learning_rate=0.1`  
  - `max_depth=3`

---

## 🧹 Preprocessing

- Replaces commas `,` with dots `.` in numeric fields
- Converts string-formatted price columns to floats
- Drops timestamp-related columns
- Extracts features `X` and target `y`

---

## 🏋️ Training

```bash
python train.py
```
- Loads dataset from `data/`
- Preprocesses using `preprocess.py`
- Trains and saves model to `model/gold_price_model.pkl`

---

## 🔮 Prediction

```bash
python predict.py
```

Sample input:
```python
{
  "priceOpen": 0.015,
  "priceHigh": 0.016,
  "priceLow": 0.014,
  "volume": 350.0
}
```

Expected output:
```
📈 Predicted priceClose: 0.015734
```

---

## 📦 Dependencies

- pandas
- xgboost
- joblib

Install with:
```bash
pip install pandas xgboost joblib
```

---

## 📁 Future Improvements

- Add timestamp-based features (e.g., trends)
- Evaluate model accuracy with RMSE, MAE
- Deploy as Streamlit app

---
