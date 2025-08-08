# 🧠 Currency Exchange Rate Predictor

This project forecasts the next-day **USD Exchange Rate** using historical financial data from multiple countries.
example - 
[!ui](assets/image.png)
## 📌 Model
- Forecasting model: **Facebook Prophet**
- Input: Date + historical USD_ExchangeRate
- Output: Next day's USD_ExchangeRate with upper and lower bounds

## 🗂️ Folder Structure

Currency_Exchange_Predictor/
├── data/
├── models/
├── train_model.py
├── predict.py
├── preprocess.py
└── README.md

## 🚀 How to Run


python train_model.py
python predict.py

📅 Prediction for: 2023-05-18
💵 USD Exchange Rate (Predicted): 5.0267
📉 Lower bound: 5.0151
📈 Upper bound: 5.0382

Dataset link - https://www.kaggle.com/datasets/frtgnn/daily-macrofinancial-pulse-of-emerging-markets