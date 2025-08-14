📈 Crypto Price Forecasting Model
This project predicts the future closing prices of cryptocurrencies using historical OHLC (Open, High, Low, Close) data.
The workflow follows a modular SSOC-style approach with separate scripts for preprocessing, training, and predicting.
[!ui](assets/image.png)
🚀 Features
Data Preprocessing – Cleans raw data, handles missing values, creates lag features and moving averages.

Model Training – Trains a machine learning model (RandomForest by default) on selected cryptocurrency data.

Prediction – Loads a trained model to forecast future close prices.

Metrics – Saves RMSE and MAE for performance evaluation.

📂 Usage
1️⃣ Preprocess Data
python preprocess.py --symbol BTCUSDT
Reads raw CSV from data/

Cleans and engineers features

Saves processed file to data/processed_<symbol>.csv

2️⃣ Train Model
python train.py --symbol BTCUSDT --model randomforest

3️⃣ Predict Future Prices
python predict.py --symbol BTCUSDT --days 7
