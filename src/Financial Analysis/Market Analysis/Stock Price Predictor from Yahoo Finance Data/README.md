# 📈 Stock Price Predictor from Yahoo Finance Data

A machine learning pipeline to predict stock prices using Yahoo Finance data, focusing on key financial indicators like volume, market cap, PE ratio, and price movement percentages. Built with Python, Pandas, Scikit-learn, and Random Forests.

---

[!ui](assets/image.png)

## 📊 Project Overview

This project uses a dataset of top stocks extracted from Yahoo Finance and trains a model to estimate the **current stock price** based on:

- 📉 Change in price
- 📊 Percentage change
- 🔁 Trading volume
- 🏢 Market capitalization
- 📐 P/E ratio

---

## 🧠 Problem Statement

Given a company’s trading indicators, can we **accurately predict its stock price**?

> Example:
> Predict price of NVIDIA stock with:
> - Change: -4.15
> - Change %: -2.33%
> - Volume: 202.637M
> - Market Cap: 4.237T
> - PE Ratio: 56.22

---

## 🗂️ Project Structure

Stock Price Predictor from Yahoo Finance Data/
│
├── data/
│ └── yahoo-stocks-data(Uncleaned_data).xlsx ← Raw input dataset
│
├── model/
│ └── stock_price_predictor.pkl ← Saved trained model
│
├── preprocess.py ← Data cleaning + feature engineering
├── train.py ← Model training + evaluation
├── predict.py ← Inference script with sample input
├── README.md ← This file

---

## ⚙️ How It Works

### 1. Data Preprocessing (`preprocess.py`)
- Cleans and parses strings like `"4.237T"`, `"202.637M"` into floats.
- Drops non-numeric fields (`symbol`, `name`).
- Handles missing or malformed values (e.g., `"--"` in PE Ratio).

### 2. Model Training (`train.py`)
- Uses Random Forest Regressor to model stock prices.
- Prints **RMSE** and **R² Score** for performance.
- Saves model in `model/stock_price_predictor.pkl`.

### 3. Prediction (`predict.py`)
- Takes a dictionary of stock indicators.
- Preprocesses the input in the same format as training.
- Predicts and prints the stock price.

---

## 🧪 Example

Run prediction directly from the script:

```bash
python predict.py

---

## ⚙️ How It Works

### 1. Data Preprocessing (`preprocess.py`)
- Cleans and parses strings like `"4.237T"`, `"202.637M"` into floats.
- Drops non-numeric fields (`symbol`, `name`).
- Handles missing or malformed values (e.g., `"--"` in PE Ratio).

### 2. Model Training (`train.py`)
- Uses Random Forest Regressor to model stock prices.
- Prints **RMSE** and **R² Score** for performance.
- Saves model in `model/stock_price_predictor.pkl`.

### 3. Prediction (`predict.py`)
- Takes a dictionary of stock indicators.
- Preprocesses the input in the same format as training.
- Predicts and prints the stock price.

---

## 🧪 Example

Run prediction directly from the script:

```bash
python predict.py
