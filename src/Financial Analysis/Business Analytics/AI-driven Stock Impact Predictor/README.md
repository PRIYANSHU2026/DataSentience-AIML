# AI-Driven Stock Impact Predictor

## Overview

The AI-Driven Stock Impact Predictor is a machine learning-based application designed to estimate the percentage impact on stock prices using historical and synthetic financial market data.

The project applies data preprocessing, feature engineering, and regression-based machine learning techniques to analyze financial indicators and market events that may influence stock performance. The system demonstrates the practical application of AI and machine learning in financial analysis and predictive analytics.

---

## Features

### Data Preprocessing
- Cleans and formats raw financial datasets
- Handles missing values
- Encodes categorical variables
- Scales numerical features
- Extracts useful date-based features

### Model Training
- Uses regression algorithms for stock impact prediction
- Supports modular and reusable training workflows
- Saves trained models for future predictions

### Evaluation Metrics
The model performance is evaluated using:
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)

### Prediction Pipeline
- Accepts new financial market input data
- Loads trained preprocessing and model files
- Predicts expected stock impact percentage

### Modular Design
The project is divided into separate scripts for:
- preprocessing
- training
- prediction

This structure improves maintainability and scalability.

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib

---

## Project Structure

```text
AI-Driven-Stock-Impact-Predictor/
│
├── data/
│   └── dataset.csv
│
├── model/
│   ├── model.pkl
│   └── preprocessor.pkl
│
├── preprocess.py
├── train.py
├── predict.py
├── requirements.txt
└── README.md
```

---

## Installation and Setup

### Clone the Repository

```bash
git clone <https://github.com/PRIYANSHU2026/DataSentience-AIML/tree/main/src/Financial%20Analysis/Business%20Analytics/AI-driven%20Stock%20Impact%20Predictor>
cd AI-Driven-Stock-Impact-Predictor
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Create Model Directory

Before training the model, create a folder named `model` in the project directory.

```bash
mkdir model
```

---

## Training the Model

Run the following commands sequentially:

```bash
python preprocess.py
python train.py
```

These scripts will:
- preprocess the dataset
- prepare training and testing data
- train the machine learning model
- save the trained model and preprocessing pipeline

---

## Running Predictions

To generate predictions using the trained model:

```bash
python predict.py
```

The prediction pipeline:
- loads the trained model
- preprocesses input data
- predicts stock impact percentage

---

## Dataset Features

| Feature | Description |
|---|---|
| Date | Market date |
| Company | Company name |
| R&D_Spending_USD_Mn | Research and development spending |
| AI_Revenue_USD_Mn | AI-related revenue |
| AI_Revenue_Growth_% | Revenue growth percentage |
| Event | Market or business event |
| Stock_Impact_% | Predicted stock impact percentage |

---

## Future Improvements

Possible enhancements for the project include:
- integration with real-time stock market APIs
- sentiment analysis using financial news data
- advanced models such as XGBoost, LightGBM, or LSTM
- dashboard visualization for analytics
- real-time prediction support

---

## License

This project is developed for educational and research purposes.