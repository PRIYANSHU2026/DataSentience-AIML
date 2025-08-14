# 💰 Credit Score Regressor

This project predicts a user's credit score based on their financial behavior using a machine learning regression model. It is built using a synthetic personal finance dataset that includes income, expenses, savings, loans, and other financial metrics.

[!ui ss](assets/image.png)
---

## 🚀 Project Overview

- **Goal**: Predict an individual's credit score using features like income, debt, savings, and demographics.
- **Model Type**: Regression
- **Algorithm Used**: Random Forest Regressor

---

## 🧠 Features Used

| Feature | Description |
|---------|-------------|
| `age`, `gender`, `education_level`, `employment_status`, `job_title` | Demographics |
| `monthly_income_usd`, `monthly_expenses_usd`, `savings_usd` | Income and savings data |
| `has_loan`, `loan_type`, `loan_amount_usd`, `monthly_emi_usd`, `loan_interest_rate_pct`, `loan_term_months` | Loan profile |
| `debt_to_income_ratio`, `savings_to_income_ratio` | Financial health indicators |
| `region` | User location |

---

## 📁 Project Structure

CreditScoreRegressor/
│
├── data/ # Dataset
├── model/ # Trained model and preprocessor
├── preprocess.py # Feature engineering and transformation
├── train.py # Model training script
├── predict.py # Inference script for prediction
└── README.md # Project overview


---

## 🛠️ How to Run

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt


python train.py
python predict.py

📊 Use Cases
Credit monitoring platforms

Pre-qualification loan scoring

Personal finance advisory tools