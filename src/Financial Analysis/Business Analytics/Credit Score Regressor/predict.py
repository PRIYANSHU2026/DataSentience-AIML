import joblib
import pandas as pd

def predict_credit_score(sample_input_dict):
    # Load model and preprocessor
    model = joblib.load('model/credit_score_model.joblib')
    preprocessor = joblib.load('model/preprocessor.joblib')

    # Convert dict to DataFrame
    df_input = pd.DataFrame([sample_input_dict])

    # Preprocess
    X_processed = preprocessor.transform(df_input)

    # Predict
    prediction = model.predict(X_processed)[0]
    return round(prediction, 2)

# Example usage
if __name__ == "__main__":
    sample = {
        'age': 34,
        'gender': 'Male',
        'education_level': 'Bachelor',
        'employment_status': 'Employed',
        'job_title': 'Engineer',
        'monthly_income_usd': 5000,
        'monthly_expenses_usd': 2500,
        'savings_usd': 150000,
        'has_loan': 'Yes',
        'loan_type': 'Mortgage',
        'loan_amount_usd': 200000,
        'loan_term_months': 120,
        'monthly_emi_usd': 1500,
        'loan_interest_rate_pct': 5.5,
        'debt_to_income_ratio': 0.3,
        'savings_to_income_ratio': 3.0,
        'region': 'North America'
    }

    score = predict_credit_score(sample)
    print(f"Predicted Credit Score: {score}")
