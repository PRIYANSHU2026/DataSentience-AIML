# predict.py
import pandas as pd
import joblib

def predict_new(data_dict):
    # Load model
    model = joblib.load("model/trained_model.pkl")

    # Convert dict to DataFrame
    df = pd.DataFrame([data_dict])

    # Convert Date to datetime and extract features
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df.drop(columns=['Date'], inplace=True)

    # Predict
    prediction = model.predict(df)
    return prediction[0]

if __name__ == "__main__":
    # Example input
    sample_input = {
        "Date": "2023-05-10",
        "Company": "OpenAI",
        "R&D_Spending_USD_Mn": 70.5,
        "AI_Revenue_USD_Mn": 45.3,
        "AI_Revenue_Growth_%": 150.2,
        "Event": "Product Launch"
    }

    pred = predict_new(sample_input)
    print(f"Predicted Stock Impact %: {pred:.2f}")
