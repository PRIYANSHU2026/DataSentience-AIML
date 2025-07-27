import joblib
import pandas as pd

def predict_candidate_offer(input_data):
    """
    input_data: dict
        {
            'Confidence Score': 8,
            'Structured Thinking Score': 7,
            'Regional Fluency Score': 6,
            'Total Score': 21,
            'Age': 25,
            'Experienced candidate - (Experience in months)': 24,
            'Last Fixed CTC (lakhs) ': 5.5,
            'Gender': 'Male',
            'Marital status': 'Unmarried',
            'Currently Employed': 'Yes',
            'Role acceptance': 'Yes',
            'Candidate is willing to relocate': 'Yes',
            'Mode of interview given by candidate?': 'In-person'
        }
    Returns: int (0 or 1)
    """
    model = joblib.load('model/model.pkl')
    label_encoders = joblib.load('model/label_encoders.pkl')

    # Define feature order
    num_cols = [
        'Confidence Score', 'Structured Thinking Score', 'Regional Fluency Score',
        'Total Score', 'Age', 'Experienced candidate - (Experience in months)',
        'Last Fixed CTC (lakhs) '
    ]
    cat_cols = [
        'Gender', 'Marital status', 'Currently Employed',
        'Role acceptance', 'Candidate is willing to relocate',
        'Mode of interview given by candidate?'
    ]

    # Create input DataFrame
    input_df = pd.DataFrame([input_data])

    # Preprocess categorical inputs (lowercase + encode)
    for col in cat_cols:
        val = str(input_df.at[0, col]).strip()


        le = label_encoders[col]
        if val not in le.classes_:
            raise ValueError(f"Unknown category '{val}' in column '{col}'. Allowed: {list(le.classes_)}")
        input_df[col] = le.transform([val])

    # Predict
    final_input = input_df[num_cols + cat_cols]
    prediction = model.predict(final_input)[0]
    return int(prediction)


# Example usage
if __name__ == "__main__":
    sample_input = {
    'Confidence Score': 9,
    'Structured Thinking Score': 8,
    'Regional Fluency Score': 7,
    'Total Score': 24,
    'Age': 27,
    'Experienced candidate - (Experience in months)': 36,
    'Last Fixed CTC (lakhs) ': 6.2,
    'Gender': 'Male',
    'Marital status': 'Unmarried',
    'Currently Employed': 'Yes',
    'Role acceptance': 'No',
    'Candidate is willing to relocate': 'Yes - Anywhere Within a State',
    'Mode of interview given by candidate?': 'Laptop'
}


    result = predict_candidate_offer(sample_input)
    print("Prediction:", "Joined ✅" if result == 1 else "Did Not Join ❌")
