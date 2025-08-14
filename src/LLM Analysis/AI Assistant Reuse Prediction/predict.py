# predict.py
import joblib
import pandas as pd

MODEL_PATH = "model/used_again_predictor.pkl"

def predict_used_again(sample_input: dict) -> int:
    """
    Predict whether a student will use AI again.
    Args:
        sample_input (dict): Features in dictionary form
    Returns:
        int: 1 if they will use again, 0 otherwise
    """
    model = joblib.load(MODEL_PATH)
    sample_df = pd.DataFrame([sample_input])
    return int(model.predict(sample_df)[0])

if __name__ == "__main__":
    sample = {
        "StudentLevel": "Undergraduate",
        "Discipline": "Computer Science",
        "SessionLengthMin": 25.5,
        "TotalPrompts": 8,
        "TaskType": "Coding",
        "AI_AssistanceLevel": 3,
        "FinalOutcome": "Assignment Completed",
        "SatisfactionRating": 4.2,
        "SessionMonth": 3,
        "SessionDay": 2
    }
    result = predict_used_again(sample)
    print("Will use AI again:" , "Yes" if result == 1 else "No")
