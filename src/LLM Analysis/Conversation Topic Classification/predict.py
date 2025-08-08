import joblib

MODEL_PATH = "model/topic_classifier.pkl"

def classify_prompt(prompt: str) -> str:
    model = joblib.load(MODEL_PATH)
    return model.predict([prompt])[0]

if __name__ == "__main__":
    sample_prompt = "How do I debug this Python error?"
    print("Prompt:", sample_prompt)
    print("Predicted Topic:", classify_prompt(sample_prompt))
