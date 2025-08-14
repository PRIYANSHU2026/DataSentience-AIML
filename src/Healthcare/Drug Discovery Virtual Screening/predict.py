import pandas as pd
import joblib

MODEL_PATH = "model/drug_classifier.pkl"

def predict_activity(sample_input: dict) -> int:
    """
    Predicts whether a compound is active (1) or inactive (0).
    """
    pipeline = joblib.load(MODEL_PATH)
    df = pd.DataFrame([sample_input])
    prediction = pipeline.predict(df)[0]
    return int(prediction)

if __name__ == "__main__":
    # Replace values with actual measurements for the compound you want to test
    sample = {
        "binding_site_size": 120.5,
        "molecular_weight": 350.2,
        "polar_surface_area": 75.3,
        "h_bond_donors": 2,
        "rotatable_bonds": 5,
        "protein_pi": 6.8,
        "hydrophobicity": 2.5,
        "h_bond_acceptors": 4,
        "mw_ratio": 0.85,
        "logp_pi_interaction": 1.2,
        "protein_length": 400,
        "logp": 3.1,
        "compound_clogp": 2.9,
        "binding_affinity": -8.5
    }

    pred = predict_activity(sample)
    print(f"Predicted Active Status: {pred} (1 = Active, 0 = Inactive)")
