import joblib
from preprocess import extract_number
import pandas as pd
# Prediction function for new car input
def predict_performance_class(car_input):
    kmeans = joblib.load("model/kmeans_model.pkl")
    scaler = joblib.load("model/scaler.pkl")

    hp = extract_number(car_input['HorsePower'])
    ts = extract_number(car_input['Total Speed'])
    ac = extract_number(car_input['Performance'])

    if any(v is None or v != v for v in [hp, ts, ac]):
        raise ValueError("❌ Invalid input: Make sure all specs are correctly formatted.")

    scaled_input = scaler.transform([[hp, ts, ac]])
    cluster = kmeans.predict(scaled_input)[0]

    # Load cluster-class mapping
    mapping_df = pd.read_csv("model/cluster_class_mapping.csv", index_col="Cluster")
    class_name = mapping_df.loc[cluster, "Class"]

    return class_name


    # Map cluster to class
    def map_cluster(c):
        # Adjust based on your KMeans centroid analysis
        return {0: "Sport", 1: "Eco", 2: "Standard"}.get(c, "Unknown")

    return map_cluster(cluster)

# Example
if __name__ == "__main__":
    sample1 = {
        "HorsePower": "1500 hp",
        "Total Speed": "500 km/h",
        "Performance": "1 sec"
    }
    sample2 = {
        "HorsePower": "500 hp",
        "Total Speed": "100 km/h",
        "Performance": "2.5 sec"
    }
    print("🚗 Predicted Performance Class:", predict_performance_class(sample1))
    print("🚗 Predicted Performance Class:", predict_performance_class(sample2))
