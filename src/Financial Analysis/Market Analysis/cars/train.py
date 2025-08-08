import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import joblib
from preprocess import preprocess_data

# Load and preprocess original dataset
df = preprocess_data("data/original_dataset.csv")  # Make sure this is your raw dataset path
X = df[['HorsePower_clean', 'TopSpeed_clean', 'Accel_clean']]

# Normalize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# KMeans Clustering
kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto')
clusters = kmeans.fit_predict(X_scaled)

# Save model + scaler
joblib.dump(kmeans, "model/kmeans_model.pkl")
joblib.dump(scaler, "model/scaler.pkl")

# Optionally save labeled dataset
df['Cluster'] = clusters
df.to_csv("data/labeled_clusters.csv", index=False)

print("✅ Training complete. Model and scaler saved.")
# Save the cluster centroids (reverse transformed to real values)
centroids = scaler.inverse_transform(kmeans.cluster_centers_)
centroid_df = pd.DataFrame(centroids, columns=['HorsePower', 'TopSpeed', 'AccelTime'])

# Auto-map cluster labels to class names
def classify(row):
    if row['HorsePower'] > 350 and row['AccelTime'] < 5:
        return 'Sport'
    elif row['HorsePower'] < 150 and row['AccelTime'] > 9:
        return 'Eco'
    else:
        return 'Standard'

centroid_df['Class'] = centroid_df.apply(classify, axis=1)
centroid_df.to_csv("model/cluster_class_mapping.csv", index_label="Cluster")
