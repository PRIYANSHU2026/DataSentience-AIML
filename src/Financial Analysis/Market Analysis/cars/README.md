# 🚗 Car Performance Classifier

This project aims to classify cars into performance categories — **Sport**, **Standard**, or **Eco** — based on numerical performance metrics like **HorsePower**, **Top Speed**, and **0–100 km/h Acceleration Time**. The project uses **unsupervised clustering (KMeans)** to group cars into performance tiers, followed by **cluster-to-label mapping** based on cluster centroids.

---

## 🎯 Objective

- Identify whether a car is:
  - **Sport**: High horsepower, fast acceleration
  - **Eco**: Low horsepower, slow acceleration, high efficiency
  - **Standard**: Moderate all-round performance

---
[!ui ss](assets/image.png)

## 📁 Project Structure
cars/
│
├── data/
│ └──<dataset>.csv
│
├── model/
│ ├── kmeans_model.pkl # Trained KMeans model
│ ├── scaler.pkl # StandardScaler object
│ └── cluster_class_mapping.csv # Mapping of cluster ID to class (Eco/Sport/Standard)
│
├── preprocess.py # Handles data cleaning and feature extraction
├── train.py # Clusters the data and builds the model
├── predict.py # Predicts class for a new car
└── README.md


---

## 🧪 Features Used

- **HorsePower** — Numeric or string (e.g., "400 hp")
- **Top Speed** — Max speed in km/h or mph
- **Performance** — Time to go 0–100 km/h (e.g., "4.2 sec")

---

## ⚙️ How It Works

### 🔹 Step 1: Preprocessing (`preprocess.py`)
- Cleans and extracts numeric values from strings (e.g., "400 hp", "4.2 sec")
- Selects only relevant features

### 🔹 Step 2: Training (`train.py`)
- Standardizes numerical features
- Applies **KMeans clustering** with `n_clusters=3`
- Assigns class labels based on cluster centroids (automatically inferred as Sport/Eco/Standard)

### 🔹 Step 3: Prediction (`predict.py`)
- Takes a dictionary input like:
  ```python
  {
    "HorsePower": "400 hp",
    "Total Speed": "320 km/h",
    "Performance": "3.5 sec"
  }
