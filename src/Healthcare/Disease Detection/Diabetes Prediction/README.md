# Diabetes Prediction Model & Web App

This project provides a **machine learning model** and an **interactive web application** for predicting the likelihood of diabetes based on key health indicators.

## Files
- `diabetes_rf_model.pkl` – Trained Random Forest model for prediction  
- `diabetes_scaler.pkl` – StandardScaler for normalizing numeric inputs  
- `gender_encoder.pkl` – LabelEncoder for gender  
- `smoking_encoder.pkl` – LabelEncoder for smoking history  
- `diabetes-dataset.csv` – Dataset used for model training/testing  
- `app.py` – Streamlit-based web application  

## Model
- **Algorithm:** RandomForestClassifier  
- **Features:**
  - Gender
  - Age
  - Hypertension
  - Heart Disease
  - Smoking History
  - BMI
  - HbA1c Level
  - Blood Glucose Level

## Web Application
An easy-to-use **Streamlit app** is included. Users can enter health details into a clean and intuitive interface to get instant predictions.

### Running the App
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
````

2. Start the application:

   ```bash
   streamlit run app.py
   ```

## Prediction Output

* **Non-Diabetic** – Low risk detected
* **Diabetic** – High risk detected

```
