# predict.py

import pandas as pd
import joblib

def predict_attrition(sample_input: dict):
    # Load trained model
    model = joblib.load("models/catboost_model.pkl")

    # Prepare the input
    input_df = pd.DataFrame([sample_input])
    input_df = input_df.astype("float32")  # Just to be safe

    # Predict
    prediction = model.predict(input_df)[0]
    return "Yes" if prediction == 1 else "No"

# Example usage
if __name__ == "__main__":
    sample_employee = {
        'Age': 35,
        'BusinessTravel': 1,
        'DailyRate': 800,
        'Department': 2,
        'DistanceFromHome': 10,
        'Education': 3,
        'EducationField': 1,
        'EnvironmentSatisfaction': 4,
        'Gender': 1,
        'HourlyRate': 70,
        'JobInvolvement': 3,
        'JobLevel': 2,
        'JobRole': 4,
        'JobSatisfaction': 4,
        'MaritalStatus': 1,
        'MonthlyIncome': 5000,
        'MonthlyRate': 20000,
        'NumCompaniesWorked': 3,
        'OverTime': 1,
        'PercentSalaryHike': 12,
        'PerformanceRating': 3,
        'RelationshipSatisfaction': 3,
        'StockOptionLevel': 1,
        'TotalWorkingYears': 10,
        'TrainingTimesLastYear': 3,
        'WorkLifeBalance': 3,
        'YearsAtCompany': 5,
        'YearsInCurrentRole': 3,
        'YearsSinceLastPromotion': 1,
        'YearsWithCurrManager': 4
    }
    sample_employee2 = {
    'Age': 26,
    'BusinessTravel': 2,               
    'DailyRate': 1250,
    'Department': 1,                 
    'DistanceFromHome': 15,
    'Education': 2,
    'EducationField': 3,             
    'EnvironmentSatisfaction': 1,
    'Gender': 0,                       #
    'HourlyRate': 48,
    'JobInvolvement': 2,
    'JobLevel': 1,
    'JobRole': 2,                      
    'JobSatisfaction': 1,
    'MaritalStatus': 0,                
    'MonthlyIncome': 2500,
    'MonthlyRate': 19000,
    'NumCompaniesWorked': 4,
    'OverTime': 1,                   
    'PercentSalaryHike': 11,
    'PerformanceRating': 3,
    'RelationshipSatisfaction': 2,
    'StockOptionLevel': 0,
    'TotalWorkingYears': 3,
    'TrainingTimesLastYear': 1,
    'WorkLifeBalance': 1,
    'YearsAtCompany': 1,
    'YearsInCurrentRole': 1,
    'YearsSinceLastPromotion': 0,
    'YearsWithCurrManager': 0
  }


    result = predict_attrition(sample_employee)
    result2 = predict_attrition(sample_employee2)
    print("🔍 Predicted Attrition for sample_data1:", result)
    print("🔍 Predicted Attrition for sample_data2:", result2)
