# Delhi NCR Housing Price Prediction

This project predicts housing prices in the Delhi NCR region using machine learning on real estate data from MagicBricks.com, acquired via Kaggle.

## Overview

- **Goal:** Predict the **log of house price** (target) based on property features such as location, area, bedrooms, and more.
- **Methods Used:**
  1. Sequential Deep Learning Model
  2. XGBoost Regression

Both models achieved high accuracy in predicting the log-price of properties.

## Dataset

- **Source:** [Kaggle – MagicBricks.com Housing Data](https://www.kaggle.com/)  
- **Features Include:** Location, size (sqft), number of bedrooms, bathrooms, price, etc.

## Approach

1. **Data Processing:**
   - Cleaned missing values, encoded categorical variables, normalized features.
2. **Modeling:**
   - Trained both a deep learning model and XGBoost regressor to predict log(price).
   - Evaluated models with RMSE, MAE, and R² on validation/test sets.
3. **Results:**
   - Both approaches performed well on log-price prediction.
   - Analyzed feature importance and provided prediction analysis in the notebook.

## Usage

1. Open `Delhi_NCR_Housing_Price_Prediction.ipynb` in Google Colab or Jupyter.
2. Install required dependencies:
!pip install pandas numpy scikit-learn xgboost matplotlib seaborn
3. Follow the notebook cells to reproduce preprocessing, modeling, and results.

## Contributing

Feedback and contributions are welcome! Please open an issue or submit a pull request.

---
