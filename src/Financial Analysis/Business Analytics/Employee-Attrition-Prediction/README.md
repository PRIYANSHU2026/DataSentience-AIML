# 🧠 Employee Attrition Prediction (HR AI)

This project predicts whether an employee is likely to **leave the organization** based on their personal and professional attributes. The model is trained on the **IBM HR Analytics Employee Attrition & Performance Dataset**, a popular dataset in the HR domain.

[!ui screenshot](assets/image.png)

this screenshot contains the predicted results the sample data contsins in the predict.py file
---

## 📊 Dataset

- **Source**: [Kaggle - IBM HR Analytics Dataset](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)
- **Target Column**: `Attrition` (`Yes` or `No`)
- **Features**: Age, Job Role, Marital Status, Overtime, Income, Travel Frequency, Job Satisfaction, and 30+ more HR-related attributes.

---

## 🧰 Tech Stack

- **Language**: Python
- **Model**: CatBoostClassifier (Gradient Boosted Decision Trees)
- **Libraries**:
  - `pandas`, `scikit-learn`, `catboost`
  - `joblib` (for model saving/loading)

---

## 🛠️ Project Structure

