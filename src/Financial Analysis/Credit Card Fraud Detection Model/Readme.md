 # Credit Card Fraud Detection

##  Project Overview  
Credit card fraud detection is a crucial task for financial institutions, as fraud can lead to significant losses.  
This project focuses on detecting fraudulent credit card transactions using **machine learning techniques**,  
specifically handling **imbalanced datasets** through **oversampling** and **undersampling** methods.  

The **Decision Tree Classifier** was chosen as the final model due to its outstanding performance, achieving near-perfect accuracy and recall.  
The model’s predictions were also visualized using a **confusion matrix** to better understand its classification performance.

---

## Dataset  
- **Source:** [Kaggle Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)  
- **Description:**  
  - Transactions made by European cardholders in September 2013.  
  - Features are numerical values obtained through **PCA transformation** (except `Time` and `Amount`).  
  - **Target Classes:**  
    - `0`:  Non-Fraud Transaction  
    - `1`:  Fraud Transaction  
  - **Size:** 284,807 transactions  
  - **Fraud cases:** Only 492 (~0.172%), making it a **highly imbalanced dataset**.

---

##  Project Workflow  

### 1. Data Loading & Exploration  
- Loaded dataset and checked data structure, missing values, and distribution of target variable.  
- Observed a strong imbalance between fraud and non-fraud transactions.

### 2. Data Preprocessing  
- Standardized `Amount` and `Time` features.  
- Removed irrelevant features that do not impact fraud detection.

### 3. Handling Class Imbalance  
Since fraud cases are very rare, two resampling strategies were applied:  
- **Oversampling:** Increased fraud cases using **SMOTE (Synthetic Minority Oversampling Technique)**.  
- **Undersampling:** Reduced majority class (non-fraud) to balance with fraud cases.  

Both techniques were tested, and the final dataset was balanced to improve model performance.

### 4. Train-Test Split  
- Split dataset into **80% training** and **20% testing** data.

### 5. Model Training  
Two machine learning models were trained and compared:  
- **Logistic Regression**  
- **Decision Tree Classifier** *(final chosen model)*

### 6. Model Evaluation  
Models were evaluated on:  
- Accuracy  
- Precision  
- Recall  
- F1-score  

### 7. Visualization  
- Plotted **Confusion Matrix** for the final Decision Tree Classifier to clearly visualize  
  the number of correct and incorrect predictions for both fraud and non-fraud classes.

---

##  Results  

| Metric        | Logistic Regression| Decision Tree Classifier|
|---------------|--------------------|--------------------------|
| Accuracy      | 0.9992             |  0.9989                  |
| Precision     | 0.8906             |0.6804                    |
| Recall        |0.6263              | 0.7252                   |
| F1-score      |0.7354              |0.7021                    |

>  **Final Model:** **Decision Tree Classifier**  
> Achieved **99.89% accuracy**, **72.52% recall**, and almost perfect precision.  
> The confusion matrix showed extremely low false negatives, which is critical in fraud detection.

---

## Confusion Matrix (Final Model)  

|               | Predicted Non-Fraud | Predicted Fraud |
|---------------|---------------------|-----------------|
| **Actual Non-Fraud** | TN (True Negatives)     | FP (False Positives) |
| **Actual Fraud**     | FN (False Negatives)    | TP (True Positives)  |

- **True Negatives (TN):** Correctly identified non-fraud transactions  
- **False Positives (FP):** Incorrectly flagged non-fraud transactions as fraud  
- **False Negatives (FN):** Missed fraud cases *(kept extremely low)*  
- **True Positives (TP):** Correctly detected fraud transactions  

---

## Libraries Used  
- Python  
- Pandas  
- NumPy  
- Matplotlib  
- Seaborn  
- Scikit-learn  
- Imbalanced-learn (SMOTE)  




