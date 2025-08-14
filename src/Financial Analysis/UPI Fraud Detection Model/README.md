## UPI Fraud Detection 

## Project Overview

In today’s fast-paced, Gen-Z-driven world, UPI (Unified Payments Interface) transactions have become as common as sending a text. With this increase in digital payments, **fraudulent transactions have also seen a rise**. 

This project simulates a basic **UPI fraud detection system**, exploring how we can use machine learning to **identify suspicious transactions**.

---

## Machine Learning Models Used

We trained and compared the performance of **three supervised ML models**:

| Model              | Description                                       |
|------------------- |---------------------------------------------------|
| Decision Tree      | Easy-to-interpret tree model.                     |
| Random Forest      | Ensemble of decision trees for improved accuracy. |
| Gradient Boosting  | Sequential ensemble that corrects previous errors.|

---

## Why Accuracy Is NOT Enough

In fraud detection, **accuracy can be misleading**, especially if fraud cases are rare.

Let’s say:
- 98% of transactions are legit (non-fraud).
- A dumb model that says "everything is legit" will be 98% accurate.

But it **won’t detect any frauds**. That's why we consider:

| Metric      | Importance                                                                |
|-------------|---------------------------------------------------------------------------|
| **Recall**  | Most important! How many actual frauds we correctly identified.           |
| **Precision**| Of all the frauds we predicted, how many were actually fraud.            |
| **F1 Score**| Balance between precision and recall.                                     |

---

## Model Evaluation Results

| Model      | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|------------|----------|-----------|--------|----------|---------|
| Decision Tree (dt_model) | 0.9077   | 0.7879    | 0.8387 | 0.8125   | 0.8840  |
| Random Forest (rf_model) | 0.9462   | 0.8750    | 0.9032 | 0.8889   | 0.9314  |
| Gradient Boosting (gb_model) | 0.9385   | 0.8286    | 0.9355 | 0.8788   | 0.9374  |

> 🏆 **Gradient Boosting** performed the best overall — especially in recall, which is key for fraud detection.

---
## ⚙️ How to Run the Project

### 1. Clone the repo
```bash
git clone https://github.com/your-username/upi-fraud-detection.git
cd upi-fraud-detection
```
---

### 2. Install dependencies
```bash
pip install -r requirements.txt
```
---

### 3. Run the notebook
```bash
jupyter notebook notebook.ipynb
```
---
## Files

| File | Purpose |
|------|---------|
| `model_rf.pkl` | Saved Random Forest model |
| `model_gb.pkl` | Saved Gradient Boosting model |
| `model_dt.pkl` | Saved Decision Tree model |
| `notebook.ipynb` | Training & evaluation notebook |
| `requirements.txt` | List of required Python packages |

---

## Contact Info

**Shreya Ramesh**-shreya.ramesh22@gmail.com
