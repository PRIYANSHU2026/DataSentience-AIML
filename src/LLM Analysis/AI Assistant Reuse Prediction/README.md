# AI Assistant Reuse Prediction

## 📌 Project Overview
This project aims to predict whether a student will **use an AI assistant again** in the future based on details of their current usage session.  
Using session metadata like student level, discipline, session duration, prompts used, and satisfaction ratings, our model learns patterns to make binary predictions:
- **1 (Yes)** → The student is likely to use the AI assistant again.
- **0 (No)** → The student is unlikely to reuse the AI assistant.
[!ui](assets/image.png)
Such predictions can help **AI education platforms** improve user experience, target interventions, and increase long-term engagement.

---

## 📂 Dataset Details
The dataset `ai_assistant_usage_student_life.csv` contains **10,000 session records** with the following columns:

| Column              | Type      | Description |
|---------------------|-----------|-------------|
| `SessionID`         | String    | Unique session identifier |
| `StudentLevel`      | String    | Academic level (Undergraduate, Graduate, etc.) |
| `Discipline`        | String    | Field of study (Computer Science, Psychology, Business, etc.) |
| `SessionDate`       | Date      | Date of the AI session |
| `SessionLengthMin`  | Float     | Session duration in minutes |
| `TotalPrompts`      | Integer   | Number of prompts used during the session |
| `TaskType`          | String    | Purpose of the session (Studying, Coding, Writing, etc.) |
| `AI_AssistanceLevel`| Integer   | Level of AI involvement (scale 1-5) |
| `FinalOutcome`      | String    | Session result (e.g., Assignment Completed, Research Drafted) |
| `UsedAgain`         | Boolean   | Target variable — whether the student reused AI |
| `SatisfactionRating`| Float     | Post-session satisfaction score (1.0 - 5.0) |

---

## 🎯 Problem Statement
We want to build a **binary classification model** that predicts the probability of a student **using AI again** based on their session characteristics.

---

## 🧠 Machine Learning Approach
**Target Variable:** `UsedAgain` (Boolean → Converted to 0 or 1)  
**Type:** Binary Classification  
**Algorithm Used:** Logistic Regression (baseline, interpretable)  

**Pipeline:**
1. **Data Preprocessing**
   - Date parsing & extraction of `SessionMonth` and `SessionDay`.
   - One-hot encoding of categorical variables.
   - Standard scaling for numerical features.
2. **Model Training**
   - Logistic Regression with `max_iter=500`.
3. **Evaluation**
   - Accuracy
   - Classification report (Precision, Recall, F1-score)

---

## 📁 Project Structure
AI Assistant Reuse Prediction/
│── data/
│ └── ai_assistant_usage_student_life.csv
│── model/
│ └── used_again_predictor.pkl
│── preprocess.py
│── train.py
│── predict.py
│── README.md

---

## ⚙️ Installation & Setup
### 1️⃣ Clone Repository
```bash
git clone https://github.com/yourusername/ai-assistant-reuse-prediction.git
cd ai-assistant-reuse-prediction
python train.py
python predict.py
