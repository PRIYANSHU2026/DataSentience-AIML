# 🧠 Fake Job Prediction - Job Ad Detection using Random Forest

## 📌 Project Overview
This project aims to detect fraudulent job postings by analyzing job descriptions and related features. Using **Natural Language Processing (NLP)** and a **Random Forest Classifier**, it predicts whether a job posting is real or fake.

The model is trained on the **Fake Job Postings Dataset** from Kaggle, containing over 18,000 job postings, of which around 800 are fraudulent.

---

## 🚀 Features
- **Data Preprocessing & Cleaning** – Removing null values, unnecessary columns, and normalizing text.
- **Lemmatization** – Converting words to their root forms for better NLP performance.
- **Feature Extraction with TF-IDF** – Transforming job descriptions into numerical features.
- **Handling Imbalanced Data** – Using `SMOTETomek` to balance real and fake job postings.
- **Random Forest Classifier** – Achieving ~96% accuracy.
- **Data Visualizations** – Understanding patterns in fake vs. real jobs.
- **Model Saving** – Using `joblib` for easy deployment.

---

## 📂 Dataset
**Source**: [Kaggle - Real or Fake: Fake Job Posting Prediction](https://www.kaggle.com/shivamb/real-or-fake-fake-jobposting-prediction)  
**Provider**: The University of the Aegean  

### Key Columns:
- `title` – Job title  
- `location` – Job location  
- `department`, `salary_range`, `company_profile` – Additional details  
- `description` – Main job description (used heavily in prediction)  
- `fraudulent` – Target label (0 = real, 1 = fake)  

---

## 🛠 Tech Stack
- Python 🐍
- Jupyter Notebook / Google Colab 📓
- Scikit-learn
- Pandas, NumPy
- NLTK (Lemmatization, Stopword removal)
- Imbalanced-learn (SMOTETomek)
- Matplotlib & Seaborn (Visualization)
- Joblib (Model Saving)
- WordCloud (NLP Visualization)

---

## 📊 Visualizations
The notebook includes:
- Distribution of **real vs fake** job postings.
- Word clouds for **real** and **fake** job descriptions.
- Feature importance plot from the Random Forest Classifier.

---

## ⚙️ Installation & Setup

1️⃣ Fork & Clone the repository  and Go Inside the Job_Detection_Model

2️⃣ Install dependencies  

```bash
pip install -r requirements.txt
```

3️⃣ Download dataset from Kaggle and place it in the project folder.

4️⃣ Run the notebook  

```bash
jupyter notebook fake_job_prediction.ipynb
```

---

## 💾 Saving & Loading the Model
Inside the notebook:
```python
import joblib

# Save model
joblib.dump(model, "fake_job_model.pkl")

# Load model
model = joblib.load("fake_job_model.pkl")
```

---

## 📈 Future Improvements
- Convert notebook into a standalone Python script for production use.
- Build a simple frontend UI for user interaction.
- Try alternative models (XGBoost, LSTM).
- Improve visualization with interactive charts (Plotly).
- Add automated testing for reliability

---

## 📜 License
This project is licensed under the MIT License.
