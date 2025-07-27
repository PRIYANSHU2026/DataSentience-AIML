Candidate Offer Acceptance Predictor
This project leverages machine learning to predict whether a job candidate will accept an offer based on various interview and background features. It’s part of the SSoC open-source initiative and is built using a Random Forest Classifier with structured + categorical data.

📌 Problem Statement
Organizations often face uncertainty when extending job offers to candidates. This project solves that by predicting candidate offer acceptance using:

Interview performance scores

Candidate background data (age, experience, CTC)

Subjective factors (willingness to relocate, marital status, etc.)

🧠 Tech Stack
Python 3

Pandas for data manipulation

Scikit-learn for ML pipeline

Joblib for model persistence

📁 Folder Structure
graphql
Copy
Edit
candidate_offer_predictor/
│
├── data/
│   └── dataset.csv               # Original dataset with candidate info
│
├── model/
│   ├── model.pkl                 # Trained RandomForestClassifier
│   └── label_encoders.pkl       # Fitted LabelEncoders for categorical features
│
├── train.py                     # Data preprocessing + model training
└── predict.py                   # Prediction script with example usage
🧪 Features Used
🧮 Numerical Features
Confidence Score

Structured Thinking Score

Regional Fluency Score

Total Score

Age

Experience in months

Last Fixed CTC (in lakhs)

🔤 Categorical Features
Gender

Marital Status

Currently Employed

Role Acceptance

Willing to Relocate

Mode of Interview

