# candidate_offer_predictor/train.py
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

# Load data
df = pd.read_csv("data/dataset.csv")

# Clean and map target
join_map = {'yes': 1, 'joined': 1, 'no': 0, 'not joined': 0}
df['Whether joined'] = df['Whether joined the company or not\n'].str.strip().str.lower()
df['joined_label'] = df['Whether joined'].map(join_map)
df = df.dropna(subset=['joined_label'])

# Clean numeric column: 'Age' may contain '32+' like strings
# Convert to int, replace '32+' with 32, etc.
df['Age'] = df['Age'].astype(str).str.extract(r'(\d+)').astype(float)
df['Experienced candidate - (Experience in months)'] = df['Experienced candidate - (Experience in months)'].astype(str).str.extract(r'(\d+)').astype(float)
df['Last Fixed CTC (lakhs) '] = df['Last Fixed CTC (lakhs) '].astype(str).str.extract(r'(\d+\.?\d*)').astype(float)

# Select features
num_cols = [
    'Confidence Score', 'Structured Thinking Score', 'Regional Fluency Score',
    'Total Score', 'Age', 'Experienced candidate - (Experience in months)',
    'Last Fixed CTC (lakhs) '
]
cat_cols = [
    'Gender', 'Marital status', 'Currently Employed',
    'Role acceptance', 'Candidate is willing to relocate',
    'Mode of interview given by candidate?'
]

# Encode categorical
df_encoded = df[num_cols + cat_cols + ['joined_label']].copy()
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
    label_encoders[col] = le

# Train-test split
X = df_encoded[num_cols + cat_cols]
y = df_encoded['joined_label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save model and encoders
joblib.dump(model, 'model/model.pkl')
joblib.dump(label_encoders, 'model/label_encoders.pkl')