import pandas as pd
import joblib
from sklearn.metrics import classification_report

df = pd.read_csv("data/labeled_topic_data.csv")
X = df['text']
y = df['topic']

model = joblib.load("model/topic_classifier.pkl")
y_pred = model.predict(X)

print("📊 Full Evaluation Report:")
print(classification_report(y, y_pred))
