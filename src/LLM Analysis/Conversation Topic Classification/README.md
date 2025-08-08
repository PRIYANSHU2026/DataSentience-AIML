# 🧠 Conversation Topic Classification

This project classifies text prompts (e.g., questions or commands) into predefined categories like:
- Translation Request
- Coding Query
- Personal Question
- Career Advice
- General Knowledge
- Technical Support
- ...and more

[!ui](assets/image.png)
It enables downstream applications to understand user intent and organize or route conversations accordingly.

---

## 🚀 How It Works

1. **Labeling** – Raw prompts are heuristically labeled using keyword rules in `preprocess.py`.
2. **Training** – A logistic regression model is trained on TF-IDF features using scikit-learn (`train.py`).
3. **Inference** – New prompts are classified in real-time using the trained model (`predict.py`).

---

## 🛠 Technologies Used

- Python
- scikit-learn
- pandas
- TfidfVectorizer
- LogisticRegression
- joblib

---

## 📈 Model Performance

The model achieves strong precision and recall across most classes. You can evaluate on the full dataset using:

```bash
python evaluate.py
```

---

## 🧪 Quick Inference Example

```bash
python predict.py
```

Input:
```
Prompt: How do I debug this Python error?
```

Output:
```
Predicted Topic: Coding Query
```

---

## ✅ Use Cases

- Chatbot message routing
- Query categorization for customer support
- Prompt filtering for LLMs
- Analytics dashboards
