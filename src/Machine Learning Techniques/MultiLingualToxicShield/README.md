# 🛡️ MultiLingualToxicShield

**MultiLingualToxicShield** is a multilingual **toxic/NSFW content detection** toolkit for text prompts and responses.  
It detects offensive, hateful, or otherwise unsafe language in **multiple languages** (English, Arabic, Hinglish, etc.) using either:
[!ui](assets/toxicity_by_language.png)
[!ui](assets/toxicity_pie_chart.png)
[!ui](assets/image.png)

- **HuggingFace Transformer Models** (default: `cardiffnlp/twitter-xlm-roberta-base-offensive`)  
- **Keyword-based fallback detector** (lightweight & offline-friendly)

The toolkit supports:
- **Dataset preprocessing**
- **Model training / scoring**
- **Prediction for new inputs or datasets**
- **Basic visualization of toxicity distribution**

---

## 📂 Project Structure
MultilingualToxicShield/
├── assets/ # Generated charts & plots
├── data/
│ ├── LLM__data.csv # Original dataset (input)
│ ├── processed_LLM_data.csv # Preprocessed dataset
│ ├── LLM_data_toxic.csv # Labeled dataset with toxicity scores
├── preprocess.py # Cleans & prepares dataset
├── train_model.py # Runs toxicity detection & generates charts
├── predict.py # Predicts toxicity for text or CSV input
└── requirements.txt # Python dependencies

---

