"""
train_model.py

- Loads processed data from data/processed_LLM_data.csv
- Runs multilingual toxicity detection using a HuggingFace model (if available)
- Falls back to keyword-based detection if model can't be loaded
- Writes labeled CSV to data/LLM_data_toxic.csv
- Generates simple visuals in assets/
"""

import os
import math
import pandas as pd
import matplotlib.pyplot as plt

MODEL_NAME = "cardiffnlp/twitter-xlm-roberta-base-offensive"  # multilingual offensive model
PROCESSED_IN = os.path.join("data", "processed_LLM_data.csv")
LABELED_OUT = os.path.join("data", "LLM_data_toxic.csv")
ASSETS_DIR = "assets"
THRESHOLD = 0.5  # probability threshold for toxic label

# Simple multilingual offensive keywords fallback (not exhaustive)
OFFENSIVE_KEYWORDS = [
    # English
    "fuck", "shit", "bitch", "asshole", "nigger", "idiot", "stupid",
    # Arabic (transliterated/common forms)
    "الكحول", "لعنة", "حمار", "غبي", "قذر",  # be careful: these might be false positives
    # Hindi / Hinglish
    "chutiya", "bhosdike", "gandu",
]

def try_load_pipeline():
    try:
        from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
        # load pipeline (this may download model files)
        print(f"[train_model] Loading model pipeline '{MODEL_NAME}' ... (may take a while)")
        classifier = pipeline("text-classification", model=MODEL_NAME, tokenizer=MODEL_NAME, return_all_scores=True, device=-1)
        print("[train_model] Model loaded successfully.")
        return classifier
    except Exception as e:
        print("[train_model] Could not load transformer model (falling back to keyword detector).")
        print("Error:", str(e))
        return None


def keyword_toxic_detector(text: str) -> float:
    """Return a pseudo-score between 0 and 1 based on keyword presence."""
    if not isinstance(text, str) or not text.strip():
        return 0.0
    t = text.lower()
    hits = sum(1 for kw in OFFENSIVE_KEYWORDS if kw in t)
    # simple mapping: more hits -> closer to 1.0
    score = 1.0 - math.exp(-hits)
    return min(1.0, score)


def classify_texts(df: pd.DataFrame, classifier) -> pd.DataFrame:
    scores = []
    backend_used = "transformers" if classifier is not None else "keyword"

    for i, txt in enumerate(df["combined_text"].astype(str).tolist()):
        if classifier is not None:
            try:
                # classifier returns list of dicts with labels and scores
                res = classifier(txt[:1000])  # limit long texts
                # res is list of labels with scores; label names vary by model
                # We'll assume the "toxic/offensive" label has the highest score among returned labels.
                # For cardiffnlp model, labels like 'NOT_OFFENSIVE'/'OFFENSIVE' may appear.
                # We'll take max score of any label containing 'OFF'/'TOXIC'/'ABUSE' etc.
                max_toxic_score = 0.0
                for label_score_list in res:  # when return_all_scores True
                    # label_score_list is a list of dicts
                    for entry in label_score_list:
                        label = entry.get("label", "").upper()
                        sc = float(entry.get("score", 0.0))
                        if any(k in label for k in ("OFFENS", "TOXIC", "ABUS", "HATE")):
                            max_toxic_score = max(max_toxic_score, sc)
                # If model label names are not clear, also fall back to picking the probability
                # of the most-confident toxic-like label found above.
                score = max_toxic_score
            except Exception:
                score = 0.0
        else:
            score = keyword_toxic_detector(txt)
        scores.append(score)

        if (i + 1) % 200 == 0 or i == len(df) - 1:
            print(f"[train_model] Processed {i+1}/{len(df)} rows.")

    df["toxic_score"] = scores
    df["label"] = df["toxic_score"].apply(lambda s: "Toxic" if s >= THRESHOLD else "Clean")
    df["detector_backend"] = backend_used
    return df


def generate_plots(df: pd.DataFrame):
    os.makedirs(ASSETS_DIR, exist_ok=True)

    # Pie chart: Toxic vs Clean
    counts = df["label"].value_counts()
    plt.figure(figsize=(6, 6))
    counts.plot.pie(autopct="%1.1f%%", ylabel="")
    plt.title("Toxic vs Clean (combined_text)")
    pie_path = os.path.join(ASSETS_DIR, "toxicity_pie_chart.png")
    plt.savefig(pie_path, bbox_inches="tight")
    plt.close()
    print(f"[train_model] Saved pie chart to {pie_path}")

    # Bar chart: toxicity rate per language
    lang_grp = df.groupby("from_language").apply(lambda d: (d["label"] == "Toxic").mean()).sort_values(ascending=False)
    plt.figure(figsize=(8, 5))
    lang_grp.plot.bar()
    plt.ylabel("Toxicity rate (fraction)")
    plt.xlabel("Language")
    plt.title("Toxicity fraction by from_language")
    bar_path = os.path.join(ASSETS_DIR, "toxicity_by_language.png")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(bar_path)
    plt.close()
    print(f"[train_model] Saved bar chart to {bar_path}")


def main():
    print("[train_model] Starting training/detection pipeline...")
    if not os.path.exists(PROCESSED_IN):
        raise FileNotFoundError(f"{PROCESSED_IN} not found. Run preprocess.py first.")

    df = pd.read_csv(PROCESSED_IN)
    print(f"[train_model] Loaded processed data: {len(df)} rows")

    classifier = try_load_pipeline()
    df_labeled = classify_texts(df, classifier)

    os.makedirs(os.path.dirname(LABELED_OUT), exist_ok=True)
    df_labeled.to_csv(LABELED_OUT, index=False)
    print(f"[train_model] Labeled data saved to {LABELED_OUT}")

    generate_plots(df_labeled)
    print("[train_model] Done.")


if __name__ == "__main__":
    main()
