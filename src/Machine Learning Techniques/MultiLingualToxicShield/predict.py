"""
predict.py

- Provides an easy interface for running toxicity detection on:
  1) a single text input (CLI)
  2) an input CSV (same format as processed, column 'combined_text')
- Writes outputs (if CSV) to data/predicted_<inputfilename>.csv
"""

import os
import sys
import argparse
import pandas as pd
from train_model import try_load_pipeline, keyword_toxic_detector, THRESHOLD

ASSETS_DIR = "assets"


def predict_single(text: str, classifier):
    # Return dict with score and label
    if classifier is not None:
        try:
            res = classifier(text[:1000])
            max_toxic_score = 0.0
            for label_score_list in res:
                for entry in label_score_list:
                    label = entry.get("label", "").upper()
                    sc = float(entry.get("score", 0.0))
                    if any(k in label for k in ("OFFENS", "TOXIC", "ABUS", "HATE")):
                        max_toxic_score = max(max_toxic_score, sc)
            score = max_toxic_score
        except Exception:
            score = keyword_toxic_detector(text)
    else:
        score = keyword_toxic_detector(text)

    label = "Toxic" if score >= THRESHOLD else "Clean"
    return {"score": float(score), "label": label}


def predict_csv(input_csv: str, classifier):
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"{input_csv} not found.")
    df = pd.read_csv(input_csv)
    if "combined_text" not in df.columns:
        raise ValueError("Input CSV must contain 'combined_text' column. Run preprocess.py first or create that column.")
    results = []
    for txt in df["combined_text"].astype(str).tolist():
        res = predict_single(txt, classifier)
        results.append(res)
    df_out = df.copy()
    df_out["toxic_score"] = [r["score"] for r in results]
    df_out["label"] = [r["label"] for r in results]
    out_path = os.path.join("data", f"predicted_{os.path.basename(input_csv)}")
    df_out.to_csv(out_path, index=False)
    print(f"[predict] Predictions saved to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Predict toxicity for text or CSV")
    parser.add_argument("--text", type=str, help="Single text to classify")
    parser.add_argument("--csv", type=str, help="CSV file (processed) to classify")
    parser.add_argument("--use-fallback", action="store_true", help="Force keyword fallback instead of loading HF model")
    args = parser.parse_args()

    classifier = None
    if not args.use_fallback:
        classifier = try_load_pipeline()

    if args.text:
        out = predict_single(args.text, classifier)
        print("Text:", args.text)
        print("Score:", out["score"])
        print("Label:", out["label"])
    elif args.csv:
        predict_csv(args.csv, classifier)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
