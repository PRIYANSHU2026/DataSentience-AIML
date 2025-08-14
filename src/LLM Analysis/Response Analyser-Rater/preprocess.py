import pandas as pd
from heuristics import get_heuristic_scores

def preprocess_data(file_path):
    df = pd.read_csv(file_path)
    df = df.dropna(subset=['text', 'response'])

    scores = df.apply(get_heuristic_scores, axis=1, result_type='expand')
    df[['completeness', 'politeness', 'relevance']] = scores
    return df
if __name__ == "__main__":
    # Make sure this path matches your actual folder structure
    file_path = "data/LLM__data.csv"
    
    # Run preprocessing
    processed_df = preprocess_data(file_path)
    
    # Print a few rows to verify
    print(processed_df[['text', 'response', 'completeness', 'politeness', 'relevance']].head())
    
    # Optionally, save the result
    processed_df.to_csv("data/LLM__scored_data.csv", index=False)
    print("✅ Scored data saved to data/LLM__scored_data.csv")