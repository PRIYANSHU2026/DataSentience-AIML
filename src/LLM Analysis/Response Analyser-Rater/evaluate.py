import pandas as pd

def evaluate_scores(data_path):
    df = pd.read_csv(data_path)

    print("\n🔍 Average Heuristic Scores:")
    print(df[['completeness', 'politeness', 'relevance']].mean())

    print("\n📊 Average Scores by Model:")
    print(df.groupby('model')[['completeness', 'politeness', 'relevance']].mean())

    print("\n🌐 Average Scores by Language:")
    print(df.groupby('from_language')[['completeness', 'politeness', 'relevance']].mean())

if __name__ == "__main__":
    evaluate_scores("data/LLM__scored_data.csv")