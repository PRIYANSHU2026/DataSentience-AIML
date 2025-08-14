from heuristics import score_completeness, score_politeness, score_relevance

def score_response(text, response):
    comp = score_completeness(response)
    poli = score_politeness(response)
    rel = score_relevance(text, response)

    print("🔍 Heuristic Scores:")
    print(f"Completeness: {comp:.2f}")
    print(f"Politeness:   {poli:.2f}")
    print(f"Relevance:    {rel:.2f}")

    overall = 0.5 * comp + 0.2 * poli + 0.3 * rel
    print(f"⭐ Overall Quality Score: {overall:.2f}")

if __name__ == "__main__":
    sample_text = input("Enter input text: ")
    sample_response = input("Enter model response: ")
    score_response(sample_text, sample_response)