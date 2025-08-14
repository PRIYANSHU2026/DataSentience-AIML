import re
import pandas as pd
def score_completeness(response):
    return min(len(response.split()) / 20, 1.0)  # Cap at 1.0

def score_politeness(response):
    polite_words = ['please', 'thank you', 'thanks', 'you’re welcome', 'glad to help']
    score = sum(1 for word in polite_words if word in response.lower())
    return min(score / 2, 1.0)

def score_relevance(text, response):
    text_set = set(re.findall(r'\w+', text.lower()))
    response_set = set(re.findall(r'\w+', response.lower()))
    intersection = text_set & response_set
    return len(intersection) / len(text_set) if text_set else 0.0

def get_heuristic_scores(row):
    if pd.isna(row['text']) or pd.isna(row['response']):
        return 0.0, 0.0, 0.0
    return (
        score_completeness(row['response']),
        score_politeness(row['response']),
        score_relevance(row['text'], row['response'])
    )
