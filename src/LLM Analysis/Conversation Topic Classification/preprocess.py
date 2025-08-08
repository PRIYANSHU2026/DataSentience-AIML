import pandas as pd

def label_topic(text):
    text = str(text).lower()

    if "translate" in text or "translation" in text:
        return "Translation Request"
    elif "how do i" in text or "what is" in text or "can you explain" in text:
        return "Educational Inquiry"
    elif "how are you" in text or "your name" in text or "where are you from" in text:
        return "Personal Question"
    elif any(word in text for word in ["python", "code", "function", "loop", "error", "debug"]):
        return "Coding Query"
    elif any(word in text for word in ["wifi", "app", "install", "device", "crash", "support"]):
        return "Technical Support"
    else:
        return "Miscellaneous"

def preprocess_and_label(file_path):
    df = pd.read_csv(file_path)
    df['topic'] = df['text'].apply(label_topic)
    df = df.dropna(subset=['text', 'topic'])
    df.to_csv("data/labeled_topic_data.csv", index=False)
    return df

if __name__ == "__main__":
    df = preprocess_and_label("data/LLM__data.csv")
    print(df[['text', 'topic']].head())
