import re
import pandas as pd

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"reuters", "", text)
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def load_and_prepare(true_path, fake_path):
    true_df = pd.read_csv(true_path)
    fake_df = pd.read_csv(fake_path)

    true_df["label"] = 1
    fake_df["label"] = 0

    df = pd.concat([true_df, fake_df], ignore_index=True)
    df = df.drop_duplicates(subset=["text"])
    df = df[df["text"].str.len() >= 10]
    df["clean_text"] = df["text"].apply(clean_text)

    return df
