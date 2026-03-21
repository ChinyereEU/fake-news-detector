import joblib
import os
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from src.preprocessing import load_and_prepare

MODEL_PATH  = "src/model_cache.joblib"
VECTOR_PATH = "src/vector_cache.joblib"

@st.cache_resource(show_spinner="Training model on first launch — this takes about 60 seconds...")
def load_model():
    if os.path.exists(MODEL_PATH) and os.path.exists(VECTOR_PATH):
        model      = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTOR_PATH)
        return model, vectorizer

    df = load_and_prepare("data/True.csv", "data/Fake.csv")

    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        sublinear_tf=True
    )

    X = vectorizer.fit_transform(df["clean_text"])
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=200,
        criterion="entropy",
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"Model accuracy: {acc:.4f}")

    joblib.dump(model,      MODEL_PATH)
    joblib.dump(vectorizer, VECTOR_PATH)

    return model, vectorizer

def predict(text, model, vectorizer):
    from src.preprocessing import clean_text
    cleaned   = clean_text(text)
    vector    = vectorizer.transform([cleaned])
    label     = model.predict(vector)[0]
    proba     = model.predict_proba(vector)[0]
    confidence = proba[label]

    feature_names = vectorizer.get_feature_names_out()
    importances   = model.feature_importances_
    indices       = vector.nonzero()[1]

    top_words = sorted(
        [(feature_names[i], importances[i]) for i in indices],
        key=lambda x: x[1],
        reverse=True
    )[:10]

    return label, confidence, top_words