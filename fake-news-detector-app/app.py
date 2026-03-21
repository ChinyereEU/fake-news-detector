import streamlit as st
from src.model import load_model, predict

st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🔍",
    layout="centered"
)

st.markdown("""
    <style>
        .main { max-width: 720px; margin: 0 auto; }
        .result-real { background: #E1F5EE; border-left: 4px solid #0F6E56;
                       padding: 1rem 1.25rem; border-radius: 4px; margin: 1rem 0; }
        .result-fake { background: #FAECE7; border-left: 4px solid #D85A30;
                       padding: 1rem 1.25rem; border-radius: 4px; margin: 1rem 0; }
        .result-label { font-size: 1.4rem; font-weight: 600; margin-bottom: 0.25rem; }
        .result-conf  { font-size: 0.9rem; color: #5F5E5A; }
        .word-pill { display: inline-block; background: #F4F3EF; color: #0F6E56;
                     font-family: monospace; font-size: 0.75rem; padding: 3px 10px;
                     border-radius: 2px; margin: 3px; }
    </style>
""", unsafe_allow_html=True)

st.title("🔍 Fake News Detector")
st.caption("Paste any news article below to classify it as Real or Fake.")
st.caption("Built with Random Forest + TF-IDF · 98.33% accuracy · Accenture × Break Through Tech AI Studio 2025")

model, vectorizer = load_model()

article = st.text_area(
    "Paste article text here",
    height=250,
    placeholder="Paste a news article here..."
)

if st.button("Analyze", use_container_width=True):
    if not article.strip():
        st.warning("Please paste some article text first.")
    elif len(article.strip()) < 50:
        st.warning("Article is too short for reliable classification. Please paste more text.")
    else:
        with st.spinner("Analyzing..."):
            label, confidence, top_words = predict(article, model, vectorizer)

        if label == 1:
            st.markdown(f"""
                <div class="result-real">
                    <div class="result-label">✅ Real News</div>
                    <div class="result-conf">Confidence: {confidence:.1%}</div>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class="result-fake">
                    <div class="result-label">❌ Fake News</div>
                    <div class="result-conf">Confidence: {confidence:.1%}</div>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("**Top contributing words:**")
        pills = " ".join([f'<span class="word-pill">{w}</span>' for w, _ in top_words])
        st.markdown(pills, unsafe_allow_html=True)

        with st.expander("About this prediction"):
            st.write(f"""
                The model analyzed the linguistic patterns in your article and compared them
                against {45000:,} training examples. The top words shown above were the
                strongest signals in your text that influenced this classification.
                Model accuracy on held-out test data: **98.33%**
            """)

st.divider()
st.caption("Chinyere E. Ugwuanyi · [Portfolio](https://chinyereeu.github.io) · [GitHub](https://github.com/ChinyereEU/fake-news-detector)")