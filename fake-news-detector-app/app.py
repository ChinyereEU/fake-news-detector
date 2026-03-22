import streamlit as st
from src.model import load_model, predict

st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🔍",
    layout="centered"
)

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500;600&display=swap');

  html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    font-size: 16px;
  }

  .header {
    padding: 2.5rem 0 1.5rem;
    border-bottom: 0.5px solid rgba(44,44,42,0.18);
    margin-bottom: 2.5rem;
  }
  .header-title {
    font-family: 'DM Serif Display', serif;
    font-size: 3rem;
    font-weight: 400;
    color: #1A1A18;
    margin-bottom: 0.4rem;
    line-height: 1.1;
  }
  .header-title em {
    font-style: italic;
    color: #1D9E75;
  }
  .header-sub {
    font-family: 'DM Mono', monospace;
    font-size: 12px;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #1D9E75;
    font-weight: 500;
    margin-bottom: 1rem;
  }
  .header-meta {
    font-size: 14px;
    color: #444441;
    font-weight: 400;
    line-height: 1.6;
  }
  .header-links {
    display: flex;
    gap: 0.75rem;
    margin-top: 1rem;
    flex-wrap: wrap;
  }
  .header-link {
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #0F6E56;
    text-decoration: none;
    padding: 0.4rem 1rem;
    border: 0.5px solid #9FE1CB;
    border-radius: 2px;
  }
  .header-link:hover {
    background: #E1F5EE;
  }

  .stats-row {
    display: flex;
    gap: 0.75rem;
    margin-bottom: 1.25rem;
    flex-wrap: wrap;
  }
  .stat-pill {
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: #444441;
    font-weight: 500;
    background: #F4F3EF;
    padding: 0.4rem 0.85rem;
    border-radius: 2px;
  }
  .stat-pill span {
    color: #0F6E56;
    font-weight: 600;
  }

  .result-card {
    border-radius: 4px;
    padding: 1.5rem 1.75rem;
    margin: 1.5rem 0;
  }
  .result-card.real {
    background: #E1F5EE;
    border-left: 4px solid #0F6E56;
  }
  .result-card.fake {
    background: #FAECE7;
    border-left: 4px solid #D85A30;
  }
  .result-label {
    font-family: 'DM Serif Display', serif;
    font-size: 1.8rem;
    font-weight: 400;
    margin-bottom: 0.5rem;
    line-height: 1.2;
  }
  .result-label.real { color: #085041; }
  .result-label.fake { color: #993C1D; }
  .result-conf {
    font-family: 'DM Mono', monospace;
    font-size: 13px;
    font-weight: 500;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-bottom: 0.85rem;
  }
  .result-conf.real { color: #0F6E56; }
  .result-conf.fake { color: #D85A30; }

  .conf-bar-bg {
    background: rgba(255,255,255,0.6);
    border-radius: 2px;
    height: 8px;
    width: 100%;
    margin-bottom: 0.5rem;
  }
  .conf-bar-fill {
    height: 8px;
    border-radius: 2px;
  }
  .conf-bar-fill.real { background: #0F6E56; }
  .conf-bar-fill.fake { background: #D85A30; }

  .words-label {
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #444441;
    margin: 1.25rem 0 0.6rem;
  }
  .word-pills {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
  }
  .word-pill {
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    font-weight: 500;
    color: #0F6E56;
    background: #E1F5EE;
    border: 0.5px solid #9FE1CB;
    padding: 0.3rem 0.75rem;
    border-radius: 2px;
  }

  .footer {
    margin-top: 3rem;
    padding-top: 1.5rem;
    border-top: 0.5px solid rgba(44,44,42,0.12);
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #5F5E5A;
    text-align: center;
  }
  .footer a {
    color: #1D9E75;
    text-decoration: none;
  }

  div[data-testid="stTextArea"] textarea {
    font-family: 'DM Sans', sans-serif;
    font-size: 15px;
    font-weight: 400;
    line-height: 1.7;
    border-radius: 4px;
    border: 0.5px solid rgba(44,44,42,0.25);
    background: #FAFAF8;
    color: #1A1A18;
  }
  div[data-testid="stTextArea"] label {
    font-family: 'DM Sans', sans-serif;
    font-size: 14px;
    font-weight: 500;
    color: #1A1A18;
  }
  div[data-testid="stButton"] button {
    font-family: 'DM Sans', sans-serif;
    font-size: 14px;
    font-weight: 600;
    letter-spacing: 0.03em;
    background: #0F6E56;
    color: white;
    border: none;
    border-radius: 2px;
    padding: 0.75rem 1.5rem;
    width: 100%;
  }
  div[data-testid="stButton"] button:hover {
    background: #085041;
  }
</style>
""", unsafe_allow_html=True)

# ── Header ──
st.markdown("""
<div class="header">
  <div class="header-sub">ML · NLP · Classification</div>
  <div class="header-title">Fake News <em>Detector</em></div>
  <div class="header-meta">
    Random Forest + TF-IDF · 98.33% accuracy · Accenture × Break Through Tech AI Studio 2025
  </div>
  <div class="header-links">
    <a class="header-link" href="https://chinyereeu.github.io" target="_blank">Portfolio</a>
    <a class="header-link" href="https://github.com/ChinyereEU/fake-news-detector" target="_blank">GitHub</a>
    <a class="header-link" href="https://www.linkedin.com/in/chinyere-ugwuanyi" target="_blank">LinkedIn</a>
  </div>
</div>
""", unsafe_allow_html=True)

model, vectorizer = load_model()

article = st.text_area(
    "Paste article text here",
    height=250,
    placeholder="Paste a news article here...",
    label_visibility="visible"
)

# ── Article stats ──
if article.strip():
    word_count = len(article.split())
    char_count = len(article)
    sentence_count = article.count('.') + article.count('!') + article.count('?')
    st.markdown(f"""
    <div class="stats-row">
      <div class="stat-pill"><span>{word_count}</span> words</div>
      <div class="stat-pill"><span>{char_count}</span> characters</div>
      <div class="stat-pill"><span>{sentence_count}</span> sentences</div>
    </div>
    """, unsafe_allow_html=True)

if st.button("Analyze", use_container_width=True):
    if not article.strip():
        st.warning("Please paste some article text first.")
    elif len(article.split()) < 20:
        st.warning("Article is too short for reliable classification. Please paste more text.")
    else:
        with st.spinner("Analyzing..."):
            label, confidence, top_words = predict(article, model, vectorizer)

        conf_pct = int(confidence * 100)
        bar_width = conf_pct

        if label == 1:
            st.markdown(f"""
            <div class="result-card real">
              <div class="result-label real">✅ Real News</div>
              <div class="result-conf real">Confidence: {conf_pct}%</div>
              <div class="conf-bar-bg">
                <div class="conf-bar-fill real" style="width:{bar_width}%"></div>
              </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-card fake">
              <div class="result-label fake">❌ Fake News</div>
              <div class="result-conf fake">Confidence: {conf_pct}%</div>
              <div class="conf-bar-bg">
                <div class="conf-bar-fill fake" style="width:{bar_width}%"></div>
              </div>
            </div>
            """, unsafe_allow_html=True)

        pills = " ".join([f'<span class="word-pill">{w}</span>' for w, _ in top_words])
        st.markdown(f"""
        <div class="words-label">Top contributing words</div>
        <div class="word-pills">{pills}</div>
        """, unsafe_allow_html=True)

        with st.expander("About this prediction"):
            st.write(f"""
                The model analyzed the linguistic patterns in your article and compared
                them against 45,000 training examples. The top words shown above were
                the strongest signals in your text that influenced this classification.
                Model accuracy on held-out test data: **98.33%**
            """)

st.markdown("""
<div class="footer">
  Chinyere E. Ugwuanyi &nbsp;·&nbsp;
  <a href="https://chinyereeu.github.io">Portfolio</a> &nbsp;·&nbsp;
  <a href="https://github.com/ChinyereEU/fake-news-detector">GitHub</a>
</div>
""", unsafe_allow_html=True)