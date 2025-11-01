import streamlit as st
import os
from modules import data_input, preprocessing, topic_modeling, sentiment_analysis, summarization, visualization
import plotly.graph_objects as go

# --- PAGE CONFIG ---
st.set_page_config(page_title="Infosys NLP Platform", layout="wide")

# --- LOAD CSS ---
def load_css(file_name="style.css"):
    try:
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.error("❌ CSS file not found.")

load_css()

st.markdown(
    """
    <div class="hero-section">
        <h1>Transform Text into <span class="highlight">Actionable Insights</span></h1>
        <p>Upload, analyze, and visualize — uncover key themes, sentiments, and summaries from your data.</p>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("<h3 class='section-title'>📂 Data Input Module</h3>", unsafe_allow_html=True)
st.markdown("<div class='input-container'>", unsafe_allow_html=True)

text_data = data_input.handle_file_upload()
user_text_input = st.text_area("Paste your text here...", placeholder="Paste articles, reports, or any text...", height=200)
analyze_button = st.button("✨ Analyze Text", use_container_width=True)

st.markdown("</div>", unsafe_allow_html=True)

# --- ANALYSIS LOGIC ---
if analyze_button:
    raw_text = text_data or user_text_input.strip()
    if not raw_text:
        st.warning("⚠️ Please upload or paste text first.")
        st.stop()

    with st.spinner("🧹 Cleaning and preprocessing text..."):
        processed_text = preprocessing.clean_text(raw_text)

    with st.spinner("🔍 Extracting key themes..."):
        topics, topic_keywords = topic_modeling.perform_lda(processed_text)

    with st.spinner("😊 Analyzing sentiment..."):
        sentiment_label, sentiment_score = sentiment_analysis.get_sentiment(processed_text)

    with st.spinner("🧾 Summarizing text..."):
        summary_text = summarization.generate_summary(raw_text)

    with st.spinner("📊 Creating visual insights..."):
        wordcloud_fig = visualization.create_wordcloud(processed_text)
        sentiment_chart = visualization.plot_sentiment(sentiment_score)
        topic_chart = visualization.plot_topics(topic_keywords)

    st.success("✅ Analysis Complete!")

    # --- DETAILED ANALYSIS FIRST ---
    st.markdown("<h2 class='section-title'>📊 Detailed Analysis</h2>", unsafe_allow_html=True)
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Summary", "Themes", "Sentiment", "Word Cloud", "Processed Text"])

    with tab1:
        st.subheader("🧾 Text Summary")
        st.write(summary_text)

    with tab2:
        st.subheader("🎯 Key Topics and Themes")
        for i, topic in enumerate(topics):
            st.markdown(f"**Topic {i+1}:** {topic}")

    with tab3:
        st.subheader("Overall Sentiment")
        color_map = {"Positive": "#00C853", "Negative": "#D50000", "Neutral": "#FFD600"}
        color = color_map.get(sentiment_label, "#999999")

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=sentiment_score * 100,
            title={'text': f"Sentiment Strength ({sentiment_label})", 'font': {'size': 18}},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': color},
                'steps': [
                    {'range': [0, 40], 'color': "rgba(213, 0, 0, 0.4)"},
                    {'range': [40, 60], 'color': "rgba(255, 214, 0, 0.4)"},
                    {'range': [60, 100], 'color': "rgba(0, 200, 83, 0.4)"}
                ]
            }
        ))
        fig.update_layout(height=250, margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig)
        st.write("💡 *Score closer to 0 = Negative, around 0.5 = Neutral, closer to 1 = Positive.*")

    with tab4:
        st.subheader("☁️ Word Cloud of Key Terms")
        st.pyplot(wordcloud_fig)

    with tab5:
        st.subheader("🧹 Cleaned & Processed Text")
        st.text_area("Processed Text", processed_text, height=250)

    # --- QUICK INSIGHTS SECOND ---
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown("<h2 class='section-title'>✨ Quick Insights</h2>", unsafe_allow_html=True)
    cols = st.columns(3)

    with cols[0]:
        st.markdown("<div class='info-box glow'><h4>🧾 Summary</h4><p>{}</p></div>".format(summary_text), unsafe_allow_html=True)

    with cols[1]:
        st.markdown(
            f"<div class='info-box glow'><h4>🎯 Sentiment</h4><h3 style='color:{color};'>{sentiment_label}</h3><p>Confidence Score: {sentiment_score:.2f}</p></div>",
            unsafe_allow_html=True
        )

    with cols[2]:
        first_topic = topics[0] if isinstance(topics, list) and topics else "No topics found"
        st.markdown(f"<div class='info-box glow'><h4>🗂 Key Topics</h4><p>{first_topic}</p></div>", unsafe_allow_html=True)

# --- FOOTER ---
st.markdown(
    """
    <div class='footer'>
         Powered by Infosys NLP Suite | Smart Text Analytics Platform
    </div>
    """,
    unsafe_allow_html=True
)
