# app.py
import streamlit as st
from dandelion import DataTXT
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd
import json
import re

# Set page config
st.set_page_config(page_title="SEO Entity & Sentiment Analyzer", layout="wide")
st.title("🔍 SEO Entity & Sentiment Analyzer (w/ Schema & Visuals)")

# Input fields
api_token = st.text_input("🔑 Enter your Dandelion API Token", type="password")
user_text = st.text_area("✍️ Enter Your Content", height=200)
ref_text = st.text_area("📄 (Optional) Enter Competitor Content", height=200)
target_keywords = st.text_input("🎯 (Optional) Target Keywords (comma-separated)")
analyze_btn = st.button("🚀 Analyze")

# Helper: clean and split into sections
def split_sections(text):
    return [s.strip() for s in re.split(r'\n{2,}', text) if s.strip()]

if api_token and user_text.strip() and analyze_btn:
    datatxt = DataTXT(token=api_token, min_confidence=0.6)
    target_kw_set = set([kw.strip().lower() for kw in target_keywords.split(",") if kw.strip()])

    # SECTION-WISE ANALYSIS
    st.header("📚 Section-wise Analysis")
    sections = split_sections(user_text)
    sec_results = []

    for i, section in enumerate(sections):
        try:
            blob = TextBlob(section)
            sent_score = blob.sentiment.polarity
            sent_label = "Positive 😊" if sent_score > 0 else "Negative 😠" if sent_score < 0 else "Neutral 😐"
            nex_result = datatxt.nex(section, include="types,uri,confidence")

            entities = [
                {
                    "label": ann.label,
                    "type": ann.types[0].split("/")[-1] if ann.types else "Thing",
                    "uri": getattr(ann, "uri", None),
                    "confidence": ann.confidence
                }
                for ann in nex_result.annotations
            ]

            sec_results.append({
                "section": section,
                "sentiment": sent_label,
                "score": sent_score,
                "entities": entities
            })

        except Exception as e:
            st.error(f"Error processing section {i+1}: {e}")

    # DISPLAY PER SECTION
    for i, res in enumerate(sec_results):
        with st.expander(f"Section {i+1} - Sentiment: {res['sentiment']} ({res['score']:.2f})"):
            for ent in res["entities"]:
                match_kw = "✅" if ent["label"].lower() in target_kw_set else ""
                st.markdown(
                    f"- **{ent['label']}** ({ent['type']}, Confidence: {ent['confidence']:.2f}) {match_kw}"
                )

    # ALL ENTITIES COMBINED
    all_entities = [ent for sec in sec_results for ent in sec["entities"]]

    if all_entities:
        st.header("📊 Visualizations & Schema")

        # WORD CLOUD
        wordcloud = WordCloud(width=800, height=300, background_color="white")
        wordcloud.generate(" ".join([ent["label"] for ent in all_entities]))
        st.subheader("☁️ Word Cloud")
        st.image(wordcloud.to_array(), use_column_width=True)

        # PIE CHART: Entity Types
        type_df = pd.DataFrame(all_entities)
        type_counts = type_df["type"].value_counts()
        st.subheader("🧩 Entity Type Distribution")
        st.bar_chart(type_counts)

        # SENTIMENT BAR PER SECTION
        st.subheader("📈 Sentiment per Section")
        sent_df = pd.DataFrame({
            "Section": [f"{i+1}" for i in range(len(sec_results))],
            "Sentiment": [sec["score"] for sec in sec_results]
        })
        st.bar_chart(sent_df.set_index("Section"))

        # SCHEMA
        st.subheader("🧾 Entity Schema Markup (JSON-LD)")
        schema = {
            "@context": "https://schema.org",
            "@type": "Article",
            "mainEntity": [
                {
                    "@type": ent["type"],
                    "name": ent["label"],
                    **({"sameAs": ent["uri"]} if ent["uri"] else {})
                }
                for ent in all_entities
            ]
        }
        schema_str = json.dumps(schema, indent=2)
        st.code(schema_str, language="json")
        st.download_button("⬇️ Download Schema", schema_str, file_name="entity_schema.json", mime="application/json")

        # OPTIONAL: LOCAL BUSINESS GENERATOR
        with st.expander("🏢 LocalBusiness Schema Generator"):
            biz_name = st.text_input("Business Name")
            url = st.text_input("Business Website")
            phone = st.text_input("Phone Number")
            address = st.text_area("Address")

            if biz_name and url:
                lb_schema = {
                    "@context": "https://schema.org",
                    "@type": "LocalBusiness",
                    "name": biz_name,
                    "url": url,
                    "telephone": phone,
                    "address": address,
                    "mainEntity": schema["mainEntity"]
                }
                lb_schema_str = json.dumps(lb_schema, indent=2)
                st.code(lb_schema_str, language="json")
                st.download_button("⬇️ Download LocalBusiness Schema", lb_schema_str, file_name="localbusiness_schema.json", mime="application/json")
else:
    st.info("🔐 Enter your API key and content, then click Analyze.")
