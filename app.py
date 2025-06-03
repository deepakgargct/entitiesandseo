import streamlit as st
from dandelion import DataTXT
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd
import json
import re

st.set_page_config(page_title="SEO Entity & Sentiment Analyzer", layout="wide")
st.title("🔍 SEO Entity & Sentiment Analyzer (Dandelion + TextBlob)")

# Input fields
api_token = st.text_input("🔑 Enter your Dandelion API Token", type="password")
user_text = st.text_area("✍️ Enter Your Content", height=200)
ref_text = st.text_area("📄 (Optional) Enter Competitor Content", height=200)
keywords = st.text_input("🔎 (Optional) Enter Target Keywords (comma-separated)")
analyze_btn = st.button("🚀 Analyze")

if api_token and user_text.strip() and analyze_btn:
    datatxt = DataTXT(token=api_token, min_confidence=0.6)

    # Split content into sections
    sections = re.split(r"\n{2,}", user_text)

    section_entities = []
    section_sentiments = []
    all_entities = []

    st.header("🧠 Section-wise Entity & Sentiment Breakdown")
    for i, sec in enumerate(sections):
        if sec.strip():
            st.subheader(f"Section {i+1}")
            try:
                result = datatxt.nex(sec, include="types,uri")
                blob = TextBlob(sec)
                polarity = blob.sentiment.polarity
                sentiment_label = "Positive 😊" if polarity > 0 else "Negative 😠" if polarity < 0 else "Neutral 😐"

                entities = [
                    {
                        "label": ann.label,
                        "type": ann.types[0].split("/")[-1] if ann.types else "Thing",
                        "uri": ann.uri if hasattr(ann, "uri") and ann.uri else None,
                        "confidence": ann.confidence
                    }
                    for ann in result.annotations
                ]

                section_entities.append(entities)
                section_sentiments.append(polarity)
                all_entities.extend(entities)

                for ent in entities:
                    uri_display = f"🔗[{ent['uri']}]({ent['uri']})" if ent["uri"] else ""
                    st.markdown(f"- **{ent['label']}** ({ent['type']}) | Confidence: `{ent['confidence']:.2f}` {uri_display}")

                st.markdown(f"**Sentiment Score:** {polarity:.3f} → {sentiment_label}")
            except Exception as e:
                st.error(f"Error processing section {i+1}: {e}")

    # Entity Keyword Matching
    if keywords.strip():
        st.header("🔍 Keyword-to-Entity Matching")
        keyword_list = [kw.strip().lower() for kw in keywords.split(",")]
        matched = [e for e in all_entities if e["label"].lower() in keyword_list]

        if matched:
            for ent in matched:
                st.markdown(f"✅ **Matched:** {ent['label']} ({ent['type']})")
        else:
            st.warning("No direct keyword matches found in extracted entities.")

    # Word cloud
    st.header("☁️ Word Cloud of Entities")
    entity_labels = [e["label"] for e in all_entities]
    if entity_labels:
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(" ".join(entity_labels))
        fig, ax = plt.subplots()
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis("off")
        st.pyplot(fig, use_container_width=True)

    # Competitor comparison
    if ref_text.strip():
        st.header("📎 Competitor Entity Comparison")
        try:
            comp_result = datatxt.nex(ref_text, include="types,uri")
            comp_entities = [
                {
                    "label": ann.label,
                    "type": ann.types[0].split("/")[-1] if ann.types else "Thing",
                    "uri": ann.uri if hasattr(ann, "uri") and ann.uri else None,
                    "confidence": ann.confidence
                }
                for ann in comp_result.annotations
            ]

            user_labels = {e["label"].lower(): e for e in all_entities}
            comp_labels = {e["label"].lower(): e for e in comp_entities}

            user_set = set(user_labels.keys())
            comp_set = set(comp_labels.keys())

            common = user_set & comp_set
            missing = comp_set - user_set
            extra = user_set - comp_set

            coverage_score = len(common) / len(comp_set.union(user_set)) if comp_set.union(user_set) else 0

            st.markdown(f"**Coverage Depth Score:** `{coverage_score:.2%}`")
            st.progress(coverage_score)

            st.subheader("📘 Entities in Your Content")
            for label in sorted(user_set):
                ent = user_labels[label]
                st.markdown(f"- ✅ **{ent['label']}** ({ent['type']}) {f'[🔗]({ent["uri"]})' if ent['uri'] else ''}")

            st.subheader("📕 Competitor Entities")
            for label in sorted(comp_set):
                ent = comp_labels[label]
                st.markdown(f"- 📌 **{ent['label']}** ({ent['type']}) {f'[🔗]({ent["uri"]})' if ent['uri'] else ''}")

            st.subheader("❗ Missing Entities from Your Content")
            for label in sorted(missing):
                ent = comp_labels[label]
                st.markdown(f"- ❌ **{ent['label']}** ({ent['type']}) {f'[🔗]({ent["uri"]})' if ent['uri'] else ''}")

        except Exception as e:
            st.error(f"Competitor analysis error: {e}")

    # Entity schema markup
    st.header("🧾 Entity Schema Markup Generator")
    if all_entities:
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

    # LocalBusiness Schema Generator link
    st.markdown("---")
    st.markdown("## 🏪 Want LocalBusiness Schema?")
    st.markdown("Use the integrated [LocalBusiness Schema Generator](#) to embed your business info with extracted entities!")
else:
    st.info("🔐 Enter your API key and content, then click Analyze.")
