import streamlit as st
from dandelion import DataTXT
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import json
import re

st.set_page_config(page_title="SEO Entity & Sentiment Analyzer", layout="wide")
st.title("🔍 SEO Entity & Sentiment Analyzer (Dandelion + TextBlob)")

# Input fields
api_token = st.text_input("🔑 Enter your Dandelion API Token", type="password")
user_text = st.text_area("✍️ Enter Your Content", height=200)
ref_text = st.text_area("📄 (Optional) Enter Competitor Content", height=200)
keyword_input = st.text_input("🔑 Enter comma-separated keywords for Keyword-to-Entity Matching")
analyze_btn = st.button("🚀 Analyze")

def split_sections(text):
    # Split by double newlines as sections
    return [sec.strip() for sec in re.split(r'\n\s*\n', text) if sec.strip()]

def extract_entities(datatxt, text):
    try:
        result = datatxt.nex(text, include="types,uri,confidence")
        entities = [
            {
                "label": ann.label,
                "type": ann.types[0].split("/")[-1] if ann.types else "Thing",
                "uri": ann.uri if hasattr(ann, "uri") and ann.uri else None,
                "confidence": ann.confidence if hasattr(ann, "confidence") else None
            }
            for ann in result.annotations
        ]
        # Deduplicate entities by label (keep highest confidence)
        unique = {}
        for ent in entities:
            key = ent["label"].lower()
            if key not in unique or (ent["confidence"] and ent["confidence"] > unique[key]["confidence"]):
                unique[key] = ent
        return list(unique.values())
    except Exception as e:
        st.error(f"Entity extraction error: {e}")
        return []

def sentiment_score(text):
    try:
        blob = TextBlob(text)
        return blob.sentiment.polarity
    except:
        return 0

def visualize_bar_chart(labels, values, title, xlabel, ylabel):
    fig, ax = plt.subplots()
    ax.bar(labels, values, color='skyblue')
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig)

if api_token and user_text.strip() and analyze_btn:
    datatxt = DataTXT(token=api_token, min_confidence=0.6)

    # Section-wise analysis
    st.header("📑 Section-wise Entity & Sentiment Analysis")
    sections = split_sections(user_text)
    section_entities = []
    section_sentiments = []
    entity_counts = []
    for i, sec in enumerate(sections, 1):
        ents = extract_entities(datatxt, sec)
        section_entities.append(ents)
        sentiment = sentiment_score(sec)
        section_sentiments.append(sentiment)
        entity_counts.append(len(ents))

        st.subheader(f"Section {i}")
        st.markdown(f"**Text:** {sec[:200]}{'...' if len(sec) > 200 else ''}")
        st.markdown(f"**Sentiment Polarity:** {sentiment:.3f}")
        if ents:
            st.markdown("**Entities:**")
            for ent in ents:
                if ent["uri"]:
                    st.markdown(f"- **[{ent['label']}]({ent['uri']})** ({ent['type']}) | Confidence: {ent['confidence']:.2f}")
                else:
                    st.markdown(f"- **{ent['label']}** ({ent['type']}) | Confidence: {ent['confidence']:.2f}")
        else:
            st.info("No entities found in this section.")

    # Visualize bar charts
    st.header("📊 Visualizations")
    visualize_bar_chart(
        labels=[f"Section {i}" for i in range(1, len(sections)+1)],
        values=entity_counts,
        title="Entity Count per Section",
        xlabel="Section",
        ylabel="Entity Count"
    )
    visualize_bar_chart(
        labels=[f"Section {i}" for i in range(1, len(sections)+1)],
        values=section_sentiments,
        title="Sentiment Polarity per Section",
        xlabel="Section",
        ylabel="Sentiment Polarity"
    )

    # Keyword-to-Entity matching
    if keyword_input.strip():
        st.header("🔑 Keyword-to-Entity Matching")
        keywords = [kw.strip().lower() for kw in keyword_input.split(",") if kw.strip()]
        all_entities = [ent for ents in section_entities for ent in ents]
        entity_labels = {ent["label"].lower(): ent for ent in all_entities}
        matched = []
        unmatched = []

        for kw in keywords:
            if kw in entity_labels:
                ent = entity_labels[kw]
                matched.append(ent)
            else:
                unmatched.append(kw)

        if matched:
            st.success(f"✅ Matched Keywords to Entities ({len(matched)}):")
            for ent in matched:
                if ent["uri"]:
                    st.markdown(f"- **[{ent['label']}]({ent['uri']})** ({ent['type']}) | Confidence: {ent['confidence']:.2f}")
                else:
                    st.markdown(f"- **{ent['label']}** ({ent['type']}) | Confidence: {ent['confidence']:.2f}")
        if unmatched:
            st.warning(f"❌ Keywords with no matching entity ({len(unmatched)}): {', '.join(unmatched)}")

    # Competitor comparison if provided
    if ref_text.strip():
        st.header("📎 Competitor Comparison")
        comp_entities = extract_entities(datatxt, ref_text)
        user_entity_labels = set([ent["label"].lower() for ent in [ent for ents in section_entities for ent in ents]])
        comp_entity_labels = set([ent["label"].lower() for ent in comp_entities])

        missing = comp_entity_labels - user_entity_labels
        extra = user_entity_labels - comp_entity_labels
        common = user_entity_labels & comp_entity_labels

        coverage_score = len(common) / len(comp_entity_labels.union(user_entity_labels)) if comp_entity_labels.union(user_entity_labels) else 0

        st.markdown(f"**Coverage Depth Score:** {coverage_score:.2%}")
        st.progress(coverage_score)

        if missing:
            st.warning("📌 Missing Entities (in your content but found in competitor):")
            for ent in comp_entities:
                if ent["label"].lower() in missing:
                    if ent["uri"]:
                        st.markdown(f"- ❗ **[{ent['label']}]({ent['uri']})** ({ent['type']})")
                    else:
                        st.markdown(f"- ❗ **{ent['label']}** ({ent['type']})")

        if extra:
            st.info("🌿 Extra Entities in Your Content (not in competitor):")
            for ents in section_entities:
                for ent in ents:
                    if ent["label"].lower() in extra:
                        if ent["uri"]:
                            st.markdown(f"- ✅ **[{ent['label']}]({ent['uri']})** ({ent['type']})")
                        else:
                            st.markdown(f"- ✅ **{ent['label']}** ({ent['type']})")

        if common:
            st.success("🎯 Shared Entities:")
            for label in sorted(common):
                st.markdown(f"- 🔁 {label}")

    # JSON-LD schema markup generation for all entities found
    all_unique_entities = {}
    for ents in section_entities:
        for ent in ents:
            key = ent["label"].lower()
            if key not in all_unique_entities or (ent["confidence"] and ent["confidence"] > all_unique_entities[key]["confidence"]):
                all_unique_entities[key] = ent

    if all_unique_entities:
        st.header("🧾 Auto-Generated Entity Schema Markup (JSON-LD)")
        schema = {
            "@context": "https://schema.org",
            "@type": "Article",
            "mainEntity": [
                {
                    "@type": ent["type"],
                    "name": ent["label"],
                    **({"sameAs": ent["uri"]} if ent["uri"] else {})
                }
                for ent in all_unique_entities.values()
            ]
        }
        schema_str = json.dumps(schema, indent=2)
        st.code(schema_str, language="json")
        st.download_button("⬇️ Download Schema", schema_str, file_name="entity_schema.json", mime="application/json")

else:
    st.info("🔐 Enter your API key and content, then click Analyze.")
