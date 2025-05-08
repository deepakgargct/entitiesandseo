
import streamlit as st
from dandelion import DataTXT
from textblob import TextBlob
import json

st.set_page_config(page_title="SEO Entity & Sentiment Analyzer", layout="wide")
st.title("🔍 SEO Entity & Sentiment Analyzer using Dandelion + TextBlob")

# Inputs
api_token = st.text_input("🔑 Enter your Dandelion API Token", type="password")
user_text = st.text_area("✍️ Enter Your Content", height=200)
ref_text = st.text_area("📄 (Optional) Enter Competitor Content", height=200)

analyze = st.button("🔍 Analyze")

if analyze and api_token and user_text.strip():
    datatxt = DataTXT(token=api_token, min_confidence=0.6)

    # Entity Extraction
    st.header("🧠 Entity Extraction")
    try:
        nex_result = datatxt.nex(user_text, include="types,uri")
        entities = [ann.label for ann in nex_result.annotations]

        if entities:
            st.success(f"✅ Found {len(entities)} Entities")
            entity_types = {}
            for ann in nex_result.annotations:
                label = ann.label
                entity_type = ann.types[0].split("/")[-1] if ann.types else "Other"
                entity_types.setdefault(entity_type, []).append(label)

            for ent_type, labels in entity_types.items():
                st.markdown(f"**{ent_type}** ({len(labels)}):")
                st.markdown(", ".join(set(labels)))
        else:
            st.warning("No entities found.")
    except Exception as e:
        st.error(f"Entity extraction error: {e}")

    # Sentiment Analysis
    st.header("💬 Sentiment Analysis (via TextBlob)")
    try:
        blob = TextBlob(user_text)
        score = blob.sentiment.polarity
        sentiment_label = (
            "Positive 😊" if score > 0 else "Negative 😠" if score < 0 else "Neutral 😐"
        )
        st.markdown(f"**Sentiment Score:** `{score:.3f}`")
        st.markdown(f"**Interpretation:** {sentiment_label}")
    except Exception as e:
        st.error(f"Sentiment analysis error: {e}")
        score = 0

    # Competitor Analysis
    if ref_text.strip():
        st.header("📎 Competitor Entity Benchmarking")
        try:
            comp_result = datatxt.nex(ref_text, include="types,uri")
            user_entity_map = {ann.label.lower(): ann for ann in nex_result.annotations}
            comp_entity_map = {ann.label.lower(): ann for ann in comp_result.annotations}

            user_labels = set(user_entity_map.keys())
            comp_labels = set(comp_entity_map.keys())

            missing = comp_labels - user_labels
            unique = user_labels - comp_labels
            common = user_labels & comp_labels

            st.subheader("📊 Coverage Benchmarking")
            st.markdown(f"- **Your Entities:** {len(user_labels)}")
            st.markdown(f"- **Competitor Entities:** {len(comp_labels)}")
            st.markdown(f"- **Overlap:** {len(common)} shared entities")
            st.markdown(f"- **Missing Entities:** {len(missing)}")
            st.markdown(f"- **Unique to You:** {len(unique)}")

            st.header("📈 SEO Recommendations")
            recommendations = []
            if missing:
                top_missing = [comp_entity_map[l].label for l in sorted(missing)[:5]]
                recommendations.append(f"Include missing entities: {', '.join(top_missing)}.")
            if unique:
                top_unique = [user_entity_map[l].label for l in sorted(unique)[:5]]
                recommendations.append(f"Highlight your unique entities: {', '.join(top_unique)}.")
            if score < 0:
                recommendations.append("Consider improving the tone to be more positive or balanced.")
            if not recommendations:
                st.success("✅ Your content is well-aligned!")
            for rec in recommendations:
                st.markdown(f"- 💡 {rec}")
        except Exception as e:
            st.error(f"Competitor entity analysis error: {e}")

    # Schema Markup
    st.header("📌 Auto-Generated Entity Schema Markup")
    schema_entities = []
    for ann in nex_result.annotations:
        ent_type = ann.types[0].split("/")[-1] if ann.types else "Thing"
        entity = {
            "@type": ent_type,
            "name": ann.label
        }
        if hasattr(ann, "uri") and ann.uri:
            entity["@id"] = ann.uri
        schema_entities.append(entity)

    json_ld = {
        "@context": "https://schema.org",
        "@graph": schema_entities
    }

    st.code(json.dumps(json_ld, indent=2), language="json")
    st.download_button("📥 Download Schema", data=json.dumps(json_ld, indent=2), file_name="entity-schema.json", mime="application/json")

else:
    st.info("🔐 Please enter your API token and content, then click 'Analyze'.")
