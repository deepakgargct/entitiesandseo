import streamlit as st
from dandelion import DataTXT
from textblob import TextBlob
import json

st.set_page_config(page_title="SEO Entity & Sentiment Analyzer", layout="wide")
st.title("🔍 SEO Entity & Sentiment Analyzer (Dandelion + TextBlob)")

# Input fields
api_token = st.text_input("🔑 Enter your Dandelion API Token", type="password")
user_text = st.text_area("✍️ Enter Your Content", height=200)
ref_text = st.text_area("📄 (Optional) Enter Competitor Content", height=200)
analyze_btn = st.button("🚀 Analyze")

# Industry-specific entity suggestions (simplified and extendable)
industry_keywords = {
    "photography": [
        "Ghost Mannequin Photography", "Flat Lay Product Photography", "Editorial Photography",
        "Retouching", "Casting", "Studio Consulting"
    ],
    "web3": ["Blockchain", "DAO", "Smart Contract", "Tokenomics", "DeFi", "NFT"]
}

def suggest_entities(user_entities, text):
    suggested = []
    for industry, terms in industry_keywords.items():
        for term in terms:
            if term.lower() not in [e.lower() for e in user_entities] and term.lower() in text.lower():
                suggested.append(term)
    return suggested

if api_token and user_text.strip() and analyze_btn:
    datatxt = DataTXT(token=api_token, min_confidence=0.6)

    # Entity extraction
    st.header("🧠 Entity Extraction")
    try:
        nex_result = datatxt.nex(user_text, include="types,uri")
        seen = set()
        entity_data = []
        for ann in nex_result.annotations:
            label = ann.label.strip()
            if label.lower() not in seen:
                seen.add(label.lower())
                entity_data.append({
                    "label": label,
                    "type": ann.types[0].split("/")[-1] if ann.types else "Thing",
                    "uri": ann.uri if hasattr(ann, "uri") and ann.uri else None
                })

        if entity_data:
            st.success(f"✅ Found {len(entity_data)} Unique Entities:")
            for ent in entity_data:
                if ent["uri"]:
                    st.markdown(f"- **[{ent['label']}]({ent['uri']})** ({ent['type']})")
                else:
                    st.markdown(f"- **{ent['label']}** ({ent['type']})")
        else:
            st.warning("No entities found.")
    except Exception as e:
        st.error(f"Entity extraction error: {e}")
        entity_data = []

    # Suggest additional entities
    st.header("🧩 Suggested Industry Entities")
    suggestions = suggest_entities([e["label"] for e in entity_data], user_text)
    if suggestions:
        for term in suggestions:
            st.markdown(f"- 📌 {term}")
    else:
        st.info("No additional relevant entities found.")

    # Sentiment analysis
    st.header("💬 Sentiment Analysis")
    try:
        blob = TextBlob(user_text)
        polarity = blob.sentiment.polarity
        sentiment_label = "Positive 😊" if polarity > 0 else "Negative 😠" if polarity < 0 else "Neutral 😐"
        st.markdown(f"**Sentiment Score:** `{polarity:.3f}`")
        st.markdown(f"**Interpretation:** {sentiment_label}")
    except Exception as e:
        st.error(f"Sentiment analysis error: {e}")

    # Competitor comparison
    if ref_text.strip():
        st.header("📎 Competitor Comparison")
        try:
            comp_result = datatxt.nex(ref_text, include="types,uri")
            user_labels = set([e["label"].lower() for e in entity_data])
            comp_entities = [{
                "label": ann.label,
                "type": ann.types[0].split("/")[-1] if ann.types else "Thing",
                "uri": ann.uri if hasattr(ann, "uri") and ann.uri else None
            } for ann in comp_result.annotations]
            comp_labels = set([e["label"].lower() for e in comp_entities])

            missing = comp_labels - user_labels
            extra = user_labels - comp_labels
            common = user_labels & comp_labels
            coverage_score = len(common) / len(comp_labels.union(user_labels)) if comp_labels.union(user_labels) else 0

            st.markdown(f"**Coverage Depth Score:** `{coverage_score:.2%}`")
            st.progress(coverage_score)

            if missing:
                st.warning("📌 Missing Entities:")
                for ent in comp_entities:
                    if ent["label"].lower() in missing:
                        st.markdown(f"- ❗ {ent['label']}")
            if extra:
                st.info("🌿 Extra Entities in Your Content:")
                for ent in entity_data:
                    if ent["label"].lower() in extra:
                        st.markdown(f"- ✅ {ent['label']}")
            if common:
                st.success("🎯 Shared Entities:")
                for label in sorted(common):
                    st.markdown(f"- 🔁 {label}")
        except Exception as e:
            st.error(f"Competitor analysis error: {e}")

    # JSON-LD Entity Schema Markup
    if entity_data:
        st.header("🧾 Entity Schema Markup Generator")
        biz_name = st.text_input("🏷️ Business Name")
        biz_desc = st.text_area("📝 Business Description")
        biz_url = st.text_input("🔗 Website URL")
        biz_logo = st.text_input("🖼️ Logo URL")
        biz_image = st.text_input("📷 Image URL")
        biz_keywords = st.text_area("🔑 Keywords (comma-separated)")

        if biz_name and biz_desc and biz_url:
            schema = {
                "@context": "https://schema.org",
                "@type": "LocalBusiness",
                "name": biz_name,
                "description": biz_desc,
                "url": biz_url,
                "logo": biz_logo if biz_logo else None,
                "image": [biz_image] if biz_image else None,
                "keywords": [k.strip() for k in biz_keywords.split(",")] if biz_keywords else [],
                "mainEntity": [
                    {
                        "@type": ent["type"],
                        "name": ent["label"],
                        **({"sameAs": ent["uri"]} if ent["uri"] else {})
                    }
                    for ent in entity_data
                ]
            }
            # Clean empty fields
            schema = {k: v for k, v in schema.items() if v}
            schema_str = json.dumps(schema, indent=2)
            st.code(schema_str, language="json")
            st.download_button("⬇️ Download Schema", schema_str, file_name="entity_schema.json", mime="application/json")
        else:
            st.info("ℹ️ Fill in business name, description, and URL to generate schema.")
else:
    st.info("🔐 Enter your API key and content, then click Analyze.")
