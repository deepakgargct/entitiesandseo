import streamlit as st
import json
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
import re
from dandelion import DataTXT

# Dandelion API setup
API_TOKEN = "YOUR_DANDELION_API_TOKEN"  # Replace with your token
nlp = DataTXT(token=API_TOKEN)

st.set_page_config(page_title="SEO Entity & Sentiment Analyzer", layout="wide")
st.title("🧠 SEO Entity & Sentiment Analyzer")

with st.expander("ℹ️ How it works"):
    st.markdown("""
    - Extracts **entities** and performs **sentiment analysis** on your text.
    - Automatically generates **Entity Schema Markup** (LocalBusiness).
    """)

# --- Text Input
content = st.text_area("📝 Paste your content here", height=300)
if st.button("🔍 Analyze"):
    if content:
        # Sentiment Analysis
        blob = TextBlob(content)
        sentiment_score = blob.sentiment.polarity
        sentiment_label = "Positive" if sentiment_score > 0 else "Negative" if sentiment_score < 0 else "Neutral"
        st.subheader("📈 Sentiment Analysis")
        st.metric("Sentiment", sentiment_label, f"{sentiment_score:.2f}")

        # Entity Extraction
        with st.spinner("Extracting entities..."):
            response = nlp.entities(content, include="types,categories,lod")
            entities = response.annotations
            if entities:
                st.subheader("🔍 Extracted Entities")
                entity_data = []
                seen = set()
                for ent in entities:
                    label = ent.spot
                    typ = ent.types[0].split(":")[-1] if ent.types else "Thing"
                    uri = ent.lod.get("wikidata") or ent.lod.get("dbpedia") or ""
                    key = (label.lower(), typ.lower())
                    if key not in seen:
                        seen.add(key)
                        entity_data.append({"label": label, "type": typ, "uri": uri})
                for e in entity_data:
                   st.markdown(f"- **{e['label']}** ({e['type']}) {'🔗[' + e['uri'] + '](' + e['uri'] + ')' if e['uri'] else ''}")
            else:
                st.info("No entities found.")
    else:
        st.warning("Please enter some text to analyze.")

# --- Entity Schema Generator (only after analysis)
if 'entity_data' in locals() and entity_data:
    st.header("🧾 Entity Schema Markup Generator")

    biz_name = st.text_input("🏷️ Business Name")
    biz_desc = st.text_area("📝 Business Description")
    biz_url = st.text_input("🔗 Website URL")
    biz_logo = st.text_input("🖼️ Logo URL")
    biz_image = st.text_input("📷 Image URL")
    biz_keywords = st.text_area("🔑 Keywords (comma-separated)")

    # Address
    st.subheader("📍 Business Address")
    street = st.text_input("Street Address")
    locality = st.text_input("Locality (City)")
    region = st.text_input("Region/State")
    postal_code = st.text_input("Postal Code")
    country = st.text_input("Country")

    # Contact
    st.subheader("📞 Contact Info")
    contact_phone = st.text_input("Phone Number")
    contact_email = st.text_input("Email")
    contact_type = st.selectbox("Contact Type", ["customer support", "sales", "technical support", "other"])

    # Social links
    st.subheader("🔗 Social Profiles")
    same_as = st.text_area("Enter one URL per line")

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
            "address": {
                "@type": "PostalAddress",
                "streetAddress": street,
                "addressLocality": locality,
                "addressRegion": region,
                "postalCode": postal_code,
                "addressCountry": country,
            },
            "contactPoint": {
                "@type": "ContactPoint",
                "telephone": contact_phone,
                "email": contact_email,
                "contactType": contact_type
            },
            "sameAs": [s.strip() for s in same_as.splitlines() if s.strip()],
            "mainEntity": [
                {
                    "@type": ent["type"],
                    "name": ent["label"],
                    **({"sameAs": ent["uri"]} if ent["uri"] else {})
                } for ent in entity_data
            ]
        }

        # Clean empty
        schema = {k: v for k, v in schema.items() if v and v != {"@type": "PostalAddress"} and v != {"@type": "ContactPoint"}}
        schema_str = json.dumps(schema, indent=2)

        st.code(schema_str, language="json")
        st.download_button("⬇️ Download Schema", schema_str, file_name="entity_schema.json", mime="application/json")
    else:
        st.info("ℹ️ Please enter business name, description, and URL to generate schema.")
