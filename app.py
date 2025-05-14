import streamlit as st
from dandelion import DataTXT
from textblob import TextBlob
import json

# Industry-based keyword suggestions
INDUSTRY_ENTITIES = {
    "photography": {
        "Ghost Mannequin Photography", "Flat Lay Product Photography", "On Model Ecommerce Photography",
        "Editorial Photography", "Retouching", "Casting & Production", "Studio Consulting"
    }
}

def deduplicate_entities(annotations):
    seen = set()
    unique_entities = []
    for ann in annotations:
        label = ann.label.strip().lower()
        if label not in seen:
            seen.add(label)
            unique_entities.append({
                "label": ann.label,
                "type": ann.types[0].split("/")[-1] if ann.types else "Thing",
                "uri": ann.uri if hasattr(ann, "uri") and ann.uri else None
            })
    return unique_entities

st.set_page_config(page_title="SEO Entity & Sentiment Analyzer", layout="wide")
st.title("🔍 SEO Entity & Sentiment Analyzer (Dandelion + TextBlob)")

api_token = st.text_input("🔑 Enter your Dandelion API Token", type="password")
user_text = st.text_area("✍️ Enter Your Content", height=200)
ref_text = st.text_area("📄 (Optional) Enter Competitor Content", height=200)
analyze_btn = st.button("🚀 Analyze")

if api_token and user_text.strip() and analyze_btn:
    datatxt = DataTXT(token=api_token, min_confidence=0.6)

    # --- Entity Extraction ---
    st.header("🧠 Entity Extraction")
    try:
        nex_result = datatxt.nex(user_text, include="types,uri")
        entity_data = deduplicate_entities(nex_result.annotations)
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

    # --- Sentiment Analysis ---
    st.header("💬 Sentiment Analysis")
    try:
        blob = TextBlob(user_text)
        polarity = blob.sentiment.polarity
        sentiment_label = "Positive 😊" if polarity > 0 else "Negative 😠" if polarity < 0 else "Neutral 😐"
        st.markdown(f"**Sentiment Score:** `{polarity:.3f}`")
        st.markdown(f"**Interpretation:** {sentiment_label}")
    except Exception as e:
        st.error(f"Sentiment analysis error: {e}")

    # --- Competitor Comparison ---
    if ref_text.strip():
        st.header("📎 Competitor Comparison")
        try:
            comp_result = datatxt.nex(ref_text, include="types,uri")
            comp_entities = deduplicate_entities(comp_result.annotations)

            user_labels = {ent["label"].lower() for ent in entity_data}
            comp_labels = {ent["label"].lower() for ent in comp_entities}

            missing = comp_labels - user_labels
            extra = user_labels - comp_labels
            common = user_labels & comp_labels
            coverage_score = len(common) / len(user_labels.union(comp_labels)) if user_labels.union(comp_labels) else 0

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

    # --- Suggested Industry Entities ---
    st.header("📌 Industry-Specific Suggestions")
    industry = "photography"  # You can make this dynamic with a dropdown later
    suggested = INDUSTRY_ENTITIES[industry] - {e["label"] for e in entity_data}
    if suggested:
        st.info("📈 You might consider adding these industry-relevant entities:")
        for s in suggested:
            st.markdown(f"- 💡 {s}")
    else:
        st.success("🚀 Your content covers common industry-specific terms!")

    # --- JSON-LD Schema Markup Generation ---
    if entity_data:
        st.header("🧾 Auto-Generated Entity Schema Markup (JSON-LD)")
        schema = {
            "@context": "https://schema.org",
            "@type": "Article",
            "mainEntity": [
                {
                    "@type": ent["type"],
                    "name": ent["label"],
                    **({"sameAs": ent["uri"]} if ent["uri"] else {})
                } for ent in entity_data
            ]
        }
        schema_str = json.dumps(schema, indent=2)
        st.code(schema_str, language="json")
        st.download_button("⬇️ Download Schema", schema_str, file_name="entity_schema.json", mime="application/json")

# --- LocalBusiness Schema Generator ---
st.header("🏢 LocalBusiness Schema Generator")
with st.form("localbiz"):
    biz_name = st.text_input("Business Name", "Hyperblack Studios")
    biz_desc = st.text_area("Description", "From ecommerce product photography to marketing campaign content...")
    biz_url = st.text_input("Website URL", "https://www.hyperblackstudios.com/")
    logo_url = st.text_input("Logo URL")
    image_urls = st.text_area("Image URLs (comma-separated)")
    sameas_links = st.text_area("Social/Other Links (comma-separated)")
    contact_url = st.text_input("Contact URL")
    keywords = st.text_area("Keywords (comma-separated)")
    address = {
        "street": st.text_input("Street Address"),
        "city": st.text_input("City"),
        "region": st.text_input("State/Region"),
        "zip": st.text_input("Postal Code"),
        "country": st.text_input("Country")
    }
    lat = st.text_input("Latitude")
    lng = st.text_input("Longitude")
    submitted = st.form_submit_button("Generate Schema")

    if submitted:
        local_schema = {
            "@context": "https://schema.org",
            "@type": "LocalBusiness",
            "name": biz_name,
            "description": biz_desc,
            "url": biz_url,
            "logo": logo_url,
            "image": [url.strip() for url in image_urls.split(",") if url.strip()],
            "sameAs": [url.strip() for url in sameas_links.split(",") if url.strip()],
            "contactPoint": {
                "@type": "ContactPoint",
                "contactType": ["Contact Sales"],
                "url": contact_url
            },
            "keywords": [kw.strip() for kw in keywords.split(",") if kw.strip()],
            "address": {
                "@type": "PostalAddress",
                "streetAddress": address["street"],
                "addressLocality": address["city"],
                "addressRegion": address["region"],
                "postalCode": address["zip"],
                "addressCountry": address["country"]
            },
            "geo": {
                "@type": "GeoCoordinates",
                "latitude": float(lat),
                "longitude": float(lng)
            }
        }
        local_schema_str = json.dumps(local_schema, indent=2)
        st.code(local_schema_str, language="json")
        st.download_button("⬇️ Download LocalBusiness Schema", local_schema_str, file_name="localbusiness_schema.json", mime="application/json")
