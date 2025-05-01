import streamlit as st
from dandelion import DataTXT
from textblob import TextBlob

# Streamlit page config
st.set_page_config(page_title="SEO Entity & Sentiment Analyzer", layout="wide")
st.title("🔍 SEO Entity & Sentiment Analyzer using Dandelion + TextBlob")

# Inputs
api_token = st.text_input("🔑 Enter your Dandelion API Token", type="password")
user_text = st.text_area("✍️ Enter Your Content", height=200)
ref_text = st.text_area("📄 (Optional) Enter Competitor Content", height=200)

if api_token and user_text.strip():
    datatxt = DataTXT(token=api_token, min_confidence=0.6)

    # Entity Extraction
    st.header("🧠 Entity Extraction")
    try:
        nex_result = datatxt.nex(user_text, include="types")
        entities = [ann.label for ann in nex_result.annotations]

        if entities:
            st.success(f"✅ Found {len(entities)} Entities:")
            st.markdown("### 🏷 Entities Detected:")
            for ann in nex_result.annotations:
                st.markdown(f"- **{ann.label}** ({ann.types[0].split('/')[-1] if ann.types else 'N/A'})")
        else:
            st.warning("No entities found.")
    except Exception as e:
        st.error(f"Entity extraction error: {e}")

    # Sentiment Analysis using TextBlob
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
        score = 0  # fallback to avoid breaking later logic

    # Competitor Comparison
    if ref_text.strip():
        st.header("📎 Competitor Analysis: Entity Comparison")
        try:
            comp_entities = datatxt.nex(ref_text, include="types")
            user_entity_labels = {ann.label.lower() for ann in nex_result.annotations}
            comp_entity_labels = {ann.label.lower() for ann in comp_entities.annotations}

            missing_entities = comp_entity_labels - user_entity_labels
            common_entities = user_entity_labels & comp_entity_labels
            unique_entities = user_entity_labels - comp_entity_labels

            if missing_entities:
                st.warning("📌 Entities your content is **missing** (used by competitor):")
                for ent in sorted(missing_entities):
                    st.markdown(f"- ❗ `{ent}`")

            if unique_entities:
                st.info("🌿 Entities unique to **your content** (not used by competitor):")
                for ent in sorted(unique_entities):
                    st.markdown(f"- ✅ `{ent}`")

            if common_entities:
                st.success("🎯 Entities common in both contents:")
                for ent in sorted(common_entities):
                    st.markdown(f"- 🔁 `{ent}`")

            # SEO Recommendations
            st.header("📈 Top SEO Recommendations")
            recommendations = []

            if missing_entities:
                recommendations.append(
                    f"Consider including missing entities like **{', '.join(sorted(missing_entities)[:5])}** to align with competitor coverage."
                )

            if unique_entities:
                recommendations.append(
                    f"Highlight unique entities such as **{', '.join(sorted(unique_entities)[:5])}** as differentiators."
                )

            if score < 0:
                recommendations.append(
                    "Your content has a negative sentiment — consider making the tone more positive or balanced."
                )

            if not recommendations:
                st.success("✅ No major improvements detected. Your content is well-aligned!")

            for rec in recommendations:
                st.markdown(f"- 💡 {rec}")

        except Exception as e:
            st.error(f"Competitor entity analysis error: {e}")
else:
    st.info("🔐 Please enter your API token and your content to begin.")
