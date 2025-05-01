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
        nex_result = datatxt.nex(user_text, include="types,uri")
        entities = [ann.label for ann in nex_result.annotations]

        if entities:
            st.success(f"✅ Found {len(entities)} Entities:")
            st.markdown("### 🏷 Entities Detected with Citations:")
            for ann in nex_result.annotations:
                label = ann.label
                entity_type = ann.types[0].split("/")[-1] if ann.types else "N/A"
                uri = ann.uri if hasattr(ann, "uri") and ann.uri else None
                if uri:
                    st.markdown(f"- **[{label}]({uri})** ({entity_type})")
                else:
                    st.markdown(f"- **{label}** ({entity_type})")
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
            comp_result = datatxt.nex(ref_text, include="types,uri")
            user_entity_map = {ann.label.lower(): ann for ann in nex_result.annotations}
            comp_entity_map = {ann.label.lower(): ann for ann in comp_result.annotations}

            user_labels = set(user_entity_map.keys())
            comp_labels = set(comp_entity_map.keys())

            missing = comp_labels - user_labels
            unique = user_labels - comp_labels
            common = user_labels & comp_labels

            if missing:
                st.warning("📌 Entities your content is **missing** (used by competitor):")
                for label in sorted(missing):
                    ann = comp_entity_map[label]
                    uri = ann.uri if hasattr(ann, "uri") and ann.uri else None
                    if uri:
                        st.markdown(f"- ❗ **[{label.title()}]({uri})**")
                    else:
                        st.markdown(f"- ❗ `{label.title()}`")

            if unique:
                st.info("🌿 Entities unique to **your content** (not used by competitor):")
                for label in sorted(unique):
                    ann = user_entity_map[label]
                    uri = ann.uri if hasattr(ann, "uri") and ann.uri else None
                    if uri:
                        st.markdown(f"- ✅ **[{label.title()}]({uri})**")
                    else:
                        st.markdown(f"- ✅ `{label.title()}`")

            if common:
                st.success("🎯 Entities common in both contents:")
                for label in sorted(common):
                    ann = user_entity_map[label]
                    uri = ann.uri if hasattr(ann, "uri") and ann.uri else None
                    if uri:
                        st.markdown(f"- 🔁 **[{label.title()}]({uri})**")
                    else:
                        st.markdown(f"- 🔁 `{label.title()}`")

            # SEO Recommendations
            st.header("📈 Top SEO Recommendations")
            recommendations = []

            if missing:
                top_missing = [f"[{comp_entity_map[l].label}]({comp_entity_map[l].uri})" 
                               if hasattr(comp_entity_map[l], "uri") and comp_entity_map[l].uri 
                               else f"`{comp_entity_map[l].label}`" for l in sorted(missing)[:5]]
                recommendations.append(
                    f"Consider including missing entities like {', '.join(top_missing)} to align with competitor coverage."
                )

            if unique:
                top_unique = [f"[{user_entity_map[l].label}]({user_entity_map[l].uri})" 
                              if hasattr(user_entity_map[l], "uri") and user_entity_map[l].uri 
                              else f"`{user_entity_map[l].label}`" for l in sorted(unique)[:5]]
                recommendations.append(
                    f"Highlight unique entities such as {', '.join(top_unique)} as differentiators."
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
