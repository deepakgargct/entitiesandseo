import streamlit as st
from dandelion import DataTXT
from textblob import TextBlob
import textstat

# Streamlit layout
st.set_page_config(page_title="SEO Content Analyzer", layout="centered")
st.title("🔍 SEO Content Analyzer with NLP")
st.write("Powered by **Dandelion.eu** + **TextBlob** + SEO heuristics.")

# API Token
token = st.text_input("🔐 Dandelion API Token", type="password")

if token:
    datatxt = DataTXT(token=token)

    # Input text
    user_input = st.text_area("✏️ Enter the content to analyze:", height=250)

    # Optional comparison input
    ref_text = st.text_area("📄 (Optional) Enter competitor/reference content to compare:", height=200)

    # Keyword for density
    keyword = st.text_input("🔑 Optional keyword to check density:")

    # Entity type filter
    entity_type_filter = st.multiselect("📌 Filter entities by type (DBpedia)", ["Person", "Place", "Organization"])

    if st.button("🚀 Run Analysis") and user_input:
        with st.spinner("Analyzing..."):

            ## LANGUAGE DETECTION
            lang_result = datatxt.li(user_input)
            lang = lang_result.get("detectedLangs", [{}])[0].get("lang", "unknown")
            st.success(f"Detected Language: `{lang}`")

            # 1. ENTITY EXTRACTION
            st.subheader("📍 Entity Extraction")
            try:
                nex_result = datatxt.nex(user_input, include="types,abstract,categories")
                filtered_entities = []
                for ann in nex_result.annotations:
                    # Extract type from DBpedia URI
                    types = [t.split("/")[-1] for t in ann.types]
                    if not entity_type_filter or any(t in types for t in entity_type_filter):
                        filtered_entities.append((ann.label, types, ann.uri))

                if filtered_entities:
                    for label, types, uri in filtered_entities:
                        st.markdown(f"- **{label}** ({', '.join(types)}) — [Link]({uri})")
                else:
                    st.info("No matching entities found.")
            except Exception as e:
                st.error(f"Entity extraction error: {e}")

            # 2. SENTIMENT ANALYSIS BY PARAGRAPH
            st.subheader("❤️ Sentiment Analysis by Section")
            paras = user_input.split("\n")
            sentiment_scores = []

            for para in paras:
                if para.strip():
                    blob = TextBlob(para)
                    polarity = blob.sentiment.polarity
                    sentiment_scores.append(polarity)
                    st.write(f"**{para.strip()[:60]}...** → Sentiment: `{polarity:.2f}`")

            avg_sent = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0

            # 3. SEMANTIC SIMILARITY
            if ref_text.strip():
                try:
                    sim_score = datatxt.sim(user_input, ref_text)
                    st.subheader("🔁 Semantic Similarity")
                    st.success(f"Similarity with reference: `{sim_score.get('similarity', 0.0):.2f}`")
                except Exception as e:
                    st.error(f"Similarity check failed: {e}")
            else:
                sim_score = {"similarity": None}

            # 4. KEYWORD DENSITY
            if keyword:
                words = user_input.lower().split()
                k_count = words.count(keyword.lower())
                density = (k_count / len(words)) * 100 if words else 0
                st.subheader("📊 Keyword Density")
                st.write(f"Keyword `{keyword}` appears **{k_count}** times — density: **{density:.2f}%**")

            # 5. READABILITY
            st.subheader("📘 Readability Score")
            try:
                score = textstat.flesch_reading_ease(user_input)
                st.write(f"Flesch Reading Ease: **{score:.2f}**")
            except Exception as e:
                st.error(f"Readability error: {e}")

            # 6. AUTOMATED RECOMMENDATIONS
            st.subheader("🧠 Content Improvement Suggestions")

            if avg_sent < -0.2:
                st.warning("⚠️ Content has a **negative tone** overall. Consider softening the language.")
            elif avg_sent > 0.5:
                st.info("😊 Content has a **positive tone**. Great for promotional messaging!")

            if len(filtered_entities) < 3:
                st.warning("📉 Few named entities detected. Consider adding more **people, places, or organizations** for topical depth.")

            if keyword and density < 1:
                st.warning(f"🔑 Keyword `{keyword}` appears infrequently. Consider reinforcing it.")
            elif keyword and density > 3:
                st.warning(f"🚨 Keyword `{keyword}` might be overused. Avoid keyword stuffing.")

            if sim_score.get("similarity") is not None and sim_score["similarity"] < 0.4:
                st.warning("🔍 Your content has **low similarity** with the reference. Make sure it's addressing the same topic.")

            if score < 50:
                st.warning("📚 Content may be too complex. Aim for a **Flesch score above 60** for better readability.")

else:
    st.info("Please enter your Dandelion API token to begin.")
