import streamlit as st
from dandelion import DataTXT
from textblob import TextBlob
import json
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from SPARQLWrapper import SPARQLWrapper, JSON
import textstat

st.set_page_config(page_title="SEO Entity & Sentiment Analyzer", layout="wide")
st.title("🔍 SEO Entity & Sentiment Analyzer (Dandelion + TextBlob)")

# Input fields
api_token = st.text_input("🔑 Enter your Dandelion API Token", type="password")
user_text = st.text_area("✍️ Enter Your Content", height=200)
ref_text = st.text_area("📄 (Optional) Enter Competitor Content", height=200)
keyword_input = st.text_input("🔎 (Optional) Enter Target Keywords (comma-separated)")
schema_type = st.selectbox("📘 Select Schema Type", ["Article", "BlogPosting", "LocalBusiness"])
author = st.text_input("👤 Author Name (optional)")
publisher = st.text_input("🏢 Publisher Name (optional)")
date_published = st.text_input("📅 Date Published (YYYY-MM-DD, optional)")
description = st.text_area("📝 Short Description (optional)", height=100)
analyze_btn = st.button("🚀 Analyze")

def fetch_related_entities_filtered(entity_labels):
    sparql = SPARQLWrapper("http://dbpedia.org/sparql")
    related_entities = set()

    for label in entity_labels:
        query = f'''
        SELECT DISTINCT ?relatedLabel WHERE {{
          ?entity rdfs:label "{label}"@en .
          ?entity dbo:wikiPageWikiLink ?related .
          ?related rdfs:label ?relatedLabel .
          ?related rdf:type ?type .
          ?related dbo:wikiPageInLinkCount ?linkCount .
          
          FILTER (lang(?relatedLabel) = 'en')
          FILTER(?type IN (dbo:Person, dbo:Place, dbo:Organization, dbo:Event, dbo:Work))
          FILTER(?linkCount > 50)
          FILTER NOT EXISTS {{ ?related dbo:wikiPageDisambiguates ?any }}
          FILTER (!regex(str(?related), "List_of"))
        }}
        LIMIT 10
        '''
        sparql.setQuery(query)
        sparql.setReturnFormat(JSON)
        try:
            results = sparql.query().convert()
            for result in results["results"]["bindings"]:
                related_label = result["relatedLabel"]["value"]
                if related_label.lower() not in [l.lower() for l in entity_labels]:
                    related_entities.add(related_label)
        except Exception as e:
            st.warning(f"SPARQL query error for entity '{label}': {e}")
    return related_entities

if api_token and user_text.strip() and analyze_btn:
    datatxt = DataTXT(token=api_token, min_confidence=0.6)

    # Entity extraction for user content
    st.header("🧠 Entity Extraction")
    try:
        nex_result = datatxt.nex(user_text, include="types,uri,confidence")
        entity_data = [
            {
                "label": ann.label,
                "type": ann.types[0].split("/")[-1] if ann.types else "Thing",
                "uri": ann.uri if hasattr(ann, "uri") and ann.uri else None,
                "confidence": round(ann.confidence, 2) if hasattr(ann, "confidence") else None
            }
            for ann in nex_result.annotations
        ]
        if entity_data:
            st.success(f"✅ Found {len(entity_data)} Entities:")
            for ent in entity_data:
                line = f"- **{ent['label']}** ({ent['type']}) - Confidence: {ent['confidence']}"
                if ent['uri']:
                    line += f" [🔗]({ent['uri']})"
                st.markdown(line)
        else:
            st.warning("No entities found.")
    except Exception as e:
        st.error(f"Entity extraction error: {e}")
        entity_data = []

    # Section-wise entity and sentiment analysis
    st.header("📑 Section-wise Entity & Sentiment Analysis")
    sections = user_text.split("\n\n")
    for i, section in enumerate(sections):
        if section.strip():
            st.subheader(f"Section {i+1}")
            blob = TextBlob(section)
            polarity = blob.sentiment.polarity
            sentiment = "Positive 😊" if polarity > 0 else "Negative 😠" if polarity < 0 else "Neutral 😐"
            st.markdown(f"- **Sentiment:** {sentiment} ({polarity:.3f})")

            try:
                sec_result = datatxt.nex(section, include="types,uri")
                section_entities = [ann.label for ann in sec_result.annotations]
                if section_entities:
                    st.markdown(f"- **Entities:** {', '.join(section_entities)}")
            except:
                pass

    # Sentiment analysis
    st.header("💬 Sentiment Analysis")
    try:
        blob = TextBlob(user_text)
        polarity = blob.sentiment.polarity
        sentiment_label = "Positive 😊" if polarity > 0 else "Negative 😠" if polarity < 0 else "Neutral 😐"
        st.markdown(f"**Sentiment Score:** {polarity:.3f}")
        st.markdown(f"**Interpretation:** {sentiment_label}")
    except Exception as e:
        st.error(f"Sentiment analysis error: {e}")

    # Readability Analysis
    st.header("📖 Readability & Optimization Suggestions")
    try:
        score = textstat.flesch_reading_ease(user_text)
        st.markdown(f"**Flesch Reading Ease Score:** {score:.2f}")
        suggestions = []
        if score < 50:
            suggestions.append("Consider simplifying your sentences to improve readability.")
        if polarity < -0.3:
            suggestions.append("Tone appears negative. Balance it with more positive framing.")
        if entity_data and keyword_input:
            keywords = [kw.strip().lower() for kw in keyword_input.split(",") if kw.strip()]
            matched = [ent["label"].lower() for ent in entity_data if ent["label"].lower() in keywords]
            if len(matched) < len(keywords):
                suggestions.append("Include more relevant entities tied to your target keywords.")
        if suggestions:
            st.warning("✏️ Optimization Suggestions:")
            for s in suggestions:
                st.markdown(f"- {s}")
        else:
            st.success("✅ No major optimization issues detected.")
    except Exception as e:
        st.error(f"Readability error: {e}")

    # Word Cloud Visualization
    st.header("☁️ Entity Word Cloud")
    try:
        wc_text = " ".join([ent["label"] for ent in entity_data])
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(wc_text)
        fig, ax = plt.subplots()
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis("off")
        st.pyplot(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Word cloud error: {e}")

    # Keyword to Entity Matching
    if keyword_input:
        st.header("🔗 Keyword-to-Entity Matching")
        keywords = [kw.strip().lower() for kw in keyword_input.split(",") if kw.strip()]
        matches = [ent for ent in entity_data if ent["label"].lower() in keywords]
        if matches:
            st.success("Entities matching target keywords:")
            for ent in matches:
                st.markdown(f"- 🎯 {ent['label']} ({ent['type']})")
        else:
            st.info("No exact keyword matches found among extracted entities.")

    # Competitor comparison if provided
    if ref_text.strip():
        st.header("📎 Competitor Comparison")
        try:
            comp_result = datatxt.nex(ref_text, include="types,uri")
            comp_entities = [
                {
                    "label": ann.label,
                    "type": ann.types[0].split("/")[-1] if ann.types else "Thing",
                    "uri": ann.uri if hasattr(ann, "uri") and ann.uri else None
                }
                for ann in comp_result.annotations
            ]
            user_labels = set([ent["label"].lower() for ent in entity_data])
            comp_labels = set([ent["label"].lower() for ent in comp_entities])

            missing = comp_labels - user_labels
            extra = user_labels - comp_labels
            common = user_labels & comp_labels

            coverage_score = len(common) / len(comp_labels.union(user_labels)) if comp_labels.union(user_labels) else 0

            st.markdown(f"**Coverage Depth Score:** {coverage_score:.2%}")
            st.progress(coverage_score)

            if missing:
                st.warning("📌 Missing Entities from Your Content:")
                for ent in comp_entities:
                    if ent["label"].lower() in missing:
                        line = f"- ❗ **{ent['label']}** ({ent['type']})"
                        if ent['uri']:
                            line += f" [🔗]({ent['uri']})"
                        st.markdown(line)

            if extra:
                st.info("🌿 Extra Entities in Your Content:")
                for ent in entity_data:
                    if ent["label"].lower() in extra:
                        line = f"- ✅ **{ent['label']}** ({ent['type']})"
                        if ent['uri']:
                            line += f" [🔗]({ent['uri']})"
                        st.markdown(line)

            if common:
                st.success("🎯 Shared Entities:")
                for ent in sorted(common):
                    st.markdown(f"- {ent}")
        except Exception as e:
            st.error(f"Competitor comparison error: {e}")

    # Suggested Missing Related Entities (DBpedia)
    if entity_data:
        st.header("💡 Suggested Missing Related Entities (DBpedia)")

        entity_labels = [ent["label"] for ent in entity_data]
        related_entities = fetch_related_entities_filtered(entity_labels)
        if related_entities:
            st.markdown(
                "Here are some related entities you might consider including to improve topic coverage:"
            )
            for ent in related_entities:
                st.markdown(f"- {ent}")
        else:
            st.info("No relevant related entities found or all related entities are already covered.")

else:
    if not api_token:
        st.info("Please enter your Dandelion API token to start analysis.")
    elif not user_text.strip():
        st.info("Please enter some content to analyze.")
