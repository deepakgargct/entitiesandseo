import streamlit as st
from dandelion import DataTXT
from textblob import TextBlob
import json
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from SPARQLWrapper import SPARQLWrapper, JSON
import textstat

st.set_page_config(page_title="SEO Entity & Sentiment Analyzer", layout="wide")
st.title("🔍 SEO Entity & Sentiment Analyzer (Dandelion + TextBlob + DBpedia)")

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

def fetch_related_entities(entity_labels):
    """
    Given a list of entity labels, query DBpedia for related entities using SPARQL.
    Returns a set of related entity labels to suggest missing entities.
    """
    sparql = SPARQLWrapper("http://dbpedia.org/sparql")
    related_entities = set()

    for label in entity_labels:
        query = f"""
        SELECT DISTINCT ?relatedLabel WHERE {{
          ?entity rdfs:label "{label}"@en .
          ?entity dbo:wikiPageWikiLink ?related .
          ?related rdfs:label ?relatedLabel .
          FILTER (lang(?relatedLabel) = 'en')
        }}
        LIMIT 10
        """
        sparql.setQuery(query)
        sparql.setReturnFormat(JSON)
        try:
            results = sparql.query().convert()
            for result in results["results"]["bindings"]:
                related_label = result["relatedLabel"]["value"]
                if related_label.lower() not in [l.lower() for l in entity_labels]:
                    related_entities.add(related_label)
        except Exception as e:
            st.error(f"SPARQL query error for entity '{label}': {e}")
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

    # Suggest missing related entities via DBpedia
    st.header("💡 Suggested Missing Related Entities (DBpedia)")
    try:
        current_entity_labels = [ent["label"] for ent in entity_data]
        related_entities = fetch_related_entities(current_entity_labels)
        if related_entities:
            st.markdown("These related entities are suggested to improve topic coverage and semantic depth:")
            for rel_ent in sorted(related_entities):
                st.markdown(f"- {rel_ent}")
        else:
            st.info("No related entities suggested. Your content covers key topics well!")
    except Exception as e:
        st.error(f"Error fetching related entities: {e}")

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
                    st.markdown(f"- 🔁 {ent}")

        except Exception as e:
            st.error(f"Competitor analysis error: {e}")

    # Enhanced JSON-LD schema markup generation
    if entity_data:
        st.header("🧾 Auto-Generated Entity Schema Markup (JSON-LD)")
        schema = {
            "@context": "https://schema.org",
            "@type": schema_type,
            **({"author": {"@type": "Person", "name": author}} if author else {}),
            **({"publisher": {"@type": "Organization", "name": publisher}} if publisher else {}),
            **({"datePublished": date_published} if date_published else {}),
            **({"description": description} if description else {}),
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

else:
    st.info("🔐 Enter your API key and content, then click Analyze.")
