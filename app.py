import streamlit as st
from dandelion import DataTXT
from textblob import TextBlob
from sentence_transformers import SentenceTransformer, util
import spacy

# Load SpaCy model
nlp = spacy.load("en_core_web_sm")

# Initialize sentence transformer for semantic similarity
model = SentenceTransformer('all-MiniLM-L6-v2')

# Streamlit page config
st.set_page_config(page_title="SEO Entity & Sentiment Analyzer", layout="wide")
st.title("🔍 SEO Entity & Sentiment Analyzer using Dandelion + TextBlob")

# Inputs
api_token = st.text_input("🔑 Enter your Dandelion API Token", type="password")
user_text = st.text_area("✍️ Enter Your Content", height=200)
target_topic = st.text_input("🔑 Enter Target Topic (e.g. 'Web3 gaming')", placeholder="Web3 gaming")
ref_text = st.text_area("📄 (Optional) Enter Competitor Content", height=200)

# Predefined entities for coverage check (simple list, can be expanded)
predefined_entities = ["blockchain", "gaming", "decentralized", "NFT", "Web3", "cryptocurrency"]

if api_token and user_text.strip():
    datatxt = DataTXT(token=api_token, min_confidence=0.6)

    # Entity Extraction
    st.header("🧠 Entity Extraction")
    try:
        nex_result = datatxt.nex(user_text, include="types,uri")
        user_entities = [ann.label.lower() for ann in nex_result.annotations]

        if user_entities:
            st.success(f"✅ Found {len(user_entities)} Entities:")
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

    # Entity Coverage Scoring (based on predefined entities)
    st.header("📊 Entity Coverage Score")
    if target_topic.strip():
        target_topic_entities = [e.lower() for e in predefined_entities]
        matched_entities = [e for e in user_entities if e in target_topic_entities]
        coverage_score = len(matched_entities) / len(target_topic_entities) * 100

        st.markdown(f"**Entities Found for '{target_topic}':**")
        st.markdown(f"Matched Entities: {', '.join(matched_entities)}")
        st.markdown(f"**Coverage Score:** {coverage_score:.2f}%")
        
    else:
        st.info("Enter a target topic to get the entity coverage score.")

    # Semantic Similarity Analysis
    if ref_text.strip():
        st.header("📎 Competitor Analysis: Semantic Similarity")
        try:
            # Compare user content to competitor content using Sentence Transformers
            user_sentences = [sent.text for sent in nlp(user_text).sents]
            ref_sentences = [sent.text for sent in nlp(ref_text).sents]

            # Encode the sentences using Sentence-Transformer
            user_embeddings = model.encode(user_sentences, convert_to_tensor=True)
            ref_embeddings = model.encode(ref_sentences, convert_to_tensor=True)

            # Compute cosine similarity
            cosine_similarities = util.pytorch_cos_sim(user_embeddings, ref_embeddings)

            # Show highest similarity scores (top 3 matches)
            top_matches = cosine_similarities.max(dim=1)[0].cpu().numpy()
            top_matches_indices = cosine_similarities.max(dim=1)[1].cpu().numpy()

            st.markdown("### Semantic Similarity Scores (User vs Competitor):")
            for idx, score in zip(top_matches_indices[:3], top_matches[:3]):
                st.markdown(f"- **{user_sentences[idx]}** → {ref_sentences[idx]} (Score: {score:.2f})")

        except Exception as e:
            st.error(f"Semantic similarity analysis error: {e}")

else:
    st.info("🔐 Please enter your API token and your content to begin.")
