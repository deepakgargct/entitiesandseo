import subprocess
import sys
import streamlit as st
from dandelion import DataTXT
from textblob import TextBlob
from sentence_transformers import SentenceTransformer, util
import spacy
import threading
from concurrent.futures import ThreadPoolExecutor

# Ensure SpaCy model is installed
try:
    spacy.load("en_core_web_sm")
except OSError:
    subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
    spacy.load("en_core_web_sm")

# Initialize Sentence Transformer for semantic similarity
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load SpaCy model
nlp = spacy.load("en_core_web_sm")

# Streamlit page config
st.set_page_config(page_title="SEO Entity & Sentiment Analyzer", layout="wide")
st.title("🔍 SEO Entity & Sentiment Analyzer using Dandelion + TextBlob")

# Inputs
api_token = st.text_input("🔑 Enter your Dandelion API Token", type="password")
user_text = st.text_area("✍️ Enter Your Content", height=200)
target_topic = st.text_input("🔑 Enter Target Topic (e.g. 'Web3 gaming')", placeholder="Web3 gaming")
ref_text = st.text_area("📄 (Optional) Enter Competitor Content", height=200)

# Predefined entities for coverage check
predefined_entities = ["blockchain", "gaming", "decentralized", "NFT", "Web3", "cryptocurrency"]

# Thread pool executor
executor = ThreadPoolExecutor(max_workers=2)

# Function to extract entities
def extract_entities():
    datatxt = DataTXT(token=api_token, min_confidence=0.6)
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

# Function to perform sentiment analysis
def analyze_sentiment():
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

# Start background tasks
if api_token and user_text.strip():
    with executor:
        future1 = executor.submit(extract_entities)
        future2 = executor.submit(analyze_sentiment)
else:
    st.info("🔐 Please enter your API token and your content to begin.")
