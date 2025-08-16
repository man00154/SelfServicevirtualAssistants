# app.py
import os
import json
import streamlit as st
import requests
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from pyvis.network import Network
import tempfile

# -------- Config ----------
MODEL_NAME = "gemini-2.0-flash-lite"
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent"

st.set_page_config(page_title="Data Center Virtual Assistant", layout="wide")
st.title("MANISH - Self-Service GenAI Data Center Assistant")

st.markdown("""
This intelligent assistant can understand natural language, provide dynamic guidance, and generate personalized, actionable support.
It can synthesize information from multiple sources and visualize dependencies using a simple LangGraph.
""")

# ---------------- Very simple RAG ----------------
class TinyRAG:
    def __init__(self):
        self.docs: List[str] = []
        self.tfidf = None
        self.vectors = None

    def index_texts(self, texts: List[str]):
        self.docs = texts
        self.tfidf = TfidfVectorizer(stop_words="english", max_features=2000)
        self.vectors = self.tfidf.fit_transform(self.docs)

    def retrieve(self, query: str, top_k: int = 3) -> List[str]:
        if self.vectors is None or not self.docs:
            return []
        qv = self.tfidf.transform([query])
        sims = cosine_similarity(qv, self.vectors)[0]
        idxs = sims.argsort()[::-1][:top_k]
        return [self.docs[i] for i in idxs if sims[i] > 0]

# ---------------- Gemini API ----------------
def generate_with_gemini(prompt: str) -> str:
    key = os.environ.get("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY", None)
    if not key:
        return f"[Local fallback] Prompt received: {prompt}"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {key}"}
    body = {"prompt": {"text": prompt}, "temperature": 0.2, "maxOutputTokens": 800}
    try:
        r = requests.post(API_URL, headers=headers, json=body, timeout=20)
        r.raise_for_status()
        resp = r.json()
        if "candidates" in resp and resp["candidates"]:
            return resp["candidates"][0].get("content", "")
        elif "output" in resp and isinstance(resp["output"], list):
            return " ".join([x.get("content","") if isinstance(x, dict) else str(x) for x in resp["output"]])
        return json.dumps(resp, indent=2)[:4000]
    except Exception as e:
        st.warning(f"Gemini API failed: {e}")
        return f"[Local fallback] Prompt received: {prompt}"

# ---------------- Very simple LangGraph ----------------
def generate_langgraph_graph(nodes: List[str], edges: List[tuple]):
    G = nx.DiGraph()
    for n in nodes:
        G.add_node(n)
    for e in edges:
        G.add_edge(e[0], e[1])
    net = Network(height="400px", width="100%", directed=True)
    net.from_nx(G)
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
    net.save_graph(tmp_file.name)
    return tmp_file.name

# ---------------- Streamlit UI ----------------
with st.sidebar:
    st.header("Knowledge / Context")
    uploaded_texts = st.file_uploader("Upload text files for context (optional)", type=["txt","md"], accept_multiple_files=True)
    paste_context = st.text_area("Or paste knowledge/context snippets (one per line)", height=150)
    st.markdown("---")
    st.markdown("**Gemini API key** (optional): set `GEMINI_API_KEY` in env or Streamlit secrets.")

# Build RAG index
rag = TinyRAG()
texts = []

if uploaded_texts:
    for f in uploaded_texts:
        try:
            txt = f.read().decode("utf-8")
        except:
            txt = f.read().decode("latin-1")
        if txt:
            texts.append(txt)

if paste_context:
    for line in paste_context.splitlines():
        if line.strip():
            texts.append(line.strip())

if not texts:
    texts.append("Default knowledge snippet: Data center operations, disaster recovery, backup validation, monitoring critical systems.")

rag.index_texts(texts)
st.success(f"Indexed {len(texts)} knowledge snippets for retrieval.")

# Chat interface
st.header("Ask the Virtual Assistant")
user_input = st.text_input("Enter your question or describe a scenario:")

if st.button("Get Answer") and user_input.strip():
    with st.spinner("Processing..."):
        # Retrieve top-k snippets
        retrieved_snippets = rag.retrieve(user_input, top_k=3)
        context_text = "\n".join([f"- {s}" for s in retrieved_snippets])
        prompt = f"You are a GenAI-powered virtual assistant for Data Center support.\nUser Query: {user_input}\nContext:\n{context_text}\nProvide a clear, actionable, and prioritized response."
        response = generate_with_gemini(prompt)
        st.subheader("Assistant Response")
        st.write(response)

        # Very simple LangGraph visualization
        nodes = [user_input[:30]] + [s[:30] for s in retrieved_snippets] + ["Response"]
        edges = [(user_input[:30], s[:30]) for s in retrieved_snippets] + [(s[:30], "Response") for s in retrieved_snippets]
        graph_file = generate_langgraph_graph(nodes, edges)
        st.subheader("Response Dependencies (LangGraph)")
        st.components.v1.html(open(graph_file, "r").read(), height=450)
