# app.py
import os
import json
from typing import List
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests
import networkx as nx
import matplotlib.pyplot as plt

# -------- Config: Gemini --------
MODEL_NAME = "gemini-2.0-flash-lite"
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent"

st.set_page_config(page_title="Virtual Assistant DC", layout="wide")
st.title("MANISH - Data Center Virtual Assistant — Self-Service AI")

# ---------------- Simple RAG ----------------
class TinyRAG:
    def __init__(self):
        self.docs: List[str] = []
        self.tfidf = None
        self.vectors = None

    def index_texts(self, texts: List[str]):
        self.docs = texts
        if self.docs:
            self.tfidf = TfidfVectorizer(stop_words="english", max_features=2000)
            self.vectors = self.tfidf.fit_transform(self.docs)

    def retrieve(self, query: str, top_k: int = 3) -> List[str]:
        if self.vectors is None or not self.docs:
            return []
        qv = self.tfidf.transform([query])
        sims = cosine_similarity(qv, self.vectors)[0]
        idxs = sims.argsort()[::-1][:top_k]
        results = [self.docs[i] for i in idxs if sims[i] > 0]
        return results

# ---------------- Gemini call with fallback ----------------
def generate_with_gemini(prompt: str) -> str:
    key = st.secrets.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not key:
        st.warning("GEMINI_API_KEY not set. Using local fallback generator.")
        return local_plan_generator(prompt)

    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {key}"}
    body = {"prompt": {"text": prompt}, "temperature": 0.2, "maxOutputTokens": 800}

    try:
        r = requests.post(API_URL, headers=headers, json=body, timeout=20)
        if r.status_code == 401:
            st.error("Gemini API Key invalid or unauthorized! Using local fallback.")
            return local_plan_generator(prompt)
        r.raise_for_status()
        resp = r.json()
        # Parse response
        if "candidates" in resp and len(resp["candidates"]) > 0:
            return resp["candidates"][0].get("content","") or resp["candidates"][0].get("display","")
        if "output" in resp and isinstance(resp["output"], list):
            return " ".join([x.get("content","") if isinstance(x, dict) else str(x) for x in resp["output"]])
        if "text" in resp:
            return resp["text"]
        return json.dumps(resp, indent=2)[:4000]
    except Exception as e:
        st.warning(f"Gemini call failed: {e}. Using local fallback.")
        return local_plan_generator(prompt)

def local_plan_generator(prompt: str) -> str:
    import re
    nums = re.findall(r"\d+\.?\d*", prompt)
    top_nums = nums[:6]
    summary = "Local fallback response:\n\n"
    summary += "- Summary: Basic guidance provided.\n"
    if top_nums:
        summary += f"- Numbers found: {', '.join(top_nums)}\n"
    summary += "- Actions:\n  1. Review critical systems.\n  2. Follow standard procedures.\n  3. Document outcomes.\n"
    summary += "Note: Set GEMINI_API_KEY to enable AI-generated plans."
    return summary

# ---------------- LangGraph visualization ----------------
def display_plan_graph(plan_text: str):
    G = nx.DiGraph()
    lines = [l.strip() for l in plan_text.split("\n") if l.strip()]
    for i, line in enumerate(lines):
        node_label = f"Step {i+1}: {line[:30]}"
        G.add_node(node_label)
        if i > 0:
            G.add_edge(f"Step {i}", node_label)
    plt.figure(figsize=(8,4))
    nx.draw(G, with_labels=True, node_color="skyblue", node_size=2500, arrows=True)
    st.pyplot(plt)

# ---------------- Streamlit UI ----------------
with st.sidebar:
    st.header("Knowledge & Context")
    uploaded_texts = st.file_uploader("Upload knowledge .txt/.md", type=["txt","md"], accept_multiple_files=True)
    paste_context = st.text_area("Or paste context snippets (one per line)", height=150)
    top_k = st.slider("RAG top-k snippets", 1, 10, 3)

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
    texts.extend([l.strip() for l in paste_context.splitlines() if l.strip()])
if not texts:
    texts.append("Default knowledge snippet: refer to data center best practices.")
rag.index_texts(texts)

st.subheader("Ask your Data Center Virtual Assistant")
user_prompt = st.text_area("Describe your issue or question:")

if st.button("Get Response"):
    with st.spinner("Generating response..."):
        # Retrieve relevant snippets
        snippets = rag.retrieve(user_prompt, top_k=top_k)
        combined_prompt = user_prompt
        if snippets:
            combined_prompt += "\n\nRelevant context:\n" + "\n".join(snippets)
        response = generate_with_gemini(combined_prompt)
        st.subheader("Assistant Response")
        st.write(response)
        st.subheader("Plan Dependency Graph")
        display_plan_graph(response)
