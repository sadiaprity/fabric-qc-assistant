"""
Fabric QC Assistant - RAG Retrieval + Reasoning Layer
========================================================
Takes a CV classifier's prediction (defect_type, confidence) and generates
a grounded QC note using retrieval-augmented generation over the knowledge base.

Uses:
- sentence-transformers (free, local, no API cost) for embeddings
- Gemini API (free tier) for the reasoning/generation step

SETUP:
    pip install sentence-transformers google-generativeai numpy

    Get a free Gemini API key: https://aistudio.google.com/apikey
    Set it as an environment variable:
        export GEMINI_API_KEY="your-key-here"
    (or set it directly in the GEMINI_API_KEY line below for local testing)
"""

import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

# ── Config ────────────────────────────────────────────────────────────────
KB_PATH = "qc_knowledge_base.json"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # small, fast, free, runs locally
GEMINI_MODEL = "gemini-2.5-flash"       # gemini-2.0-flash was deprecated June 1, 2026 -- still free tier
CONFIDENCE_THRESHOLD = 0.85             # matches kb028 in your knowledge base

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
genai.configure(api_key=GEMINI_API_KEY)


# ── Load knowledge base + build embeddings once at startup ─────────────────
class QCKnowledgeBase:
    def __init__(self, kb_path: str):
        with open(kb_path, "r") as f:
            self.entries = json.load(f)
        print(f"Loaded {len(self.entries)} knowledge base entries")

        self.embedder = SentenceTransformer(EMBEDDING_MODEL)
        texts = [e["content"] for e in self.entries]
        self.embeddings = self.embedder.encode(texts, normalize_embeddings=True)

    def search(self, query: str, top_k: int = 3, defect_type_filter: str = None):
        """Cosine similarity search, optionally restricted to a defect_type."""
        query_vec = self.embedder.encode([query], normalize_embeddings=True)[0]
        sims = self.embeddings @ query_vec  # cosine similarity (both normalized)

        candidates = list(enumerate(sims))
        if defect_type_filter:
            candidates = [
                (i, s) for i, s in candidates
                if self.entries[i]["defect_type"] in (defect_type_filter, "general")
            ]
        candidates.sort(key=lambda x: x[1], reverse=True)
        top = candidates[:top_k]
        return [self.entries[i] for i, _ in top]


# ── Query decomposition (the "advanced RAG" piece tied to your survey) ────
def decompose_query(defect_type: str, confidence: float):
    """
    Instead of one flat retrieval query, split into sub-questions so retrieval
    covers both 'what is this defect' and 'what should be done about it'
    rather than relying on a single similarity match to cover everything.
    """
    return [
        f"What does the {defect_type} defect mean and what causes it?",
        f"What is the severity and recommended action for {defect_type}?",
        f"How should low-confidence predictions be handled?" if confidence < CONFIDENCE_THRESHOLD else None,
    ]


def retrieve_context(kb: QCKnowledgeBase, defect_type: str, confidence: float):
    sub_queries = [q for q in decompose_query(defect_type, confidence) if q]
    retrieved = []
    seen_ids = set()
    for q in sub_queries:
        results = kb.search(q, top_k=2, defect_type_filter=defect_type)
        for r in results:
            if r["id"] not in seen_ids:
                retrieved.append(r)
                seen_ids.add(r["id"])
    return retrieved


# ── Generation: turn retrieved context into a QC note ───────────────────────
def generate_qc_note(defect_type: str, confidence: float, retrieved_context: list):
    context_text = "\n".join(f"- {e['content']}" for e in retrieved_context)

    low_confidence_flag = confidence < CONFIDENCE_THRESHOLD

    prompt = f"""You are a garment quality-control assistant. A computer vision model
has classified a fabric image as follows:

Predicted defect: {defect_type}
Model confidence: {confidence:.1%}
{"NOTE: confidence is below the 85% threshold -- flag for human review." if low_confidence_flag else ""}

Relevant QC knowledge retrieved for this defect type:
{context_text}

Write a short, plain-language QC note (3-4 sentences) for a quality inspector. Include:
1. What the defect is
2. Its severity level
3. The recommended action
4. If confidence is below 85%, explicitly recommend human verification before acting

Keep it concise and professional, no markdown formatting."""

    model = genai.GenerativeModel(GEMINI_MODEL)
    response = model.generate_content(prompt)
    return response.text.strip()


# ── Full pipeline: classifier output -> QC note ────────────────────────────
def get_qc_note(defect_type: str, confidence: float, kb: QCKnowledgeBase):
    """
    Main entry point. Call this from your Streamlit app after the classifier
    predicts a defect_type and confidence score.
    """
    if defect_type == "defect free" and confidence >= CONFIDENCE_THRESHOLD:
        return {
            "defect_type": defect_type,
            "confidence": confidence,
            "note": "Fabric passed inspection with no defects detected. Proceed to next production stage.",
            "needs_human_review": False,
        }

    retrieved = retrieve_context(kb, defect_type, confidence)
    note = generate_qc_note(defect_type, confidence, retrieved)

    return {
        "defect_type": defect_type,
        "confidence": confidence,
        "note": note,
        "needs_human_review": confidence < CONFIDENCE_THRESHOLD,
        "sources": [e["id"] for e in retrieved],
    }


# ── Example usage ────────────────────────────────────────────────────────
if __name__ == "__main__":
    kb = QCKnowledgeBase(KB_PATH)

    # Simulate a classifier output (in your real pipeline, this comes from
    # the ResNet50 model's prediction + softmax confidence)
    test_cases = [
        ("hole", 0.975),
        ("stain", 0.79),      # below threshold -- should flag for human review
        ("broken stitch", 0.91),
    ]

    for defect_type, confidence in test_cases:
        result = get_qc_note(defect_type, confidence, kb)
        print(f"\n{'='*60}")
        print(f"Defect: {result['defect_type']} ({result['confidence']:.1%})")
        print(f"Needs human review: {result['needs_human_review']}")
        print(f"Note: {result['note']}")
        if "sources" in result:
            print(f"Sources used: {result['sources']}")
