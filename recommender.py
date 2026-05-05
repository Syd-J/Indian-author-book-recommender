"""
BookRecommender — load the index and answer queries.

Two justification modes:
  - justify(): template-based, no API key needed. Default.
  - justify_llm(): uses an LLM client you pass in. Off by default.
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Minimal stoplist for keyword overlap in template justifications.
_STOP = set(
    "a an the and or but if while of in on at to for with from by is are "
    "was were be been being have has had do does did this that these those "
    "i you he she it we they my your his her our their about as into over "
    "under up down out so than then very can just also more most some any "
    "all no not me him us them what which who whose why how when where "
    "book books novel novels story stories author authors read reading want "
    "looking like love feel feels something".split()
)


class BookRecommender:
    def __init__(self, data_dir: str = "data"):
        d = Path(data_dir)
        self.df = pd.read_parquet(d / "books.parquet").fillna("")
        self.index = faiss.read_index(str(d / "books.faiss"))
        self.model = SentenceTransformer(MODEL_NAME)

    # --- core search ---

    def search(
        self,
        query: str,
        k: int = 10,
        author_filter: Optional[str] = None,
        min_score: float = 0.0,
    ) -> List[Dict]:
        """Return top-k hits as a list of dicts."""
        q = self.model.encode(
            [query], normalize_embeddings=True, convert_to_numpy=True
        ).astype("float32")

        # Over-fetch when we're going to filter post-hoc.
        fetch_k = k * 8 if author_filter else k * 2
        scores, idxs = self.index.search(q, fetch_k)

        results: List[Dict] = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx < 0 or score < min_score:
                continue
            row = self.df.iloc[int(idx)]
            if author_filter and author_filter.lower() not in str(row["authors"]).lower():
                continue
            results.append({
                "score": float(score),
                "title": str(row["title"]),
                "authors": str(row["authors"]),
                "year": str(row["first_publish_year"]),
                "subjects": str(row["subjects"]),
                "description": str(row["description"]),
                "cover_url": str(row["cover_url"]),
                "work_key": str(row["work_key"]),
            })
            if len(results) >= k:
                break
        return results

    # --- justification ---

    def justify(self, query: str, book: Dict) -> str:
        """Template 'why?' — keyword overlap + subjects + score.

        No API key required. Good enough for the demo and for the report's
        ablation (template vs LLM).
        """
        q_words = {w.lower().strip(".,!?;:'\"()") for w in query.split()}
        q_words = {w for w in q_words if len(w) > 2 and w not in _STOP}

        haystack = f"{book['title']} {book['subjects']} {book['description']}".lower()
        overlap = sorted(w for w in q_words if w in haystack)

        bits: List[str] = []
        if overlap:
            bits.append(f"matches your interest in **{', '.join(overlap[:5])}**")

        subj_list = [s.strip() for s in book["subjects"].split(";") if s.strip()][:3]
        if subj_list:
            bits.append(f"tagged *{', '.join(subj_list)}*")

        if book["authors"]:
            bits.append(f"by {book['authors']}")

        bits.append(f"similarity {book['score']:.3f}")

        return "Recommended because it " + "; ".join(bits) + "."

    def justify_llm(self, query: str, book: Dict, client=None, model: str = "claude-haiku-4-5-20251001") -> str:
        """Optional LLM justification.

        Pass any Anthropic-compatible client:
            from anthropic import Anthropic
            rec.justify_llm(query, book, client=Anthropic())
        """
        if client is None:
            return self.justify(query, book)

        prompt = (
            f"User query: {query}\n\n"
            f"Book: {book['title']} by {book['authors']}\n"
            f"Subjects: {book['subjects'][:300]}\n"
            f"Description: {book['description'][:600]}\n\n"
            "In ONE sentence under 30 words, explain why this book matches the user's "
            "request. Be specific. Don't restate title or author."
        )
        try:
            msg = client.messages.create(
                model=model,
                max_tokens=80,
                messages=[{"role": "user", "content": prompt}],
            )
            return msg.content[0].text.strip()
        except Exception as e:
            return self.justify(query, book) + f"  _(llm fallback: {e})_"