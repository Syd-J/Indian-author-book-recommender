"""
Encode books with sentence-transformers and build a FAISS index.

Cosine similarity is implemented as inner product on L2-normalized vectors,
so we use IndexFlatIP. Exact (not approximate) — fine up to ~50k items.

Run after data_fetcher.py:
    python build_index.py

Outputs (in data/):
    embeddings.npy   - (N, 384) float32, L2-normalized
    books.faiss      - FAISS index
    books.parquet    - the metadata, aligned 1:1 with the index rows
"""
from __future__ import annotations
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def make_doc_text(row: pd.Series) -> str:
    """One text blob per book — what we actually embed.

    Putting the title and author up front gives them extra weight under
    bag-of-features-y encoders. Description and subjects fill in the semantic
    body. Field labels ('Title:', 'Author:') are deliberate — MiniLM has seen
    enough structured text on the web that they help.
    """
    parts = [
        f"Title: {row['title']}",
        f"Author: {row['authors']}" if row["authors"] else "",
        f"Subjects: {row['subjects']}" if row["subjects"] else "",
        f"Description: {row['description']}" if row["description"] else "",
    ]
    return ". ".join(p for p in parts if p)


def main(csv_path: str = "data/books.csv", out_dir: str = "data"):
    out = Path(out_dir)
    out.mkdir(exist_ok=True)

    df = pd.read_csv(csv_path).fillna("")
    df["first_publish_year"] = pd.to_numeric(df["first_publish_year"], errors="coerce")
    df = df[df["title"].astype(str).str.len() > 0].reset_index(drop=True)
    print(f"Loaded {len(df)} books")

    texts = [make_doc_text(r) for _, r in df.iterrows()]

    print(f"Loading {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)

    print("Encoding...")
    emb = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype("float32")

    print(f"Embeddings: {emb.shape}, dtype={emb.dtype}")

    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    print(f"FAISS index size: {index.ntotal}")

    np.save(out / "embeddings.npy", emb)
    faiss.write_index(index, str(out / "books.faiss"))
    df.to_parquet(out / "books.parquet")
    print(f"Saved to {out_dir}/")


if __name__ == "__main__":
    main()