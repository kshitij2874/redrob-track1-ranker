"""
Small helper used by app.py (the sandbox demo) to embed an arbitrary, small
batch of candidates on the fly using the *already-fitted* TF-IDF + TruncatedSVD
models saved by precompute_embeddings.py.

This keeps the demo self-contained (no need to ship the 36MB full-corpus
candidate_embeddings.npz to the sandbox) while using the exact same vector
space the full-dataset ranking run uses, so semantic-similarity numbers from
the demo are directly comparable to rank.py's output.

Not used by rank.py itself, which loads the precomputed full-corpus
embeddings directly for speed.
"""

from __future__ import annotations

import os

import numpy as np
from joblib import load
from sklearn.preprocessing import normalize

from text_utils import candidate_narrative_text, jd_core_text


def load_models(artifacts_dir: str):
    vectorizer = load(os.path.join(artifacts_dir, "vectorizer.joblib"))
    svd = load(os.path.join(artifacts_dir, "svd.joblib"))
    return vectorizer, svd


def jd_embedding(vectorizer, svd) -> np.ndarray:
    vec = vectorizer.transform([jd_core_text()])
    emb = svd.transform(vec)
    return normalize(emb, norm="l2", axis=1).astype(np.float32)[0]


def embed_candidates(candidates: list[dict], vectorizer, svd) -> np.ndarray:
    texts = [candidate_narrative_text(c) for c in candidates]
    vecs = vectorizer.transform(texts)
    emb = svd.transform(vecs)
    return normalize(emb, norm="l2", axis=1).astype(np.float32)
