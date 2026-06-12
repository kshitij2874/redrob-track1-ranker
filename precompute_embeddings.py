"""
Offline pre-computation step (NOT part of the 5-minute reproduce budget --
see submission_spec.docx Section on compute constraints, and README.md for
how this fits into the overall pipeline).

What it does:
  1. Streams candidates.jsonl once and builds the same "narrative text" per
     candidate that text_utils.candidate_narrative_text() would build live.
  2. Fits a TF-IDF vectorizer + TruncatedSVD (LSA) on the full 100K-candidate
     corpus. This is our "embedding model" -- chosen instead of a neural
     sentence-transformer because (a) it is CPU-only and trains in a couple
     of minutes on this corpus, (b) it has zero network dependency at
     inference time, (c) it is fully reproducible with a fixed random_state,
     and (d) for a recruiting-domain corpus where the JD and resumes share a
     lot of explicit vocabulary (skill names, tool names, role titles), a
     vocabulary-grounded LSA space is a strong, fast, and highly explainable
     semantic-similarity baseline.
  3. Projects the JD's core narrative (text_utils.jd_core_text()) into the
     same LSA space.
  4. L2-normalizes every vector so that cosine similarity == dot product at
     ranking time (cheap, no per-pair normalization needed in rank.py).
  5. Saves three artifacts to ./artifacts/:
       - candidate_embeddings.npz  (candidate_ids[100000], embeddings[100000 x N_COMPONENTS] float32)
       - jd_embedding.npy          (embeddings[N_COMPONENTS] float32)
       - vectorizer.joblib, svd.joblib  (the fitted models, for documentation /
         in case the JD text needs to be re-projected later)

Usage:
    python precompute_embeddings.py --candidates ./candidates.jsonl --out-dir ./artifacts

Runtime: ~2-4 minutes on a 4-core CPU laptop for 100K candidates, ~1.5-2GB peak RAM.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
from joblib import dump
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from text_utils import candidate_narrative_text, jd_core_text

N_COMPONENTS = 96
RANDOM_STATE = 42


def iter_candidates(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def iter_narrative_texts(path: str):
    """Single-pass generator -- TfidfVectorizer.fit_transform consumes this
    in one pass (it builds the vocabulary and the count matrix together), so
    we never hold all 100K narrative strings in memory at once."""
    for candidate in iter_candidates(path):
        yield candidate_narrative_text(candidate)


def iter_candidate_ids(path: str):
    for candidate in iter_candidates(path):
        yield candidate["candidate_id"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True, help="Path to candidates.jsonl")
    parser.add_argument("--out-dir", default="./artifacts", help="Where to write embedding artifacts")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    t0 = time.time()
    print("[1/4] Fitting TF-IDF vectorizer over candidate narrative texts (single pass)...")
    vectorizer = TfidfVectorizer(
        max_features=30000,
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.5,
        sublinear_tf=True,
        stop_words="english",
    )
    X = vectorizer.fit_transform(iter_narrative_texts(args.candidates))
    print(f"    TF-IDF matrix: {X.shape}, nnz={X.nnz:,} ({time.time()-t0:.1f}s)")

    print("[2/4] Fitting TruncatedSVD (LSA) ...")
    svd = TruncatedSVD(n_components=N_COMPONENTS, random_state=RANDOM_STATE)
    candidate_emb = svd.fit_transform(X)
    candidate_emb = normalize(candidate_emb, norm="l2", axis=1).astype(np.float32)
    explained = svd.explained_variance_ratio_.sum()
    print(f"    LSA embeddings: {candidate_emb.shape}, explained variance ~= {explained:.3f} "
          f"({time.time()-t0:.1f}s)")

    print("[3/4] Projecting JD core text into the same LSA space ...")
    jd_vec = vectorizer.transform([jd_core_text()])
    jd_emb = svd.transform(jd_vec)
    jd_emb = normalize(jd_emb, norm="l2", axis=1).astype(np.float32)[0]

    print("[4/4] Saving artifacts ...")
    candidate_ids = np.array(list(iter_candidate_ids(args.candidates)), dtype="<U12")
    np.savez_compressed(
        os.path.join(args.out_dir, "candidate_embeddings.npz"),
        candidate_ids=candidate_ids,
        embeddings=candidate_emb,
    )
    np.save(os.path.join(args.out_dir, "jd_embedding.npy"), jd_emb)
    dump(vectorizer, os.path.join(args.out_dir, "vectorizer.joblib"), compress=3)
    dump(svd, os.path.join(args.out_dir, "svd.joblib"), compress=3)

    # quick sanity check: top cosine-sim candidates against the JD
    sims = candidate_emb @ jd_emb
    top_idx = np.argsort(-sims)[:10]
    print("\nTop-10 candidates by raw semantic similarity to JD (sanity check only):")
    for i in top_idx:
        print(f"    {candidate_ids[i]}  sim={sims[i]:.3f}")

    print(f"\nDone in {time.time()-t0:.1f}s. Artifacts written to {args.out_dir}/")


if __name__ == "__main__":
    main()
