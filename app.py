"""
Sandbox demo (Gradio) for the Redrob Track 1 candidate ranker.

Per submission_spec.docx Section 10.5, this sandbox:
  - accepts a small candidate sample (<=100 candidates) via upload, or uses
    the bundled sample_candidates_100.jsonl (first 100 rows of the official
    candidates.jsonl) if nothing is uploaded
  - runs the ranking system end-to-end and produces a ranked CSV
  - completes well within the 5-minute / CPU-only / no-network compute budget
    (a 100-candidate run takes well under 5 seconds)

It reuses the *exact same* scoring code as rank.py (score_candidate,
generate_reasoning, jd_profile.json) -- embeddings are computed on the fly by
fitting a TF-IDF + TruncatedSVD model from the bundled 100-candidate sample at
startup (no pre-built binary artifacts required for the Space).

Run locally:
    python app.py
Deploy: push this file + jd_profile.json + features.py + text_utils.py +
embed.py + rank.py + sample_candidates_100.jsonl + requirements.txt to a
HuggingFace Space (SDK: gradio). No artifacts/ directory needed.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import date

import gradio as gr
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from embed import embed_candidates, jd_embedding
from rank import generate_reasoning, score_candidate
from text_utils import candidate_narrative_text, jd_core_text

HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLE_PATH = os.path.join(HERE, "sample_candidates_100.jsonl")

with open(os.path.join(HERE, "jd_profile.json"), "r", encoding="utf-8") as f:
    JD_PROFILE = json.load(f)
REFERENCE_DATE = date.fromisoformat(JD_PROFILE["behavioral_signals"]["reference_date"])


def _load_jsonl(path_or_file) -> list[dict]:
    candidates = []
    with open(path_or_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                candidates.append(json.loads(line))
    return candidates


def _build_models():
    """Fit TF-IDF + TruncatedSVD from the bundled sample at startup."""
    candidates = _load_jsonl(SAMPLE_PATH)
    texts = [candidate_narrative_text(c) for c in candidates] + [jd_core_text()]
    vectorizer = TfidfVectorizer(max_features=8000, sublinear_tf=True, min_df=1)
    matrix = vectorizer.fit_transform(texts)
    n_components = min(100, len(texts) - 1)
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    svd.fit(matrix)
    return vectorizer, svd


VECTORIZER, SVD = _build_models()
JD_EMB = jd_embedding(VECTORIZER, SVD)


def run_ranking(uploaded_file):
    if uploaded_file is not None:
        candidates = _load_jsonl(uploaded_file.name)
    else:
        candidates = _load_jsonl(SAMPLE_PATH)

    if len(candidates) > 100:
        candidates = candidates[:100]

    embs = embed_candidates(candidates, VECTORIZER, SVD)
    id_to_idx = {c["candidate_id"]: i for i, c in enumerate(candidates)}

    scored = []
    for c in candidates:
        final_score, feats = score_candidate(c, JD_PROFILE, id_to_idx, embs, JD_EMB, REFERENCE_DATE)
        scored.append((final_score, c["candidate_id"], feats))

    scored.sort(key=lambda x: (-x[0], x[1]))

    rows = []
    for rank, (final_score, cid, feats) in enumerate(scored, start=1):
        reasoning = generate_reasoning(feats, JD_PROFILE)
        rows.append([cid, rank, round(final_score, 4), reasoning])

    out_path = os.path.join(tempfile.gettempdir(), "submission_demo.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        import csv
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        writer.writerows(rows)

    headers = ["candidate_id", "rank", "score", "reasoning"]
    return rows, out_path


with gr.Blocks(title="Redrob Track 1 - Candidate Ranker (Sandbox)") as demo:
    gr.Markdown(
        "# Redrob Track 1 -- Intelligent Candidate Discovery & Ranking (Sandbox)\n"
        "Upload a `.jsonl` file of up to 100 candidates in the "
        "`candidate_schema.json` format, or click **Run** to rank the "
        "bundled 100-candidate sample (`sample_candidates_100.jsonl`, the "
        "first 100 rows of the official dataset).\n\n"
        "This sandbox runs the same `score_candidate` / `generate_reasoning` "
        "code as `rank.py`, with embeddings computed on the fly from the "
        "saved TF-IDF + TruncatedSVD models. Full-dataset (100K) runs use "
        "`rank.py` directly with precomputed embeddings -- see the README."
    )

    with gr.Row():
        file_in = gr.File(label="Candidates JSONL (optional, max 100 rows)", file_types=[".jsonl"])
        run_btn = gr.Button("Run ranking", variant="primary")

    table = gr.Dataframe(
        headers=["candidate_id", "rank", "score", "reasoning"],
        label="Ranked candidates",
        wrap=True,
    )
    download = gr.File(label="Download submission.csv")

    run_btn.click(fn=run_ranking, inputs=[file_in], outputs=[table, download])

if __name__ == "__main__":
    demo.launch()
