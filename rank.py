"""
rank.py -- Track 1 "Intelligent Candidate Discovery & Ranking" submission.

Single reproduce command (per submission_spec.docx):

    python rank.py --candidates ./candidates.jsonl --out ./submission.csv

Reads the 100K-row candidates.jsonl, scores every candidate against the
Senior AI Engineer JD (encoded in jd_profile.json), and writes the top-100
ranked candidates to a CSV with columns: candidate_id, rank, score, reasoning.

Design summary (full detail in README.md):

  final_score = base_score * disqualifier_multiplier * behavioral_multiplier

  base_score (0-100) = 100 * (
        0.35 * semantic_similarity      (LSA cosine sim vs JD, precomputed)
      + 0.30 * skill_match              (anti-stuffing weighted category coverage)
      + 0.15 * experience_fit           (years_of_experience vs JD's 5-9yr band)
      + 0.10 * title_relevance          (current_title vs JD title taxonomy)
      + 0.07 * location_score           (target-city / India / relocation)
      + 0.03 * notice_score             (notice_period_days vs JD preference)
  )

  disqualifier_multiplier  in (0, 1] -- soft penalties for JD red flags
                                          (pure-consulting-only, research-only,
                                          CV/speech-only, architecture-only,
                                          title-chasing, LangChain-only AI exp)
  behavioral_multiplier    in [0.55, 1.12] -- engagement/availability signal
                                              ("perfect on paper but inactive")

  Honeypot-suspect candidates (per honeypot_heuristics) get an additional
  0.02x multiplier, forcing them to the bottom regardless of how good their
  listed skills look.

Compute profile: streaming, single pass over candidates.jsonl, O(1) extra
memory per candidate plus a fixed-size (100-item) heap for the running
top-100. Precomputed semantic-similarity embeddings (artifacts/) are loaded
once (~36MB). On a 4-core / 16GB CPU-only machine this runs in well under a
minute for 100K candidates -- comfortably inside the 5-minute budget.
"""

from __future__ import annotations

import argparse
import csv
import heapq
import json
import os
import time
from datetime import date

import numpy as np

from features import extract_features
from text_utils import jd_core_text  # noqa: F401  (kept for documentation parity)

WEIGHTS = {
    "semantic": 0.35,
    "skill": 0.30,
    "experience": 0.15,
    "title": 0.10,
    "location": 0.07,
    "notice": 0.03,
}

HONEYPOT_MULTIPLIER = 0.02


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_jd_profile(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_embeddings(artifacts_dir: str):
    npz = np.load(os.path.join(artifacts_dir, "candidate_embeddings.npz"))
    ids = npz["candidate_ids"]
    embs = npz["embeddings"]
    id_to_idx = {cid: i for i, cid in enumerate(ids)}
    jd_emb = np.load(os.path.join(artifacts_dir, "jd_embedding.npy"))
    return id_to_idx, embs, jd_emb


def iter_candidates(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------

def score_candidate(candidate: dict, jd_profile: dict, id_to_idx, embs, jd_emb,
                     reference_date: date) -> tuple[float, dict]:
    feats = extract_features(candidate, jd_profile, reference_date)

    idx = id_to_idx.get(candidate["candidate_id"])
    raw_sim = float(embs[idx] @ jd_emb) if idx is not None else 0.0
    semantic_score = max(0.0, raw_sim)  # negative cosine sims -> "no relevant signal"

    base = 100.0 * (
        WEIGHTS["semantic"] * semantic_score
        + WEIGHTS["skill"] * feats["skill_match"]["score"]
        + WEIGHTS["experience"] * feats["experience_fit"]
        + WEIGHTS["title"] * feats["title_score"]
        + WEIGHTS["location"] * feats["location_score"]
        + WEIGHTS["notice"] * feats["notice_score"]
    )

    final = base * feats["disqualifier_multiplier"] * feats["behavioral_multiplier"]
    if feats["is_honeypot"]:
        final *= HONEYPOT_MULTIPLIER

    feats["raw_semantic_sim"] = raw_sim
    feats["semantic_score"] = semantic_score
    feats["base_score"] = base
    return final, feats


# ---------------------------------------------------------------------------
# reasoning generation -- every clause below is sourced directly from a field
# in `feats` (which is itself sourced directly from the candidate record), so
# nothing here is hallucinated.
# ---------------------------------------------------------------------------

def _experience_clause(feats: dict, jd_profile: dict) -> str:
    band = jd_profile["experience_band"]
    yoe = feats["years_of_experience"]
    if band["ideal_min"] <= yoe <= band["ideal_max"]:
        return f"{yoe:g} yrs experience (squarely in JD's {band['ideal_min']}-{band['ideal_max']}yr ideal band)"
    if band["acceptable_min"] <= yoe <= band["acceptable_max"]:
        return f"{yoe:g} yrs experience (within JD's {band['acceptable_min']}-{band['acceptable_max']}yr acceptable range)"
    return f"{yoe:g} yrs experience (outside JD's {band['acceptable_min']}-{band['acceptable_max']}yr range)"


def _title_clause(feats: dict) -> str:
    tier = feats["title_tier"]
    title = feats["current_title"]
    company = feats["current_company"]
    if tier == "high":
        return f"current title '{title}' at {company} closely matches the target role"
    if tier == "medium":
        return f"current title '{title}' at {company} is adjacent to the target role"
    return f"current title '{title}' at {company} does not directly match the target AI/ML titles, but is evaluated on career-history content"


def _skill_clause(feats: dict) -> str:
    sm = feats["skill_match"]
    req = sm["matched_required_skills"]
    nice = sm["matched_nice_skills"]
    verified = sm.get("assessment_verified_skills", [])
    parts = []
    if req:
        parts.append("substantiated (endorsed/long-duration) match on required skills: " + ", ".join(req[:6]))
    else:
        parts.append("no substantiated match on the JD's required embeddings/retrieval, vector-DB, or eval-framework skills")
    if nice:
        parts.append("plus nice-to-have skills: " + ", ".join(nice[:4]))
    if verified:
        parts.append("platform skill assessment corroborates: " + ", ".join(verified[:4]))
    return "; ".join(parts)


def _location_clause(feats: dict, jd_profile: dict) -> str:
    loc = feats["location"]
    if feats["in_target_city"]:
        return f"based in {loc} (one of the JD's target cities)"
    return f"based in {loc}"


def _notice_clause(feats: dict, jd_profile: dict) -> str:
    days = feats["notice_period_days"]
    cfg = jd_profile["notice_period"]
    if days <= cfg["ideal_max_days"]:
        return f"{days}-day notice period (within JD's preferred sub-{cfg['ideal_max_days']}-day window)"
    if days <= cfg["acceptable_max_days"]:
        return f"{days}-day notice period (coverable via buyout per JD)"
    return f"{days}-day notice period (longer than JD prefers)"


def _behavioral_clause(feats: dict) -> str:
    bf = feats["behavioral_facts"]
    days_inactive = bf["days_inactive"]
    rr = bf["recruiter_response_rate"]
    if days_inactive is None:
        recency = "no last-active date on record"
    elif days_inactive <= 60:
        recency = f"active recently (last active {days_inactive} days ago)"
    elif days_inactive <= 180:
        recency = f"semi-active (last active {days_inactive} days ago)"
    else:
        recency = f"inactive for a long time (last active {days_inactive} days ago)"
    return f"{recency}, {rr*100:.0f}% recruiter response rate"


def generate_reasoning(feats: dict, jd_profile: dict) -> str:
    if feats["is_honeypot"]:
        return (
            "HONEYPOT SUSPECT -- " + "; ".join(feats["honeypot_reasons"])
            + ". Score forced near zero despite any surface-level keyword matches."
        )

    clauses = [
        _experience_clause(feats, jd_profile),
        _title_clause(feats),
        _skill_clause(feats),
        _location_clause(feats, jd_profile),
        _notice_clause(feats, jd_profile),
        _behavioral_clause(feats),
        f"semantic similarity to JD narrative: {feats['raw_semantic_sim']:.2f}",
    ]
    if feats["concerns"]:
        clauses.append("Concerns: " + "; ".join(feats["concerns"]))

    return ". ".join(c[0].upper() + c[1:] for c in clauses) + "."


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True, help="Path to candidates.jsonl")
    parser.add_argument("--out", required=True, help="Path to write submission.csv")
    parser.add_argument("--jd-profile", default=os.path.join(os.path.dirname(__file__), "jd_profile.json"))
    parser.add_argument("--artifacts-dir", default=os.path.join(os.path.dirname(__file__), "artifacts"))
    parser.add_argument("--reference-date", default=None,
                         help="YYYY-MM-DD reference date for recency scoring "
                              "(default: jd_profile.behavioral_signals.reference_date, "
                              "for fully deterministic/reproducible output)")
    parser.add_argument("--top-n", type=int, default=100)
    args = parser.parse_args()

    t0 = time.time()
    jd_profile = load_jd_profile(args.jd_profile)

    if args.reference_date:
        reference_date = date.fromisoformat(args.reference_date)
    else:
        reference_date = date.fromisoformat(jd_profile["behavioral_signals"]["reference_date"])

    print(f"Loading precomputed embeddings from {args.artifacts_dir} ...")
    id_to_idx, embs, jd_emb = load_embeddings(args.artifacts_dir)
    print(f"  {embs.shape[0]:,} candidate embeddings loaded ({embs.shape[1]} dims)")

    print(f"Scoring candidates from {args.candidates} (streaming, top-{args.top_n}) ...")

    # Min-heap of (final_score, -candidate_id_num, candidate_id, feats).
    # heap[0] is always the *worst* kept candidate: lowest score, and among
    # equal scores, the one with the largest candidate_id (since -id_num is
    # smallest for the largest id). This makes "pop the worst when a better
    # candidate comes along" and "is `item` better than the current worst"
    # both simple tuple comparisons, while the final ascending-candidate_id
    # tie-break (required by validate_submission.py) falls out of the
    # explicit sort below.
    heap: list[tuple[float, int, str, dict]] = []
    n_seen = 0
    n_honeypots = 0

    for candidate in iter_candidates(args.candidates):
        n_seen += 1
        final_score, feats = score_candidate(candidate, jd_profile, id_to_idx, embs, jd_emb, reference_date)
        if feats["is_honeypot"]:
            n_honeypots += 1

        cid = candidate["candidate_id"]
        cid_num = int(cid.split("_")[1])
        item = (final_score, -cid_num, cid, feats)
        if len(heap) < args.top_n:
            heapq.heappush(heap, item)
        else:
            if item > heap[0]:
                heapq.heapreplace(heap, item)

    print(f"  scored {n_seen:,} candidates ({n_honeypots} honeypot-suspects flagged)")

    # Final ordering: score descending, candidate_id ascending for ties.
    ranked = sorted(heap, key=lambda x: (-x[0], x[2]))

    print(f"Writing top-{len(ranked)} to {args.out} ...")
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for rank, (final_score, _, cid, feats) in enumerate(ranked, start=1):
            reasoning = generate_reasoning(feats, jd_profile)
            writer.writerow([cid, rank, f"{final_score:.4f}", reasoning])

    print(f"Done in {time.time()-t0:.1f}s.")


if __name__ == "__main__":
    main()
