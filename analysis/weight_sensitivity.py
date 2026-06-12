"""
Weight sensitivity analysis for rank.py's base_score formula.

base_score = 100 * ( w_semantic * semantic_score
                    + w_skill    * skill_match
                    + w_exp      * experience_fit
                    + w_title    * title_score
                    + w_location * location_score
                    + w_notice   * notice_score )

This script does a single streaming pass over candidates.jsonl to compute the
six raw component scores (plus disqualifier_multiplier, behavioral_multiplier,
honeypot flag) for every candidate -- the *expensive* part -- and caches them
as numpy arrays. It then re-scores the entire dataset under several alternative
weight vectors using pure numpy (near-instant), and reports for each variant:

  - Jaccard overlap of the top-100 candidate set vs. the baseline weights
  - Spearman rank correlation (over the union of both top-100 sets) vs. baseline
  - How many candidates newly enter / drop out of the top-100

This is meant to answer: "how sensitive is the final top-100 to the exact
values of the 0.35/0.30/0.15/0.10/0.07/0.03 weights?" -- i.e. is the ranking
robust to reasonable re-calibration, or is it riding on one fragile knob.

Usage:
    python analysis/weight_sensitivity.py --candidates ./candidates.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import date

import numpy as np

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from features import extract_features  # noqa: E402
from rank import load_embeddings, load_jd_profile  # noqa: E402

BASELINE = {
    "semantic": 0.35, "skill": 0.30, "experience": 0.15,
    "title": 0.10, "location": 0.07, "notice": 0.03,
}

VARIANTS = {
    "baseline": BASELINE,
    "semantic_heavy": {"semantic": 0.50, "skill": 0.20, "experience": 0.12, "title": 0.08, "location": 0.06, "notice": 0.04},
    "skill_heavy":    {"semantic": 0.20, "skill": 0.45, "experience": 0.15, "title": 0.10, "location": 0.07, "notice": 0.03},
    "experience_heavy": {"semantic": 0.30, "skill": 0.25, "experience": 0.25, "title": 0.10, "location": 0.07, "notice": 0.03},
    "title_minimized": {"semantic": 0.40, "skill": 0.32, "experience": 0.16, "title": 0.02, "location": 0.07, "notice": 0.03},
    "more_uniform":   {"semantic": 0.25, "skill": 0.25, "experience": 0.20, "title": 0.15, "location": 0.10, "notice": 0.05},
}


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = a.argsort().argsort().astype(float)
    rb = b.argsort().argsort().astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = np.sqrt((ra**2).sum() * (rb**2).sum())
    if denom == 0:
        return float("nan")
    return float((ra * rb).sum() / denom)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--jd-profile", default=os.path.join(os.path.dirname(__file__), "..", "jd_profile.json"))
    parser.add_argument("--artifacts-dir", default=os.path.join(os.path.dirname(__file__), "..", "artifacts"))
    parser.add_argument("--top-n", type=int, default=100)
    args = parser.parse_args()

    jd_profile = load_jd_profile(args.jd_profile)
    reference_date = date.fromisoformat(jd_profile["behavioral_signals"]["reference_date"])

    id_to_idx, embs, jd_emb = load_embeddings(args.artifacts_dir)

    t0 = time.time()
    cids, semantic, skill, exp, title, loc, notice, disq, behav, honeypot = ([] for _ in range(10))

    with open(args.candidates, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            c = json.loads(line)
            feats = extract_features(c, jd_profile, reference_date)
            cid = c["candidate_id"]
            idx = id_to_idx.get(cid)
            raw_sim = float(embs[idx] @ jd_emb) if idx is not None else 0.0

            cids.append(cid)
            semantic.append(max(0.0, raw_sim))
            skill.append(feats["skill_match"]["score"])
            exp.append(feats["experience_fit"])
            title.append(feats["title_score"])
            loc.append(feats["location_score"])
            notice.append(feats["notice_score"])
            disq.append(feats["disqualifier_multiplier"])
            behav.append(feats["behavioral_multiplier"])
            honeypot.append(0.02 if feats["is_honeypot"] else 1.0)

    print(f"Computed components for {len(cids):,} candidates in {time.time()-t0:.1f}s")

    cids = np.array(cids)
    components = {
        "semantic": np.array(semantic), "skill": np.array(skill), "experience": np.array(exp),
        "title": np.array(title), "location": np.array(loc), "notice": np.array(notice),
    }
    mults = np.array(disq) * np.array(behav) * np.array(honeypot)

    def score_for(weights):
        base = sum(weights[k] * components[k] for k in components) * 100.0
        return base * mults

    baseline_scores = score_for(BASELINE)
    baseline_order = np.argsort(-baseline_scores, kind="stable")
    baseline_top = set(cids[baseline_order[:args.top_n]])

    print(f"\n{'variant':<18} {'top-100 Jaccard':>16} {'Spearman (union)':>18} {'in/out vs baseline':>20}")
    for name, w in VARIANTS.items():
        scores = score_for(w)
        order = np.argsort(-scores, kind="stable")
        top = set(cids[order[:args.top_n]])

        jaccard = len(top & baseline_top) / len(top | baseline_top)
        union_idx = [i for i, cid in enumerate(cids) if cid in (top | baseline_top)]
        sp = spearman(baseline_scores[union_idx], scores[union_idx])

        new_in = len(top - baseline_top)
        dropped = len(baseline_top - top)
        print(f"{name:<18} {jaccard:>16.3f} {sp:>18.3f} {f'+{new_in}/-{dropped}':>20}")


if __name__ == "__main__":
    main()
