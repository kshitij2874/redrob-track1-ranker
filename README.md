# Redrob AI -- Track 1: Intelligent Candidate Discovery & Ranking

A CPU-only, no-network, explainable ranking pipeline that scores 100,000
candidate profiles against the "Senior AI Engineer -- Founding Team" job
description and produces the top-100 ranked CSV required by
`submission_spec.docx`.

## TL;DR -- reproduce

```bash
pip install -r requirements.txt   # numpy is the only runtime dependency for rank.py
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
```

Runtime on a 4-core CPU / <4GB RAM box: **~3 seconds** for all 100,000
candidates (well inside the 5-minute / 16GB / CPU-only / no-network budget).
`rank.py` does a single streaming pass over `candidates.jsonl` and loads a
36MB precomputed embeddings file from `artifacts/`.

## How the JD was read

`job_description.docx` (Senior AI Engineer, Founding Team, Redrob AI) was
distilled into [`jd_profile.json`](jd_profile.json) -- a structured, checkable
rubric. Every threshold in that file is annotated with the JD sentence it
came from, so the scoring logic is traceable back to the original document.
Highlights:

- **Experience band**: 5-9 years acceptable, 6-8 ideal, soft falloff outside
  that range (JD: *"we'll seriously consider candidates outside the band if
  other signals are strong"* -- never a hard cutoff).
- **Title relevance is one signal, not a filter**: the JD explicitly calls
  out a "Tier 5" candidate with no AI title but strong production career
  history, who must score well via the semantic-similarity component.
- **Explicit JD red flags** become *soft* multiplicative penalties, not hard
  rejects: pure-IT-services-consulting careers, research-only backgrounds
  with no production evidence, CV/speech/robotics-only experience without
  NLP/IR, long-tenured "architecture-only" titles, title-chasing across short
  stints, and "LangChain-calls-OpenAI for <12 months" as the *only* AI
  experience.
- **Location**: Pune/Noida (preferred), Hyderabad/Mumbai/Delhi-NCR/Bangalore
  (welcome), other India (lower), outside India (lowest, with a relocation
  bonus -- JD: *"we don't sponsor work visas"*).
- **Notice period**: sub-30-day preferred, up to 60 days coverable via
  buyout, longer is in-scope but raises the bar.
- **Behavioral signals**: the JD's closing note -- *"a perfect-on-paper
  candidate who hasn't logged in for 6 months and has a 5% recruiter
  response rate is, for hiring purposes, not actually available"* -- is
  implemented as a multiplier in `[0.55, 1.12]` driven by `last_active_date`,
  `recruiter_response_rate`, `interview_completion_rate`,
  `offer_acceptance_rate`, `open_to_work_flag`, and
  `profile_completeness_score` from `redrob_signals`.

## Scoring formula

```
final_score = base_score(0-100) * disqualifier_multiplier * behavioral_multiplier
                                  * (0.02 if honeypot_suspect else 1)

base_score = 100 * ( 0.35 * semantic_similarity
                    + 0.30 * skill_match
                    + 0.15 * experience_fit
                    + 0.10 * title_relevance
                    + 0.07 * location_score
                    + 0.03 * notice_score )
```

| Component | What it measures | Where it lives |
|---|---|---|
| `semantic_similarity` | Cosine similarity (LSA / TF-IDF + TruncatedSVD) between the candidate's full narrative (headline, summary, every career_history description, skill list) and a hand-written narrative of the JD's core requirements | `precompute_embeddings.py`, `text_utils.py` |
| `skill_match` | Anti-stuffing-weighted coverage of the JD's required skill categories (embeddings/retrieval, vector-DB/hybrid-search, Python, eval frameworks) plus nice-to-haves | `features.py: skill_match_score` |
| `experience_fit` | `years_of_experience` vs the 5-9yr / 6-8yr ideal band, soft falloff outside | `features.py: experience_fit_score` |
| `title_relevance` | `current_title` vs a 3-tier title taxonomy from the JD | `features.py: title_relevance_score` |
| `location_score` | `location`/`country`/`willing_to_relocate` vs the JD's city preferences | `features.py: location_score` |
| `notice_score` | `notice_period_days` vs the JD's preference | `features.py: notice_score` |
| `disqualifier_multiplier` | Soft penalties for the 6 explicit JD red flags listed above | `features.py: detect_disqualifiers` |
| `behavioral_multiplier` | Engagement/availability signal from `redrob_signals` | `features.py: behavioral_multiplier` |
| `honeypot_suspect` | Internal-consistency red flags (see below) | `features.py: is_honeypot` |

### Why TF-IDF + TruncatedSVD instead of a neural embedding model

The "embeddings" component intentionally uses TF-IDF + TruncatedSVD (LSA)
rather than `sentence-transformers`/`torch`:

1. **Compute constraints** -- the ranking step must run CPU-only, with no
   network access, in under 5 minutes and 16GB RAM. LSA fits this trivially
   (the entire 100K-candidate embedding precompute takes ~20 seconds on 4
   CPU cores).
2. **Reproducibility** -- a fixed `random_state` and a single `fit_transform`
   pass make the embeddings byte-for-byte reproducible without downloading
   any model weights.
3. **Domain fit** -- this corpus and the JD share a large, specific
   vocabulary (tool names, skill names, role titles: "Pinecone", "BM25",
   "Sentence Transformers", "Senior AI Engineer", ...). A vocabulary-grounded
   LSA space captures this well, and the JD itself praises candidates who
   "understood retrieval and ranking before it became fashionable" -- TF-IDF
   + LSA *is* a hybrid retrieval baseline, which fits the narrative.
4. **Explainability** -- every other component of `base_score` is a literal,
   checkable fact about the candidate. Keeping the embedding model simple and
   local keeps the *entire* pipeline auditable end-to-end, which directly
   serves the "Explainability & Data Validation" judging criterion.

`precompute_embeddings.py` is the offline step (not counted against the
5-minute budget, per submission_spec.docx): it streams `candidates.jsonl`
once, fits TF-IDF (max 30K features, uni+bigrams, `min_df=3`, `max_df=0.5`,
sublinear TF) + TruncatedSVD (96 components, ~65% explained variance),
projects the JD's core narrative into the same space, L2-normalizes
everything, and writes:

- `artifacts/candidate_embeddings.npz` (candidate_id -> 96-dim vector, 36MB)
- `artifacts/jd_embedding.npy`
- `artifacts/vectorizer.joblib`, `artifacts/svd.joblib` (the fitted models,
  used by the sandbox demo to embed new/uploaded candidates on the fly)

To regenerate:
```bash
python precompute_embeddings.py --candidates ./candidates.jsonl --out-dir ./artifacts
```

### Upgrade path: neural (sentence-transformer) embeddings

This sandbox has no internet access, so the embedding model can't be
downloaded here -- but if you have network access on your own machine, the
TF-IDF/LSA embedding step can be swapped for a small CPU-friendly
sentence-transformer (e.g. `all-MiniLM-L6-v2`, ~80MB, 384-dim) **without
violating the no-network ranking constraint**, because the model only needs
network access *once*, at build time:

1. **One-time, on a machine with internet**: download and save the model
   locally so it never needs to hit the network again:
   ```python
   from sentence_transformers import SentenceTransformer
   model = SentenceTransformer("all-MiniLM-L6-v2")
   model.save("artifacts/minilm")  # ~80MB, commit to the repo (git-lfs if needed)
   ```
2. **`precompute_embeddings.py`** loads `artifacts/minilm` from disk (no
   network) and calls `model.encode(candidate_narrative_text(c), batch_size=64)`
   for all 100K candidates instead of `vectorizer.transform` + `svd.transform`.
   On 4 CPU cores, MiniLM throughput is roughly 200-400 short texts/sec, so
   100K candidates take on the order of 5-10 minutes -- still fine as an
   *offline* pre-computation step (not counted against the 5-minute ranking
   budget), though notably slower than the ~20s LSA fit.
3. **`rank.py` / `app.py`** load `artifacts/minilm` from disk at runtime
   (`SentenceTransformer("artifacts/minilm")`, `local_files_only=True`) and
   use the same precomputed `.npz` lookup pattern -- no code changes beyond
   `embed.py`/`precompute_embeddings.py`.
4. **Trade-off**: adds `sentence-transformers` + a CPU-only `torch` wheel to
   `requirements.txt` (several hundred MB), versus the current
   numpy-only runtime footprint. Given that, a **hybrid** approach is likely
   the best ROI: keep TF-IDF/LSA for the vocabulary-specific tool/skill-name
   matching it's good at, and blend in MiniLM cosine similarity (e.g.
   `0.5 * lsa_sim + 0.5 * minilm_sim`) specifically to catch "Tier 5
   plain-language" candidates whose career-history descriptions are
   semantically relevant but don't share the JD's exact vocabulary -- the
   case TF-IDF is weakest on. We're not implementing this here because we
   can't validate it without internet access to download and test the model,
   and an unvalidated change to the core similarity signal is a bigger risk
   than the LSA baseline, which is fully reproducible and already validated
   end-to-end.

### Weight calibration & sensitivity analysis

The `base_score` weights (0.35 semantic / 0.30 skill / 0.15 experience / 0.10
title / 0.07 location / 0.03 notice) were initially set by hand from the JD's
emphasis (semantic similarity and skill match dominate; title is explicitly
"one signal, not a filter" per the JD's Tier-5 example). `analysis/weight_sensitivity.py`
checks how sensitive the resulting top-100 is to these exact values: it
computes all six raw component scores once for all 100K candidates, then
re-scores the whole dataset under several alternative weight vectors using
pure numpy, and reports the top-100 Jaccard overlap and Spearman rank
correlation (over the union of both top-100 sets) against the baseline.

```bash
python analysis/weight_sensitivity.py --candidates ./candidates.jsonl
```

| Variant | top-100 Jaccard | Spearman (union) | Change |
|---|---|---|---|
| baseline (0.35/0.30/0.15/0.10/0.07/0.03) | 1.000 | 1.000 | -- |
| semantic_heavy (0.50/0.20/0.12/0.08/0.06/0.04) | 0.786 | 0.893 | +12/-12 |
| skill_heavy (0.20/0.45/0.15/0.10/0.07/0.03) | 0.835 | 0.855 | +9/-9 |
| experience_heavy (0.30/0.25/0.25/0.10/0.07/0.03) | 0.887 | 0.946 | +6/-6 |
| title_minimized (0.40/0.32/0.16/0.02/0.07/0.03) | 0.961 | 0.986 | +2/-2 |
| more_uniform (0.25/0.25/0.20/0.15/0.10/0.05) | 0.869 | 0.906 | +7/-7 |

Takeaways:
- The ranking is **most sensitive to the semantic/skill split** (the two
  largest weights, 65% combined) -- reallocating between them moves ~9-12 of
  the top-100. This matches expectation: these are the two components that
  most directly encode "does this person actually do the job."
- The ranking is **least sensitive to the title weight** (Jaccard 0.961 even
  when cut from 0.10 to 0.02) -- empirical confirmation that the JD's "title
  is one signal, not a filter" framing is correctly implemented; title isn't
  doing hidden heavy lifting.
- All variants retain 78-96% top-100 overlap with Spearman >=0.85 -- the
  ranking is directionally stable under +/-30-60% reweighting of any single
  component, i.e. it isn't riding on one fragile, untunable knob. We kept the
  baseline weights (no change) since they're the most directly traceable to
  specific JD language and no alternative produced a result we could argue is
  *more* correct without ground-truth labels to check against.

## Anti-keyword-stuffing and honeypot defenses

**Skill credit weighting** (`features.py: _skill_credit`): a skill only
counts at full weight if it has either >=3 endorsements or >=12 months of
`duration_months`. A skill that's merely *listed* (0 endorsements, ~0
duration) counts at 30% weight. This directly targets the classic trap
candidate -- e.g. a Marketing Manager whose skills list includes every AI
buzzword with zero endorsements and zero duration -- without penalizing
candidates who genuinely have a skill but happen to have low endorsement
counts on the platform (skill match also requires *category* coverage, so a
single strong skill in a category is enough).

**Platform skill-assessment corroboration** (`features.py: _skill_credit`,
new): `redrob_signals.skill_assessment_scores` is a dict of
`skill_name -> 0-100` platform-administered test scores (present for ~25% of
candidates, ~1-2 skills each). A skill that fails the endorsement/duration
test above but has an assessment score >= 60 is upgraded from 30% to 75%
credit -- a keyword-stuffer can paste a skill name onto their profile but
can't fake a passing score on a proctored test. This is an *upgrade-only*
signal (it never lowers a score), so it specifically helps genuine "Tier 5
plain-language" candidates whose real competency hasn't yet accrued
endorsements or tenure on the platform, without creating any new way to game
the ranking. When it fires, the matched skill is surfaced in the reasoning
as "platform skill assessment corroborates: ...". (On the full 100K dataset
this affects 1 candidate -- the signal is rare but free, since it can only
help and is directly traceable to a schema field.)

**Honeypot detection** (`features.py: is_honeypot`), per
`honeypot_heuristics` in `jd_profile.json`:
- >=3 skills claimed at "expert" proficiency with **0 duration_months and 0
  endorsements** simultaneously. If `skill_assessment_scores` also rates any
  of those same "expert" skills below 35/100, that's surfaced as
  corroborating detail in the reasoning (independent confirmation from a
  proctored test that the claim is fake) -- but it is *not* a separate
  trigger, to avoid a false positive from one noisy assessment alone.
- `career_history` durations sum to >1.6x the candidate's stated
  `years_of_experience`, with a >3-year absolute gap.

Flagged candidates get an additional 0.02x multiplier on `final_score`,
forcing them to the bottom regardless of how good their listed skills look.
The reasoning column for any such candidate (none currently land in the
top-100 -- see Results) explicitly says `HONEYPOT SUSPECT` and cites the
specific triggering fact.

`submission_spec.docx` describes ~80 honeypots dataset-wide ("8 years of
experience at a company founded 3 years ago; 'expert' proficiency in 10
skills with 0 years used"), forced to relevance tier 0 in the hidden ground
truth, and a Stage-3 disqualification rule of *honeypot rate > 10% in
top-100*. We currently flag 23/100,000 honeypot-suspects and **0 land in the
top-100** -- comfortably under the 10% threshold. The spec also notes "we
expect a good ranking system to naturally avoid them; you don't need to
special-case them" -- consistent with that, the primary defense here is that
honeypots score poorly on every *other* component (semantic similarity,
substantiated skill match, coherent career history) before the 0.02x
multiplier ever applies; `is_honeypot` is a secondary, explicit safety net.
We did not expand the heuristic set further: a third heuristic risks false
positives on genuinely strong candidates (a single bad multiplier of 0.02x is
catastrophic), and the existing two checks already give a 0% top-100 rate
with a wide margin.

**Disqualifier multipliers** (`features.py: detect_disqualifiers`) catch the
6 narrative traps described in the JD (pure-consulting, research-only,
CV/speech/robotics-only, architecture-only with long tenure, title-chasing,
LangChain-only recent AI experience). Each is a *soft* multiplier (0.35-0.55)
combined multiplicatively, matching the JD's "we will probably not move
forward" (probabilistic, not absolute) language.

## Reasoning / explainability

Every field used in the per-candidate `reasoning` string in `submission.csv`
(`rank.py: generate_reasoning`) is read directly from that candidate's record
or from the deterministic features computed from it -- years of experience,
current title/company, the specific matched skills (with their
endorsement/duration evidence already filtered through anti-stuffing),
location, notice period, last-active recency and recruiter response rate, any
triggered disqualifier concerns, and the raw semantic-similarity score. Stage
4's "no hallucinated claims" check should pass by construction: nothing is
generated by an LLM, and no claim appears that isn't traceable to a schema
field.

## Results (full 100K run)

- 100,000 candidates scored in ~3 seconds.
- 23 honeypot-suspects flagged dataset-wide; **0 appear in the top-100**
  (well under the Stage-3 disqualification threshold of >10% in top-100).
- 0 of the top-100 trigger any disqualifier concern.
- Top-100 `final_score` range: 76.85 - 96.40 (mean 82.09) -- unchanged after
  adding the `skill_assessment_scores` corroboration signal, which affects 1
  candidate dataset-wide (not in the top-100).
- `validate_submission.py` passes: exactly 100 rows, ranks 1-100 unique,
  scores non-increasing, ties broken by `candidate_id` ascending,
  `candidate_id` format `CAND_[0-9]{7}`.
- Weight sensitivity: top-100 set retains 78-96% Jaccard overlap and
  Spearman >=0.85 under +/-30-60% reweighting of any single base_score
  component (see "Weight calibration & sensitivity analysis" above).

Sample top-1 reasoning (`CAND_0006567`, score 96.40):

> 7.9 yrs experience (squarely in JD's 6-8yr ideal band). Current title
> 'Senior AI Engineer' at Meta closely matches the target role. Substantiated
> (endorsed/long-duration) match on required skills: Vector Representations,
> BM25, Search & Discovery, Search Backend, Python, Ranking Systems; plus
> nice-to-have skills: Recommendation Systems, NLP, Kubernetes. Based in
> Noida, Uttar Pradesh (one of the JD's target cities). 60-day notice period
> (coverable via buyout per JD). Active recently (last active 40 days ago),
> 79% recruiter response rate. Semantic similarity to JD narrative: 0.79.

## Repo structure

```
jd_profile.json             Structured, annotated rubric distilled from job_description.docx
text_utils.py                Shared text-construction helpers (candidate narrative + JD narrative)
features.py                  Per-candidate rule-based feature extraction, scoring, disqualifiers, honeypot detection
precompute_embeddings.py      Offline: TF-IDF + TruncatedSVD over the 100K-candidate corpus -> artifacts/
embed.py                      Helper: embed an arbitrary small batch of candidates with the saved models (used by app.py)
rank.py                       THE 5-MINUTE REPRODUCE SCRIPT: streaming top-100 ranker -> submission.csv
app.py                        Gradio sandbox demo (Section 10.5) -- ranks an uploaded/sample <=100-candidate batch
sample_candidates_100.jsonl   First 100 rows of candidates.jsonl, used by the sandbox demo as a default sample
artifacts/                    Precomputed embeddings + fitted vectorizer/SVD (generated by precompute_embeddings.py)
analysis/weight_sensitivity.py  Offline: re-scores all 100K candidates under alternative base_score weight vectors
                                 to check top-100 stability (see "Weight calibration & sensitivity analysis" above)
requirements.txt
submission_metadata.yaml
```

## Sandbox demo

`app.py` is a small Gradio app: upload a `.jsonl` of up to 100 candidates (or
just click Run to use the bundled `sample_candidates_100.jsonl`), and it
ranks them using the same `score_candidate` / `generate_reasoning` code as
`rank.py`, with embeddings computed on the fly via the saved
`vectorizer.joblib` + `svd.joblib`. Run locally with `python app.py`, or
deploy to HuggingFace Spaces (SDK: `gradio`) -- see `submission_metadata.yaml`
for the hosted link.

## Compute profile

- CPU-only, no GPU, no network calls during ranking.
- `rank.py`: ~3s / 100K candidates, <500MB peak RAM (96-dim embeddings for
  100K candidates = ~36MB, plus a fixed-size 100-item heap).
- `precompute_embeddings.py`: ~20s / 100K candidates, ~1.5-2GB peak RAM.
  This is the "pre-computation, documented separately, not counted in the
  5-minute budget" step referenced in `submission_spec.docx`.
