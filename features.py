"""
Per-candidate structured feature extraction and rule-based sub-scores.

Everything here is pure Python over a single candidate dict (as parsed from
candidates.jsonl) plus the jd_profile.json rubric. No ML, no I/O, no network --
this is the "rules" half of the hybrid ranker. The "embeddings" half (LSA
semantic similarity) lives in precompute_embeddings.py / rank.py.

Design goal: every number this module produces should be traceable back to a
literal field in the candidate's profile, so rank.py can build reasoning
strings that are guaranteed not to hallucinate (Stage 4 manual review checks
exactly this).
"""

from __future__ import annotations

from datetime import date, datetime


# ---------------------------------------------------------------------------
# small parsing helpers
# ---------------------------------------------------------------------------

def _parse_date(s: str | None) -> date | None:
    if not s:
        return None
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except ValueError:
        return None


def _skill_index(candidate: dict) -> dict:
    """name -> skill dict, case-insensitive."""
    return {s["name"].lower(): s for s in candidate.get("skills", [])}


def _skill_credit(skill: dict | None, jd_profile: dict, assessment_score: float | None = None) -> float:
    """How much "trust" do we put in a listed skill?

    Anti-keyword-stuffing: a skill with real endorsements or real time-on-tool
    counts fully; a bare listing with 0 endorsements and ~0 duration counts
    for a fraction. This directly targets the "candidate who has all the AI
    keywords listed as skills but whose title is Marketing Manager" trap --
    such candidates typically have 0/0 on every AI skill.

    `assessment_score` (0-100, from redrob_signals.skill_assessment_scores) is
    a platform-administered test score for this skill, when present. A
    keyword-stuffer cannot fake a proctored test, so a high score on an
    otherwise-unsubstantiated skill upgrades it from "bare listing" credit to
    "verified" credit -- this specifically helps genuine candidates whose real
    competency hasn't yet accrued endorsements/tenure on the platform.
    """
    if skill is None:
        return 0.0
    cfg = jd_profile["skills"]["anti_stuffing"]
    endorsed = skill.get("endorsements", 0) >= cfg["min_endorsements_for_full_credit"]
    seasoned = skill.get("duration_months", 0) >= cfg["min_duration_months_for_full_credit"]
    if endorsed or seasoned:
        return 1.0
    if assessment_score is not None and assessment_score >= cfg["assessment_score_verify_threshold"]:
        return cfg["assessment_verified_credit"]
    return 0.3


def _category_score(skill_idx: dict, skill_names: list[str], jd_profile: dict, sas_idx: dict) -> tuple[float, list[str], list[str]]:
    """Best single-skill credit within a category (operational depth in ANY
    one tool counts -- JD: 'the specific tech doesn't matter; the operational
    experience does'). Returns (score, matched_skill_names_with_credit,
    assessment_verified_skill_names)."""
    best = 0.0
    matched = []
    assessment_verified = []
    for name in skill_names:
        sk = skill_idx.get(name.lower())
        if sk is None:
            continue
        assessment_score = sas_idx.get(name.lower())
        credit = _skill_credit(sk, jd_profile, assessment_score)
        if credit > 0:
            matched.append(sk["name"])
        if credit == jd_profile["skills"]["anti_stuffing"]["assessment_verified_credit"]:
            assessment_verified.append(sk["name"])
        best = max(best, credit)
    return best, matched, assessment_verified


# ---------------------------------------------------------------------------
# sub-scores
# ---------------------------------------------------------------------------

def skill_match_score(candidate: dict, jd_profile: dict) -> dict:
    skills_cfg = jd_profile["skills"]
    skill_idx = _skill_index(candidate)
    rs = candidate.get("redrob_signals", {}) or {}
    sas = rs.get("skill_assessment_scores") or {}
    sas_idx = {k.lower(): v for k, v in sas.items()}

    emb_score, emb_matched, emb_verified = _category_score(skill_idx, skills_cfg["required_embeddings_retrieval"], jd_profile, sas_idx)
    vdb_score, vdb_matched, vdb_verified = _category_score(skill_idx, skills_cfg["required_vector_db_hybrid_search"], jd_profile, sas_idx)
    py_score, py_matched, py_verified = _category_score(skill_idx, skills_cfg["required_python"], jd_profile, sas_idx)
    eval_score, eval_matched, eval_verified = _category_score(skill_idx, skills_cfg["required_eval_frameworks"], jd_profile, sas_idx)

    nice_llm_score, nice_llm_matched, nice_llm_verified = _category_score(skill_idx, skills_cfg["nice_to_have_llm_finetuning"], jd_profile, sas_idx)
    nice_rec_score, nice_rec_matched, nice_rec_verified = _category_score(skill_idx, skills_cfg["nice_to_have_recommendation_nlp"], jd_profile, sas_idx)
    nice_dist_score, nice_dist_matched, nice_dist_verified = _category_score(skill_idx, skills_cfg["nice_to_have_distributed_systems"], jd_profile, sas_idx)
    nice_avg = (nice_llm_score + nice_rec_score + nice_dist_score) / 3.0

    score = (
        0.30 * emb_score
        + 0.30 * vdb_score
        + 0.15 * py_score
        + 0.15 * eval_score
        + 0.10 * nice_avg
    )

    matched_required = list(dict.fromkeys(emb_matched + vdb_matched + py_matched + eval_matched))
    matched_nice = list(dict.fromkeys(nice_llm_matched + nice_rec_matched + nice_dist_matched))
    assessment_verified = list(dict.fromkeys(
        emb_verified + vdb_verified + py_verified + eval_verified
        + nice_llm_verified + nice_rec_verified + nice_dist_verified
    ))

    return {
        "score": score,
        "matched_required_skills": matched_required,
        "matched_nice_skills": matched_nice,
        "assessment_verified_skills": assessment_verified,
        "embeddings_retrieval_hit": emb_score > 0,
        "vector_db_hit": vdb_score > 0,
        "eval_framework_hit": eval_score > 0,
    }


def experience_fit_score(yoe: float, jd_profile: dict) -> float:
    band = jd_profile["experience_band"]
    if band["ideal_min"] <= yoe <= band["ideal_max"]:
        return 1.0
    if band["acceptable_min"] <= yoe <= band["acceptable_max"]:
        return 0.85
    # soft falloff outside the acceptable band
    if yoe < band["acceptable_min"]:
        gap = band["acceptable_min"] - yoe
    else:
        gap = yoe - band["acceptable_max"]
    return max(0.30, 1.0 - 0.12 * gap)


def title_relevance_score(current_title: str, jd_profile: dict) -> tuple[float, str]:
    cfg = jd_profile["title_relevance"]
    if current_title in cfg["high"]:
        return 1.0, "high"
    if current_title in cfg["medium"]:
        return 0.7, "medium"
    return 0.4, "neutral"


def location_score(profile: dict, redrob_signals: dict, jd_profile: dict) -> tuple[float, bool]:
    cfg = jd_profile["location"]
    location = profile.get("location", "")
    country = profile.get("country", "")
    is_target_city = any(city.lower() in location.lower() for city in cfg["target_cities"])
    if is_target_city:
        return cfg["target_city_score"], True
    if country == "India":
        return cfg["india_other_score"], False
    score = cfg["outside_india_base_score"]
    if redrob_signals.get("willing_to_relocate"):
        score = min(1.0, score + cfg["outside_india_relocate_bonus"])
    return score, False


def notice_score(notice_period_days: int, jd_profile: dict) -> float:
    cfg = jd_profile["notice_period"]
    if notice_period_days <= cfg["ideal_max_days"]:
        return 1.0
    if notice_period_days <= cfg["acceptable_max_days"]:
        return 0.8
    if notice_period_days <= 90:
        return 0.6
    return 0.4


def behavioral_multiplier(redrob_signals: dict, jd_profile: dict, reference_date: date) -> tuple[float, dict]:
    """JD: 'a perfect-on-paper candidate who hasn't logged in for 6 months and
    has a 5% recruiter response rate is, for hiring purposes, not actually
    available. Down-weight them appropriately.'

    Combines several engagement signals into a single multiplier.
    """
    cfg = jd_profile["behavioral_signals"]

    last_active = _parse_date(redrob_signals.get("last_active_date"))
    if last_active is None:
        recency_score = 0.5
        days_inactive = None
    else:
        days_inactive = (reference_date - last_active).days
        if days_inactive <= cfg["stale_after_days"]:
            recency_score = 1.0
        elif days_inactive <= cfg["very_stale_after_days"]:
            # linear falloff between stale and very-stale thresholds
            span = cfg["very_stale_after_days"] - cfg["stale_after_days"]
            recency_score = 1.0 - 0.6 * (days_inactive - cfg["stale_after_days"]) / span
        else:
            recency_score = 0.25

    response_rate = redrob_signals.get("recruiter_response_rate", 0.0)
    open_to_work = 1.0 if redrob_signals.get("open_to_work_flag") else 0.6
    interview_completion = redrob_signals.get("interview_completion_rate", 0.0)

    offer_rate = redrob_signals.get("offer_acceptance_rate", -1)
    offer_component = 0.5 if offer_rate < 0 else offer_rate

    completeness = redrob_signals.get("profile_completeness_score", 0) / 100.0

    raw = (
        0.30 * recency_score
        + 0.25 * response_rate
        + 0.15 * open_to_work
        + 0.15 * interview_completion
        + 0.10 * offer_component
        + 0.05 * completeness
    )
    raw = min(1.0, max(0.0, raw))

    lo, hi = cfg["min_multiplier"], cfg["max_multiplier"]
    multiplier = lo + raw * (hi - lo)

    return multiplier, {
        "days_inactive": days_inactive,
        "recruiter_response_rate": response_rate,
        "open_to_work_flag": redrob_signals.get("open_to_work_flag"),
        "interview_completion_rate": interview_completion,
    }


# ---------------------------------------------------------------------------
# disqualifier / red-flag detection
# ---------------------------------------------------------------------------

PRODUCTION_KEYWORDS = (
    "production", "deployed", "deploy", "shipped", "ship ", "launched",
    "scale", "users", "live ", "in prod",
)


def _seniority_index(title: str, ladder: list[str]) -> int:
    title_l = title.lower()
    for i, rung in enumerate(ladder):
        if rung.lower() in title_l:
            return i
    return -1


def detect_disqualifiers(candidate: dict, skill_idx: dict, jd_profile: dict) -> tuple[float, list[str]]:
    """Returns (multiplier in (0,1], list of human-readable concern strings).

    These are *soft* penalties (JD language is "we will probably not move
    forward", not an automatic zero) -- multiple flags compound
    multiplicatively but never fully zero out a candidate by themselves
    (honeypots are handled separately and DO go to ~0).
    """
    p = candidate["profile"]
    history = candidate.get("career_history", [])
    cfg = jd_profile["disqualifier_industries"]

    multiplier = 1.0
    concerns: list[str] = []

    # 1. Pure consulting career (every role at an "IT Services" employer)
    if history and all(ch.get("industry") in cfg["pure_consulting_firms_proxy"] for ch in history):
        multiplier *= 0.35
        concerns.append(
            f"entire career history ({len(history)} role(s)) is at IT-services "
            f"employers ('{history[0]['company']}' etc.) with no product-company experience"
        )

    # 2. Pure research, no production evidence
    if p.get("current_industry") in cfg["research_only_proxy"]:
        text = " ".join(ch.get("description", "") for ch in history).lower()
        if not any(kw in text for kw in PRODUCTION_KEYWORDS):
            multiplier *= 0.40
            concerns.append("research-only background with no mention of production deployment")

    # 3. Computer vision / speech / robotics without NLP/IR
    cv_skills_hit = any(
        skill_idx.get(s.lower()) for s in cfg["cv_speech_robotics_proxy_skills"]
    )
    ir_nlp_skills = {"information retrieval", "information retrieval systems", "nlp",
                     "recommendation systems", "search & discovery", "rag",
                     "ranking systems", "semantic search", "vector search"}
    has_ir_nlp = any(s in skill_idx for s in ir_nlp_skills)
    if (p.get("current_industry") in cfg["cv_speech_robotics_proxy_industries"] or cv_skills_hit) and not has_ir_nlp:
        multiplier *= 0.45
        concerns.append("computer-vision/speech/robotics background without NLP or information-retrieval exposure")

    # 4. Architecture/management role for a long time -- "hasn't written code"
    title_l = p.get("current_title", "")
    if any(t.lower() in title_l.lower() for t in jd_profile["architecture_only_titles"]):
        current_role = next((ch for ch in history if ch.get("is_current")), None)
        duration = current_role.get("duration_months", 0) if current_role else 0
        if duration > 18:
            multiplier *= 0.5
            concerns.append(
                f"has been in a {p.get('current_title')} role for {duration} months "
                f"-- JD wants someone still writing production code"
            )

    # 5. Title-chasing: rapid seniority escalation across short stints
    ladder = jd_profile["seniority_escalation_titles"]
    short_stints = [ch for ch in history if ch.get("duration_months", 0) < 18]
    if len(short_stints) >= 3:
        sorted_hist = sorted(history, key=lambda ch: ch.get("start_date") or "")
        idxs = [_seniority_index(ch.get("title", ""), ladder) for ch in sorted_hist]
        idxs = [i for i in idxs if i >= 0]
        if len(idxs) >= 3 and all(b >= a for a, b in zip(idxs, idxs[1:])) and idxs[-1] > idxs[0]:
            multiplier *= 0.55
            concerns.append(
                f"career shows {len(short_stints)} roles under 18 months with steadily "
                f"escalating titles -- reads as title-chasing"
            )

    # 6. "LangChain-only" recent AI experience
    weak = jd_profile["skills"]["weak_signal_only"]
    weak_hits = [skill_idx[w.lower()] for w in weak if w.lower() in skill_idx]
    if weak_hits:
        recent_weak_only = all(s.get("duration_months", 0) < 12 for s in weak_hits)
        strong_categories = (
            jd_profile["skills"]["required_embeddings_retrieval"]
            + jd_profile["skills"]["required_vector_db_hybrid_search"]
            + jd_profile["skills"]["required_eval_frameworks"]
        )
        has_strong = any(s.lower() in skill_idx for s in strong_categories)
        deep_nlp = skill_idx.get("nlp") and skill_idx["nlp"].get("duration_months", 0) >= 24
        deep_rec = skill_idx.get("recommendation systems") and skill_idx["recommendation systems"].get("duration_months", 0) >= 24
        if recent_weak_only and not has_strong and not deep_nlp and not deep_rec:
            multiplier *= 0.5
            concerns.append(
                "AI exposure looks limited to recent (<12mo) LangChain/prompting work "
                "with no deeper retrieval or ranking background"
            )

    return multiplier, concerns


# ---------------------------------------------------------------------------
# honeypot detection
# ---------------------------------------------------------------------------

def is_honeypot(candidate: dict, jd_profile: dict) -> tuple[bool, list[str]]:
    cfg = jd_profile["honeypot_heuristics"]
    anti_stuffing_cfg = jd_profile["skills"]["anti_stuffing"]
    p = candidate["profile"]
    skills = candidate.get("skills", [])
    history = candidate.get("career_history", [])
    rs = candidate.get("redrob_signals", {}) or {}
    sas = rs.get("skill_assessment_scores") or {}
    sas_idx = {k.lower(): v for k, v in sas.items()}
    reasons = []

    zero_evidence_experts = [
        s for s in skills
        if s.get("proficiency") == "expert" and s.get("duration_months", 0) == 0 and s.get("endorsements", 0) == 0
    ]
    expert_zero = len(zero_evidence_experts)
    if expert_zero >= cfg["expert_zero_evidence_skill_threshold"]:
        detail = f"{expert_zero} skills claimed at 'expert' level with 0 duration and 0 endorsements"
        # Corroborating signal (not an independent trigger): if the
        # platform's own proctored assessment also scored these "expert"
        # skills as low, surface that fact -- a keyword-stuffer can list a
        # skill but can't fake a low-scoring test result into a high one.
        low_assessed = [
            s["name"] for s in zero_evidence_experts
            if sas_idx.get(s["name"].lower(), 100) < anti_stuffing_cfg["expert_claim_low_assessment_threshold"]
        ]
        if low_assessed:
            detail += (
                f"; platform assessment also scored {', '.join(low_assessed)} "
                f"below {anti_stuffing_cfg['expert_claim_low_assessment_threshold']}/100, "
                f"corroborating the unsubstantiated claim"
            )
        reasons.append(detail)

    yoe = p.get("years_of_experience", 0)
    total_months = sum(ch.get("duration_months", 0) for ch in history)
    if yoe > 0 and total_months / 12.0 > yoe * cfg["yoe_overstate_ratio"] and (total_months / 12.0 - yoe) > cfg["yoe_overstate_min_gap_years"]:
        reasons.append(
            f"career_history sums to {total_months/12:.1f} years but profile claims "
            f"{yoe} years of experience"
        )

    return (len(reasons) > 0), reasons


# ---------------------------------------------------------------------------
# top-level entry point
# ---------------------------------------------------------------------------

def extract_features(candidate: dict, jd_profile: dict, reference_date: date) -> dict:
    p = candidate["profile"]
    rs = candidate["redrob_signals"]
    skill_idx = _skill_index(candidate)

    skills_res = skill_match_score(candidate, jd_profile)
    exp_score = experience_fit_score(p.get("years_of_experience", 0), jd_profile)
    title_score, title_tier = title_relevance_score(p.get("current_title", ""), jd_profile)
    loc_score, in_target_city = location_score(p, rs, jd_profile)
    notice = notice_score(rs.get("notice_period_days", 0), jd_profile)
    behav_mult, behav_facts = behavioral_multiplier(rs, jd_profile, reference_date)
    disq_mult, concerns = detect_disqualifiers(candidate, skill_idx, jd_profile)
    honeypot, honeypot_reasons = is_honeypot(candidate, jd_profile)

    return {
        "candidate_id": candidate["candidate_id"],
        "skill_match": skills_res,
        "experience_fit": exp_score,
        "title_score": title_score,
        "title_tier": title_tier,
        "location_score": loc_score,
        "in_target_city": in_target_city,
        "notice_score": notice,
        "behavioral_multiplier": behav_mult,
        "behavioral_facts": behav_facts,
        "disqualifier_multiplier": disq_mult,
        "concerns": concerns,
        "is_honeypot": honeypot,
        "honeypot_reasons": honeypot_reasons,
        # raw facts needed for reasoning text
        "current_title": p.get("current_title", ""),
        "current_company": p.get("current_company", ""),
        "current_industry": p.get("current_industry", ""),
        "years_of_experience": p.get("years_of_experience", 0),
        "location": p.get("location", ""),
        "notice_period_days": rs.get("notice_period_days", 0),
    }
