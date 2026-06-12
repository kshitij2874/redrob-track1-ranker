"""
Shared text-construction helpers.

Both the offline embedding precompute step (precompute_embeddings.py) and the
online ranking step (rank.py, via features.py) need to build the *same* kind
of "narrative text" for a candidate, and the offline step needs the JD's core
narrative text too. Keeping this in one place guarantees the vector space the
candidates were embedded into matches the vector the JD query is projected
into at ranking time.
"""

from __future__ import annotations


def candidate_narrative_text(candidate: dict) -> str:
    """Build a single text blob capturing what a candidate has *actually done*.

    Deliberately weighted toward career_history descriptions (where the
    "Tier 5 plain-language" signal lives -- e.g. someone who built a
    recommendation system but never wrote the word "RAG") rather than just
    the skills list (which is where keyword-stuffing lives).
    """
    p = candidate["profile"]
    parts = [
        p.get("headline", ""),
        p.get("summary", ""),
        f"Current role: {p.get('current_title','')} at {p.get('current_company','')} "
        f"({p.get('current_industry','')}).",
    ]

    for ch in candidate.get("career_history", []):
        parts.append(
            f"{ch.get('title','')} at {ch.get('company','')} "
            f"({ch.get('industry','')}, {ch.get('duration_months',0)} months): "
            f"{ch.get('description','')}"
        )

    # Skills are included too, but only once each -- the embedding shouldn't
    # be dominated by a long flat skill list (that's the keyword-stuffing
    # vector; we want the *narrative* vector).
    skill_names = [s["name"] for s in candidate.get("skills", [])]
    if skill_names:
        parts.append("Skills: " + ", ".join(skill_names) + ".")

    return "\n".join(p for p in parts if p)


def jd_core_text() -> str:
    """The JD's core narrative, condensed to what actually drives semantic fit.

    Source: job_description.docx ("Senior AI Engineer - Founding Team", Redrob AI).
    This is the text the candidate vectors are compared against via cosine
    similarity in the LSA embedding space. It intentionally mirrors the
    *language* candidates would use if they'd done this kind of work --
    i.e. it describes the work, not just a skills checklist -- so that a
    Tier-5 candidate who "built a recommendation system at a product
    company" but never wrote the word "RAG" still lands close to this
    vector.
    """
    return (
        "Senior AI Engineer, founding engineering team at an AI-native talent "
        "intelligence / recruiting platform. Owns the intelligence layer: the "
        "ranking, retrieval, and matching systems that decide what recruiters "
        "see when they search for candidates and what candidates see when they "
        "search for roles.\n"
        "Audited an existing BM25 plus rule-based scoring system and shipped a "
        "v2 ranking system that improved recruiter engagement, using "
        "embeddings, hybrid retrieval combining dense vector search with "
        "lexical search, and LLM-based re-ranking.\n"
        "Built production experience with embeddings-based retrieval systems "
        "using sentence-transformers, OpenAI embeddings, BGE, or E5 deployed to "
        "real users -- handled embedding drift, index refresh, and retrieval "
        "quality regressions in production.\n"
        "Production experience with vector databases or hybrid search "
        "infrastructure such as Pinecone, Weaviate, Qdrant, Milvus, OpenSearch, "
        "Elasticsearch, or FAISS. Strong Python and code quality.\n"
        "Designed evaluation frameworks for ranking systems: NDCG, MRR, MAP, "
        "offline-to-online correlation, A/B test interpretation. Set up offline "
        "benchmarks, online A/B testing, and recruiter-feedback loops.\n"
        "Has shipped at least one end-to-end ranking, search, recommendation, "
        "or personalization system to real users at meaningful scale at a "
        "product company (not a pure consulting or pure research environment). "
        "Has opinions about hybrid versus dense retrieval, offline versus "
        "online evaluation, and when to fine-tune versus prompt an LLM, backed "
        "by systems actually built and deployed.\n"
        "Bonus: LLM fine-tuning with LoRA, QLoRA, or PEFT; learning-to-rank "
        "models such as XGBoost-based or neural rankers; distributed systems or "
        "large-scale inference optimization (Kafka, Spark, Kubernetes); prior "
        "HR-tech, recruiting-tech, or marketplace product experience; "
        "open-source contributions in AI/ML."
    )
