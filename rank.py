#!/usr/bin/env python3
import json
import re
import argparse
import pandas as pd
import numpy as np

CORE_SKILLS = {
    "embeddings": 3.5, "vector search": 3.5, "vector database": 3.0,
    "retrieval": 4.0, "ranking": 4.0, "search": 2.5,
    "llm": 2.0, "fine-tuning llms": 2.5, "lora": 2.0, "qlora": 2.0, "peft": 1.5,
    "python": 3.0, "ndcg": 3.0, "mrr": 2.5, "map": 2.5, "a/b testing": 2.0,
    "evaluation": 3.0, "sentence-transformers": 2.0, "bge": 1.5, "e5": 1.5,
    "pinecone": 2.0, "weaviate": 2.0, "qdrant": 2.0, "milvus": 2.0,
    "opensearch": 2.0, "elasticsearch": 2.0, "faiss": 2.0,
    "rag": 1.5, "recommendation systems": 2.5, "recommendation": 2.0,
    "ir": 2.0, "nlp": 2.0, "langchain": -1.0
}

PREFERRED_TITLES = {
    "ai engineer": 6, "ml engineer": 6, "machine learning engineer": 6,
    "applied scientist": 4, "data scientist": 2.5, "search engineer": 5,
    "relevance engineer": 6, "backend engineer": 2.5, "software engineer": 2.0,
    "nlp engineer": 5, "ranking engineer": 6, "recommendation engineer": 6,
    "llm engineer": 3.5
}

PRODUCT_INDUSTRIES = {
    "Software", "SaaS", "AI/ML", "E-commerce", "Fintech", "AdTech",
    "HealthTech", "HealthTech AI", "Conversational AI", "AI Services",
    "EdTech", "Gaming", "Insurance Tech"
}

CONSULTING_COMPANIES = {
    "TCS", "Infosys", "Wipro", "Accenture", "Cognizant", "Capgemini",
    "HCL", "Tech Mahindra", "Mindtree", "Mphasis", "LTIMindtree"
}

BAD_DOMAINS = {
    "computer vision", "cv", "speech recognition", "robotics",
    "tts", "asr", "object detection", "image classification"
}

GOOD_LOCATIONS = {
    "Pune", "Noida", "Mumbai", "Hyderabad", "Delhi NCR", "Delhi",
    "Gurgaon", "Gurugram", "Bengaluru", "Bangalore"
}

def norm(s):
    return re.sub(r"[^a-z0-9+.#/-]+", " ", str(s).lower()).strip()

def any_kw(text, kws):
    t = norm(text)
    return any(k in t for k in kws)

def score_skill_match(skills, texts):
    names = [norm(s.get("name", "")) for s in skills]
    full = " ".join([norm(t) for t in texts] + names)
    score = 0.0
    hits = []

    for k, w in CORE_SKILLS.items():
        if k in full:
            score += w
            hits.append(k)

    for s in skills:
        n = norm(s.get("name", ""))
        if n in CORE_SKILLS:
            prof = s.get("proficiency", "intermediate")
            prof_mult = {"beginner": 0.6, "intermediate": 0.85, "advanced": 1.0, "expert": 1.1}.get(prof, 0.85)
            end = min(s.get("endorsements", 0), 50) / 50
            dur = min(s.get("duration_months", 0), 60) / 60
            score += CORE_SKILLS[n] * (0.4 * prof_mult + 0.3 * end + 0.3 * dur)

    return score, hits

def score_title(title, headline):
    t = norm(title + " " + headline)
    score = 0.0
    hits = []
    for k, w in PREFERRED_TITLES.items():
        if k in t:
            score += w
            hits.append(k)
    return score, hits

def company_productness(history, current_industry):
    prod = 0
    consult = 0
    for h in history:
        if h.get("industry") in PRODUCT_INDUSTRIES:
            prod += 1
        if h.get("company", "") in CONSULTING_COMPANIES or h.get("industry") == "IT Services":
            consult += 1
    if current_industry in PRODUCT_INDUSTRIES:
        prod += 1
    return prod, consult

def tenure_features(history):
    durs = [h.get("duration_months", 0) for h in history if h.get("duration_months", 0) is not None]
    if not durs:
        return 0.0, 0, 0
    return float(np.mean(durs)), sum(d < 18 for d in durs), max(durs)

def behavioral_score(sig):
    s = 0.0
    s += sig.get("profile_completeness_score", 0) / 100 * 2.0
    s += 1.2 if sig.get("open_to_work_flag") else 0
    s += min(sig.get("recruiter_response_rate", 0), 1) * 2.0
    s += min(sig.get("interview_completion_rate", 0), 1) * 1.2
    s += min(sig.get("offer_acceptance_rate", 0), 1) * 0.6
    s += min(sig.get("saved_by_recruiters_30d", 0), 20) / 20 * 1.0
    s += min(sig.get("search_appearance_30d", 0), 500) / 500 * 0.6
    s += min(sig.get("github_activity_score", 0), 100) / 100 * 1.1

    if sig.get("verified_email"):
        s += 0.2
    if sig.get("verified_phone"):
        s += 0.2
    if sig.get("linkedin_connected"):
        s += 0.2

    npd = sig.get("notice_period_days", 90)
    if npd <= 30:
        s += 1.0
    elif npd <= 60:
        s += 0.4
    elif npd >= 90:
        s -= 0.6

    if sig.get("willing_to_relocate"):
        s += 0.6
    if sig.get("preferred_work_mode", "") in ("hybrid", "onsite"):
        s += 0.3
    return s

def yoe_score(y):
    if 5 <= y <= 9:
        return 4.5
    if 4 <= y < 5 or 9 < y <= 11:
        return 3.2
    if 3 <= y < 4 or 11 < y <= 13:
        return 1.8
    return 0.5 if y >= 2 else 0.0

def location_score(loc, willing):
    if any(k.lower() in (loc or "").lower() for k in [x.lower() for x in GOOD_LOCATIONS]):
        return 1.8
    return 0.8 if willing else 0.0

def jd_penalties(obj, full_text):
    p = 0.0
    reasons = []
    history = obj.get("career_history", [])
    prod, consult = company_productness(history, obj["profile"].get("current_industry", ""))

    if consult >= max(2, len(history)) and prod == 0:
        p += 4.0
        reasons.append("services-only background")

    title = norm(obj["profile"].get("current_title", ""))
    if "research" in title and not any_kw(full_text, ["deployed", "production", "shipped", "users", "pipeline", "latency"]):
        p += 3.0
        reasons.append("research-heavy without production evidence")

    if "langchain" in full_text and not any_kw(full_text, ["retrieval", "ranking", "search", "ndcg", "mrr", "map", "elasticsearch", "faiss", "qdrant", "weaviate", "pinecone", "opensearch"]):
        p += 2.5
        reasons.append("framework-heavy, weak IR evidence")

    bad_count = sum(1 for b in BAD_DOMAINS if b in full_text)
    if bad_count >= 2 and not any_kw(full_text, ["nlp", "retrieval", "ranking", "search", "recommendation"]):
        p += 2.8
        reasons.append("adjacent AI domain without NLP/IR depth")

    avg_tenure, short, _ = tenure_features(history)
    if short >= 2 and avg_tenure < 20:
        p += 1.8
        reasons.append("job-hopping pattern")

    if obj["redrob_signals"].get("avg_response_time_hours", 0) > 168:
        p += 0.8
        reasons.append("slow recruiter response time")

    return p, reasons

def make_reason(obj, title_hits, skill_hits, penalty_reasons):
    parts = []
    title = obj["profile"].get("current_title", "")
    y = obj["profile"].get("years_of_experience", 0)
    parts.append(f"{title} with {y:.1f} years")

    if title_hits:
        parts.append(f"title alignment: {', '.join(title_hits[:2])}")

    if skill_hits:
        uniq = []
        for h in skill_hits:
            if h not in uniq:
                uniq.append(h)
        parts.append(f"evidence of {', '.join(uniq[:4])}")

    sig = obj["redrob_signals"]
    if sig.get("open_to_work_flag"):
        parts.append("active/open-to-work")
    if sig.get("notice_period_days", 999) <= 30:
        parts.append("short notice period")
    if sig.get("saved_by_recruiters_30d", 0) >= 8:
        parts.append("strong recruiter interest")
    if penalty_reasons:
        parts.append("watchouts: " + ", ".join(penalty_reasons[:2]))

    return ("; ".join(parts)[:220]).rstrip(" ;.") + "."

def rank_candidates(candidates_path, out_path):
    records = []

    with open(candidates_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            prof = obj["profile"]
            hist = obj.get("career_history", [])
            skills = obj.get("skills", [])
            sig = obj["redrob_signals"]

            texts = [
                prof.get("headline", ""),
                prof.get("summary", ""),
                prof.get("current_title", ""),
                prof.get("current_industry", "")
            ]
            texts += [
                h.get("title", "") + " " + h.get("description", "") + " " + h.get("industry", "")
                for h in hist
            ]

            full_text = " ".join(norm(t) for t in texts + [s.get("name", "") for s in skills])

            skill_s, skill_hits = score_skill_match(skills, texts)
            title_s, title_hits = score_title(prof.get("current_title", ""), prof.get("headline", ""))
            y_s = yoe_score(prof.get("years_of_experience", 0))
            loc_s = location_score(prof.get("location", ""), sig.get("willing_to_relocate", False))
            beh_s = behavioral_score(sig)
            prod, consult = company_productness(hist, prof.get("current_industry", ""))
            prod_s = min(prod, 4) * 1.2 - min(consult, 4) * 0.2

            eval_s = sum(
                w for kw, w in [
                    ("ndcg", 1.8), ("mrr", 1.2), ("map", 1.2),
                    ("a/b testing", 1.0), ("evaluation", 1.0),
                    ("offline benchmark", 1.0), ("relevance", 1.0)
                ] if kw in full_text
            )

            ship_s = sum(
                w for kw, w in [
                    ("production", 1.2), ("deployed", 1.2), ("shipped", 1.2),
                    ("real users", 1.0), ("latency", 0.8), ("pipeline", 0.6),
                    ("scale", 0.8), ("online", 0.6), ("retrieval quality", 1.1),
                    ("recommendation", 0.8)
                ] if kw in full_text
            )

            penalty, penalty_reasons = jd_penalties(obj, full_text)

            total = (
                0.32 * skill_s +
                0.18 * title_s +
                0.11 * y_s +
                0.07 * loc_s +
                0.12 * beh_s +
                0.10 * prod_s +
                0.05 * eval_s +
                0.08 * ship_s -
                0.12 * penalty
            )

            records.append((total, obj, title_hits, skill_hits, penalty_reasons))

    records.sort(key=lambda r: (-r[0], r[1]["candidate_id"]))
    top = records[:100]

    raw = [r[0] for r in top]
    mn, mx = min(raw), max(raw)

    rows = []
    for score_raw, obj, title_hits, skill_hits, penalty_reasons in top:
        score = 0.4 if mx == mn else 0.4 + 0.599 * (score_raw - mn) / (mx - mn)
        score = round(float(score), 3)
        rows.append({
            "candidate_id": obj["candidate_id"],
            "score": score,
            "reasoning": make_reason(obj, title_hits, skill_hits, penalty_reasons)
        })

    df = pd.DataFrame(rows)
    df = df.sort_values(by=["score", "candidate_id"], ascending=[False, True]).reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)
    df = df[["candidate_id", "rank", "score", "reasoning"]]

    df.to_csv(out_path, index=False)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default="candidates.jsonl")
    ap.add_argument("--out", default="submission.csv")
    args = ap.parse_args()
    rank_candidates(args.candidates, args.out)

if __name__ == "__main__":
    main()