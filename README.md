# Redrob Ranker

An offline, explainable candidate ranking pipeline built for the **Redrob AI / India Runs** hiring challenge. The goal is simple: rank candidates the way a strong recruiter would — not by raw keyword overlap, but by combining role-fit, production evidence, relevance depth, and hiring-readiness signals into a shortlist that is practical to trust and easy to validate.[web:99][web:60]

This project targets the challenge setting of generating a valid **top-100 candidate shortlist** from the provided dataset and exporting it in the required submission format for validator checks.[web:97]

## Why this exists

Keyword-only matching breaks down fast in hiring. Two candidates may mention the same tools, but one has actually built production retrieval systems, evaluated ranking quality with proper metrics, and shown stronger signals of recruiter readiness. This project tries to capture that difference.

The ranker is intentionally designed as a **lightweight offline system**:
- No external API calls during ranking.
- No GPU dependency.
- Reproducible on a regular laptop or desktop.
- Fully compatible with the provided submission validator and CSV format requirements.[web:97]

## Problem framing

The challenge asks for a system that can:
- Understand a job description beyond shallow keyword extraction.
- Evaluate candidates using the full picture, not just a skill list.
- Produce a shortlist that a recruiter can review with confidence.[web:60][web:99]

For this submission, the ranking logic was tailored to a **Senior AI Engineer–style role** where stronger candidates typically show evidence of:
- AI / ML engineering depth.
- Retrieval, ranking, recommendation, or search relevance work.
- Embeddings, vector search, or related infrastructure.
- Python and production deployment maturity.
- Evaluation literacy, including metrics such as NDCG, MRR, MAP, or A/B testing.
- Product-building experience rather than only services-based delivery.

## Solution overview

The current implementation uses a **multi-signal heuristic ranker**. Instead of doing heavy model inference at scoring time, it parses each candidate profile, extracts structured signals, computes a weighted score, and generates a ranked shortlist.

At a high level, the pipeline does this:

1. Read every candidate from `candidates.jsonl`.
2. Extract evidence from profile fields, skills, career history, and Redrob platform signals.
3. Score each candidate across multiple recruiter-relevant dimensions.
4. Apply penalty rules for weak-fit or suspicious-fit patterns.
5. Rank candidates globally.
6. Select the top 100.
7. Normalize output scores.
8. Export a validator-compliant CSV.

This makes the system fast, deterministic, and easy to reason about in an open repository.

## How the ranker thinks

The core philosophy is: **evidence beats mention frequency**.

A candidate should rank higher not because they wrote “LLM” ten times, but because their profile shows stronger evidence of the kind of work the role actually needs.

### Positive signals

The ranker rewards signals such as:
- **Title alignment** — AI Engineer, ML Engineer, NLP Engineer, Search Engineer, Recommendation Engineer, etc.
- **Skill evidence** — retrieval, ranking, embeddings, vector search, vector databases, NLP, evaluation metrics, Python.
- **Production evidence** — deployed systems, shipped work, latency, pipelines, scale, real users.
- **Evaluation depth** — NDCG, MRR, MAP, A/B testing, relevance evaluation, offline benchmark language.
- **Experience fit** — especially mid-senior profiles with appropriate years of experience.
- **Product-company exposure** — preference for candidates with product-building context.
- **Behavioral / hireability signals** — recruiter response rate, interview completion, open-to-work flag, notice period, recruiter saves, GitHub activity.

### Negative signals

The ranker penalizes profiles that look superficially relevant but weak in recruiter terms, such as:
- Services-only background without meaningful product exposure.
- Research-heavy profiles with little production evidence.
- Framework-heavy but IR-light profiles.
- Adjacent AI domains without real retrieval / NLP / ranking depth.
- Unstable tenure patterns or job-hopping signals.

## Scoring design

The implementation combines multiple components into a single final ranking score. The exact weights are encoded in `rank.py`, but the dimensions are conceptually grouped like this:

- Skill match
- Title alignment
- Years-of-experience fit
- Location / relocation fit
- Behavioral fit
- Product-company exposure
- Evaluation evidence
- Production / shipping evidence
- Penalty terms

This is deliberately closer to how a recruiter reasons than a pure semantic similarity score.

## Explainability

Each shortlisted candidate receives a compact `reasoning` string in the final CSV. The explanation is built only from observed candidate fields and deterministic scoring logic.

That means the output reasoning is:
- grounded in the data,
- reproducible,
- and not dependent on live LLM generation.

The aim is not perfect natural language explanation, but **trustworthy shortlist evidence**.

## Repository structure

```text
redrob-ranker/
├── candidates.jsonl
├── candidate_schema.json
├── rank.py
├── validate_submission.py
├── final-valid.csv
├── submission_metadata.yaml
└── README.md
```

Recommended additions for a public repo:

```text
redrob-ranker/
├── .gitignore
├── requirements.txt
├── candidates.jsonl              # keep local only if dataset sharing is restricted
├── candidate_schema.json
├── rank.py
├── validate_submission.py
├── submission_metadata.yaml
├── final-valid.csv
└── README.md
```

## Quick start

### 1) Clone the repository

```bash
git clone https://github.com/dev-xplor/ranker.git
cd ranker
```

### 2) Create a virtual environment

**Linux / macOS**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows**
```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3) Install dependencies

```bash
pip install pandas numpy
```

Or use a `requirements.txt` containing:

```txt
pandas
numpy
```

### 4) Place the official dataset files

Put the official `candidates.jsonl` file in the project root.

### 5) Run the ranker

```bash
python rank.py --candidates ./candidates.jsonl --out ./final-valid.csv
```

### 6) Validate the submission

```bash
python validate_submission.py ./final-valid.csv
```

A successful run should print:

```bash
Submission is valid.
```

## Output format

The final CSV follows the challenge submission format:

| Column | Description |
|---|---|
| `candidate_id` | Unique candidate identifier |
| `rank` | Final rank from 1 to 100 |
| `score` | Non-increasing numeric score |
| `reasoning` | Short explanation for why the candidate was ranked there |

Special care is taken to re-sort the final shortlist by:
1. rounded `score` descending,
2. `candidate_id` ascending for ties,

so the submission passes validator tie-break checks.[web:52]

## Design choices

### Why heuristics instead of heavy LLM inference?

Because the challenge is ultimately about ranking quality under realistic constraints, not about building the flashiest architecture. A well-designed heuristic system can be:
- easier to reproduce,
- cheaper to run,
- faster to debug,
- and more honest about what evidence it is using.

This repo favors a system that can be inspected line-by-line and improved iteratively.

### Why behavioral signals?

A recruiter rarely looks at skills in isolation. Profile completeness, response rate, notice period, and recent recruiter interest can meaningfully affect shortlist quality in practice. The solution uses those signals as modifiers, not replacements for technical fit.

### Why penalties?

In ranking problems, false positives matter. A superficially relevant profile can consume recruiter attention even if it is actually a poor fit. Penalty rules help the system distinguish between “mentions the right words” and “looks genuinely matchable.”

## Known limitations

This is a compact baseline designed to run locally and submit cleanly, not a final production hiring engine.

Current limitations include:
- No learned ranking model yet.
- No embedding-based semantic retrieval layer in the current offline version.
- Limited synonym expansion compared to a production ontology.
- Heuristic company and industry mapping can still be improved.
- Reasoning strings are concise and functional, not deeply narrative.

## Improvement ideas

If extended further, the next upgrades would be:
- Add a semantic retrieval pre-stage using embeddings.
- Build a richer title and skill synonym normalization layer.
- Add honeypot / suspicious-profile detection heuristics.
- Learn feature weights from recruiter feedback or labeled examples.
- Add a small analysis dashboard to inspect top-ranked candidates and score breakdowns.

## Reproducibility

This project is designed to be reproducible with one command plus the official dataset.

Core reproducibility characteristics:
- deterministic scoring logic,
- local execution,
- minimal dependencies,
- validator-compliant CSV output,
- no hidden inference service.

## Open-source notes

If this repo is public, make sure dataset licensing and challenge rules allow the dataset file to be included directly. If not, keep `candidates.jsonl` out of version control and document how users should obtain it.

A minimal `.gitignore` could include:

```gitignore
.venv/
__pycache__/
*.pyc
candidates.jsonl
submission.csv
final-valid.csv
```

## Submission assets

Typical submission bundle:
- `rank.py`
- `final-valid.csv`
- `validate_submission.py`
- `submission_metadata.yaml`
- `README.md`

## Final note

This project is an attempt to make candidate ranking feel a little less like string matching and a little more like recruiter judgment — still lightweight, still explainable, and still practical enough to run on a local machine.
