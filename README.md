# ResumeIQ — AI Resume Screener

Parses a resume, matches it against a job description, and scores the fit
using both a rule-based weighted formula and an LLM semantic evaluation.

## Architecture

```
                ┌────────────────────┐
                │   Resume (PDF)      │
                └─────────┬───────────┘
                           │  pdfplumber
                           ▼
                ┌────────────────────┐
                │  Text Extraction    │  backend.py: extract_resume_text()
                └─────────┬───────────┘
                           │
                           ▼
                ┌────────────────────┐
                │  Entity Extraction  │  backend.py: extract_resume_entities()
                │  - BERT NER model   │  (loads my_resume_bert/ if present)
                │    (if trained model│
                │    folder present)  │
                │  - else: regex /    │
                │    keyword fallback │
                └─────────┬───────────┘
                           │  name, email, phone, skills,
                           │  companies, education, experience
                           ▼
        ┌──────────────────┴───────────────────┐
        ▼                                       ▼
┌───────────────────┐                 ┌───────────────────────┐
│ Rule-based Scoring │                 │  LLM Semantic Match    │
│ backend.py:        │                 │  llm_matcher.py:        │
│ score_resume()      │                 │  get_llm_match()        │
│ Skills 60% ·         │                 │  Gemini 2.0 Flash       │
│ Experience 25% ·     │                 │  → 1-10 score +          │
│ Education 15%        │                 │    justification +       │
│                       │                 │    strengths/gaps        │
└──────────┬────────────┘                 └────────────┬─────────────┘
           │                                            │
           └───────────────────┬────────────────────────┘
                                ▼
                  backend.py: get_full_match_report()
                                │
                                ▼
                  ┌─────────────────────────┐
                  │   SQLite (db.py)          │
                  │   Every screening is       │
                  │   persisted: resume data,  │
                  │   JD, both scores           │
                  └──────────────┬─────────────┘
                                 │
                 ┌───────────────┴────────────────┐
                 ▼                                 ▼
     ┌─────────────────────┐          ┌─────────────────────────┐
     │  FastAPI (api.py)     │          │  Streamlit UI (app.py)   │
     │  REST endpoints,       │          │  Dashboard, charts,       │
     │  Swagger docs at        │          │  past-screenings sidebar  │
     │  /docs                  │          │  (optional per spec)      │
     └─────────────────────┘          └─────────────────────────┘
```

## Why two scoring methods?

- **Rule-based (`score_resume`)** — deterministic, fast, explainable. Good
  for exact skill/keyword coverage and a reproducible numeric baseline.
- **LLM (`get_llm_match`)** — reads the *actual* resume and JD text and
  judges fit semantically (transferable skills, seniority, phrasing that
  regex can't catch), and writes a plain-language justification a recruiter
  can read directly.

Both are shown side by side so neither one is a black box — you can see
where they agree and where they diverge.

## The LLM prompt

Sent to Gemini (`llm_matcher.py`), truncated to keep token usage low
(resume: 6000 chars, JD: 3000 chars):

```
You are an expert technical recruiter. Compare the following resume
with the job description and rate the candidate's fit on a scale of 1-10,
with clear justification.

Respond with ONLY a valid JSON object (no markdown fences, no extra text)
in exactly this shape:

{
  "llm_score": <integer 1-10>,
  "justification": "<2-4 sentence explanation of the rating>",
  "key_strengths": ["<short phrase>", "<short phrase>", "<short phrase>"],
  "key_gaps": ["<short phrase>", "<short phrase>"],
  "recommendation": "<one of: Strong Fit, Moderate Fit, Weak Fit>"
}

RESUME:
"""
{resume_text}
"""

JOB DESCRIPTION:
"""
{jd_text}
"""
```

The response is parsed as strict JSON (code fences stripped if the model
adds them). If the API key is missing or the call fails for any reason,
`get_llm_match()` returns `"available": False` with an error message —
the rule-based score and the rest of the app keep working regardless.

## Project structure

| File | Purpose |
|---|---|
| `backend.py` | PDF text extraction, BERT/regex entity extraction, rule-based scoring, and `get_full_match_report()` which combines rule-based + LLM results |
| `llm_matcher.py` | Gemini API call for semantic match scoring + justification |
| `db.py` | SQLite persistence — every screening (resume, JD, both scores) is saved |
| `api.py` | FastAPI REST backend — `/api/screen`, `/api/screenings`, `/api/parse-jd` |
| `app.py` | Streamlit dashboard (optional frontend) — charts, tabs, past-screenings sidebar |
| `datasets.json` | Training/reference data used for the BERT NER model |

## Setup

```bash
pip install -r requirements.txt
export GEMINI_API_KEY="your-key-here"   # free key: https://aistudio.google.com/apikey
```

**Run the Streamlit dashboard:**
```bash
streamlit run app.py
```

**Run the API instead (or alongside):**
```bash
uvicorn api:app --reload --port 8000
# interactive docs: http://localhost:8000/docs
```

## Notes on the NER model

`extract_resume_entities()` looks for a trained BERT model folder at
`my_resume_bert/`. If it's not present (e.g. not included in the repo due
to size), extraction automatically falls back to the regex/keyword-based
extractor — the app works either way, just with lower precision on
unusual resume formats without the trained model.
