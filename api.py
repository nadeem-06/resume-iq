"""
api.py

FastAPI backend for ResumeIQ. Wraps the existing extraction, scoring, and
LLM-matching logic from backend.py behind REST endpoints, and persists each
run via db.py. This satisfies the PDF's "Backend API (Node.js/Python/Java)"
requirement — the Streamlit app (app.py) remains the optional frontend
dashboard and can keep working exactly as before, unchanged.

Run with:
    pip install fastapi uvicorn python-multipart
    uvicorn api:app --reload --port 8000

Then open http://localhost:8000/docs for interactive Swagger docs.
"""

import os
import tempfile

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel

from backend import (
    extract_resume_text,
    extract_resume_entities,
    parse_job_description,
    get_full_match_report,
)
from db import init_db, save_screening, get_all_screenings, get_screening_by_id

app = FastAPI(
    title="ResumeIQ API",
    description="Parses resumes, matches them against a job description, and scores fit (rule-based + LLM).",
    version="1.0.0",
)


@app.on_event("startup")
def _startup():
    init_db()


# ------------------------------------------------------------
# Response models
# ------------------------------------------------------------
class ScreeningSummary(BaseModel):
    id: int
    created_at: str
    candidate_name: str | None
    candidate_email: str | None
    candidate_phone: str | None
    rule_score: float | None
    llm_score: int | None
    llm_recommendation: str | None


# ------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------
@app.get("/")
def health_check():
    return {"status": "ok", "service": "ResumeIQ API"}


@app.post("/api/parse-jd")
def parse_jd(jd_text: str = Form(...)):
    """Parses a job description and returns required skills + min experience."""
    if not jd_text.strip():
        raise HTTPException(status_code=400, detail="jd_text cannot be empty.")
    return parse_job_description(jd_text)


@app.post("/api/screen")
async def screen_resume(
    resume: UploadFile = File(...),
    jd_text: str = Form(...),
):
    """
    Full pipeline: extract resume text -> extract entities -> parse JD ->
    rule-based score -> LLM semantic match -> persist to DB -> return report.
    """
    if not jd_text.strip():
        raise HTTPException(status_code=400, detail="jd_text cannot be empty.")
    if resume.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Resume must be a PDF file.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(await resume.read())
        tmp_path = tmp.name

    try:
        resume_text = extract_resume_text(tmp_path)
        resume_data = extract_resume_entities(resume_text)
        job_data = parse_job_description(jd_text)
        rule_score, breakdown, llm_result = get_full_match_report(
            resume_text, jd_text, resume_data, job_data
        )
    finally:
        os.unlink(tmp_path)

    screening_id = save_screening(
        resume_text, resume_data, jd_text, job_data,
        rule_score, breakdown, llm_result,
    )

    return {
        "screening_id": screening_id,
        "resume_data": resume_data,
        "job_data": job_data,
        "rule_score": rule_score,
        "breakdown": breakdown,
        "llm_result": llm_result,
    }


@app.get("/api/screenings", response_model=list[ScreeningSummary])
def list_screenings(limit: int = 25):
    """Returns recent screenings, most recent first."""
    return get_all_screenings(limit=limit)


@app.get("/api/screenings/{screening_id}")
def get_screening(screening_id: int):
    """Returns the full stored record for one screening, including parsed JSON."""
    record = get_screening_by_id(screening_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Screening not found.")
    return record