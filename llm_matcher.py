"""
llm_matcher.py

LLM-based semantic matching between a resume and a job description,
using Google Gemini. This is separate from the rule-based scoring in
backend.py — it adds the "LLM computes a match score with justification"
requirement on top of the existing extraction pipeline.

Setup:
    pip install google-genai
    export GEMINI_API_KEY="your-key-here"   (get a free key at https://aistudio.google.com/apikey)
"""

import os
import re
import json

from google import genai
from google.genai import types

GEMINI_MODEL = "gemini-3.6-flash"  # fast + free-tier friendly


def _get_client():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return None
    return genai.Client(api_key=api_key)


def _build_prompt(resume_text: str, jd_text: str) -> str:
    return f"""You are an expert technical recruiter. Compare the following resume
with the job description and rate the candidate's fit on a scale of 1-10,
with clear justification.

Evaluate the candidate based on:
- Required technical skills
- Programming languages
- Frameworks and technologies
- Relevant experience
- Projects
- Education
- Job responsibilities
- Missing or weak requirements
- Overall relevance

Do not invent skills or experience that are not present in the resume.

Give a realistic score from 1-10. Different candidates should receive
different scores when their qualifications differ.

Respond with ONLY a valid JSON object (no markdown fences, no extra text)
in exactly this shape:

{{
  "llm_score": <integer 1-10>,
  "justification": "<2-4 sentence explanation of the rating>",
  "key_strengths": ["<short phrase>", "<short phrase>", "<short phrase>"],
  "key_gaps": ["<short phrase>", "<short phrase>"],
  "recommendation": "<one of: Strong Fit, Moderate Fit, Weak Fit>"
}}

RESUME:
\"\"\"
{resume_text[:6000]}
\"\"\"

JOB DESCRIPTION:
\"\"\"
{jd_text[:3000]}
\"\"\"
"""


def _extract_json(raw_text: str) -> dict | None:
    """Strip markdown code fences etc. and parse the JSON object."""
    cleaned = raw_text.strip()

    cleaned = re.sub(r"^```(?:json)?", "", cleaned.strip())
    cleaned = re.sub(r"```$", "", cleaned.strip())

    match = re.search(r"\{.*\}", cleaned, re.DOTALL)

    if not match:
        return None

    try:
        return json.loads(match.group())

    except json.JSONDecodeError:
        return None


def get_llm_match(resume_text: str, jd_text: str) -> dict:
    """
    Calls Gemini to semantically compare a resume against a job description.

    Returns a dict:
        {
          "llm_score": int | None,
          "justification": str,
          "key_strengths": list[str],
          "key_gaps": list[str],
          "recommendation": str,
          "available": bool,
          "error": str | None,
        }

    This never raises — callers can always render the result, and the rest
    of the app (rule-based score) keeps working even if the LLM call fails.
    """

    client = _get_client()

    if client is None:
        return {
            "llm_score": None,
            "justification": "",
            "key_strengths": [],
            "key_gaps": [],
            "recommendation": "",
            "available": False,
            "error": "GEMINI_API_KEY not set.",
        }

    prompt = _build_prompt(resume_text, jd_text)

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=2048,
            ),
        )

        raw_text = response.text or ""

    except Exception as e:
        return {
            "llm_score": None,
            "justification": "",
            "key_strengths": [],
            "key_gaps": [],
            "recommendation": "",
            "available": False,
            "error": f"Gemini API call failed: {e}",
        }

    parsed = _extract_json(raw_text)

    if parsed is None:
        return {
            "llm_score": None,
            "justification": raw_text.strip()[:500]
            if raw_text.strip()
            else "(empty response from model)",
            "key_strengths": [],
            "key_gaps": [],
            "recommendation": "",
            "available": False,
            "error": (
                "Could not parse JSON from LLM response. "
                f"Raw: {raw_text.strip()[:200]!r}"
            ),
        }

    # Validate and normalize score
    llm_score = parsed.get("llm_score")

    try:
        if llm_score is not None:
            llm_score = int(llm_score)

            if llm_score < 1 or llm_score > 10:
                llm_score = None

    except (ValueError, TypeError):
        llm_score = None

    return {
        "llm_score": llm_score,
        "justification": parsed.get("justification", ""),
        "key_strengths": parsed.get("key_strengths", []),
        "key_gaps": parsed.get("key_gaps", []),
        "recommendation": parsed.get("recommendation", ""),
        "available": llm_score is not None,
        "error": None if llm_score is not None else "Invalid LLM score returned.",
    }