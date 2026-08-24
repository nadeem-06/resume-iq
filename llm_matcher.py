"""
llm_matcher.py

LLM-based semantic matching between a resume and a job description,
using Groq.

This is separate from the rule-based scoring in backend.py.
It adds the LLM semantic match score, justification,
strengths, gaps, and recommendation.

Setup:
    pip install groq

For local runs:
    set GROQ_API_KEY="your-key-here"

For Render:
    Add GROQ_API_KEY under Environment Variables.
"""

import os
import re
import json

from groq import Groq


# Groq model used for semantic resume matching
GROQ_MODEL = "openai/gpt-oss-20b"


def _get_client():
    api_key = os.environ.get("GROQ_API_KEY")

    if not api_key:
        return None

    return Groq(api_key=api_key)


def _build_prompt(resume_text: str, jd_text: str) -> str:
    return f"""You are an expert technical recruiter.

Compare the following resume with the job description and evaluate
how well the candidate fits the role.

Consider:
- Required technical skills
- Programming languages
- Frameworks and technologies
- Relevant experience
- Projects
- Education
- Job responsibilities
- Overall relevance
- Missing or weak requirements

Do not invent skills, experience, education, or projects
that are not present in the resume.

Give a realistic score from 1 to 10.

Respond with ONLY a valid JSON object.
Do not use markdown.
Do not include ```json.
Do not include any text outside the JSON object.

The response MUST follow exactly this structure:

{{
  "llm_score": 8,
  "justification": "2-4 sentence explanation of why the candidate received this score.",
  "key_strengths": [
    "short strength",
    "short strength",
    "short strength"
  ],
  "key_gaps": [
    "short gap",
    "short gap"
  ],
  "recommendation": "Strong Fit"
}}

The recommendation MUST be exactly one of:
- Strong Fit
- Moderate Fit
- Weak Fit

The llm_score MUST be an integer from 1 to 10.

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
    """
    Remove markdown fences if present and extract
    the JSON object from the model response.
    """

    if not raw_text:
        return None

    cleaned = raw_text.strip()

    # Remove ```json ... ```
    cleaned = re.sub(
        r"^```(?:json)?\s*",
        "",
        cleaned,
        flags=re.IGNORECASE
    )

    cleaned = re.sub(
        r"\s*```$",
        "",
        cleaned
    )

    # Find JSON object
    match = re.search(
        r"\{.*\}",
        cleaned,
        re.DOTALL
    )

    if not match:
        return None

    json_text = match.group(0)

    try:
        return json.loads(json_text)

    except json.JSONDecodeError:
        return None


def get_llm_match(resume_text: str, jd_text: str) -> dict:
    """
    Calls Groq to semantically compare a resume
    against a job description.

    Returns:

        {
            "llm_score": int | None,
            "justification": str,
            "key_strengths": list[str],
            "key_gaps": list[str],
            "recommendation": str,
            "available": bool,
            "error": str | None
        }

    This function never raises an exception.
    The rule-based scoring system can continue working
    even if the LLM call fails.
    """

    client = _get_client()

    # API key missing
    if client is None:
        return {
            "llm_score": None,
            "justification": "",
            "key_strengths": [],
            "key_gaps": [],
            "recommendation": "",
            "available": False,
            "error": "GROQ_API_KEY not set.",
        }

    prompt = _build_prompt(
        resume_text,
        jd_text
    )

    try:

        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert technical recruiter. "
                        "Return only valid JSON."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=0.2,
            max_completion_tokens=2048,
            response_format={
                "type": "json_object"
            },
        )

        raw_text = (
            response.choices[0].message.content
            if response.choices
            else ""
        )

        raw_text = raw_text or ""

    except Exception as e:

        print(
            f"[GROQ ERROR] {e}",
            flush=True
        )

        return {
            "llm_score": None,
            "justification": "",
            "key_strengths": [],
            "key_gaps": [],
            "recommendation": "",
            "available": False,
            "error": f"Groq API call failed: {e}",
        }

    # Parse JSON
    parsed = _extract_json(raw_text)

    if parsed is None:

        print(
            f"[GROQ JSON ERROR] Raw response: {raw_text[:1000]}",
            flush=True
        )

        return {
            "llm_score": None,
            "justification": (
                raw_text.strip()[:500]
                if raw_text.strip()
                else "(empty response from model)"
            ),
            "key_strengths": [],
            "key_gaps": [],
            "recommendation": "",
            "available": False,
            "error": (
                "Could not parse JSON from Groq response. "
                f"Raw: {raw_text.strip()[:200]!r}"
            ),
        }

    # Validate score
    llm_score = parsed.get("llm_score")

    try:

        if llm_score is not None:
            llm_score = int(llm_score)

            if llm_score < 1 or llm_score > 10:
                llm_score = None

    except (ValueError, TypeError):

        llm_score = None

    # Normalize strengths
    key_strengths = parsed.get(
        "key_strengths",
        []
    )

    if not isinstance(key_strengths, list):
        key_strengths = []

    # Normalize gaps
    key_gaps = parsed.get(
        "key_gaps",
        []
    )

    if not isinstance(key_gaps, list):
        key_gaps = []

    # Recommendation
    recommendation = parsed.get(
        "recommendation",
        ""
    )

    if recommendation not in [
        "Strong Fit",
        "Moderate Fit",
        "Weak Fit"
    ]:
        recommendation = ""

    return {
        "llm_score": llm_score,
        "justification": parsed.get(
            "justification",
            ""
        ),
        "key_strengths": key_strengths,
        "key_gaps": key_gaps,
        "recommendation": recommendation,
        "available": llm_score is not None,
        "error": (
            None
            if llm_score is not None
            else "Invalid LLM score returned."
        ),
    }