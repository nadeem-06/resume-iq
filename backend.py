import os
import re
import json
import pdfplumber
import streamlit as st

from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForTokenClassification,
)

MODEL_PATH = "my_resume_bert"

# ------------------------------------------------------------
# MODEL LOADING (Inference Only)
# ------------------------------------------------------------

@st.cache_resource
def get_pipeline():
    """
    Loads the trained BERT model only once.
    Streamlit caches it to reduce memory usage.
    """

    if not os.path.isdir(MODEL_PATH):
        return None

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_PATH
    )

    return pipeline(
        "token-classification",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple",
    )


# ------------------------------------------------------------
# SKILL DATABASE
# ------------------------------------------------------------

ALL_SKILLS = [

    # Programming Languages
    "python",
    "java",
    "c++",
    "c#",
    "javascript",
    "typescript",
    "go",
    "rust",
    "kotlin",
    "swift",
    "r",
    "matlab",
    "scala",
    "php",
    "cobol",
    "perl",
    "ruby",
    "bash",
    "shell",
    "powershell",
    "dart",
    "lua",
    "haskell",
    "elixir",

    # Frontend
    "html",
    "css",
    "bootstrap",
    "tailwind",
    "react",
    "angular",
    "vue",
    "angularjs",
    "jquery",
    "d3.js",
    "next.js",
    "nuxt",
    "svelte",
    "webpack",
    "vite",
    "graphql",

    # Backend
    "node.js",
    "express",
    "spring",
    "spring boot",
    "django",
    "flask",
    "fastapi",
    "hibernate",
    "jsp",
    "servlet",
    "jdbc",
    "rails",
    "laravel",
    "asp.net",
    ".net",
    "microservices",
    "rest api",
    "soap",

    # Databases
    "sql",
    "mysql",
    "postgresql",
    "mongodb",
    "oracle",
    "teradata",
    "redis",
    "cassandra",
    "sqlite",
    "dynamodb",
    "elasticsearch",
    "firebase",
    "snowflake",
    "bigquery",

    # AI / ML
    "machine learning",
    "deep learning",
    "data science",
    "natural language processing",
    "computer vision",
    "tensorflow",
    "pytorch",
    "keras",
    "scikit-learn",
    "pandas",
    "numpy",
    "pyspark",
    "spark",
    "hadoop",
    "hive",
    "tableau",
    "power bi",
    "data analysis",
    "data engineering",
    "mlops",
    "llm",
    "generative ai",
    "transformers",
    "hugging face",

    # Cloud
    "aws",
    "azure",
    "gcp",
    "docker",
    "kubernetes",
    "git",
    "github",
    "gitlab",
    "linux",
    "ci/cd",
    "jenkins",
    "terraform",
    "ansible",
    "helm",
    "prometheus",
    "grafana",

    # Testing
    "selenium",
    "junit",
    "pytest",
    "postman",
    "jira",
    "confluence",
    "nifi",
    "jcl",
    "mainframe",

    # CS Fundamentals
    "data structures",
    "algorithms",
    "object oriented programming",
    "system design",
    "design patterns",
    "agile",
    "scrum",
]
SKILL_ALIASES = {
    "ml": "machine learning",
    "dl": "deep learning",
    "nlp": "natural language processing",
    "cv": "computer vision",
    "js": "javascript",
    "ts": "typescript",
    "dsa": "data structures",
    "ai": "machine learning",
    "k8s": "kubernetes",
    "oop": "object oriented programming",
    "tf": "tensorflow",
    "sk": "scikit-learn",
    "sklearn": "scikit-learn",
    "pg": "postgresql",
    "postgres": "postgresql",
    "gai": "generative ai",
    "llms": "llm",
    "hf": "hugging face",
}

EDUCATION_KEYWORDS = [
    "college",
    "university",
    "institute",
    "academy",
    "degree",
    "bachelor",
    "master",
    "phd",
    "doctorate",
    "school",
    "b.tech",
    "m.tech",
    "b.e",
    "m.e",
    "bsc",
    "msc",
    "b.sc",
    "m.sc",
    "engineering",
    "technology",
    "mathematics",
    "physics",
    "chemistry",
    "arts",
    "science",
    "diploma",
    "certification",
    "graduate",
    "undergraduate",
    "postgraduate",
]

# ------------------------------------------------------------
# PDF TEXT EXTRACTION
# ------------------------------------------------------------

def extract_resume_text(path: str) -> str:
    """
    Extract full text from PDF.
    """
    text = ""

    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"

    return text.strip()


# ------------------------------------------------------------
# EXPERIENCE EXTRACTION
# ------------------------------------------------------------

def extract_experience_from_text(text: str) -> float:

    candidates = []

    patterns = [
        r"(\d+\.?\d*)\s*[\+]?\s*years?\s+of\s+(?:total\s+|IT\s+|relevant\s+)?experience",
        r"(?:experience|minimum|min)[:\s]+(\d+\.?\d*)\s*[\+]?\s*years?",
        r"(\d+\.?\d*)\s*[\+]?\s*years?\s*(?:of\s+)?(?:experience|exp)\b",
    ]

    for pattern in patterns:
        for match in re.finditer(pattern, text, re.I):
            try:
                candidates.append(float(match.group(1)))
            except:
                pass

    for match in re.finditer(
        r"(\d+)\s*months?\s+of\s+(?:experience|exp)",
        text,
        re.I,
    ):
        try:
            candidates.append(round(float(match.group(1)) / 12, 1))
        except:
            pass

    current_year = 2025

    ranges = re.findall(
        r"(20\d{2}|19\d{2})\s*[-–—to]+\s*(20\d{2}|19\d{2}|present|current|now|till date)",
        text,
        re.I,
    )

    total = 0

    for start, end in ranges:

        try:
            s = int(start)

            if re.match(r"present|current|now|till", end, re.I):
                e = current_year
            else:
                e = int(end)

            if s <= e:
                total += e - s

        except:
            pass

    if total > 0:
        candidates.append(total)

    valid = [x for x in candidates if 0 < x <= 50]

    if valid:
        return round(max(valid), 1)

    return 0.0


# ------------------------------------------------------------
# MODEL HELPERS
# ------------------------------------------------------------

def normalize_skills(raw_tokens):

    cleaned = set()

    combined = " ".join(raw_tokens).lower()

    for alias, full in SKILL_ALIASES.items():
        combined = re.sub(
            r"\b" + re.escape(alias) + r"\b",
            full,
            combined,
        )

    for skill in ALL_SKILLS:

        if re.search(
            r"\b" + re.escape(skill) + r"\b",
            combined,
            re.I,
        ):
            cleaned.add(skill.title())

    return sorted(cleaned)


def _extract_skills_from_text(text):

    text = text.lower()

    for alias, full in SKILL_ALIASES.items():
        text = re.sub(
            r"\b" + re.escape(alias) + r"\b",
            full,
            text,
        )

    skills = set()

    for skill in ALL_SKILLS:

        if re.search(
            r"\b" + re.escape(skill) + r"\b",
            text,
            re.I,
        ):
            skills.add(skill.title())

    return sorted(skills)
# ------------------------------------------------------------
# CONTACT INFO EXTRACTION
# ------------------------------------------------------------

def _extract_email(text: str) -> str | None:
    match = re.search(
        r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}",
        text,
    )
    return match.group() if match else None


def _extract_phone(text: str) -> str | None:
    match = re.search(
        r"(\+?\d[\d\s\-().]{7,}\d)",
        text,
    )
    return match.group().strip() if match else None


def _extract_name(text: str) -> str | None:

    for line in text.splitlines():

        line = line.strip()

        if not line:
            continue

        words = line.split()

        if (
            2 <= len(words) <= 5
            and all(
                w[0].isupper()
                for w in words
                if w.isalpha()
            )
        ):

            if not any(
                keyword in line.lower()
                for keyword in [
                    "resume",
                    "curriculum",
                    "vitae",
                    "cv",
                    "profile",
                    "summary",
                ]
            ):
                return line

    return None


# ------------------------------------------------------------
# TEXT CHUNKING
# ------------------------------------------------------------

def _chunk_text(text: str, max_chars: int = 1500):

    chunks = []

    start = 0

    while start < len(text):

        end = start + max_chars

        chunks.append(text[start:end])

        start += max_chars - 200

    return chunks


# ------------------------------------------------------------
# EDUCATION EXTRACTION
# ------------------------------------------------------------

def _extract_education_from_text(text: str):

    education = []

    for line in text.splitlines():

        line = line.strip()

        if not line:
            continue

        lower = line.lower()

        if any(keyword in lower for keyword in EDUCATION_KEYWORDS):

            if len(line) < 200:
                education.append(line)

    seen = set()

    unique = []

    for item in education:

        if item.lower() not in seen:
            seen.add(item.lower())
            unique.append(item)

    return unique[:5]


# ------------------------------------------------------------
# RESUME ENTITY EXTRACTION
# ------------------------------------------------------------

def extract_resume_entities(text: str, ner_pipe=None):

    active_pipe = ner_pipe or get_pipeline()

    results = []

    if active_pipe is not None:

        for chunk in _chunk_text(text):

            try:
                results.extend(active_pipe(chunk))
            except Exception:
                pass

    resume = {
        "name": _extract_name(text),
        "email": _extract_email(text),
        "phone": _extract_phone(text),
        "skills": [],
        "companies": [],
        "designation": None,
        "experience_years": 0.0,
        "education": [],
    }

    raw_skill_tokens = []

    for result in results:

        label = result["entity_group"].lower()

        word = result["word"].strip()

        if not word:
            continue

        if word.startswith("##"):
            continue

        education_entity = (
            any(
                key in label
                for key in [
                    "degree",
                    "college",
                    "education",
                ]
            )
            or any(
                key in word.lower()
                for key in EDUCATION_KEYWORDS
            )
        )

        if education_entity:

            if word not in resume["education"]:
                resume["education"].append(word)

        elif "skill" in label:

            raw_skill_tokens.append(word)

        elif any(
            key in label
            for key in [
                "compan",
                "organization",
                "org",
                "employer",
            ]
        ):

            if word not in resume["companies"]:
                resume["companies"].append(word)

        elif any(
            key in label
            for key in [
                "designation",
                "role",
                "title",
                "position",
            ]
        ):

            if resume["designation"] is None:
                resume["designation"] = word

    if raw_skill_tokens:
        resume["skills"] = normalize_skills(raw_skill_tokens)
    else:
        resume["skills"] = _extract_skills_from_text(text)

    if not resume["education"]:
        resume["education"] = _extract_education_from_text(text)

    resume["experience_years"] = extract_experience_from_text(text)

    return resume
# ------------------------------------------------------------
# JOB DESCRIPTION PARSING
# ------------------------------------------------------------

def parse_job_description(jd_text: str) -> dict:

    job = {
        "required_skills": [],
        "min_experience": 0.0,
    }

    normalized = jd_text.lower()

    for alias, full in SKILL_ALIASES.items():
        normalized = re.sub(
            r"\b" + re.escape(alias) + r"\b",
            full,
            normalized,
        )

    for skill in ALL_SKILLS:

        if re.search(
            r"\b" + re.escape(skill) + r"\b",
            normalized,
            re.I,
        ):
            job["required_skills"].append(skill.title())

    job["min_experience"] = extract_experience_from_text(jd_text)

    return job


# ------------------------------------------------------------
# RESUME SCORING
# ------------------------------------------------------------

def score_resume(resume: dict, job: dict):

    score = 0.0

    breakdown = {}

    resume_skills = {s.lower() for s in resume["skills"]}
    jd_skills = {s.lower() for s in job["required_skills"]}

    if jd_skills:

        matched = resume_skills & jd_skills
        missing = jd_skills - resume_skills
        extra = resume_skills - jd_skills

        skill_score = (
            len(matched) / len(jd_skills)
        ) * 60

    else:

        matched = set()
        missing = set()
        extra = resume_skills

        skill_score = 0.0

    score += skill_score

    breakdown["matched_skills"] = sorted(matched)
    breakdown["missing_skills"] = sorted(missing)
    breakdown["extra_skills"] = sorted(extra)
    breakdown["skill_score"] = round(skill_score, 2)

    candidate_exp = resume["experience_years"]
    required_exp = job["min_experience"]

    if required_exp == 0:

        exp_score = 25 if candidate_exp > 0 else 12.5

    elif candidate_exp >= required_exp:

        exp_score = 25

    else:

        exp_score = (
            candidate_exp / required_exp
        ) * 25

    score += exp_score

    breakdown["experience_score"] = round(exp_score, 2)
    breakdown["candidate_experience"] = candidate_exp
    breakdown["required_experience"] = required_exp

    education_score = (
        15
        if resume.get("education")
        else 0
    )

    score += education_score

    breakdown["education_score"] = education_score

    matched_count = len(matched)
    predicted_count = len(resume_skills)
    required_count = len(jd_skills)

    precision = (
        matched_count / predicted_count
        if predicted_count
        else 0
    )

    recall = (
        matched_count / required_count
        if required_count
        else 0
    )

    if precision + recall:

        f1 = (
            2
            * precision
            * recall
            / (precision + recall)
        )

    else:

        f1 = 0

    breakdown["evaluation_metrics"] = {

        "precision": round(
            precision,
            4,
        ),

        "recall": round(
            recall,
            4,
        ),

        "f1_score": round(
            f1,
            4,
        ),

        "skill_match_rate": round(
            recall * 100,
            2,
        ),

        "experience_gap": round(
            candidate_exp - required_exp,
            2,
        ),

        "total_resume_skills": predicted_count,

        "total_required_skills": required_count,

        "total_matched_skills": matched_count,

        "coverage_score": round(
            score,
            2,
        ),
    }

    return round(score, 2), breakdown