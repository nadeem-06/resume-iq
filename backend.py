import os
import re
import json
import numpy as np
import pdfplumber

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
    pipeline,
)
from seqeval.metrics import f1_score, precision_score, recall_score

MODEL_PATH = "my_resume_bert"
DATASET_PATH = "dataset.json"

ALL_SKILLS = [
    # Programming Languages
    "python", "java", "c++", "c#", "javascript", "typescript", "go", "rust",
    "kotlin", "swift", "r", "matlab", "scala", "php", "cobol", "perl",
    "ruby", "bash", "shell", "powershell", "dart", "lua", "haskell", "elixir",
    # Web / Frontend
    "html", "css", "bootstrap", "tailwind", "react", "angular", "vue",
    "angularjs", "jquery", "d3.js", "next.js", "nuxt", "svelte",
    "webpack", "vite", "graphql",
    # Backend / Frameworks
    "node.js", "express", "spring", "spring boot", "django", "flask",
    "fastapi", "hibernate", "jsp", "servlet", "jdbc", "rails", "laravel",
    "asp.net", ".net", "microservices", "rest api", "soap",
    # Databases
    "sql", "mysql", "postgresql", "mongodb", "oracle", "teradata",
    "redis", "cassandra", "sqlite", "dynamodb", "elasticsearch", "firebase",
    "snowflake", "bigquery",
    # AI / ML / Data
    "machine learning", "deep learning", "data science",
    "natural language processing", "computer vision",
    "tensorflow", "pytorch", "keras", "scikit-learn",
    "pandas", "numpy", "pyspark", "spark", "hadoop", "hive",
    "tableau", "power bi", "data analysis", "data engineering",
    "mlops", "llm", "generative ai", "transformers", "hugging face",
    # Cloud / DevOps
    "aws", "azure", "gcp", "docker", "kubernetes", "git", "github",
    "gitlab", "linux", "ci/cd", "jenkins", "terraform", "ansible",
    "helm", "prometheus", "grafana",
    # Testing / Tools
    "selenium", "junit", "pytest", "postman", "jira", "confluence",
    "nifi", "jcl", "mainframe",
    # CS Fundamentals
    "data structures", "algorithms", "object oriented programming",
    "system design", "design patterns", "agile", "scrum",
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
    "college", "university", "institute", "academy", "degree",
    "bachelor", "master", "phd", "doctorate", "school",
    "b.tech", "m.tech", "b.e", "m.e", "bsc", "msc", "b.sc", "m.sc",
    "engineering", "technology", "mathematics", "physics",
    "chemistry", "arts", "science", "diploma", "certification",
    "graduate", "undergraduate", "postgraduate",
]

# ─────────────────────────────────────────────
#  MODEL TRAINING / LOADING
# ─────────────────────────────────────────────

def train_or_load_model():
    if os.path.isdir(MODEL_PATH):
        print(f"Loading existing model from {MODEL_PATH}")
        return _build_pipeline()

    if not os.path.exists(DATASET_PATH):
        print("Model and dataset not found - using rule-based extraction.")
        return None

    print(f"Model not found - starting training from {DATASET_PATH}...")

    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        raw = f.read().strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = [json.loads(line) for line in raw.splitlines() if line.strip()]

    data = [d for d in data if d.get("content", "").strip() and d.get("annotation")]
    for item in data:
        item["content"] = item["content"].encode("utf-8", errors="ignore").decode("utf-8")

    label_set = set()
    for item in data:
        for ann in item["annotation"]:
            if ann.get("label"):
                label_set.add(ann["label"][0].upper())

    label_list = sorted(
        ["O"] + [f"B-{lbl}" for lbl in label_set] + [f"I-{lbl}" for lbl in label_set]
    )
    label2id = {lbl: idx for idx, lbl in enumerate(label_list)}
    id2label = {idx: lbl for lbl, idx in label2id.items()}

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    def tokenize_and_align(example):
        tokenized = tokenizer(
            example["content"],
            truncation=True,
            max_length=512,
            return_offsets_mapping=True,
        )
        labels = ["O"] * len(tokenized["input_ids"])

        for ann in example["annotation"]:
            if not ann.get("label"):
                continue
            label_name = ann["label"][0].upper()
            start = ann["points"][0]["start"]
            end = ann["points"][0]["end"]
            if start is None or end is None or start >= end:
                continue
            for idx, (offset_start, offset_end) in enumerate(tokenized["offset_mapping"]):
                if offset_start < end and offset_end > start:
                    labels[idx] = (
                        f"B-{label_name}" if offset_start == start else f"I-{label_name}"
                    )

        tokenized["labels"] = [label2id.get(lbl, label2id["O"]) for lbl in labels]
        tokenized.pop("offset_mapping")
        return tokenized

    dataset = Dataset.from_list(data).map(tokenize_and_align, batched=False)
    split = dataset.train_test_split(test_size=0.1, seed=42)

    model = AutoModelForTokenClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id,
    )

    def compute_metrics(prediction_output):
        predictions, labels = prediction_output
        predictions = np.argmax(predictions, axis=2)
        true_labels = [
            [id2label[lbl] for lbl in label_row if lbl != -100]
            for label_row in labels
        ]
        true_preds = [
            [id2label[pred] for pred, lbl in zip(pred_row, label_row) if lbl != -100]
            for pred_row, label_row in zip(predictions, labels)
        ]
        return {
            "f1": f1_score(true_labels, true_preds),
            "precision": precision_score(true_labels, true_preds),
            "recall": recall_score(true_labels, true_preds),
        }

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir="./results",
            eval_strategy="epoch",
            save_strategy="epoch",
            num_train_epochs=3,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            learning_rate=2e-5,
            weight_decay=0.01,
            warmup_ratio=0.1,
            label_smoothing_factor=0.1,
            load_best_model_at_end=True,
            logging_dir="./logs",
        ),
        train_dataset=split["train"],
        eval_dataset=split["test"],
        tokenizer=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=compute_metrics,
    )

    trainer.train()
    model.save_pretrained(MODEL_PATH)
    tokenizer.save_pretrained(MODEL_PATH)
    print(f"Model trained and saved to {MODEL_PATH}")
    return _build_pipeline()


def _build_pipeline():
    return pipeline(
        "token-classification",
        model=MODEL_PATH,
        tokenizer=MODEL_PATH,
        aggregation_strategy="simple",
    )


ner_pipeline = train_or_load_model()

# ─────────────────────────────────────────────
#  PDF TEXT EXTRACTION
# ─────────────────────────────────────────────

def extract_resume_text(path: str) -> str:
    """Extract full text from PDF, page by page."""
    text = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text.strip()

# ─────────────────────────────────────────────
#  EXPERIENCE EXTRACTION
# ─────────────────────────────────────────────

def extract_experience_from_text(text: str) -> float:
    """
    Extracts years of experience using multiple patterns.
    Handles: '5 years', '5+ years', '6 months', date ranges like '2019 – 2023'.
    Returns the highest/most credible value found.
    """
    candidates = []

    # Pattern 1: explicit "X years of experience"
    for pattern in [
        r"(\d+\.?\d*)\s*[\+]?\s*years?\s+of\s+(?:total\s+|IT\s+|relevant\s+)?experience",
        r"(?:experience|minimum|min)[:\s]+(\d+\.?\d*)\s*[\+]?\s*years?",
        r"(\d+\.?\d*)\s*[\+]?\s*years?\s*(?:of\s+)?(?:experience|exp)\b",
    ]:
        for m in re.finditer(pattern, text, re.I):
            try:
                candidates.append(float(m.group(1)))
            except ValueError:
                pass

    # Pattern 2: months → convert to years
    for m in re.finditer(r"(\d+)\s*months?\s+of\s+(?:experience|exp)", text, re.I):
        try:
            candidates.append(round(float(m.group(1)) / 12, 1))
        except ValueError:
            pass

    # Pattern 3: year ranges like "2019 – 2023" or "Jan 2018 - Dec 2022"
    current_year = 2025
    year_ranges = re.findall(
        r"(20\d{2}|19\d{2})\s*[-–—to]+\s*(20\d{2}|19\d{2}|present|current|now|till date)",
        text,
        re.I,
    )
    total_range_years = 0.0
    for start_y, end_y in year_ranges:
        try:
            s = int(start_y)
            e = current_year if re.match(r"present|current|now|till", end_y, re.I) else int(end_y)
            if 1990 <= s <= current_year and s <= e:
                total_range_years += e - s
        except ValueError:
            pass
    if total_range_years > 0:
        candidates.append(round(total_range_years, 1))

    if not candidates:
        return 0.0

    # Return the most credible value (cap at 50 to filter noise)
    valid = [c for c in candidates if 0 < c <= 50]
    return round(max(valid), 1) if valid else 0.0

# ─────────────────────────────────────────────
#  SKILL NORMALIZATION
# ─────────────────────────────────────────────

def normalize_skills(raw_tokens: list[str]) -> list[str]:
    """Map raw NER tokens to canonical skill names."""
    cleaned = set()
    combined_text = " ".join(raw_tokens).lower()

    for alias, full in SKILL_ALIASES.items():
        combined_text = re.sub(r"\b" + re.escape(alias) + r"\b", full, combined_text)

    for skill in ALL_SKILLS:
        pattern = r"\b" + re.escape(skill) + r"\b"
        if re.search(pattern, combined_text, re.I):
            cleaned.add(skill.title())

    return sorted(cleaned)


def _extract_skills_from_text(text: str) -> list[str]:
    """Rule-based fallback: scan full text for skill keywords."""
    normalized = text.lower()
    for alias, full in SKILL_ALIASES.items():
        normalized = re.sub(r"\b" + re.escape(alias) + r"\b", full, normalized)

    matches = set()
    for skill in ALL_SKILLS:
        pattern = r"\b" + re.escape(skill) + r"\b"
        if re.search(pattern, normalized, re.I):
            matches.add(skill.title())
    return sorted(matches)

# ─────────────────────────────────────────────
#  CONTACT INFO EXTRACTION
# ─────────────────────────────────────────────

def _extract_email(text: str) -> str | None:
    m = re.search(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", text)
    return m.group() if m else None


def _extract_phone(text: str) -> str | None:
    m = re.search(r"(\+?\d[\d\s\-().]{7,}\d)", text)
    return m.group().strip() if m else None


def _extract_name(text: str) -> str | None:
    """
    Heuristic: the candidate name is usually on the first non-empty line
    and contains 2-4 capitalized words.
    """
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        words = line.split()
        if 2 <= len(words) <= 5 and all(w[0].isupper() for w in words if w.isalpha()):
            # Exclude lines that look like section headers or addresses
            if not any(kw in line.lower() for kw in ["resume", "curriculum", "vitae", "cv", "profile", "summary"]):
                return line
    return None

# ─────────────────────────────────────────────
#  CORE ENTITY EXTRACTION
# ─────────────────────────────────────────────

def _chunk_text(text: str, max_chars: int = 1500) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + max_chars
        chunks.append(text[start:end])
        start += max_chars - 200
    return chunks


def extract_resume_entities(text: str, ner_pipe=None) -> dict:
    """
    Extract structured data from resume text using BERT NER (if available)
    with rule-based fallbacks for every field.
    """
    active_pipe = ner_pipe or ner_pipeline
    results = []

    if active_pipe is not None:
        for chunk in _chunk_text(text, max_chars=1500):
            results.extend(active_pipe(chunk))

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
        if not word or word.startswith("##"):
            continue

        is_education = (
            any(kw in label for kw in ["degree", "college", "education"])
            or any(kw in word.lower() for kw in EDUCATION_KEYWORDS)
        )

        if is_education:
            if word not in resume["education"]:
                resume["education"].append(word)
        elif "skill" in label:
            raw_skill_tokens.append(word)
        elif any(kw in label for kw in ["compan", "employer", "org", "organization"]):
            if word not in resume["companies"]:
                resume["companies"].append(word)
        elif any(kw in label for kw in ["designation", "title", "role", "position"]):
            if resume["designation"] is None:
                resume["designation"] = word

    # Skills: prefer NER output, fallback to rule-based
    resume["skills"] = (
        normalize_skills(raw_skill_tokens) if raw_skill_tokens
        else _extract_skills_from_text(text)
    )

    # Education fallback: scan text if NER found nothing
    if not resume["education"]:
        resume["education"] = _extract_education_from_text(text)

    # Experience
    resume["experience_years"] = extract_experience_from_text(text)

    return resume


def _extract_education_from_text(text: str) -> list[str]:
    """Rule-based education extraction: find lines containing degree keywords."""
    edu_entries = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        lower = line.lower()
        if any(kw in lower for kw in EDUCATION_KEYWORDS):
            if len(line) < 200:  # Exclude huge paragraph blobs
                edu_entries.append(line)
    # Deduplicate while preserving order
    seen = set()
    unique = []
    for e in edu_entries:
        if e.lower() not in seen:
            seen.add(e.lower())
            unique.append(e)
    return unique[:5]  # Cap at 5

# ─────────────────────────────────────────────
#  JOB DESCRIPTION PARSING
# ─────────────────────────────────────────────

def parse_job_description(jd_text: str) -> dict:
    """Extract required skills and minimum experience from job description."""
    job = {
        "required_skills": [],
        "min_experience": 0.0,
    }

    normalized = jd_text.lower()
    for alias, full in SKILL_ALIASES.items():
        normalized = re.sub(r"\b" + re.escape(alias) + r"\b", full, normalized)

    for skill in ALL_SKILLS:
        if re.search(r"\b" + re.escape(skill) + r"\b", normalized, re.I):
            job["required_skills"].append(skill.title())

    job["min_experience"] = extract_experience_from_text(jd_text)
    return job

# ─────────────────────────────────────────────
#  SCORING ENGINE
# ─────────────────────────────────────────────

def score_resume(resume: dict, job: dict) -> tuple[float, dict]:
    """
    Score a resume against a job description.

    Weights:
      - Skills match : 60 pts
      - Experience   : 25 pts
      - Education    : 15 pts

    Evaluation metrics (informational, not added to score):
      - Precision  = matched / predicted
      - Recall     = matched / required   (= skill match rate)
      - F1         = harmonic mean of precision & recall
      - Experience gap
    """
    score = 0.0
    breakdown = {}

    # ── Skills ──────────────────────────────
    resume_set = {s.lower() for s in resume["skills"]}
    job_set = {s.lower() for s in job["required_skills"]}

    if job_set:
        matched = resume_set & job_set
        missing = job_set - resume_set
        extra   = resume_set - job_set          # skills candidate has beyond JD
        skill_score = (len(matched) / len(job_set)) * 60
    else:
        matched, missing, extra = set(), set(), resume_set
        skill_score = 0.0

    breakdown["matched_skills"] = sorted(matched)
    breakdown["missing_skills"] = sorted(missing)
    breakdown["extra_skills"]   = sorted(extra)
    breakdown["skill_score"]    = round(skill_score, 2)
    score += skill_score

    # ── Experience ──────────────────────────
    candidate_exp = resume["experience_years"]
    required_exp  = job["min_experience"]

    if required_exp == 0:
        # No requirement stated → full marks if candidate has any experience
        exp_score = 25.0 if candidate_exp > 0 else 12.5
    elif candidate_exp >= required_exp:
        exp_score = 25.0
    else:
        # Partial credit: proportional to how close they are
        exp_score = round((candidate_exp / required_exp) * 25, 2)

    breakdown["experience_score"]     = exp_score
    breakdown["candidate_experience"] = candidate_exp
    breakdown["required_experience"]  = required_exp
    score += exp_score

    # ── Education ───────────────────────────
    edu_score = 15.0 if resume.get("education") else 0.0
    breakdown["education_score"] = edu_score
    score += edu_score

    # ── Evaluation Metrics ──────────────────
    matched_count   = len(matched)
    predicted_count = len(resume_set)
    required_count  = len(job_set)

    precision = matched_count / predicted_count if predicted_count else 0.0
    recall    = matched_count / required_count  if required_count  else 0.0
    f1_value  = (
        (2 * precision * recall) / (precision + recall)
        if (precision + recall) > 0 else 0.0
    )
    skill_match_rate  = recall * 100
    exp_gap           = round(candidate_exp - required_exp, 2)

    # Coverage score: how well this resume covers the JD holistically
    coverage_score = round(score, 2)

    breakdown["evaluation_metrics"] = {
        "precision":         round(precision, 4),
        "recall":            round(recall, 4),
        "f1_score":          round(f1_value, 4),
        "skill_match_rate":  round(skill_match_rate, 2),   # % of JD skills matched
        "experience_gap":    exp_gap,                       # +ve = exceeds req
        "total_resume_skills":   predicted_count,
        "total_required_skills": required_count,
        "total_matched_skills":  matched_count,
        "coverage_score":        coverage_score,            # same as final score
    }

    return round(score, 2), breakdown