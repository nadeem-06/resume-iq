"""
db.py

SQLite persistence layer for ResumeIQ. Stores every parsed resume, the job
description it was matched against, and the resulting scores (rule-based +
LLM) so past screenings aren't lost when the Streamlit session ends.

No setup needed — creates resumeiq.db (a single file) on first run.
"""

import json
import sqlite3
from datetime import datetime, timezone

DB_PATH = "resumeiq.db"


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_connection()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS screenings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            candidate_name TEXT,
            candidate_email TEXT,
            candidate_phone TEXT,
            resume_text TEXT,
            resume_data_json TEXT,
            jd_text TEXT,
            job_data_json TEXT,
            rule_score REAL,
            breakdown_json TEXT,
            llm_score INTEGER,
            llm_justification TEXT,
            llm_recommendation TEXT,
            llm_result_json TEXT
        )
        """
    )
    conn.commit()
    conn.close()


def save_screening(resume_text, resume_data, jd_text, job_data, rule_score, breakdown, llm_result):
    """Persists one screening run. Returns the new row's id."""
    init_db()
    conn = get_connection()
    cur = conn.execute(
        """
        INSERT INTO screenings (
            created_at, candidate_name, candidate_email, candidate_phone,
            resume_text, resume_data_json, jd_text, job_data_json,
            rule_score, breakdown_json,
            llm_score, llm_justification, llm_recommendation, llm_result_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            datetime.now(timezone.utc).isoformat(),
            resume_data.get("name"),
            resume_data.get("email"),
            resume_data.get("phone"),
            resume_text,
            json.dumps(resume_data),
            jd_text,
            json.dumps(job_data),
            rule_score,
            json.dumps(breakdown),
            llm_result.get("llm_score"),
            llm_result.get("justification"),
            llm_result.get("recommendation"),
            json.dumps(llm_result),
        ),
    )
    conn.commit()
    row_id = cur.lastrowid
    conn.close()
    return row_id


def get_all_screenings(limit: int = 100):
    """Returns past screenings, most recent first, as a list of dicts."""
    init_db()
    conn = get_connection()
    rows = conn.execute(
        """
        SELECT id, created_at, candidate_name, candidate_email, candidate_phone,
               rule_score, llm_score, llm_recommendation
        FROM screenings
        ORDER BY id DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_screening_by_id(screening_id: int):
    """Returns the full stored record (including parsed JSON blobs) for one screening."""
    init_db()
    conn = get_connection()
    row = conn.execute(
        "SELECT * FROM screenings WHERE id = ?", (screening_id,)
    ).fetchone()
    conn.close()
    if row is None:
        return None
    record = dict(row)
    record["resume_data"] = json.loads(record.pop("resume_data_json"))
    record["job_data"] = json.loads(record.pop("job_data_json"))
    record["breakdown"] = json.loads(record.pop("breakdown_json"))
    record["llm_result"] = json.loads(record.pop("llm_result_json"))
    return record