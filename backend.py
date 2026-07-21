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