import io
import os
import re
from dataclasses import dataclass
from typing import List

import nltk
import pandas as pd
from flask import Flask, render_template, request
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from werkzeug.utils import secure_filename


def ensure_nltk_resources() -> None:
    resources = [
        ("tokenizers/punkt", "punkt"),
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
        ("corpora/omw-1.4", "omw-1.4"),
    ]
    for path, name in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(name, quiet=True)


ensure_nltk_resources()
STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()


@dataclass
class CandidateResult:
    filename: str
    semantic_score: float
    skill_score: float
    final_score: float
    matched_skills: List[str]


def read_pdf(file_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(file_bytes))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = word_tokenize(text)
    cleaned = [LEMMATIZER.lemmatize(t) for t in tokens if t not in STOP_WORDS and len(t) > 2]
    return " ".join(cleaned)


def normalize_skill(skill: str) -> str:
    return re.sub(r"\s+", " ", skill.strip().lower())


def extract_skills(text: str, skill_catalog: List[str]) -> List[str]:
    normalized_text = text.lower()
    matched = [s for s in skill_catalog if normalize_skill(s) in normalized_text]
    return sorted(set(matched))


def score_candidates(job_description: str, resumes: List[dict], skill_catalog: List[str]) -> List[CandidateResult]:
    processed_docs = [preprocess_text(job_description)] + [preprocess_text(r["text"]) for r in resumes]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    matrix = vectorizer.fit_transform(processed_docs)

    job_vector = matrix[0:1]
    resume_vectors = matrix[1:]
    semantic_scores = cosine_similarity(resume_vectors, job_vector).flatten()

    jd_skills = set(extract_skills(job_description, skill_catalog))
    results: List[CandidateResult] = []
    for idx, resume in enumerate(resumes):
        resume_skills = set(extract_skills(resume["text"], skill_catalog))
        overlap = len(jd_skills & resume_skills) / len(jd_skills) if jd_skills else 0.0
        semantic = float(semantic_scores[idx])
        final = (0.75 * semantic) + (0.25 * overlap)
        results.append(
            CandidateResult(
                filename=resume["name"],
                semantic_score=semantic * 100,
                skill_score=overlap * 100,
                final_score=final * 100,
                matched_skills=sorted(jd_skills & resume_skills),
            )
        )
    return sorted(results, key=lambda x: x.final_score, reverse=True)


def parse_uploaded_file(file_storage) -> str:
    raw = file_storage.read()
    if file_storage.filename.lower().endswith(".pdf"):
        return read_pdf(raw)
    return raw.decode("utf-8", errors="ignore")


def load_jobs_dataset(path: str = "data/job_roles_dataset.csv") -> pd.DataFrame:
    return pd.read_csv(path)


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024


@app.route("/", methods=["GET"])
def index():
    jobs = load_jobs_dataset()
    return render_template("index.html", jobs=jobs.to_dict("records"), results=None, selected_jd="", error=None)


@app.route("/screen", methods=["POST"])
def screen():
    jobs = load_jobs_dataset()
    selected_job = request.form.get("job_role", "")
    custom_jd = request.form.get("job_description", "").strip()
    files = request.files.getlist("resumes")

    jd_text = custom_jd
    if not jd_text and selected_job:
        match = jobs[jobs["job_role"] == selected_job]
        if not match.empty:
            jd_text = str(match.iloc[0]["job_description"])

    if not jd_text:
        return render_template(
            "index.html",
            jobs=jobs.to_dict("records"),
            results=None,
            selected_jd="",
            error="Please pick a role or paste a job description.",
        )

    if not files or files[0].filename == "":
        return render_template(
            "index.html",
            jobs=jobs.to_dict("records"),
            results=None,
            selected_jd=jd_text,
            error="Please upload at least one resume (.pdf or .txt).",
        )

    skill_catalog = sorted(
        set(
            skill.strip()
            for skill_cell in jobs["required_skills"].fillna("")
            for skill in str(skill_cell).split(",")
            if skill.strip()
        )
    )

    resumes = []
    for file in files:
        filename = secure_filename(file.filename)
        if not filename:
            continue
        ext = os.path.splitext(filename)[1].lower()
        if ext not in {".pdf", ".txt"}:
            continue
        resumes.append({"name": filename, "text": parse_uploaded_file(file)})

    if not resumes:
        return render_template(
            "index.html",
            jobs=jobs.to_dict("records"),
            results=None,
            selected_jd=jd_text,
            error="Only .pdf and .txt files are supported.",
        )

    results = score_candidates(jd_text, resumes, skill_catalog)
    return render_template(
        "index.html",
        jobs=jobs.to_dict("records"),
        results=results,
        selected_jd=jd_text,
        error=None,
    )


if __name__ == "__main__":
    app.run(debug=True)
