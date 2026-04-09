import io
import re
from dataclasses import dataclass
from typing import List, Tuple

import pandas as pd
import streamlit as st
from pypdf import PdfReader
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


COMMON_SKILLS = {
    "python", "java", "sql", "aws", "azure", "gcp", "docker", "kubernetes",
    "nlp", "machine learning", "deep learning", "pytorch", "tensorflow", "scikit-learn",
    "data analysis", "data visualization", "excel", "tableau", "power bi", "spark",
    "hadoop", "git", "linux", "rest api", "flask", "django", "fastapi", "streamlit",
    "communication", "leadership", "project management", "statistics", "llm", "transformers",
}


@dataclass
class CandidateResult:
    name: str
    semantic_score: float
    skill_overlap_score: float
    final_score: float
    matched_skills: List[str]


def read_pdf(file_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(file_bytes))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s+#.-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_skills(text: str) -> List[str]:
    t = normalize_text(text)
    found = [skill for skill in COMMON_SKILLS if skill in t]
    return sorted(set(found))


def score_resumes(job_description: str, resumes: List[Tuple[str, str]]) -> List[CandidateResult]:
    docs = [normalize_text(job_description)] + [normalize_text(text) for _, text in resumes]

    vectorizer = TfidfVectorizer(stop_words=list(ENGLISH_STOP_WORDS), ngram_range=(1, 2), min_df=1)
    tfidf = vectorizer.fit_transform(docs)

    jd_vec = tfidf[0:1]
    resume_vecs = tfidf[1:]
    semantic_scores = cosine_similarity(resume_vecs, jd_vec).flatten()

    jd_skills = set(extract_skills(job_description))

    results: List[CandidateResult] = []
    for i, (name, resume_text) in enumerate(resumes):
        resume_skills = set(extract_skills(resume_text))
        if jd_skills:
            overlap = len(jd_skills.intersection(resume_skills)) / len(jd_skills)
        else:
            overlap = 0.0

        semantic = float(semantic_scores[i])
        final = (0.7 * semantic) + (0.3 * overlap)

        results.append(
            CandidateResult(
                name=name,
                semantic_score=semantic * 100,
                skill_overlap_score=overlap * 100,
                final_score=final * 100,
                matched_skills=sorted(jd_skills.intersection(resume_skills)),
            )
        )

    return sorted(results, key=lambda x: x.final_score, reverse=True)


def read_resume_file(uploaded_file) -> str:
    data = uploaded_file.read()
    if uploaded_file.name.lower().endswith(".pdf"):
        return read_pdf(data)
    return data.decode("utf-8", errors="ignore")


def app() -> None:
    st.set_page_config(page_title="NLP Resume Screener", page_icon="🧠", layout="wide")

    st.markdown(
        """
        <style>
        .main-title {font-size: 2.2rem; font-weight: 700; color: #0f172a; margin-bottom: 0.25rem;}
        .subtitle {color: #475569; margin-bottom: 1.5rem;}
        .card {background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 12px; padding: 1rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="main-title">🧠 NLP Resume Screening App</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Rank candidates with TF-IDF semantic matching + skill overlap scoring.</div>',
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.2, 1])

    with left:
        st.markdown("#### 1) Paste Job Description")
        job_description = st.text_area(
            "Job Description",
            height=280,
            placeholder="Paste the role requirements, must-have skills, responsibilities...",
            label_visibility="collapsed",
        )

    with right:
        st.markdown("#### 2) Upload Resumes")
        files = st.file_uploader(
            "Upload .pdf or .txt resumes",
            type=["pdf", "txt"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )
        st.caption("Tip: include skill keywords in JD for better matching quality.")

    run = st.button("Screen Candidates", type="primary", use_container_width=True)

    if run:
        if not job_description.strip():
            st.error("Please provide a job description.")
            return
        if not files:
            st.error("Please upload at least one resume.")
            return

        resumes: List[Tuple[str, str]] = []
        for f in files:
            resumes.append((f.name, read_resume_file(f)))

        results = score_resumes(job_description, resumes)

        st.markdown("---")
        st.markdown("### Ranked Candidates")

        df = pd.DataFrame(
            [
                {
                    "Rank": idx + 1,
                    "Candidate": r.name,
                    "Final Score": round(r.final_score, 2),
                    "Semantic Match": round(r.semantic_score, 2),
                    "Skill Match": round(r.skill_overlap_score, 2),
                    "Matched Skills": ", ".join(r.matched_skills) if r.matched_skills else "-",
                }
                for idx, r in enumerate(results)
            ]
        )

        st.dataframe(df, use_container_width=True, hide_index=True)

        st.markdown("### Top 3 Snapshot")
        for r in results[:3]:
            st.markdown(f"**{r.name}**")
            st.progress(min(max(r.final_score / 100, 0.0), 1.0), text=f"Overall: {r.final_score:.1f}%")
            st.caption(
                f"Semantic: {r.semantic_score:.1f}% | Skill overlap: {r.skill_overlap_score:.1f}% | "
                f"Matched skills: {', '.join(r.matched_skills) if r.matched_skills else 'None'}"
            )


if __name__ == "__main__":
    app()
