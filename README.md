# NLP Resume Screening App

A simple, clean **NLP-based resume screening web app** built with **Streamlit**.

## Features

- Upload multiple resumes (`.pdf` or `.txt`)
- Paste a job description
- NLP-based ranking using:
  - **TF-IDF + Cosine Similarity** (semantic match)
  - **Skill overlap score** from job-description skills
- Final weighted score:
  - `70% semantic match`
  - `30% skill overlap`
- Clean UI with ranked table + top candidate snapshot

## Tech Stack

- Python
- Streamlit
- scikit-learn
- pandas
- pypdf

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL shown by Streamlit (usually `http://localhost:8501`).

## Notes

- Better results come from clear job descriptions with explicit required skills.
- This is a starter screening tool; for production use, add stronger parsing, model calibration, and bias checks.
