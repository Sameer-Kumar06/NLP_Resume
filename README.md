# NLP Resume Screening App (Flask + NLTK)

A simple NLP-based resume screening web app with a clean UI, built using **Flask** and **NLTK**.

## What this project includes

- Clean Flask UI for screening resumes
- Dataset-driven role selection (`data/job_roles_dataset.csv`)
- Optional custom job description input
- Upload multiple resumes (`.pdf`, `.txt`)
- NLP scoring with:
  - **NLTK preprocessing** (tokenization, stopword removal, lemmatization)
  - **TF-IDF + cosine similarity** for semantic matching
  - **Skill overlap score** using dataset skill catalog
- Final weighted score (`75% semantic + 25% skills`)

## Project structure

- `app.py` – Flask app and NLP pipeline
- `templates/index.html` – UI template
- `static/styles.css` – styling
- `data/job_roles_dataset.csv` – sample job role dataset

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:5000`.

## Notes

- On first run, NLTK resources are auto-downloaded if missing.
- This is a starter screening tool; tune scoring and skills taxonomy for production.
