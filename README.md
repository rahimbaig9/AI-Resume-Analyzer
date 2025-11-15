# AI Resume Analyzer

Upload a resume PDF → get skills, ATS-like score, job-match score, and suggestions.

## Features
- PDF text extraction (pdfplumber)
- Embedding-based skill extraction (sentence-transformers)
- Job-match score (semantic similarity)
- ATS-like score and improvement suggestions
- Clean Tailwind UI; easy to deploy

## Run locally
1. Backend
```bash
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -m nltk.downloader stopwords punkt
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
