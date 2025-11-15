# backend/main.py
import io
import os
from typing import Optional, List
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pdfplumber
from sentence_transformers import SentenceTransformer, util
import nltk
from nltk.corpus import stopwords
import re
import json

# Initialize resources
MODEL_NAME = os.environ.get("EMBED_MODEL", "all-MiniLM-L6-v2")
model = SentenceTransformer(MODEL_NAME)
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
STOPWORDS = set(stopwords.words("english"))

app = FastAPI(title="AI Resume Analyzer")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# A small curated skill seed list (expandable). Use in README to show how to extend.
SEED_SKILLS = [
    "python","java","c++","javascript","react","nodejs","fastapi","flask","docker","kubernetes",
    "aws","gcp","azure","sql","postgresql","mongodb","tensorflow","pytorch","nlp","computer vision",
    "deep learning","machine learning","data analysis","pandas","numpy","scikit-learn"
]

class AnalysisResult(BaseModel):
    skills: List[str]
    skill_scores: dict
    top_sentences: List[str]
    job_match_score: float
    ats_score: float
    suggestions: List[str]

# Utilities
def extract_text_from_pdf(file_bytes: bytes) -> str:
    text_parts = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            txt = page.extract_text()
            if txt:
                text_parts.append(txt)
    return "\n".join(text_parts)

def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def sentence_chunk(text: str, max_sent_len=200):
    sentences = nltk.tokenize.sent_tokenize(text)
    chunks = []
    cur = ""
    for s in sentences:
        if len(cur) + len(s) < max_sent_len:
            cur += " " + s
        else:
            if cur.strip():
                chunks.append(cur.strip())
            cur = s
    if cur.strip():
        chunks.append(cur.strip())
    return chunks

def simple_keyword_skills(text: str, seed_skills=SEED_SKILLS):
    found = {}
    low = text.lower()
    for skill in seed_skills:
        if skill.lower() in low:
            found[skill] = found.get(skill, 0) + 1
    return found

def embedding_skill_extraction(text: str, seed_skills=SEED_SKILLS, top_k=10):
    sentences = sentence_chunk(text)
    if not sentences:
        return {}, []
    sent_emb = model.encode(sentences, convert_to_tensor=True)
    skill_emb = model.encode(seed_skills, convert_to_tensor=True)
    sim = util.cos_sim(skill_emb, sent_emb)  # shape: (len(skills), len(sentences))
    # Aggregate max sim per skill
    skill_scores = {}
    for i, skill in enumerate(seed_skills):
        max_score = float(sim[i].max())
        if max_score > 0.45:  # threshold; tweak as needed
            skill_scores[skill] = max_score
    # top sentences that matter
    # for display, grab sentences with high similarity to any skill
    max_per_sentence = util.cos_sim(sent_emb, skill_emb).max(axis=1).values.tolist()
    top_idx = sorted(range(len(sentences)), key=lambda i: max_per_sentence[i], reverse=True)[:5]
    top_sentences = [sentences[i] for i in top_idx]
    return skill_scores, top_sentences

def compute_job_match(resume_text: str, job_text: str):
    if not job_text or len(job_text.strip()) < 10:
        return 0.0
    a = model.encode([resume_text], convert_to_tensor=True)
    b = model.encode([job_text], convert_to_tensor=True)
    sim = util.cos_sim(a, b).item()
    # normalize to 0-100
    return round(float((sim + 1) / 2 * 100), 2)

def compute_ats_score(resume_text: str, found_skills: dict, job_text: str):
    # simple ATS-like heuristics:
    # - keyword coverage: percentage of job keywords present
    # - presence of contact/email
    # - length heuristic (200-800 words preferred)
    words = nltk.word_tokenize(resume_text)
    word_count = len(words)
    score = 0
    # length score
    if 200 <= word_count <= 800:
        score += 30
    else:
        # penalize extreme lengths
        score += max(0, 30 - abs(400 - word_count) / 20)
    # contact/email check
    if re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", resume_text):
        score += 20
    # keyword coverage from job_text
    if job_text and len(job_text.strip()) > 10:
        job_words = set([w.lower() for w in nltk.word_tokenize(job_text) if w.isalpha() and w.lower() not in STOPWORDS])
        matched = 0
        for skill in found_skills.keys():
            if skill.lower() in job_words:
                matched += 1
        if len(job_words) == 0:
            kw_score = 0
        else:
            kw_score = min(30, (matched / max(1, len(job_words))) * 30)
        score += kw_score
    else:
        # no job provided: give some base skill coverage
        score += min(20, len(found_skills) * 3)
    # normalize to 0-100
    return round(min(100, score), 2)

def generate_suggestions(found_skills: dict, top_sentences: List[str], job_text: str):
    suggestions = []
    if not found_skills:
        suggestions.append("Add a skills section listing technologies, frameworks, and tools you've used.")
    else:
        if len(found_skills) < 5:
            suggestions.append("Add more specific skills (libraries, tools) and quantify experience: e.g., 'Pandas (2 years)'.")
    if job_text and len(job_text.strip()) > 10:
        suggestions.append("Tailor the top bullet points to match the job description keywords for higher ATS match.")
    if any(len(s) < 40 for s in top_sentences):
        suggestions.append("Expand short achievement bullets into 2-3 lines explaining outcome and metrics.")
    suggestions.append("Use action verbs and quantify results (%, counts, speedups).")
    suggestions.append("Consider adding a short summary (2-3 lines) at top emphasizing role + years + top skills.")
    return suggestions

@app.post("/analyze", response_model=AnalysisResult)
async def analyze_resume(
    file: UploadFile = File(...),
    job_description: Optional[str] = Form(None)
):
    content = await file.read()
    text = extract_text_from_pdf(content)
    text = clean_text(text)
    if not text:
        return AnalysisResult(
            skills=[],
            skill_scores={},
            top_sentences=[],
            job_match_score=0.0,
            ats_score=0.0,
            suggestions=["Could not extract text from the uploaded PDF."]
        )
    # 1) embedding-based skill extraction
    skill_scores, top_sentences = embedding_skill_extraction(text)
    # fallback: keyword scan
    if not skill_scores:
        skill_scores = simple_keyword_skills(text)
    # prepare results
    skills_sorted = sorted(skill_scores.items(), key=lambda x: x[1], reverse=True)
    skills_list = [s for s, score in skills_sorted]
    # compute match and ats score
    job_match = compute_job_match(text, job_description or "")
    ats = compute_ats_score(text, skill_scores, job_description or "")
    suggestions = generate_suggestions(skill_scores, top_sentences, job_description or "")
    # return numeric-friendly skill_scores
    numeric_skill_scores = {k: round(float(v), 3) for k, v in skill_scores.items()}
    return AnalysisResult(
        skills=skills_list,
        skill_scores=numeric_skill_scores,
        top_sentences=top_sentences,
        job_match_score=job_match,
        ats_score=ats,
        suggestions=suggestions
    )
