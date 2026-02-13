from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import torch
import re

model = None
vectorizer = None


# -----------------------------
# Load model only once
# -----------------------------
def load_model():
    global model, vectorizer

    if model is None:
        model = SentenceTransformer("all-MiniLM-L6-v2")

    if vectorizer is None:
        vectorizer = TfidfVectorizer(stop_words="english")

    return model


# -----------------------------
# Text cleaning
# -----------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# -----------------------------
# TF-IDF Similarity
# -----------------------------
def tfidf_score(correct, student):
    correct = clean_text(correct)
    student = clean_text(student)

    tfidf = vectorizer.fit_transform([correct, student])
    score = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]

    return float(score)


# -----------------------------
# SBERT Similarity (Batch Encode)
# -----------------------------
def sbert_score(correct, student):
    m = load_model()

    sentences = [correct, student]
    embeddings = m.encode(sentences, convert_to_tensor=True)

    score = util.cos_sim(embeddings[0], embeddings[1])

    return float(score.item())


# -----------------------------
# Final Hybrid Score
# -----------------------------
def final_score(correct, student):

    if not correct.strip() or not student.strip():
        return 0.0

    t = tfidf_score(correct, student)
    s = sbert_score(correct, student)

    # Weighted hybrid scoring
    final = (0.2 * t + 0.8 * s) * 100

    # Clamp score between 0 and 100
    final = max(0, min(100, final))

    return round(final, 2)
