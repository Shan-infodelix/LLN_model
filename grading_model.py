import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import re
import torch
import language_tool_python
from sentence_transformers import SentenceTransformer, CrossEncoder, util
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity



def clean_text(text):
    if pd.isna(text):
        return ""

    # Convert to string
    text = str(text)

    # Lowercase
    text = text.lower()

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)

    # Normalize punctuation (remove special characters)
    text = re.sub(r'[^\w\s]', '', text)

    # Strip leading/trailing spaces
    text = text.strip()

    return text

# @title Default title text
def remove_stopwords(text):
    tokens = text.split()
    filtered_tokens = [word for word in tokens if word not in ENGLISH_STOP_WORDS]
    return " ".join(filtered_tokens)




def llm_api(question,answer_guide,student_ans):
  pass
# ===============================
# GLOBAL MODELS (Load once)
# ===============================

sbert_model = None
cross_model = None
grammar_tool = None


def load_models():
    global sbert_model, cross_model, grammar_tool

    if sbert_model is None:
        sbert_model = SentenceTransformer("all-MiniLM-L6-v2")

    if cross_model is None:
        cross_model = CrossEncoder("cross-encoder/stsb-roberta-base")

    if grammar_tool is None:
        grammar_tool = language_tool_python.LanguageTool('en-US')


# ===============================
# TF-IDF Score
# ===============================

def tfidf_score(sample_ans, student_ans):
      # Additional preprocessing ONLY for TF-IDF
    tf_correct = remove_stopwords(sample_ans)
    tf_student = remove_stopwords(student_ans)
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf = vectorizer.fit_transform([tf_correct, tf_student])
    return cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]


# ===============================
# SBERT Score
# ===============================

def sbert_score(sample_ans, student_ans):
    emb1 = sbert_model.encode(sample_ans, convert_to_tensor=True)
    emb2 = sbert_model.encode(student_ans, convert_to_tensor=True)
    return util.cos_sim(emb1, emb2).item()


# ===============================
# Cross Encoder Score
# ===============================

def cross_score(sample_ans, student_ans):
    score = cross_model.predict([(sample_ans, student_ans)])
    return float(score[0])


# ===============================
# Rubric Keyword Score
# ===============================

def rubric_score(sample_ans, student_ans):
    keywords = set(re.findall(r'\b\w+\b', sample_ans.lower()))
    student_words = set(re.findall(r'\b\w+\b', student_ans.lower()))

    overlap = keywords.intersection(student_words)

    return len(overlap) / len(keywords) if keywords else 0


# ===============================
# Grammar Score
# ===============================

def grammar_score(text):
    matches = grammar_tool.check(text)
    word_count = len(text.split())
    error_rate = len(matches) / word_count if word_count else 0
    return max(0, 1 - error_rate)


# ===============================
# Final Aggregation
# ===============================

def final_score(sample_ans, student_ans,question,answer_guid):
    is_sample = False
    if(sample_ans == ""):
        sample_ans = llm_api(question,answer_guid,student_ans)
        is_sample = True
    load_models()
    ans1 = clean_text(sample_ans)
    ans2 = clean_text(student_ans)

    t = tfidf_score(ans1, ans2)
    s = sbert_score(ans1, ans2)
    c = cross_score(ans1, ans2)
    r = rubric_score(ans1, ans2)
    g = grammar_score(student_ans)

    # Weighted industrial formula
    final = (
        0.25 * s +
        0.25 * c +
        0.15 * t +
        0.20 * r +
        0.15 * g
    ) * 100

    confidence = min(100, abs(s - c) * 100)
    response = {
            "final_score": round(final, 2),
            "semantic_score": round(s * 100, 2),
            "cross_score": round(c * 100, 2),
            "tfidf_score": round(t * 100, 2),
            "rubric_score": round(r * 100, 2),
            "grammar_score": round(g * 100, 2),
            "confidence": round(confidence, 2)
        }
    if is_sample:
        response["sample ans"] = sample_ans

    return response

