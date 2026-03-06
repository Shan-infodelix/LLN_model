# app/feature_engineering.py

import re
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import util

# Cleans and normalizes text by converting to lowercase, removing extra spaces, special characters, and trimming whitespace.
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()



# Removes English stopwords from the text to retain only meaningful words.
def remove_stopwords(text):
    tokens = text.split()
    return " ".join([w for w in tokens if w not in ENGLISH_STOP_WORDS])


# Computes cosine similarity between TF-IDF vectors of sample and student answers after stopword removal.
def tfidf_score(sample, student):
    tf_sample = remove_stopwords(sample)
    tf_student = remove_stopwords(student)

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf = vectorizer.fit_transform([tf_sample, tf_student])
    return cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]

# Calculates keyword overlap ratio between sample and student answers as a rubric-based similarity score.
def rubric_score(sample, student):
    keywords = set(re.findall(r'\b\w+\b', sample.lower()))
    student_words = set(re.findall(r'\b\w+\b', student.lower()))
    overlap = keywords.intersection(student_words)
    return len(overlap) / len(keywords) if keywords else 0


# Evaluates grammatical quality by computing error rate per word and converting it into a normalized grammar score.
def grammar_score(text, grammar_tool):

    matches = grammar_tool.check(text)
    word_count = len(text.split())
    error_rate = len(matches) / word_count if word_count else 0
    return max(0, 1 - error_rate)


# Generates semantic similarity score using Instructor embedding model with normalized vector cosine similarity.
def instructor_score(sample, student, instructor_model):
    instruction = "Represent the answer for grading comparison:"
    emb1 = instructor_model.encode([[instruction, sample]], convert_to_tensor=True, normalize_embeddings=True)
    emb2 = instructor_model.encode([[instruction, student]], convert_to_tensor=True, normalize_embeddings=True)
    return util.cos_sim(emb1, emb2).item()


# Computes cross-encoder similarity score between sample and student answers.
def cross_score(sample, student, cross_model):
    return float(cross_model.predict([(sample, student)])[0])


# Uses DeBERTa NLI model to calculate entailment-based similarity score by combining entailment, neutral, and contradiction probabilities.
def deberta_nli_score(sample, student, tokenizer, model):
    inputs = tokenizer(sample, student, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)

    probs = F.softmax(outputs.logits, dim=1)[0]
    label_map = model.config.label2id

    entailment = probs[label_map["entailment"]].item()
    neutral = probs[label_map["neutral"]].item()
    contradiction = probs[label_map["contradiction"]].item()

    return max(0,entailment + 0.5 * neutral - 0.25 * contradiction)



















