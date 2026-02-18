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
import requests
import json
import os
from dotenv import load_dotenv
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

load_dotenv() 


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



def generate_sample_answer_prompt(question, answer_guide):
    
    prompt = f"""
You are a subject-matter expert and academic examiner.

Question:
{question}

Answer Guide:
{answer_guide}

Task:
Generate a high-quality model answer suitable for full marks (100%).

Instructions:
- The answer must fully satisfy the answer guide.
- Cover all key concepts mentioned in the guide.
- Maintain academic tone and clarity.
- Keep the answer concise but complete.
- Do NOT include explanations about the task.
- Do NOT mention that this is a generated answer.
- Return only the final model answer text.

Model Answer:
"""
    
    return prompt


def llm_api(question,answer_guide):

    prompt = generate_sample_answer_prompt(question,answer_guide)
    load_dotenv()

    API_KEY = os.getenv("OPENROUTER_API_KEY")

    response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": "openrouter/aurora-alpha",
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "reasoning": {"enabled": True},
            "temperature": 0.3  # more deterministic academic output
        }
    )

    if response.status_code != 200:
        raise Exception(f"LLM API Error: {response.text}")

    result = response.json()

    sample_answer = result["choices"][0]["message"]["content"].strip()

    return sample_answer
# ===============================
# GLOBAL MODELS (Load once)
# ===============================

instructor_model = None
cross_model = None
grammar_tool = None
deberta_model = None
deberta_tokenizer = None


def load_models():
    global instructor_model, cross_model, grammar_tool,deberta_model,deberta_tokenizer

    if instructor_model is None:
        instructor_model = SentenceTransformer("hkunlp/instructor-xl")

    if cross_model is None:
        cross_model = CrossEncoder("cross-encoder/stsb-roberta-base")

    if grammar_tool is None:
        grammar_tool = language_tool_python.LanguageTool('en-US')

    if deberta_model is None:
        # Load tokenizer and model
        deberta_model_name = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
        deberta_tokenizer = AutoTokenizer.from_pretrained(deberta_model_name)
        deberta_model = AutoModelForSequenceClassification.from_pretrained(deberta_model_name)
        deberta_model.eval()


# ===============================
# deberta_nli_score
# ===============================

def deberta_nli_score(sample_answer, student_answer):

    inputs = deberta_tokenizer(
        sample_answer,
        student_answer,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )

    with torch.no_grad():
        outputs = deberta_model(**inputs)

    probs = F.softmax(outputs.logits, dim=1)[0]

    label_map = deberta_model.config.label2id

    entailment = probs[label_map["entailment"]].item()
    neutral = probs[label_map["neutral"]].item()
    contradiction = probs[label_map["contradiction"]].item()

    # Better scoring logic
    score = entailment + 0.5*neutral-0.25*contradiction

    return score


# ===============================
# TF-IDF Score
# ===============================

def tfidf_score(sample_ans, student_ans):
      # Additional preprocessing ONLY for TF-IDF
    tf_sample = remove_stopwords(sample_ans)
    tf_student = remove_stopwords(student_ans)
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf = vectorizer.fit_transform([tf_sample, tf_student])
    return cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]


# ===============================
# semantic_score
# ===============================

def instructor_score(sample_ans, student_ans):

    instruction = "Represent the answer for grading comparison:"

    emb1 = instructor_model.encode(
        [[instruction, sample_ans]],
        convert_to_tensor=True,
        normalize_embeddings=True
    )

    emb2 = instructor_model.encode(
        [[instruction, student_ans]],
        convert_to_tensor=True,
        normalize_embeddings=True
    )

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
        sample_ans = llm_api(question,answer_guid)
        is_sample = True
    load_models()
    ans1 = clean_text(sample_ans)
    ans2 = clean_text(student_ans)

    t = tfidf_score(ans1, ans2)
    s = instructor_score(ans1, ans2)
    d = deberta_nli_score(sample_ans, student_ans)
    c = cross_score(ans1, ans2)
    r = rubric_score(ans1, ans2)
    g = grammar_score(student_ans)

    # Weighted industrial formula

    final = (
        0.30 * s +
        0.25 * c +
        0.15 * d +
        0.10 * t +
        0.10 * r +
        0.10 * g
    ) * 100
    if(c < 0.15):
        final = 0
        s = c
    features = [s, c, d]
    variance = np.var(features)
    confidence = (1 - variance) * 100
    response = {
          "final_score": round(final, 2),
          "semantic_score": round(s * 100, 2),
          "cross_score": round(c * 100, 2),
          "deberta_model_score": round(d * 100,2),
          "tfidf_score": round(t * 100, 2),
          "rubric_score": round(r * 100, 2),
          "grammar_score": round(g * 100, 2),
          "confidence": round(confidence, 2)
        }
    if is_sample:
        response["sample ans"] = sample_ans

    return response

