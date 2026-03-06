# app/scorer.py

import numpy as np
from app.load_models import load_models
from app.feature_engineering import *
from app.LLM_service import llm_api
import json
import joblib
import math


# Loads the current model version from configuration, retrieves the corresponding weight file name, 
# and returns the weight parameters from the selected JSON file.
def load_weights():

    with open("models/current_version.json") as f:
        current = json.load(f)

    curr_weight = current["weights"]

    with open(f"models/{curr_weight}") as f:
        weight_data = json.load(f)
        return weight_data
    

# convert value into a probability between 0 and 1.
def sigmoid(z):
    return 1 / (1 + math.exp(-z))





# Generates a final evaluation score by computing multiple similarity and quality metrics, 
# applying trained weights, passing through sigmoid for probability, determining prediction based on threshold, 
# and returning detailed scoring with confidence.
def final_score(sample_ans, student_ans, question, answer_guid,checking_mode):

    is_sample = False
    if sample_ans == "":
        sample_ans = llm_api(question, answer_guid)
        is_sample = True

    instructor_model, cross_model, grammar_tool, deberta_model, deberta_tokenizer = load_models()

    ans1 = clean_text(sample_ans)
    ans2 = clean_text(student_ans)

    t = tfidf_score(ans1, ans2)
    s = instructor_score(ans1, ans2, instructor_model)
    d = deberta_nli_score(sample_ans, student_ans, deberta_tokenizer, deberta_model)
    c = cross_score(ans1, ans2, cross_model)
    r = rubric_score(ans1, ans2)
    g = grammar_score(student_ans, grammar_tool)

    # -----------------------------
    # Load trained weight layer
    # -----------------------------
    weight_data = load_weights()

    # -----------------------------
    # Build full feature dictionary
    # -----------------------------
    feature_dict = {
        "semantic_score": s,
        "cross_score": c,
        "deberta_score": d,
        "tf_idf_score": t,
        "rubic_score": r,
        "grammar_score": g,
        "mode_strict": 1 if checking_mode == "strict" else 0,
        "mode_moderate": 1 if checking_mode == "moderate" else 0,
        "mode_light": 1 if checking_mode == "light" else 0
    }

    # Keep correct order
    values = [feature_dict[f] for f in weight_data["features"]]

    # Linear prediction
    score = sum(w * v for w, v in zip(weight_data["weights"], values)) + weight_data["bias"]
    # Clip between 0–1
    prob = sigmoid(score)

    # ---------------------------------
    # 6️⃣ Mode-based threshold
    # ---------------------------------
    threshold = 0.5

    prediction = 1 if prob >= threshold else 0

    # -----------------------------
    # Confidence (agreement-based)
    # -----------------------------
    agreement = [s, c, d]
    variance = np.var(agreement)
    confidence = (1 - variance) * 100

    response = {
        "final_probability": round(prob * 100, 2),
        "final_prediction": prediction,
        "semantic_score": round(s * 100, 2),
        "cross_score": round(c * 100, 2),
        "deberta_model_score": round(d * 100, 2),
        "tfidf_score": round(t * 100, 2),
        "rubric_score": round(r * 100, 2),
        "grammar_score": round(g * 100, 2),
        "confidence": round(confidence, 2)
    }

    if is_sample:
        response["sample_ans"] = sample_ans

    return response