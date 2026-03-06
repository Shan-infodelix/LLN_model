# app/load_models.py

import json
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import language_tool_python


# Initializes global model variables to enable lazy loading and reuse across requests.
instructor_model = None
cross_model = None
grammar_tool = None
deberta_model = None
deberta_tokenizer = None

# Loads model version configuration and lazily initializes Instructor embedding model, Cross-Encoder, Grammar tool, and version-controlled DeBERTa NLI model, then returns all loaded components.
def load_models():
    global instructor_model, cross_model, grammar_tool, deberta_model, deberta_tokenizer

    # Load version controller
    with open("models/current_version.json") as f:
        versions = json.load(f)

    # 1️⃣ Instructor (can stay fixed or versioned later)
    if instructor_model is None:
        instructor_model = SentenceTransformer("hkunlp/instructor-xl")

    # 2️⃣ Cross Encoder (Load from saved version)
    if cross_model is None:
        cross_model = CrossEncoder(f"models/{versions['cross']}")

    # 3️⃣ Grammar Tool
    if grammar_tool is None:
        grammar_tool = language_tool_python.LanguageTool("en-US")

    # 4️⃣ DeBERTa (Load from saved version)
    if deberta_model is None:
        deberta_tokenizer = AutoTokenizer.from_pretrained(
            f"models/{versions['deberta']}"
        )

        deberta_model = AutoModelForSequenceClassification.from_pretrained(
            f"models/{versions['deberta']}"
        )

        deberta_model.eval()

    return instructor_model, cross_model, grammar_tool, deberta_model, deberta_tokenizer