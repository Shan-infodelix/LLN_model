import os
import json
from sentence_transformers import CrossEncoder
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Create models folder if not exists and run this file first

# Initializes the model environment by creating the models directory, downloading and saving the base Cross-Encoder and DeBERTa models locally, 
# generating initial logistic weight configuration, and creating the version control file for production setup.
os.makedirs("models", exist_ok=True)

print("Saving Cross Encoder...")

cross_model = CrossEncoder("cross-encoder/stsb-roberta-base")
cross_model.save("models/cross_v1")

print("Saving DeBERTa...")

model_name = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

model.save_pretrained("models/deberta_v1")
tokenizer.save_pretrained("models/deberta_v1")

print("Creating initial weights...")

weights_data = {
    "features": [
        "semantic_score",
        "cross_score",
        "deberta_score",
        "tf_idf_score",
        "rubic_score",
        "grammar_score",
        "mode_strict",
        "mode_moderate",
        "mode_light"
    ],
    "weights": [
        0.35,
        0.30,
        0.25,
        0.15,
        0.15,
        0.10,
        -0.25,
        -0.10,
        0.15
    ],
    "bias": -0.3,
    "version": "manual_stronger_v1"
}

# save json file
with open("models/weights_v1.json", "w") as f:
    json.dump(weights_data, f, indent=4)

print("Creating version controller...")

version_data = {
    "cross": "cross_v1",
    "deberta": "deberta_v1",
    "weights": "weights_v1.json"
}

with open("models/current_version.json", "w") as f:
    json.dump(version_data, f, indent=4)

print("✅ Initial model setup complete.")