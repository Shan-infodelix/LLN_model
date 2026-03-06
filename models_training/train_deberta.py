# models_training/train_deberta.py

import json
import re
import os
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments
)
from datasets import Dataset




# ===============================================================================
# Temporary placeholder: DeBERTa training pipeline reserved for future implementation and currently not active in production.
# ================================================================================




def fine_tune_deberta(df, model_version):

    print("🚀 Fine-tuning DeBERTa...")

    # 1️⃣ Load current base version
    with open("models/current_version.json") as f:
        versions = json.load(f)

    base_version = versions["deberta"]
    model_path = f"models/{base_version}"

    print(f"Loading base model: {base_version}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    # 2️⃣ Prepare dataset
    df["text"] = df["student_answer"] + " [SEP] " + df["sample_answer"]
    df["label"] = df["true_score"]  # already normalized

    dataset = Dataset.from_pandas(df[["text", "label"]])

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=512
        )

    dataset = dataset.map(tokenize, batched=True)

    dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "label"]
    )

    # 3️⃣ Define new version name
    new_version = f"deberta_{model_version}"
    output_path = f"models/{new_version}"
    os.makedirs(output_path, exist_ok=True)

    # 4️⃣ Training args
    training_args = TrainingArguments(
        output_dir=output_path,
        num_train_epochs=2,
        per_device_train_batch_size=8,
        learning_rate=2e-5,
        logging_steps=10,
        save_strategy="no"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset
    )

    # 5️⃣ Train
    trainer.train()

    # 6️⃣ Save
    trainer.save_model(output_path)
    tokenizer.save_pretrained(output_path)

    print(f"✅ Saved new model: {new_version}")

    return new_version