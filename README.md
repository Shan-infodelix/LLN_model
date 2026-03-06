# Subjective Answer Auto Scoring System (V2.0)

An **AI-powered system for automatic evaluation of subjective answers** using Natural Language Processing and Transformer-based models.

The system predicts whether a student's answer is **correct or incorrect**, assigns a score, and continuously improves using **trainer feedback and automated retraining**.

---

# Overview

Evaluating subjective answers manually is time-consuming and inconsistent.
This project automates the evaluation process using **multiple NLP models and feature engineering techniques**.

The system combines:

* Semantic similarity
* Cross Encoder scoring
* DeBERTa classification
* TF-IDF similarity
* Grammar evaluation
* Rubric scoring

The final score is calculated using a **weighted logistic scoring system**.

---

# Key Features

* Automatic subjective answer evaluation
* Multi-model NLP scoring pipeline
* Weighted scoring system
* Trainer feedback system
* Continuous model improvement
* Automatic retraining
* Model version control
* Production-ready ML pipeline

---

# Project Structure

```
LLM_MODEL
│
├── app
│   ├── feature_engineering.py
│   ├── LLM_service.py
│   ├── load_models.py
│   ├── main.py
│   └── score.py
│
├── database
│   ├── db.py
│   └── prepare_dataset.py
│
├── models
│   ├── cross_v1
│   ├── deberta_v1
│   ├── weights_v1.json
│   ├── current_version.json
│   └── __init__.py
│
├── models_training
│   ├── train_cross.py
│   ├── train_deberta.py
│   ├── train_pipeline.py
│   └── train_weight.py
│
├── checkpoints
│
├── init_models.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

# Model Architecture

The scoring system generates multiple features from a student's answer.

### Feature Extraction

* Semantic similarity score
* Cross Encoder similarity
* DeBERTa inference score
* TF-IDF similarity
* Rubric score
* Grammar score

### Scoring Formula

Final score is computed using weighted scoring:

```
score = Σ(weight_i × feature_i) + bias
```

Weights are stored in:

```
models/weights_v1.json
```

---

# Continuous Learning Pipeline

The system uses a **Human-in-the-Loop training pipeline**.

### Workflow

1. Student submits an answer.
2. The model predicts whether the answer is **correct or incorrect**.
3. A trainer reviews the prediction.
4. Trainer feedback is stored in the database.
5. Each feedback becomes a **new labeled training datapoint**.
6. When dataset size reaches a predefined **batch threshold**, training is triggered automatically.
7. A new model is trained using updated data.
8. The new model is compared with the **current production model**.

### Model Replacement Logic

If the new model performs **better**:

* Deploy the new model
* Update version control

If performance is **worse**:

* Ignore the new model
* Keep the current model

The same process applies to **logistic regression weight optimization**.

---

# Continuous Learning Architecture

```
Student Answer
      │
      ▼
Model Prediction
      │
      ▼
Trainer Verification
      │
      ▼
Feedback Stored in Database
      │
      ▼
Dataset Growth
      │
      ▼
Batch Threshold Reached
      │
      ▼
Automatic Training Trigger
      │
      ▼
New Model Evaluation
      │
 ┌───────────────┐
 │ Performance   │
 │ Comparison    │
 └───────────────┘
      │
 ┌────┴────┐
 ▼         ▼
Better     Worse
Model      Model
 ▼          ▼
Deploy      Ignore
```

---

# Model Initialization

Before running the system, initialize the models.

Run:

```
python init_models.py
```

This script will:

* Download the Cross Encoder model
* Download the DeBERTa model
* Create initial scoring weights
* Create the version control configuration

---

# Running the System

Install dependencies:

```
pip install -r requirements.txt
```

Initialize models:

```
python init_models.py
```

Run the application:

```
python app/main.py
```

---

# Model Version Control

Model versions are managed using:

```
models/current_version.json
```

Example configuration:

```
{
  "cross": "cross_v1",
  "deberta": "deberta_v1",
  "weights": "weights_v1.json"
}
```

This allows easy switching between model versions.

---

# Training System

Training scripts are located in:

```
models_training/
```

Available training components:

* Cross Encoder fine-tuning
* DeBERTa fine-tuning
* Logistic regression weight optimization
* Training pipeline automation

---

# Technologies Used

* Python
* HuggingFace Transformers
* Sentence Transformers
* Scikit-learn
* NLP feature engineering
* TF-IDF
* Logistic scoring models

---

# Future Improvements

* Explainable scoring system
* Answer feedback generation
* Instructor analytics dashboard
* Rubric auto-generation
* Model monitoring system

---

# Author

SHAN

AI / NLP Project

Subjective Answer Auto Scoring System V2.0
