# SWIFTLLN: Subjective Answer Auto-Scoring Model V2.0

SWIFTLLN is an **AI-powered automated grading system for subjective answers** designed for scalable educational platforms.
It evaluates descriptive student responses using a **multi-model ensemble architecture** and continuously improves through **trainer feedback and adaptive retraining**.

The system integrates semantic similarity models, transformer-based evaluation, statistical methods, rubric analysis, and grammar scoring to produce an accurate prediction of whether an answer is **correct or incorrect**.

The model acts as the **backend evaluation engine for the SWIFTLLN assessment platform**.

---

# Project Overview

Evaluating subjective answers manually is:

* time-consuming
* inconsistent
* difficult to scale

SWIFTLLN V2.0 solves this problem by implementing an **adaptive automated grading framework** that combines multiple NLP models with a meta-learning layer.

The system evaluates answers based on:

* semantic understanding
* contextual alignment
* logical correctness
* keyword coverage
* grammatical quality

Each evaluation module produces a score which is combined using a **logistic regression meta-learner** to produce the final prediction.

---

# Key Capabilities

• Automated subjective answer grading
• Multi-model NLP scoring pipeline
• Logistic regression meta-learning
• Batch API processing
• Trainer feedback correction system
• Continuous model retraining
• Controlled model deployment strategy
• Model version management
• LLM fallback for missing reference answers

---

# System Architecture

The architecture follows a modular pipeline:

```text
API Request
     │
     ▼
Authentication Layer
     │
     ▼
Multi-Model Scoring Engine
     │
     ▼
Feature Vector Construction
     │
     ▼
Logistic Regression Meta-Learner
     │
     ▼
Probability Prediction
     │
     ▼
Database Storage
     │
     ▼
Trainer Feedback API
     │
     ▼
Retraining Pipeline
     │
     ▼
Model Comparison
     │
     ▼
Production Deployment
```

This architecture enables **continuous learning and controlled improvement of model accuracy**.

---

# Multi-Model Scoring Engine

The system evaluates answers using six independent modules:

| Module                              | Purpose                       |
| ----------------------------------- | ----------------------------- |
| Semantic Similarity (Instructor-XL) | Measures conceptual alignment |
| Cross-Encoder                       | Contextual pair evaluation    |
| DeBERTa (NLI)                       | Logical inference validation  |
| TF-IDF Similarity                   | Statistical text similarity   |
| Rubric Scoring                      | Domain keyword coverage       |
| Grammar Evaluation                  | Linguistic correctness        |

Each module produces a normalized score:

```
si ∈ [0,1]
```

These scores form the feature vector used by the meta-learning model.

---

# Feature Vector

The system constructs a structured feature vector:

```
X = [semantic, cross, deberta, tfidf, rubric, grammar,
     mode_strict, mode_moderate, mode_light]
```

Grading modes are encoded using **one-hot encoding**.

---

# Logistic Regression Meta-Learner

Instead of manually assigning weights, Version 2.0 learns optimal weights using logistic regression.

Linear model:

```
z = Σ(wi * xi) + b
```

Probability calculation:

```
P(y=1) = 1 / (1 + e^-z)
```

Prediction rule:

```
Correct if P(y=1) > 0.5
Otherwise Incorrect
```

This probabilistic framework improves interpretability and adaptive learning.

---

# Continuous Learning Pipeline

SWIFTLLN uses a **Human-in-the-Loop learning system**.

### Workflow

1. Student answer is evaluated by the model.
2. Prediction and sub-scores are stored in the database.
3. Trainer reviews the result.
4. Trainer may override the prediction.
5. Trainer feedback becomes new training data.

When enough feedback samples accumulate, the system triggers retraining.

---

# Training Trigger Logic

Retraining is activated when:

```
Untrained Feedback Samples ≥ Threshold
```

Training pipeline:

1. Fetch untrained feedback data
2. Merge with historical training dataset
3. Fine-tune cross-encoder model
4. Recalculate logistic regression weights
5. Evaluate new model performance

---

# Controlled Model Deployment

To maintain stability, the system compares the **new model** with the **current production model**.

If:

```
New Model Performance > Current Model
```

Then:

* Update model version
* Deploy new model

Otherwise:

* Discard update
* Keep existing model

This prevents performance degradation in production systems.

---

# LLM Fallback Reference Answer Generation

If a reference answer is missing, the system automatically generates one using an LLM.

Condition:

```
if sample_answer == None
```

Then:

```
reference_answer = LLM(question)
```

This ensures the grading pipeline always has a reference answer for evaluation.

---

# Database Architecture

The system uses a relational database to support continuous learning.

Main tables:

| Table             | Purpose                                 |
| ----------------- | --------------------------------------- |
| Question          | Stores question metadata                |
| Student Answer    | Stores student responses                |
| Model Score       | Stores model predictions and sub-scores |
| Trainer Score     | Stores human corrections                |
| Training Metadata | Tracks training runs                    |
| Cross Model       | Stores cross-encoder versions           |
| Weight Model      | Stores logistic regression weights      |

This design supports **traceability, retraining, and version management**.

---

# API Input Format

Example batch request:

```json
{
 "submissions":[
  {
   "question_id":"Q1",
   "question":"Explain photosynthesis",
   "student_answer_id":"A1",
   "student_answer":"Plants produce energy using sunlight",
   "sample_answer":"...",
   "grading_mode":"moderate"
  }
 ]
}
```

---

# API Response Format

```json
{
 "results":[
  {
   "question_id":"Q1",
   "student_answer_id":"A1",
   "final_prediction":"Correct",
   "probability":0.87,
   "semantic_score":0.82,
   "cross_score":0.91,
   "deberta_score":0.88,
   "tfidf_score":0.76,
   "rubric_score":0.80,
   "grammar_score":0.90
  }
 ]
}
```

---

# Technologies Used

* Python
* HuggingFace Transformers
* Sentence Transformers
* DeBERTa NLI models
* Scikit-learn
* TF-IDF
* Logistic Regression
* NLP Feature Engineering
* REST APIs

---

# Applications

The system can be deployed in:

* Universities
* Online examination platforms
* EdTech platforms
* Corporate training systems
* Learning Management Systems

---

# Future Improvements

• Multi-class scoring (not just binary correct/incorrect)
• Multilingual answer evaluation
• Explainable AI feedback for students
• Automated rubric extraction
• Instructor analytics dashboard

---

# Author

SHAN
AI/ML Intern — INFODELIX

Project: **SWIFTLLN: Subjective Answer Auto-Scoring Model V2.0**
