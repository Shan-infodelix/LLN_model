import os
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from app.score import final_score
from app.load_models import load_models
from database.db import insert_trainer_score,insert_student_answer,insert_Question,insert_model_score
from dotenv import load_dotenv
from database.prepare_dataset import  get_untrained_sample_count
import subprocess
import sys
from typing import List
load_dotenv() 

app = FastAPI()


# Defines the data schema for a single student answer submission
class AnswerItem(BaseModel):
    question_id:str
    question:str
    answer_Guide: str
    student_answer_id: str
    student_answer: str
    sample_answer: str
    grading_mode: str
    
# Defines the request schema for processing multiple student answer submissions in a single batch.
class BatchAnswerRequest(BaseModel):
    submissions: List[AnswerItem]

# Read API key from environment variable
API_KEY = os.getenv("MY_KEY")

@app.get("/")
def root():
    return {"status": "Model is running"}



# API endpoint to securely process batch answer submissions, compute final scores, store question/answer/model data in the database, and return structured grading results.
@app.post("/grade")
def grade_answer_batch(data: BatchAnswerRequest, x_api_key: str = Header(None)):

    if x_api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Unauthorized")

    results = []

    for item in data.submissions:

        score = final_score(
            item.sample_answer,
            item.student_answer,
            item.question,
            item.answer_Guide,
            item.grading_mode
        )

        generate_sample_answer = score.get("sample_ans") or item.sample_answer

        insert_Question(
            question_id=item.question_id,
            question=item.question,
            sample_answer=generate_sample_answer
        )

        insert_student_answer(
            student_answer_id=item.student_answer_id,
            question_id=item.question_id,
            student_answer=item.student_answer
        )

        insert_model_score({
            "question_id": item.question_id,
            "student_answer_id": item.student_answer_id,
            "grading_mode": item.grading_mode,
            "semantic_score": score["semantic_score"],
            "cross_score": score["cross_score"],
            "deberta_score": score["deberta_model_score"],
            "tf_idf_score": score["tfidf_score"],
            "rubic_score": score["rubric_score"],
            "grammar_score": score["grammar_score"]
        })

        results.append({
            "question_id": item.question_id,
            "student_answer_id": item.student_answer_id,
            **score
        })

    return {"results": results}

# Defines the data schema for a trainer’s feedback
class TrainerFeedback(BaseModel):
    question_id: str
    student_answer_id: str
    trainer_score: float
# Defines the request schema for submitting multiple trainer feedback entries in a single batch.
class BatchFeedbackRequest(BaseModel):
    Feedback: List[TrainerFeedback]




BATCH_SIZE = 1000


# Launches the model training pipeline as a separate background process.
def trigger_training_worker():
    subprocess.Popen(
        [sys.executable, "-m", "models_training.train_pipeline"]
    )

# API endpoint to store trainer feedback securely, monitor untrained sample count, and trigger model retraining automatically when the batch threshold is reached.
@app.post("/trainer-feedback")
def trainer_feedback(data: BatchFeedbackRequest, x_api_key: str = Header(None)):

    if x_api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Unauthorized")

    # 1️⃣ Insert trainer score
    for item in data.Feedback:
        insert_trainer_score(
            question_id=item.question_id,
            student_answer_id=item.student_answer_id,
            trainer_score=item.trainer_score
        )

    # 2️⃣ Check untrained sample count
    count = get_untrained_sample_count()

    # 3️⃣ Trigger training if threshold reached
    if count >= BATCH_SIZE:
        trigger_training_worker()

    return {
        "message": "Trainer feedback stored successfully",
        "untrained_samples": count
    }
