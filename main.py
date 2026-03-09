import os
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from grading_model import final_score, load_models
from dotenv import load_dotenv
from typing import List
load_dotenv() 

app = FastAPI()

# Load model once at startup
@app.on_event("startup")
def startup_event():
    load_models()

class AnswerRequest(BaseModel):
    question_id:str
    student_answer_id:str
    sample_answer: str
    student_answer: str
    question:str
    answer_Guide: str
class BatchAnswerRequest(BaseModel):
    submissions: List[AnswerRequest]

# Read API key from environment variable
API_KEY = os.getenv("MY_KEY")

@app.get("/")
def root():
    return {"status": "Model is running"}

@app.post("/grade")
def grade_answer(data: BatchAnswerRequest, x_api_key: str = Header(None)):

    if x_api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Unauthorized")
    results = []
    for item in data.submissions:
        score = final_score(item.sample_answer, item.student_answer,item.question,item.answer_Guide)
        results.append({
            "question_id": item.question_id,
            "student_answer_id": item.student_answer_id,
            **score
        })

    return {"scores": results}
