import os
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from grading_model import final_score, load_models
from dotenv import load_dotenv
load_dotenv() 

app = FastAPI()

# Load model once at startup
@app.on_event("startup")
def startup_event():
    load_models()

class AnswerRequest(BaseModel):
    sample_answer: str
    student_answer: str
    question:str
    answer_Guide: str

# Read API key from environment variable
API_KEY = os.getenv("MY_KEY")

@app.get("/")
def root():
    return {"status": "Model is running"}

@app.post("/grade")
def grade_answer(data: AnswerRequest, x_api_key: str = Header(None)):

    if x_api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Unauthorized")

    score = final_score(data.sample_answer, data.student_answer,data.question,data.answer_Guide)

    return {"score": score}
