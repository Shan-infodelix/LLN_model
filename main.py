from fastapi import FastAPI
from pydantic import BaseModel
from grading_model import final_score

from fastapi import Header, HTTPException

app = FastAPI()

class AnswerRequest(BaseModel):
    sample_answer: str
    student_answer: str




API_KEY = "your_secret_key"

@app.post("/grade")
def grade_answer(data: AnswerRequest, x_api_key: str = Header(None)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Unauthorized")

    score = final_score(data.correct_answer, data.student_answer)
    return {"score": score}

