from fastapi import FastAPI
from pydantic import BaseModel
from grading_model import final_score

app = FastAPI()

class AnswerRequest(BaseModel):
    sample_answer: str
    student_answer: str

@app.post("/grade")
def grade_answer(data: AnswerRequest):
    score = final_score(data.sample_answer, data.student_answer)
    return {"score": score}
