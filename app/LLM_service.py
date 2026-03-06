# app/llm_service.py

import os
import requests
from dotenv import load_dotenv

load_dotenv()


# Generates a structured academic prompt to create a high-quality model answer based on the question and answer guide.
def generate_sample_answer_prompt(question, answer_guide):
    
    prompt = f"""
You are a subject-matter expert and academic examiner.

Question:
{question}

Answer Guide:
{answer_guide}

Task:
Generate a high-quality model answer suitable for full marks (100%).

Instructions:
- The answer must fully satisfy the answer guide.
- Cover all key concepts mentioned in the guide.
- Maintain academic tone and clarity.
- Keep the answer concise but complete.
- Do NOT include explanations about the task.
- Do NOT mention that this is a generated answer.
- Return only the final model answer text.

Model Answer:
"""
    
    return prompt

# Calls the LLM API with the generated prompt to produce a deterministic, full-mark model answer and returns the extracted response text.
def llm_api(question,answer_guide):

    prompt = generate_sample_answer_prompt(question,answer_guide)
    load_dotenv()

    API_KEY = os.getenv("OPENROUTER_API_KEY")

    response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": "arcee-ai/trinity-large-preview:free",
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "reasoning": {"enabled": True},
            "temperature": 0.3  # more deterministic academic output
        }
    )

    if response.status_code != 200:
        raise Exception(f"LLM API Error: {response.text}")

    result = response.json()

    sample_answer = result["choices"][0]["message"]["content"].strip()

    return sample_answer