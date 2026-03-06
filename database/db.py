# database/db.py

import mysql.connector
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()


# ===============================
# Database Connection
# ===============================
def get_connection():
    return mysql.connector.connect(
        host=os.getenv("DB_HOST"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        database=os.getenv("DB_NAME")
    )


# ===============================
# Inserts a new question with its sample answer into the database or updates it if the question ID already exists, then commits and closes the connection.
# ===============================
def insert_Question(question_id, question, sample_answer):

    conn = get_connection()
    cursor = conn.cursor()

    query = """
        INSERT INTO Question (
            question_id,
            question,
            sample_answer,
            created_at
        )
        VALUES (%s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            question = VALUES(question),
            sample_answer = VALUES(sample_answer)
    """

    values = (
        question_id,
        question,
        sample_answer,
        datetime.now()
    )

    cursor.execute(query, values)
    conn.commit()

    cursor.close()
    conn.close()


# ===============================
# Inserts a student answer into the database and ignores duplicates based on the primary key, then commits and closes the connection.
# ===============================
def insert_student_answer(student_answer_id, question_id, student_answer):

    conn = get_connection()
    cursor = conn.cursor()

    query = """
        INSERT INTO Student_answer (
            student_answer_id,
            question_id,
            student_answer,
            created_at
        )
        VALUES (%s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            student_answer_id = student_answer_id
    """

    values = (
        student_answer_id,
        question_id,
        student_answer,
        datetime.now()
    )

    cursor.execute(query, values)
    conn.commit()

    cursor.close()
    conn.close()


# ===============================
# Inserts or updates the trainer’s assigned score for a student answer in the database and records the latest timestamp.
# ===============================
def insert_trainer_score(question_id, student_answer_id, trainer_score):

    conn = get_connection()
    cursor = conn.cursor()

    query = """
        INSERT INTO Trainer_Score (
            question_id,
            student_answer_id,
            trainer_score,
            created_at
        )
        VALUES (%s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            trainer_score = VALUES(trainer_score),
            created_at = VALUES(created_at)
    """

    values = (
        question_id,
        student_answer_id,
        trainer_score,
        datetime.now()
    )

    cursor.execute(query, values)
    conn.commit()

    cursor.close()
    conn.close()


# ===============================
# Inserts or updates the model-generated feature scores for a student answer based on grading mode, then commits and closes the connection.
# ===============================
def insert_model_score(data):

    conn = get_connection()
    cursor = conn.cursor()

    query = """
        INSERT INTO Model_Score (
            question_id,
            student_answer_id,
            grading_mode,
            semantic_score,
            cross_score,
            deberta_score,
            tf_idf_score,
            rubic_score,
            grammar_score
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            grading_mode = VALUES(grading_mode),
            semantic_score = VALUES(semantic_score),
            cross_score = VALUES(cross_score),
            deberta_score = VALUES(deberta_score),
            tf_idf_score = VALUES(tf_idf_score),
            rubic_score = VALUES(rubic_score),
            grammar_score = VALUES(grammar_score)
    """

    values = (
        data["question_id"],
        data["student_answer_id"],
        data["grading_mode"],
        data["semantic_score"],
        data["cross_score"],
        data["deberta_score"],
        data["tf_idf_score"],
        data["rubic_score"],
        data["grammar_score"]
    )

    cursor.execute(query, values)
    conn.commit()

    cursor.close()
    conn.close()