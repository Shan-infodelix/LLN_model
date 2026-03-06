import mysql.connector
from datetime import datetime
import os
from dotenv import load_dotenv
import pandas as pd
import json

load_dotenv()

# Establishes and returns a MySQL database connection using environment configuration variables.
def get_connection():
    return mysql.connector.connect(
        host=os.getenv("DB_HOST"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        database=os.getenv("DB_NAME")
    )


# Fetches all trainer-labeled samples that have not yet been used for model training, including feature scores and true scores.
def fetch_untrained_samples():
    conn = get_connection()


    query = """
    SELECT 
        q.question_id,
        s.student_answer_id,
        q.sample_answer,
        s.student_answer,
        t.trainer_score AS true_score,

        m.grading_mode,
        m.semantic_score,
        m.cross_score,
        m.deberta_score,
        m.tf_idf_score,
        m.rubic_score,
        m.grammar_score

    FROM Question q

    JOIN Student_answer s 
        ON q.question_id = s.question_id

    JOIN Trainer_Score t
        ON s.question_id = t.question_id
        AND s.student_answer_id = t.student_answer_id

    JOIN Model_Score m
        ON s.question_id = m.question_id
        AND s.student_answer_id = m.student_answer_id

    WHERE 
        t.trained_in_version IS NULL
    """

    df = pd.read_sql(query, conn)
    conn.close()
    return df


# Retrieves previously trained samples for the current model version to support incremental or evaluation-based training.
def fetch_previous_trained_samples():
    conn = get_connection()
    with open("models/current_version.json") as f:
        versions = json.load(f)

    current_version = versions["cross"][6:]


    query = """
    SELECT 
        q.question_id,
        s.student_answer_id,
        q.sample_answer,
        s.student_answer,
        t.trainer_score AS true_score,

        m.grading_mode,
        m.semantic_score,
        m.cross_score,
        m.deberta_score,
        m.tf_idf_score,
        m.rubic_score,
        m.grammar_score

    FROM Question q

    JOIN Student_answer s 
        ON q.question_id = s.question_id

    JOIN Trainer_Score t
        ON s.question_id = t.question_id
        AND s.student_answer_id = t.student_answer_id

    JOIN Model_Score m
        ON s.question_id = m.question_id
        AND s.student_answer_id = m.student_answer_id

    WHERE 
        t.trained_in_version = %s
    """

    df = pd.read_sql(query, conn,params=[current_version])
    conn.close()
    return df


# Marks given samples as trained by updating their trained version in the Trainer_Score table.
def mark_samples_as_trained(df, version):
    conn = get_connection()
    cursor = conn.cursor()

    query = """
        UPDATE Trainer_score
        SET trained_in_version = %s
        WHERE question_id = %s AND student_answer_id = %s
    """

    for _, row in df.iterrows():
        cursor.execute(query, (version, row["question_id"], row["student_answer_id"]))

    conn.commit()
    cursor.close()
    conn.close()



# Stores training session metadata including model version, sample count, timestamps, and training status.
def save_training_metadata(model_name, model_version, sample_count, status):
    conn = get_connection()
    cursor = conn.cursor()

    query = """
        INSERT INTO Training_metadata (
            model_name,
            model_version,
            trained_on_samples,
            training_started,
            training_completed,
            status
        )
        VALUES (%s, %s, %s, NOW(), NOW(), %s)
    """

    cursor.execute(query, (
        model_name,
        model_version,
        sample_count,
        status
    ))

    conn.commit()
    training_id = cursor.lastrowid

    cursor.close()
    conn.close()

    return training_id


import json


# Saves cross-encoder validation and test metrics, archives previous metrics if needed, and tracks generalization gap.
def save_cross_model_metrics(training_id, metrics_dict_val, metrics_dict_test, gap, is_current=False):
    conn = get_connection()
    cursor = conn.cursor()

    # Archive only same model type (optional improvement)
    if is_current:
        cursor.execute("""
            UPDATE Cross_Model_Metrics
            SET status = 'archived'
            WHERE training_id IN (
                SELECT id FROM Training_metadata
            )
        """)

    query = """
        INSERT INTO Cross_Model_Metrics (
            training_id,
            validation_metrics,
            test_metrics,
            generalization_gap,
            status
        )
        VALUES (%s, %s, %s, %s, %s)
    """

    cursor.execute(query, (
        training_id,
        json.dumps(metrics_dict_val),
        json.dumps(metrics_dict_test),
        gap,
        'current' if is_current else 'archived'
    ))

    conn.commit()
    cursor.close()
    conn.close()



# Saves trained weight-model parameters (loss, accuracy, weights, bias), archiving previous production versions if required.
def save_weight_model_metrics(training_id, loss, accuracy, weights_dict, bias, is_current=False):
    conn = get_connection()
    cursor = conn.cursor()

    # If production model → archive previous
    if is_current:
        cursor.execute("""
            UPDATE Weight_Model_Metrics
            SET status = 'archived'
        """)

    query = """
        INSERT INTO Weight_Model_Metrics(
            training_id,
            loss,
            accuracy,
            weights,
            bias,
            status
        )
        VALUES (%s, %s, %s, %s, %s, %s)
    """

    cursor.execute(query, (
        training_id,
        loss,
        accuracy,
        json.dumps(weights_dict),
        bias,
        'current' if is_current else 'archived'
    ))

    conn.commit()
    cursor.close()
    conn.close()




# Returns the total number of trainer-labeled samples that are not yet used for training.
def get_untrained_sample_count():
    conn = get_connection()
    cursor = conn.cursor()

    query = """
        SELECT COUNT(*)
        FROM Trainer_score
        WHERE trained_in_version IS NULL
    """

    cursor.execute(query)
    count = cursor.fetchone()[0]

    cursor.close()
    conn.close()

    return count