import json
import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score


# Computes the sigmoid activation function using NumPy to convert linear outputs into probability values between 0 and 1.
def sigmoid(z):
    z = np.array(z, dtype=np.float64)  # ensure numeric
    return 1 / (1 + np.exp(-z))


# Computes regularized logistic loss by applying sigmoid to linear predictions, calculating binary cross-entropy, and adding L2 weight penalty.
def compute_log_loss(input, output, weights, bias):

    input = np.array(input, dtype=np.float64)
    output = np.array(output, dtype=np.float64)
    weights = np.array(weights, dtype=np.float64)
    bias = float(bias)

    z = np.dot(input, weights) + bias
    probs = sigmoid(z)

    eps = 1e-8
    loss = -np.mean(
        output * np.log(probs + eps) +
        (1 - output) * np.log(1 - probs + eps)
    )
    C = 1.0
    lambda_reg = 1.0 / C
    l2_penalty = (lambda_reg / 2.0) * np.sum(weights ** 2)
    return float(loss + l2_penalty)


# Trains and validates a logistic regression weight layer on extracted feature scores, 
# compares it with the existing production weight model using validation loss and accuracy, 
# and saves the new version only if it performs better.
def train_weight_layer(df, model_version):

    print("🚀 Training Logistic Weight Layer...")

    # -----------------------------
    #  1️⃣ Performs one-hot encoding for grading modes and prepares normalized feature matrix and labels.
    # -----------------------------
    df = pd.get_dummies(df, columns=["grading_mode"], prefix="mode")

    for col in ["mode_strict", "mode_moderate", "mode_light"]:
        if col not in df.columns:
            df[col] = 0

    feature_columns = [
        "semantic_score",
        "cross_score",
        "deberta_score",
        "tf_idf_score",
        "rubic_score",
        "grammar_score",
        "mode_strict",
        "mode_moderate",
        "mode_light"
    ]


    Input_feature = df[feature_columns]/100.0
    output_value = df["true_score"].astype(int)   # Must be 0 or 1


    Input_train, Input_test, output_train, output_test = train_test_split(
        Input_feature, output_value, test_size=0.3, random_state=42
    )


    # -----------------------------
    # 2️⃣ Train Logistic Regression
    # -----------------------------
    model = LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="lbfgs",
        max_iter=1000,
        class_weight="balanced" 
    )

    model.fit(Input_train, output_train)

    new_weights = model.coef_[0].tolist()
    new_bias = float(model.intercept_[0])
    # 2️⃣ computes validation loss.
    new_loss = compute_log_loss(Input_test.values, output_test.values, new_weights, new_bias)


    # Predict probabilities
    score_probability = model.predict_proba(Input_test)[:, 1]

    # Convert to class labels (threshold = 0.5)
    Answer_prediction = (score_probability >= 0.5).astype(int)


    # computes accuracy.
    new_accuracy = accuracy_score(output_test, Answer_prediction)

    print("New Model Validation Loss:", new_loss)
    print("New Model Validation Accuracy:", new_accuracy)

    # Loads the current production weight model, evaluates it on validation data by computing loss and accuracy, and prints its performance for comparison with the newly trained model.
    currect_path = "models/current_version.json"
    loss = None
    accuracy = None
    if os.path.exists(currect_path):

        with open(currect_path) as f:
            current = json.load(f)

        currect_weight_file = current["weights"]
        current_version = currect_weight_file.replace("weights_", "").replace(".json", "")

        with open(f"models/{currect_weight_file}") as f:
            currect_model = json.load(f)

        currect_weights = np.array(currect_model["weights"])
        currect_bias = currect_model["bias"]

        currect_loss = compute_log_loss(
            Input_test.values,
            output_test.values,
            currect_weights,
            currect_bias
        )
        currect_probability = sigmoid(np.dot(Input_test.values, currect_weights) + currect_bias)
        currect_prediction = (currect_probability >= 0.5).astype(int)

        currect_accuracy = accuracy_score(output_test, currect_prediction)

        print("Previous Model Validation Accuracy:", currect_accuracy)
        print("Previous Model Validation Loss:", currect_loss)

        # Compare
        if new_loss < currect_loss and new_accuracy >= currect_accuracy:
            loss = new_loss
            accuracy = new_accuracy
            print("✅ New model is better. Updating version.")
        else:
            print("❌ Previous model is better. Keeping old.")
            return current_version,loss,accuracy,None

    # -----------------------------
    # 3️⃣ Save model weights
    # -----------------------------
    save_dict = {
        "features": feature_columns,
        "weights": new_weights,
        "bias": new_bias,
        "version": model_version
    }

    os.makedirs("models", exist_ok=True)

    weight_path = f"models/weights_{model_version}.json"
    with open(weight_path, "w") as f:
        json.dump(save_dict, f, indent=4)

    print(f"✅ Logistic weight layer saved: weights_{model_version}.json")



    return model_version,loss,accuracy,save_dict