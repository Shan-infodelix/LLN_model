
# models_training/train_cross.py

import os
import json
import torch
import torch.nn.functional as F
from datetime import datetime
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from sentence_transformers import CrossEncoder
from sentence_transformers.readers import InputExample
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sentence_transformers import losses
from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score


# Fine-tunes the Cross-Encoder model on newly labeled data, evaluates it against the current production model using validation/test metrics and generalization gap,
#  applies overfitting check, and saves the new version only if it outperforms the existing model.
def fine_tune_cross(df, model_version):

    print("🚀 Fine-tuning Cross Encoder...")

    # -----------------------------
    #  Load Current Model
    # -----------------------------
    with open("models/current_version.json") as f:
        versions = json.load(f)

    current_version = versions["cross"]
    model_path = f"models/{current_version}"
    model_curr = CrossEncoder(model_path, num_labels=1)
    model = CrossEncoder(model_path, num_labels=1)

    # -----------------------------
    #  Train / Val / Test Split
    # -----------------------------
    train_df, temp_df = train_test_split(
        df, test_size=0.3, random_state=42
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=42
    )
    # -----------------------------
    #  Convert to InputExamples
    # -----------------------------
    def create_examples(dataframe):
        return [
            InputExample(
                texts=[row["student_answer"], row["sample_answer"]],
                label=float(row["true_score"])
            )
            for _, row in dataframe.iterrows()
        ]

    train_examples = create_examples(train_df)
    val_examples = create_examples(val_df)
    test_examples = create_examples(test_df)

    # -----------------------------
    # Creates PyTorch DataLoaders for training, validation, and testing datasets with batching and appropriate shuffling strategy.
    # -----------------------------
    train_dataloader = DataLoader(
        train_examples,
        shuffle=True,
        batch_size=16
    )

    val_dataloader = DataLoader(
        val_examples,
        shuffle=False,
        batch_size=16
    )

    test_dataloader = DataLoader(
        test_examples,
        shuffle=False,
        batch_size=16
    )
    # -----------------------------
    # Train Model
    # -----------------------------
    model.fit(
        train_dataloader=train_dataloader,
        epochs=3,
        warmup_steps=50,
        show_progress_bar=True,
    )



    # Evaluates the model on given examples by computing regression metrics (MSE, RMSE, Pearson, Spearman) and classification accuracy after rounding predictions.
    def evaluate(model, examples):

        texts = [example.texts for example in examples]
        labels = np.array([example.label for example in examples])

        predictions = model.predict(texts)

        mse = np.mean((predictions - labels) ** 2)
        rmse = np.sqrt(mse)
        pearson_corr, _ = pearsonr(predictions, labels)
        spearman_corr, _ = spearmanr(predictions, labels)

        # 🔹 Convert to integer class (rounding)
        pred_classes = np.rint(predictions).astype(int)
        true_classes = labels.astype(int)

        accuracy = accuracy_score(true_classes, pred_classes)

        return {
            "mse": float(mse),
            "rmse": float(rmse),
            "pearson": float(pearson_corr),
            "spearman": float(spearman_corr),
            "accuracy": float(accuracy)
        }
    val_metrics_fineTune_model = evaluate(model, val_examples)
    test_metrics_fineTune_model = evaluate(model, test_examples)
    val_metrics_curr_model = evaluate(model_curr, val_examples)
    test_metrics_curr_model = evaluate(model_curr, test_examples)


    # Compares two models using validation metrics (with defined lower/higher rules), resolves ties using test metrics,
    #  evaluates generalization gap via RMSE difference, and returns the selected best-performing model with comparison details.
    def select_best_model(
        model1_name,
        val1, test1,
        model2_name,
        val2, test2
    ):
        """
        Select best model using validation metrics
        and confirm using test metrics.
        """
        # Compares two models metric-wise using predefined performance rules (lower or higher is better) and returns the total win count for each model.
        def score_model(m1, m2):
            score1 = 0
            score2 = 0

            lower_is_better = ["mse", "rmse"]
            higher_is_better = ["pearson", "spearman", "accuracy"]

            for metric in m1.keys():

                if metric in lower_is_better:
                    if m1[metric] < m2[metric]:
                        score1 += 1
                    elif m2[metric] < m1[metric]:
                        score2 += 1

                elif metric in higher_is_better:
                    if m1[metric] > m2[metric]:
                        score1 += 1
                    elif m2[metric] > m1[metric]:
                        score2 += 1

            return score1, score2


        print("\n📊 Metric Rules:")
        print("MSE → Lower ↓")
        print("RMSE → Lower ↓")
        print("Pearson → Higher ↑")
        print("Spearman → Higher ↑")


        # Step 1: Validation comparison
        val_score1, val_score2 = score_model(val1, val2)

        print("\n📈 Validation Score:")
        print(f"{model1_name}: {val_score1}")
        print(f"{model2_name}: {val_score2}")

        if val_score1 > val_score2:
            selected = model1_name
            selected_test = test1
            other_test = test2
        elif val_score2 > val_score1:
            selected = model2_name
            selected_test = test2
            other_test = test1
        else:
            print("\n⚠ Validation Tie — checking test metrics")
            test_score1, test_score2 = score_model(test1, test2)
            selected = model1_name if test_score1 > test_score2 else model2_name


        print(f"\n🎯 Selected Based on Validation: {selected}")

        # Step 2: Check generalization gap
        def generalization_gap(val, test):
            return abs(val["rmse"] - test["rmse"])

        gap1 = generalization_gap(val1, test1)
        gap2 = generalization_gap(val2, test2)

        print("\n🔍 Generalization Gap (|Val RMSE - Test RMSE|)")
        print(f"{model1_name}: {gap1:.4f}")
        print(f"{model2_name}: {gap2:.4f}")

        return {
            "selected_model": selected,
            "val_scores": (val_score1, val_score2),
            "generalization_gap": {
                model1_name: gap1,
                model2_name: gap2
            }
        }

    result = select_best_model(
        "FineTune_Model",
        val_metrics_fineTune_model,
        test_metrics_fineTune_model,
        "Current_Model",
        val_metrics_curr_model,
        test_metrics_curr_model
    )
    THRESHOLD = 0.1

    selected = result['selected_model']
    gap = result['generalization_gap'][selected]

    if gap < THRESHOLD:
        final_model = selected
    else:
        print("⚠ Overfitting detected. Falling back to Current_Model")
        final_model = "Current_Model"


    if(final_model == 'FineTune_Model'):
        # -----------------------------
        #  Save New Version
        # -----------------------------
        new_version = f"cross_{model_version}"
        save_path = f"models/{new_version}"

        os.makedirs(save_path, exist_ok=True)
        model.save(save_path)
        return model_version,val_metrics_fineTune_model,test_metrics_fineTune_model,gap

    else:
        version = current_version.replace("cross_", "")
        return version,None,None,None