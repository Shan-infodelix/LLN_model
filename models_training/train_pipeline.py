import pandas as pd
import uuid
from database.prepare_dataset import fetch_untrained_samples
from models_training.train_cross import fine_tune_cross
from models_training.train_deberta import fine_tune_deberta
from models_training.train_weight import train_weight_layer
from database.prepare_dataset import mark_samples_as_trained, save_training_metadata,fetch_previous_trained_samples,save_cross_model_metrics,save_weight_model_metrics
import json

BATCH_SIZE = 25



# Updates the active model configuration by setting the new Cross-Encoder and weight versions
def update_cross_version(cross_v):
    with open("models/current_version.json") as f:
        versions = json.load(f)

    deberta_v = versions["deberta"]
    weight_v = versions["weights"]
    version_data = {
        "cross": cross_v,
        "deberta": deberta_v,
        "weights": weight_v
    }

    with open("models/current_version.json", "w") as f:
        json.dump(version_data, f, indent=4)


def update_weight_version(weight_v):
    with open("models/current_version.json") as f:
        versions = json.load(f)

    deberta_v = versions["deberta"]
    cross_v = versions["cross"]
    version_data = {
        "cross": cross_v,
        "deberta": deberta_v,
        "weights": weight_v
    }

    with open("models/current_version.json", "w") as f:
        json.dump(version_data, f, indent=4)


def main():

    # Step 1: Pull only untrained labeled samples from DB
    null_df = fetch_untrained_samples()

    if len(null_df) < BATCH_SIZE:
        print("Not enough new samples.")
        return

    # Retrieves previously trained samples for the current model version from the database for incremental training or evaluation.
    pre_df = fetch_previous_trained_samples()

    import pandas as pd

    df = pd.concat([null_df, pre_df], ignore_index=True)


    print(f"Starting training on {len(df)} samples...")


    # Generate new version name
    model_version = "v_" + str(uuid.uuid4())[:8]

    # Fine-tunes the Cross-Encoder model on new data and returns the selected model version along with validation metrics, test metrics, and generalization gap.
    new_cross_version,val_metrics,test_metrics,generalization_gap = fine_tune_cross(df, model_version)


    # Stores metadata of the completed Cross-Encoder training session, including version, sample count, and status, and returns the training record ID.
    cross_training_id = save_training_metadata(
        model_name="cross_encoder",
        model_version=model_version,
        sample_count=len(df),
        status="completed"
    )

    # If fine-tuned model metrics are available, save validation/test performance and generalization gap as the current production Cross-Encoder metrics.
    if val_metrics != None or test_metrics != None or generalization_gap != None:
        save_cross_model_metrics(
            training_id=cross_training_id,
            metrics_dict_val=val_metrics,
            metrics_dict_test=test_metrics,
            gap=generalization_gap,
            is_current=True
        )
        # Updates the cross version.
        update_cross_version(
        f"cross_{new_cross_version}"
        )



    # Trains the logistic regression weight layer on the dataset and returns the new version name along with loss, accuracy, and learned weight parameters.
    new_weight_version,new_loss,new_accuracy,dict = train_weight_layer(df, model_version)


    # Saves logistic weight-layer training metadata and, if training outputs exist, stores the learned weights and bias as the current production model.
    logistic_training_id = save_training_metadata(
        model_name="logistic",
        model_version=model_version,
        sample_count=len(df),
        status="completed"
    )
    if new_loss != None or new_accuracy != None or dict != None:
        save_weight_model_metrics(
            training_id=logistic_training_id,
            loss=new_loss,
            accuracy=new_accuracy,
            weights_dict=dict.get("weights"),
            bias=dict.get("bias"),
            is_current=True
        )
         # Updates the weight version.
        update_weight_version(f"weights_{new_weight_version}.json");
   

    
    # Step 4: Mark samples as trained
    mark_samples_as_trained(df, model_version)

    print(f"Training completed successfully. Version: {model_version}")


if __name__ == "__main__":
    main()








