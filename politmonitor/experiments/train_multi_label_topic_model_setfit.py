import argparse
import os
import sys
from datetime import datetime

from datasets import Dataset
import pandas as pd
from datasets import load_dataset
from setfit import SetFitModel, SetFitTrainer, sample_dataset
from sklearn.metrics import f1_score

from training_data_handler import TrainingDataHandler

tdh = TrainingDataHandler()

# Load a SetFit model from Hub
# model_id = "sentence-transformers/all-mpnet-base-v2"
model_id = "clips/mfaq"

tdh = TrainingDataHandler()


def prepare_data():
    tdh.get_training_data(language='all', affair_text_scope=['zh', 'ch'], affair_attachment_category='all',
                          running_mode='experimental')
    dataset = tdh.training_data

    train_dataset = dataset["train"]
    validation_dataset = dataset["validation"]
    predict_dataset = dataset["test"]

    label_list = sorted(list(tdh.label2id.keys()))

    label2id = tdh.label2id
    id2label = tdh.id2label

    return label_list, label2id, id2label, train_dataset, validation_dataset, predict_dataset


def model_init(params):
    params = params or {}
    max_iter = params.get("max_iter", 12)
    solver = params.get("solver", "liblinear")
    params = {
        "head_params": {
            "max_iter": max_iter,
            "solver": solver,
        }
    }
    return SetFitModel.from_pretrained(model_id, multi_target_strategy="one-vs-rest", **params)


def hp_space(trial):  # Training parameters
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True),
        "num_epochs": trial.suggest_int("num_epochs", 1, 5),
        "batch_size": trial.suggest_categorical("batch_size", [4, 8, 16, 32, 64]),
        "seed": trial.suggest_int("seed", 1, 40),
        "num_iterations": trial.suggest_categorical("num_iterations", [5, 10, 20]),
        "max_iter": trial.suggest_int("max_iter", 50, 300),
        "solver": trial.suggest_categorical("solver", ["newton-cg", "lbfgs", "liblinear"]),
    }


def convert_id2label(label_output: list, id2label: dict) -> list:
    label_output = list(label_output)

    label_output_indices_of_positive_values = list()

    for n, lp in enumerate(label_output):
        if lp == 1:
            label_output_indices_of_positive_values.append(n)

    label_output_as_labels = [id2label[label_id] for label_id in label_output_indices_of_positive_values]

    return label_output_as_labels


def training(save_model=True, metric="macro-f1"):
    label_list, label2id, id2label, train_dataset, validation_dataset, df_test = prepare_data()

    # Create trainer
    trainer = SetFitTrainer(
        metric=metric,
        metric_kwargs={"average": "macro"},
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        model_init=model_init,
        column_mapping={"input": "text", "one_hot_representation": "label"}
    )

    best_run = trainer.hyperparameter_search(direction="maximize", hp_space=hp_space, n_trials=5)

    trainer.apply_hyperparameters(best_run.hyperparameters, final_model=True)

    # Train and evaluate
    trainer.train()
    metrics = trainer.evaluate()
    print("*** Evaluation ***")
    print(metrics)

    if str(save_model).lower() == "true":
        path_to_best_model = os.path.join(os.path.dirname(__file__), "setfit/best_model")
        if os.path.exists(path_to_best_model) == False:
            os.mkdir(path_to_best_model)
        trainer.model._save_pretrained(path_to_best_model)

    print('Making predictions on test set.')

    predictions = trainer.model.predict_proba(df_test.Tweet.tolist())
    predictions = [[1 if p > 0.5 else 0 for p in pred] for pred in predictions]

    df_test['predicted_labels'] = predictions
    df_test['labels_as_text'] = df_test.labels.apply(lambda x: convert_id2label(x, id2label))
    df_test['predicted_label_as_text'] = df_test.predicted_labels.apply(lambda x: convert_id2label(x, id2label))
    print(df_test.head())

    f1 = f1_score(df_test.labels.tolist(), df_test.predicted_labels.tolist(), average="macro")

    return f1, trainer.model, df_test


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('-sm', '--save_model', help='', default=True)

    args = parser.parse_args()

    if os.path.exists('reports') == False:
        os.mkdir('reports')

    f1, loaded_model, df_test = training(metric='f1', save_model=bool(args.save_model))
    time_stamp = datetime.now().isoformat()
    df_test.to_excel('reports/predictions_from' + time_stamp + '.xlsx', index=False)
    with open('reports/f1_for_emotion_detection' + '_from_' + time_stamp + '.txt', 'w') as f:
        print(f1, file=f)
    print('F1', f1)
