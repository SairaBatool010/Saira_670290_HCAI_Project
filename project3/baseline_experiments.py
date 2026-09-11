import json
import os

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline

from .config import BASELINE_PARAM_GRID, RANDOM_STATE, TASK1_DIR


def build_classifier_pipeline():
    return Pipeline(
        steps=[
            ("vectorizer", TfidfVectorizer()),
            ("classifier", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
        ]
    )


def _evaluate_model(model, X, y):
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)
    report = classification_report(y, predictions, output_dict=True, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y, predictions)),
        "macro_precision": float(precision_score(y, predictions, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y, predictions, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y, predictions, average="macro", zero_division=0)),
        "per_class": {
            str(label): {
                "precision": float(report[str(label)]["precision"]),
                "recall": float(report[str(label)]["recall"]),
                "f1": float(report[str(label)]["f1-score"]),
                "support": int(report[str(label)]["support"]),
            }
            for label in sorted(set(y))
        },
        "confusion_matrix": confusion_matrix(y, predictions).tolist(),
        "predictions": predictions,
        "probabilities": probabilities,
    }


def run_hyperparameter_search(splits, verbose=True):
    best_score = -1.0
    best_params = None
    search_rows = []
    grid = list(ParameterGrid(BASELINE_PARAM_GRID))

    for index, params in enumerate(grid, start=1):
        if verbose:
            print(f"[Task1] Hyperparameter search {index}/{len(grid)}: {params}")
        model = build_classifier_pipeline()
        model.set_params(**params)
        model.fit(splits["X_train_fit"], splits["y_train_fit"])
        val_metrics = _evaluate_model(model, splits["X_val"], splits["y_val"])
        row = {
            "params": params,
            "val_macro_f1": val_metrics["macro_f1"],
            "val_accuracy": val_metrics["accuracy"],
        }
        search_rows.append(row)
        if val_metrics["macro_f1"] > best_score:
            best_score = val_metrics["macro_f1"]
            best_params = params

    return {
        "best_params": best_params,
        "best_val_macro_f1": best_score,
        "search_rows": search_rows,
    }


def train_final_baseline(splits, best_params):
    final_model = build_classifier_pipeline()
    final_model.set_params(**best_params)
    final_model.fit(splits["X_train_full"], splits["y_train_full"])
    return final_model


def run_task1_baseline(splits):
    search = run_hyperparameter_search(splits)
    final_model = train_final_baseline(splits, search["best_params"])

    val_metrics = _evaluate_model(final_model, splits["X_val"], splits["y_val"])
    test_metrics = _evaluate_model(final_model, splits["X_test"], splits["y_test"])

    return {
        "model": final_model,
        "best_params": search["best_params"],
        "hyperparameter_search": search,
        "validation_metrics": {
            key: val_metrics[key]
            for key in [
                "accuracy",
                "macro_precision",
                "macro_recall",
                "macro_f1",
                "per_class",
                "confusion_matrix",
            ]
        },
        "test_metrics": {
            key: test_metrics[key]
            for key in [
                "accuracy",
                "macro_precision",
                "macro_recall",
                "macro_f1",
                "per_class",
                "confusion_matrix",
            ]
        },
        "validation_predictions": val_metrics["predictions"],
        "validation_probabilities": val_metrics["probabilities"],
        "test_predictions": test_metrics["predictions"],
        "test_probabilities": test_metrics["probabilities"],
    }


def save_task1_artifacts(result, splits, output_dir=TASK1_DIR):
    os.makedirs(output_dir, exist_ok=True)
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    model_path = os.path.join(output_dir, "task1_model.joblib")
    joblib.dump(result["model"], model_path)

    config = {
        "random_state": splits["random_state"],
        "val_size": splits["val_size"],
        "best_params": result["best_params"],
        "model_type": "TF-IDF + LogisticRegression",
    }
    config_path = os.path.join(output_dir, "task1_config.json")
    with open(config_path, "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)

    metrics = {
        "validation": result["validation_metrics"],
        "test_official": result["test_metrics"],
        "selection_criterion": "validation macro F1",
        "best_val_macro_f1": result["hyperparameter_search"]["best_val_macro_f1"],
    }
    metrics_path = os.path.join(output_dir, "task1_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    search_path = os.path.join(output_dir, "task1_hyperparameter_search.json")
    with open(search_path, "w", encoding="utf-8") as handle:
        json.dump(result["hyperparameter_search"]["search_rows"], handle, indent=2)

    return {
        "model_path": model_path,
        "config_path": config_path,
        "metrics_path": metrics_path,
        "search_path": search_path,
        "plots_dir": plots_dir,
    }
