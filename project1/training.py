import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from django.conf import settings
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC


CLASSIFICATION_MODELS = {
    "logistic_regression": {
        "label": "Logistic Regression",
        "param": "C",
        "param_label": "Regularization strength (C)",
        "param_description": "Inverse regularization strength — smaller values apply stronger regularization (simpler model), larger values fit the training data more closely.",
        "default_values": "0.1,1,10",
        "builder": lambda value: LogisticRegression(C=float(value), max_iter=1000),
    },
    "knn_classifier": {
        "label": "K-Nearest Neighbors",
        "param": "n_neighbors",
        "param_label": "Number of neighbors (k)",
        "param_description": "Number of nearest neighbors (k) used to vote on the predicted class. Small k can overfit noise; large k can oversmooth class boundaries.",
        "default_values": "3,5,7,9",
        "builder": lambda value: KNeighborsClassifier(n_neighbors=int(value)),
    },
    "svm": {
        "label": "Support Vector Machine",
        "param": "C",
        "param_label": "Regularization strength (C)",
        "param_description": "Regularization strength controlling the margin — smaller values allow a wider margin with more misclassifications, larger values fit the training data more tightly.",
        "default_values": "0.1,1,10",
        "builder": lambda value: SVC(C=float(value)),
    },
    "random_forest_classifier": {
        "label": "Random Forest",
        "param": "n_estimators",
        "param_label": "Number of trees (n_estimators)",
        "param_description": "Number of decision trees in the forest. More trees are generally more stable but slower to train.",
        "default_values": "10,50,100",
        "builder": lambda value: RandomForestClassifier(n_estimators=int(value), random_state=42),
    },
}

REGRESSION_MODELS = {
    "linear_regression": {
        "label": "Linear Regression",
        "param": None,
        "param_label": "No hyperparameter for this model",
        "param_description": "Ordinary linear regression has no hyperparameter to tune here — leave this field blank.",
        "default_values": "",
        "builder": lambda value: LinearRegression(),
    },
    "ridge": {
        "label": "Ridge Regression",
        "param": "alpha",
        "param_label": "Regularization strength (alpha)",
        "param_description": "L2 regularization strength — larger values shrink coefficients more, reducing overfitting at the cost of fitting the training data less tightly.",
        "default_values": "0.1,1,10",
        "builder": lambda value: Ridge(alpha=float(value)),
    },
    "knn_regressor": {
        "label": "K-Nearest Neighbors",
        "param": "n_neighbors",
        "param_label": "Number of neighbors (k)",
        "param_description": "Number of nearest neighbors (k) averaged to produce the predicted value. Small k can overfit noise; large k can oversmooth the trend.",
        "default_values": "3,5,7,9",
        "builder": lambda value: KNeighborsRegressor(n_neighbors=int(value)),
    },
    "random_forest_regressor": {
        "label": "Random Forest",
        "param": "n_estimators",
        "param_label": "Number of trees (n_estimators)",
        "param_description": "Number of decision trees in the forest. More trees are generally more stable but slower to train.",
        "default_values": "10,50,100",
        "builder": lambda value: RandomForestRegressor(n_estimators=int(value), random_state=42),
    },
}

CLASSIFICATION_METRICS = {
    "accuracy": ("Accuracy", accuracy_score, True),
    "f1_weighted": ("F1 score (weighted)", lambda y_true, y_pred: f1_score(y_true, y_pred, average="weighted"), True),
}

REGRESSION_METRICS = {
    "r2": ("R² score", r2_score, True),
    "mse": ("Mean squared error", mean_squared_error, False),
}


def detect_problem_type(label_series):
    if label_series.dtype == "object" or label_series.nunique() <= 10:
        return "classification"
    return "regression"


def prepare_dataset(df):
    feature_names = list(df.columns[:-1])
    label_name = df.columns[-1]
    all_numeric = df[feature_names].select_dtypes(include=[np.number])
    excluded_columns = [name for name in feature_names if name not in all_numeric.columns]
    X = all_numeric
    if X.empty:
        raise ValueError("No numeric feature columns were found for training.")

    if X.isnull().values.any() or df[label_name].isnull().values.any():
        raise ValueError(
            "The dataset contains missing values. Please clean the CSV (remove or fill missing "
            "cells) before training."
        )

    y = df[label_name]
    problem_type = detect_problem_type(y)

    label_encoder = None
    if problem_type == "classification" and not pd.api.types.is_numeric_dtype(y):
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y.astype(str))

    return X, y, label_name, problem_type, label_encoder, excluded_columns


def parse_hyperparameter_values(raw_values, model_config):
    if not model_config["param"]:
        return [None]

    if not raw_values.strip():
        raw_values = model_config["default_values"]

    values = []
    is_integer_param = model_config["param"] in ("n_neighbors", "n_estimators")
    for item in raw_values.split(","):
        cleaned = item.strip()
        if not cleaned:
            continue
        try:
            values.append(int(cleaned) if is_integer_param else float(cleaned))
        except ValueError:
            raise ValueError(
                f'"{cleaned}" is not a valid value for {model_config["param"]}. '
                f"Please enter comma-separated numbers, e.g. {model_config['default_values']}."
            )

    if not values:
        raise ValueError("Please provide at least one hyperparameter value.")
    return values


def _plot_score_by_hyperparameter(results, hyperparameter_name, metric_label, filename):
    labels = [str(r["hyperparameter"]) for r in results]
    scores = [r["score"] for r in results]

    plt.figure(figsize=(7, 5))
    plt.bar(labels, scores, color="#4C72B0")
    plt.xlabel(hyperparameter_name if hyperparameter_name != "none" else "Configuration")
    plt.ylabel(metric_label)
    plt.title(f"{metric_label} by {hyperparameter_name}" if hyperparameter_name != "none" else metric_label)
    plt.tight_layout()
    image_path = os.path.join(settings.MEDIA_ROOT, filename)
    plt.savefig(image_path)
    plt.close()
    return settings.MEDIA_URL + filename


def _plot_confusion_matrix(y_test, predictions, class_labels, filename):
    matrix = confusion_matrix(y_test, predictions)
    plt.figure(figsize=(6, 5))
    plt.imshow(matrix, cmap="Blues")
    plt.colorbar()
    tick_positions = range(len(class_labels))
    plt.xticks(tick_positions, class_labels, rotation=45, ha="right")
    plt.yticks(tick_positions, class_labels)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("Confusion matrix (best model)")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            plt.text(j, i, str(matrix[i, j]), ha="center", va="center", color="black")
    plt.tight_layout()
    image_path = os.path.join(settings.MEDIA_ROOT, filename)
    plt.savefig(image_path)
    plt.close()
    return settings.MEDIA_URL + filename


def _plot_predicted_vs_actual(y_test, predictions, filename):
    plt.figure(figsize=(6, 5))
    plt.scatter(y_test, predictions, alpha=0.7, color="#4C72B0")
    lims = [min(min(y_test), min(predictions)), max(max(y_test), max(predictions))]
    plt.plot(lims, lims, "--", color="gray", linewidth=1)
    plt.xlabel("Actual value")
    plt.ylabel("Predicted value")
    plt.title("Predicted vs. actual (best model)")
    plt.tight_layout()
    image_path = os.path.join(settings.MEDIA_ROOT, filename)
    plt.savefig(image_path)
    plt.close()
    return settings.MEDIA_URL + filename


def run_training(df, model_key, test_size, hyperparameter_values, metric_key):
    X, y, label_name, problem_type, label_encoder, excluded_columns = prepare_dataset(df)
    models = CLASSIFICATION_MODELS if problem_type == "classification" else REGRESSION_MODELS
    metrics = CLASSIFICATION_METRICS if problem_type == "classification" else REGRESSION_METRICS

    if model_key not in models:
        raise ValueError("Selected model is not available for this problem type.")
    if metric_key not in metrics:
        raise ValueError("Selected metric is not available for this problem type.")

    min_required_samples = 10
    if len(X) < min_required_samples:
        raise ValueError(
            f"The dataset only has {len(X)} row(s). At least {min_required_samples} are needed "
            "to train and evaluate a model."
        )

    n_test = round(len(X) * test_size)
    if n_test < 1 or (len(X) - n_test) < 1:
        raise ValueError(
            "The chosen test set size leaves no samples for training or testing. "
            "Please choose a value between 0.1 and 0.5 that fits your dataset size."
        )

    model_config = models[model_key]
    metric_label, metric_fn, higher_is_better = metrics[metric_key]
    parsed_values = parse_hyperparameter_values(hyperparameter_values, model_config)

    stratify = y if problem_type == "classification" else None
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=stratify
        )
    except ValueError:
        # Stratification fails if a class has too few members for the split; fall back.
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

    results = []
    best_predictions = None
    for value in parsed_values:
        model = model_config["builder"](value)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        score = metric_fn(y_test, predictions)
        results.append(
            {
                "hyperparameter": value if value is not None else "default",
                "score": round(float(score), 4),
                "predictions": predictions,
            }
        )

    if higher_is_better:
        best_result = max(results, key=lambda item: item["score"])
    else:
        best_result = min(results, key=lambda item: item["score"])
    best_predictions = best_result.pop("predictions")
    for result in results:
        result.pop("predictions", None)

    score_plot_url = _plot_score_by_hyperparameter(
        results, model_config["param"] or "none", metric_label, "project1_train_scores.png"
    )

    result_plot_url = None
    if problem_type == "classification":
        class_labels = (
            list(label_encoder.classes_) if label_encoder is not None else sorted(set(y_test) | set(best_predictions))
        )
        result_plot_url = _plot_confusion_matrix(
            y_test, best_predictions, class_labels, "project1_confusion_matrix.png"
        )
    else:
        result_plot_url = _plot_predicted_vs_actual(
            y_test, best_predictions, "project1_predicted_vs_actual.png"
        )

    return {
        "label_name": label_name,
        "problem_type": problem_type,
        "model_name": model_config["label"],
        "hyperparameter_name": model_config["param"] or "none",
        "metric_label": metric_label,
        "test_size": test_size,
        "train_percent": round((1 - test_size) * 100),
        "test_percent": round(test_size * 100),
        "results": results,
        "best_result": best_result,
        "feature_count": X.shape[1],
        "sample_count": X.shape[0],
        "classes": list(label_encoder.classes_) if label_encoder is not None else None,
        "excluded_columns": excluded_columns,
        "score_plot_url": score_plot_url,
        "result_plot_url": result_plot_url,
    }
