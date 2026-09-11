import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from .data import build_preprocessor


TREE_LEAF_OPTIONS = [2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 30, 50, None]
LOGISTIC_C_OPTIONS = [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 100.0]

# The two complexity measures (leaf count vs. L1 coefficient norm) live on very
# different scales, so a shared lambda range leaves one slider mostly "dead."
# These ceilings were picked so the full slider range covers every transition
# in each model bank's accuracy-vs-complexity trade-off (see selection_score).
LAMBDA_MAX = {"tree": 0.01, "logistic": 0.05}


def tree_complexity(model):
    return int(model.named_steps["classifier"].get_n_leaves())


def logistic_complexity(model):
    coefficients = model.named_steps["classifier"].coef_.ravel()
    return float(np.sum(np.abs(coefficients)))


def train_tree_models(dataset):
    models = []
    for max_leaf_nodes in TREE_LEAF_OPTIONS:
        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                (
                    "classifier",
                    DecisionTreeClassifier(
                        max_leaf_nodes=max_leaf_nodes,
                        random_state=42,
                    ),
                ),
            ]
        )
        pipeline.fit(dataset["X_train"], dataset["y_train"])
        y_pred = pipeline.predict(dataset["X_test"])
        accuracy = accuracy_score(dataset["y_test"], y_pred)
        complexity = tree_complexity(pipeline)
        models.append(
            {
                "id": f"tree_{max_leaf_nodes if max_leaf_nodes is not None else 'full'}",
                "model_type": "tree",
                "setting_label": f"max_leaf_nodes={max_leaf_nodes if max_leaf_nodes is not None else 'None'}",
                "parameter": max_leaf_nodes,
                "pipeline": pipeline,
                "accuracy": float(accuracy),
                "complexity": float(complexity),
            }
        )
    return models


def train_logistic_models(dataset):
    models = []
    for c_value in LOGISTIC_C_OPTIONS:
        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                (
                    "classifier",
                    LogisticRegression(
                        C=c_value,
                        max_iter=5000,
                        random_state=42,
                    ),
                ),
            ]
        )
        pipeline.fit(dataset["X_train"], dataset["y_train"])
        y_pred = pipeline.predict(dataset["X_test"])
        accuracy = accuracy_score(dataset["y_test"], y_pred)
        complexity = logistic_complexity(pipeline)
        models.append(
            {
                "id": f"logistic_{c_value}",
                "model_type": "logistic",
                "setting_label": f"C={c_value}",
                "parameter": c_value,
                "pipeline": pipeline,
                "accuracy": float(accuracy),
                "complexity": float(complexity),
            }
        )
    return models


def selection_score(accuracy, complexity, lambda_value):
    return float(accuracy - lambda_value * complexity)


def select_model(models, lambda_value):
    scored_models = []
    for model_info in models:
        score = selection_score(model_info["accuracy"], model_info["complexity"], lambda_value)
        scored_models.append({**model_info, "selection_score": score})

    best = max(scored_models, key=lambda item: item["selection_score"])
    return best, scored_models


def get_model_bank(dataset, model_type):
    if model_type == "tree":
        return train_tree_models(dataset)
    return train_logistic_models(dataset)


def complexity_label(model_type):
    if model_type == "tree":
        return "Number of leaves"
    return "L1 norm of coefficients"
