import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score

from .config import CLASS_NAMES, RANDOM_STATE, TASK3_DIR
from .prediction_utils import build_deferral_feature_matrix, extract_prediction_features


def _team_predictions(ai_pred, expert_pred, defer_decisions):
    return np.where(defer_decisions, expert_pred, ai_pred)


def _per_class_defer_rate(y_true, defer_decisions):
    """Defer rate broken down by the article's true class -- lets us check whether a
    deferral policy concentrates its deferrals where the expert is actually strong,
    rather than deferring uniformly at random across classes."""
    rates = {}
    deferred = defer_decisions.astype(bool)
    for label in range(len(CLASS_NAMES)):
        mask = y_true == label
        rates[CLASS_NAMES[label]] = float(deferred[mask].mean()) if mask.any() else 0.0
    return rates


def _deferral_metrics(y_true, ai_pred, expert_pred, defer_decisions):
    team_pred = _team_predictions(ai_pred, expert_pred, defer_decisions)
    deferred = defer_decisions.astype(bool)
    not_deferred = ~deferred

    ai_correct = ai_pred == y_true
    expert_correct = expert_pred == y_true

    helpful = deferred & (~ai_correct) & expert_correct
    unnecessary = deferred & ai_correct & (~expert_correct)

    metrics = {
        "team_accuracy": float(accuracy_score(y_true, team_pred)),
        "ai_accuracy": float(accuracy_score(y_true, ai_pred)),
        "expert_accuracy": float(accuracy_score(y_true, expert_pred)),
        "defer_rate": float(deferred.mean()),
        "coverage": float(not_deferred.mean()),
        "ai_handled_accuracy": float(accuracy_score(y_true[not_deferred], ai_pred[not_deferred]))
        if not_deferred.any()
        else 0.0,
        "deferred_accuracy": float(accuracy_score(y_true[deferred], expert_pred[deferred]))
        if deferred.any()
        else 0.0,
        "defer_helpful_count": int(helpful.sum()),
        "defer_unnecessary_count": int(unnecessary.sum()),
        "defer_precision": float(precision_score(deferred, expert_correct, zero_division=0)),
        "defer_recall": float(recall_score(deferred, (~ai_correct) & expert_correct, zero_division=0)),
        "per_class_defer_rate": _per_class_defer_rate(y_true, deferred),
    }
    return metrics


def strategy_always_ai(ai_pred, expert_pred):
    return np.zeros(len(ai_pred), dtype=bool)


def strategy_always_expert(ai_pred, expert_pred):
    return np.ones(len(ai_pred), dtype=bool)


def strategy_confidence_threshold(features, threshold):
    return features["confidence"] < threshold


def tune_confidence_threshold(features, y_true, ai_pred, expert_pred):
    best_threshold = 0.5
    best_accuracy = -1.0
    threshold_rows = []
    for threshold in np.linspace(0.2, 0.95, 16):
        defer = strategy_confidence_threshold(features, threshold)
        metrics = _deferral_metrics(y_true, ai_pred, expert_pred, defer)
        row = {"threshold": round(float(threshold), 3), **metrics}
        threshold_rows.append(row)
        if metrics["team_accuracy"] > best_accuracy:
            best_accuracy = metrics["team_accuracy"]
            best_threshold = float(threshold)
    return best_threshold, threshold_rows


def tune_probability_threshold(probabilities, y_true, ai_pred, expert_pred, thresholds=None):
    """Pick the deferral probability cutoff that maximizes team accuracy.

    The "helpful defer" event (expert right AND AI wrong) is rare, so a plain
    classifier's default 0.5 cutoff on predict() almost never fires: the fitted
    probability of the positive class rarely exceeds 0.5 even when it is the
    single best-scoring class among cases worth deferring. Scanning the cutoff
    directly against team accuracy (the metric we actually care about) avoids
    this and mirrors the confidence-threshold baseline's own tuning approach.
    """
    if thresholds is None:
        thresholds = np.concatenate([np.linspace(0.05, 0.95, 19), np.linspace(0.96, 0.995, 8)])
    best_threshold = 0.5
    best_accuracy = -1.0
    threshold_rows = []
    for threshold in thresholds:
        defer = probabilities >= threshold
        metrics = _deferral_metrics(y_true, ai_pred, expert_pred, defer)
        row = {"threshold": round(float(threshold), 3), **metrics}
        threshold_rows.append(row)
        if metrics["team_accuracy"] > best_accuracy:
            best_accuracy = metrics["team_accuracy"]
            best_threshold = float(threshold)
    return best_threshold, threshold_rows


def per_class_accuracy(condition_labels, y_true, y_pred, num_classes):
    """Accuracy of y_pred vs y_true, grouped by condition_labels.

    For the deferral feature this must be conditioned on the AI's *predicted* class
    (condition_labels = ai_pred), not the article's true class: at inference time the
    true label is exactly what we don't have, only the AI's prediction. Conditioning
    on the true label instead silently mixes in every other true class's articles that
    the AI *misclassified* as this one -- a population the expert's competence for the
    true class says nothing about, and on which the expert has no special reliability.
    That mismatch was large enough in practice to make the learned deferral model
    defer close to indiscriminately within a class instead of concentrating on the
    genuinely beneficial cases.
    """
    accuracy_by_class = {}
    for label in range(num_classes):
        mask = condition_labels == label
        accuracy_by_class[label] = float((y_pred[mask] == y_true[mask]).mean()) if mask.any() else 0.5
    return accuracy_by_class


def train_learned_deferral(features, y_true, ai_pred, expert_pred, competence_by_class):
    defer_labels = ((expert_pred == y_true) & (ai_pred != y_true)).astype(int)
    if len(np.unique(defer_labels)) < 2:
        return None
    matrix = build_deferral_feature_matrix(features, competence_by_class)
    model = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE, class_weight="balanced")
    model.fit(matrix, defer_labels)
    return model


def learned_deferral_probabilities(defer_model, features, competence_by_class):
    matrix = build_deferral_feature_matrix(features, competence_by_class)
    return defer_model.predict_proba(matrix)[:, 1]


def predict_learned_deferral(defer_model, features, competence_by_class, threshold=0.5):
    if defer_model is None:
        return np.zeros(len(features["confidence"]), dtype=bool)
    probabilities = learned_deferral_probabilities(defer_model, features, competence_by_class)
    return probabilities >= threshold


def run_deferral_experiments(classifier, expert, splits, output_dir=TASK3_DIR):
    os.makedirs(output_dir, exist_ok=True)
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    val_probs = classifier.predict_proba(splits["X_val"])
    test_probs = classifier.predict_proba(splits["X_test"])
    val_features = extract_prediction_features(val_probs)
    test_features = extract_prediction_features(test_probs)

    val_ai = val_features["predicted_class"]
    test_ai = test_features["predicted_class"]
    val_expert = expert.predict(splits["X_val"])
    test_expert = expert.predict(splits["X_test"])

    train_fit_probs = classifier.predict_proba(splits["X_train_fit"])
    train_fit_features = extract_prediction_features(train_fit_probs)
    train_fit_ai = train_fit_features["predicted_class"]
    train_fit_expert = expert.predict(splits["X_train_fit"])

    best_threshold, threshold_search = tune_confidence_threshold(
        val_features, splits["y_val"], val_ai, val_expert
    )

    # Task 3 has full access to expert labels, so the expert's competence can be
    # measured directly on train_fit and given to the deferral model as a feature --
    # this tells it *where this specific expert tends to be right*, which AI-internal
    # confidence/entropy/margin alone cannot capture. Conditioned on the AI's own
    # predicted class (not the true class), matching what's actually available when
    # the deferral model has to make a decision at inference time.
    competence_by_class = per_class_accuracy(
        train_fit_ai, splits["y_train_fit"], train_fit_expert, num_classes=len(CLASS_NAMES)
    )

    defer_model = train_learned_deferral(
        train_fit_features,
        splits["y_train_fit"],
        train_fit_ai,
        train_fit_expert,
        competence_by_class,
    )

    if defer_model is not None:
        val_defer_probs = learned_deferral_probabilities(defer_model, val_features, competence_by_class)
        learned_threshold, learned_threshold_search = tune_probability_threshold(
            val_defer_probs, splits["y_val"], val_ai, val_expert
        )
    else:
        learned_threshold, learned_threshold_search = 0.5, []

    strategies = {
        "always_ai": {
            "description": "Baseline: AI handles every article.",
            "validation_defer": strategy_always_ai(val_ai, val_expert),
            "test_defer": strategy_always_ai(test_ai, test_expert),
        },
        "always_expert": {
            "description": "Baseline: defer every article to the expert.",
            "validation_defer": strategy_always_expert(val_ai, val_expert),
            "test_defer": strategy_always_expert(test_ai, test_expert),
        },
        "confidence_threshold": {
            "description": (
                f"Defer when AI confidence < {best_threshold:.3f} "
                "(threshold tuned on validation)."
            ),
            "validation_defer": strategy_confidence_threshold(val_features, best_threshold),
            "test_defer": strategy_confidence_threshold(test_features, best_threshold),
            "best_threshold": best_threshold,
        },
        "learned_deferral": {
            "description": (
                "Logistic regression deferral model (class_weight=balanced) using AI "
                "confidence, entropy, margin, predicted class, and the expert's measured "
                "per-class competence (from train_fit) as features, with its deferral "
                f"probability cutoff ({learned_threshold:.3f}) tuned on validation team "
                "accuracy rather than a default 0.5 cutoff. Trained on train_fit with "
                "historical expert outcomes; inference never uses the expert answer."
            ),
            "validation_defer": predict_learned_deferral(
                defer_model, val_features, competence_by_class, learned_threshold
            ),
            "test_defer": predict_learned_deferral(
                defer_model, test_features, competence_by_class, learned_threshold
            ),
            "best_threshold": learned_threshold,
        },
    }

    results = {
        "strategies": {},
        "best_threshold": best_threshold,
        "threshold_search": threshold_search,
        "learned_deferral_threshold": learned_threshold,
        "learned_deferral_threshold_search": learned_threshold_search,
    }
    for name, config in strategies.items():
        val_metrics = _deferral_metrics(
            splits["y_val"], val_ai, val_expert, config["validation_defer"]
        )
        test_metrics = _deferral_metrics(
            splits["y_test"], test_ai, test_expert, config["test_defer"]
        )
        results["strategies"][name] = {
            "description": config["description"],
            "validation": val_metrics,
            "test_official": test_metrics,
        }
        if "best_threshold" in config:
            results["strategies"][name]["best_threshold"] = config["best_threshold"]

    _plot_strategy_comparison(results, os.path.join(plots_dir, "strategy_comparison_test.png"))
    _plot_threshold_search(threshold_search, os.path.join(plots_dir, "threshold_search_validation.png"))

    metrics_path = os.path.join(output_dir, "task3_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    import joblib

    model_path = os.path.join(output_dir, "task3_deferral_model.joblib")
    joblib.dump(defer_model, model_path)

    config = {
        "random_state": splits["random_state"],
        "best_confidence_threshold": best_threshold,
        "learned_deferral_threshold": learned_threshold,
        "expert_competence_by_class": {str(k): v for k, v in competence_by_class.items()},
        "strategy_names": list(strategies.keys()),
    }
    config_path = os.path.join(output_dir, "task3_config.json")
    with open(config_path, "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)

    return {
        "results": results,
        "defer_model": defer_model,
        "best_threshold": best_threshold,
        "artifact_paths": {
            "metrics_path": metrics_path,
            "config_path": config_path,
            "model_path": model_path,
            "plots_dir": plots_dir,
        },
    }


def _plot_strategy_comparison(results, output_path):
    names = list(results["strategies"].keys())
    accuracies = [results["strategies"][name]["test_official"]["team_accuracy"] for name in names]
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.bar(names, accuracies, color=["#457b9d", "#2a9d8f", "#e9c46a", "#e76f51"])
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("Team accuracy (test)")
    axis.set_title("Task 3 — Deferral strategy comparison")
    axis.tick_params(axis="x", rotation=15)
    figure.tight_layout()
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def _plot_threshold_search(rows, output_path):
    thresholds = [row["threshold"] for row in rows]
    team_acc = [row["team_accuracy"] for row in rows]
    defer_rate = [row["defer_rate"] for row in rows]
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(thresholds, team_acc, marker="o", label="Team accuracy")
    axis.plot(thresholds, defer_rate, marker="s", label="Deferral rate")
    axis.set_xlabel("Confidence threshold")
    axis.set_ylabel("Score")
    axis.set_title("Validation threshold search")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)
