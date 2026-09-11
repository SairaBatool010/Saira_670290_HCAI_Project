import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .config import AL_BATCH_SIZE, AL_QUERY_BUDGETS, CLASS_NAMES, NUM_CLASSES, RANDOM_STATE, TASK4_DIR
from .deferral_experiments import (
    _deferral_metrics,
    learned_deferral_probabilities,
    predict_learned_deferral,
    train_learned_deferral,
    tune_probability_threshold,
)
from .prediction_utils import extract_prediction_features


def _estimate_competence(y_true, expert_pred):
    estimates = {}
    for label in range(NUM_CLASSES):
        mask = y_true == label
        if mask.any():
            estimates[label] = float((expert_pred[mask] == y_true[mask]).mean())
        else:
            estimates[label] = 0.5
    return estimates


def _select_random(remaining, count, rng):
    remaining_list = list(remaining)
    chosen = rng.choice(remaining_list, size=min(count, len(remaining_list)), replace=False)
    return [int(i) for i in chosen]


def _select_by_score(remaining, scores, count):
    ranked = sorted(((scores[i], i) for i in remaining), reverse=True)
    return [index for _, index in ranked[:count]]


def _run_competence_aware_selection(
    remaining,
    features,
    y_train,
    expert,
    texts,
    budget,
    batch_size,
    rng,
):
    queried = []
    expert_cache = {}
    competence = {label: 0.5 for label in range(NUM_CLASSES)}
    pool = set(remaining)

    while len(queried) < budget and pool:
        batch = min(batch_size, budget - len(queried), len(pool))
        scores = {}
        for index in pool:
            pred_class = int(features["predicted_class"][index])
            uncertainty = 1.0 - features["confidence"][index]
            unknown = 1.0 - competence[pred_class]
            scores[index] = uncertainty * (0.5 + unknown)

        selected = _select_by_score(pool, scores, batch)
        for index in selected:
            expert_cache[index] = int(expert.predict([texts[index]])[0])
        queried.extend(selected)
        pool -= set(selected)

        queried_expert = np.array([expert_cache[i] for i in queried], dtype=int)
        competence = _estimate_competence(y_train[queried], queried_expert)

    return queried, competence


def _evaluate_budget(
    classifier,
    expert,
    splits,
    train_fit_features,
    train_fit_ai,
    queried_indices,
    competence_estimate,
    true_competence,
):
    texts = splits["X_train_fit"]
    y_train = splits["y_train_fit"]

    queried_texts = [texts[i] for i in queried_indices]
    queried_y = y_train[queried_indices]
    queried_expert = expert.predict(queried_texts)
    queried_features = {
        key: value[queried_indices] if isinstance(value, np.ndarray) else value
        for key, value in train_fit_features.items()
    }
    queried_ai = train_fit_ai[queried_indices]

    # Only the running competence estimate from queried points is available here
    # (never the true/full profile) -- this is the Task 4 constraint in action, and
    # it is exactly the "expert competence discovery" signal Task 4 asks the active
    # learning strategy to produce, now actually put to use by the deferral model.
    competence_by_class = {int(k): v for k, v in competence_estimate.items()}

    defer_model = train_learned_deferral(
        queried_features, queried_y, queried_ai, queried_expert, competence_by_class
    )

    # The deferral probability cutoff is tuned in-sample on the queried subset only
    # (the same data used to fit the model): under the Task 4 constraint, expert
    # labels beyond the query budget are unavailable, so a held-out validation split
    # for threshold tuning is not an option here (unlike Task 3's confidence and
    # learned-deferral baselines, which may use the full validation set).
    if defer_model is not None:
        queried_probs = learned_deferral_probabilities(defer_model, queried_features, competence_by_class)
        threshold, _ = tune_probability_threshold(queried_probs, queried_y, queried_ai, queried_expert)
    else:
        threshold = 0.5

    test_probs = classifier.predict_proba(splits["X_test"])
    test_features = extract_prediction_features(test_probs)
    test_ai = test_features["predicted_class"]
    test_expert = expert.predict(splits["X_test"])
    test_defer = predict_learned_deferral(defer_model, test_features, competence_by_class, threshold)
    test_metrics = _deferral_metrics(splits["y_test"], test_ai, test_expert, test_defer)

    competence_mae = float(
        np.mean(
            [
                abs(float(competence_estimate.get(str(i), competence_estimate.get(i, 0.5))) - true_competence[i])
                for i in range(NUM_CLASSES)
            ]
        )
    )

    return {
        "queried_count": len(queried_indices),
        "competence_estimate": {str(k): round(v, 4) for k, v in competence_estimate.items()},
        "competence_mae_vs_true": competence_mae,
        "learned_deferral_threshold": round(float(threshold), 3),
        "test_metrics": test_metrics,
    }


def run_active_learning_experiments(classifier, expert, splits, output_dir=TASK4_DIR):
    os.makedirs(output_dir, exist_ok=True)
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    texts = splits["X_train_fit"]
    y_train = splits["y_train_fit"]
    n_pool = len(texts)
    pool_indices = np.arange(n_pool)

    train_fit_probs = classifier.predict_proba(texts)
    train_fit_features = extract_prediction_features(train_fit_probs)
    train_fit_ai = train_fit_features["predicted_class"]

    true_expert_preds = expert.predict(texts)
    true_competence = _estimate_competence(y_train, true_expert_preds)

    rng = np.random.default_rng(RANDOM_STATE)
    strategies = {}

    for budget in AL_QUERY_BUDGETS:
        if budget > n_pool:
            continue

        # Random
        random_indices = _select_random(set(pool_indices), budget, rng)
        random_comp = _estimate_competence(
            y_train[random_indices], expert.predict([texts[i] for i in random_indices])
        )
        strategies.setdefault("random", []).append(
            {
                "budget": budget,
                **_evaluate_budget(
                    classifier,
                    expert,
                    splits,
                    train_fit_features,
                    train_fit_ai,
                    random_indices,
                    {str(k): v for k, v in random_comp.items()},
                    true_competence,
                ),
            }
        )

        # Uncertainty
        uncertainty_scores = {i: 1.0 - train_fit_features["confidence"][i] for i in pool_indices}
        uncertainty_indices = _select_by_score(set(pool_indices), uncertainty_scores, budget)
        unc_comp = _estimate_competence(
            y_train[uncertainty_indices],
            expert.predict([texts[i] for i in uncertainty_indices]),
        )
        strategies.setdefault("uncertainty", []).append(
            {
                "budget": budget,
                **_evaluate_budget(
                    classifier,
                    expert,
                    splits,
                    train_fit_features,
                    train_fit_ai,
                    uncertainty_indices,
                    {str(k): v for k, v in unc_comp.items()},
                    true_competence,
                ),
            }
        )

        # Entropy
        entropy_scores = {i: train_fit_features["entropy"][i] for i in pool_indices}
        entropy_indices = _select_by_score(set(pool_indices), entropy_scores, budget)
        ent_comp = _estimate_competence(
            y_train[entropy_indices],
            expert.predict([texts[i] for i in entropy_indices]),
        )
        strategies.setdefault("entropy", []).append(
            {
                "budget": budget,
                **_evaluate_budget(
                    classifier,
                    expert,
                    splits,
                    train_fit_features,
                    train_fit_ai,
                    entropy_indices,
                    {str(k): v for k, v in ent_comp.items()},
                    true_competence,
                ),
            }
        )

        # Competence-aware
        comp_indices, comp_estimates = _run_competence_aware_selection(
            pool_indices,
            train_fit_features,
            y_train,
            expert,
            texts,
            budget,
            AL_BATCH_SIZE,
            rng,
        )
        strategies.setdefault("competence_aware", []).append(
            {
                "budget": budget,
                **_evaluate_budget(
                    classifier,
                    expert,
                    splits,
                    train_fit_features,
                    train_fit_ai,
                    comp_indices,
                    {str(k): round(v, 4) for k, v in comp_estimates.items()},
                    true_competence,
                ),
            }
        )

    _plot_learning_curves(strategies, os.path.join(plots_dir, "learning_curves_team_accuracy.png"))
    _plot_competence_mae(strategies, os.path.join(plots_dir, "learning_curves_competence_mae.png"))

    results = {
        "query_budgets": AL_QUERY_BUDGETS,
        "batch_size": AL_BATCH_SIZE,
        "true_expert_competence_train_fit": {CLASS_NAMES[k]: round(v, 4) for k, v in true_competence.items()},
        "strategies": strategies,
    }

    metrics_path = os.path.join(output_dir, "task4_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    config_path = os.path.join(output_dir, "task4_config.json")
    with open(config_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "random_state": splits["random_state"],
                "query_budgets": AL_QUERY_BUDGETS,
                "pool_size": n_pool,
                "strategy_names": list(strategies.keys()),
            },
            handle,
            indent=2,
        )

    return {
        "results": results,
        "artifact_paths": {
            "metrics_path": metrics_path,
            "config_path": config_path,
            "plots_dir": plots_dir,
        },
    }


def _plot_learning_curves(strategies, output_path):
    figure, axis = plt.subplots(figsize=(9, 5))
    colors = {"random": "#457b9d", "uncertainty": "#e76f51", "entropy": "#f4a261", "competence_aware": "#2a9d8f"}
    for name, rows in strategies.items():
        budgets = [row["budget"] for row in rows]
        acc = [row["test_metrics"]["team_accuracy"] for row in rows]
        axis.plot(budgets, acc, marker="o", label=name.replace("_", " "), color=colors.get(name))
    axis.set_xlabel("Expert queries")
    axis.set_ylabel("Team accuracy (test)")
    axis.set_title("Task 4 — Active learning learning curves")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def _plot_competence_mae(strategies, output_path):
    figure, axis = plt.subplots(figsize=(9, 5))
    colors = {"random": "#457b9d", "uncertainty": "#e76f51", "entropy": "#f4a261", "competence_aware": "#2a9d8f"}
    for name, rows in strategies.items():
        budgets = [row["budget"] for row in rows]
        mae = [row["competence_mae_vs_true"] for row in rows]
        axis.plot(budgets, mae, marker="s", label=name.replace("_", " "), color=colors.get(name))
    axis.set_xlabel("Expert queries")
    axis.set_ylabel("Competence estimation MAE")
    axis.set_title("Task 4 — Competence estimation quality")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)
