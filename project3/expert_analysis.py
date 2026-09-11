import json
import os

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from .config import CLASS_NAMES, TASK2_DIR


def evaluate_predictions(y_true, y_pred):
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "per_class": {
            str(label): {
                "precision": float(report[str(label)]["precision"]),
                "recall": float(report[str(label)]["recall"]),
                "f1": float(report[str(label)]["f1-score"]),
                "support": int(report[str(label)]["support"]),
            }
            for label in sorted(set(y_true))
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def compare_ai_and_expert(y_true, ai_pred, expert_pred, class_names):
    ai_correct = ai_pred == y_true
    expert_correct = expert_pred == y_true

    categories = {
        "ai_correct_expert_wrong": int(np.sum(ai_correct & ~expert_correct)),
        "ai_wrong_expert_correct": int(np.sum(~ai_correct & expert_correct)),
        "both_correct": int(np.sum(ai_correct & expert_correct)),
        "both_wrong": int(np.sum(~ai_correct & ~expert_correct)),
        "disagree_total": int(np.sum(ai_pred != expert_pred)),
    }

    per_class = {}
    for label in range(len(class_names)):
        mask = y_true == label
        if not mask.any():
            continue
        per_class[class_names[label]] = {
            "expert_accuracy": float((expert_pred[mask] == y_true[mask]).mean()),
            "ai_accuracy": float((ai_pred[mask] == y_true[mask]).mean()),
            "expert_beats_ai_count": int(
                np.sum((expert_pred[mask] == y_true[mask]) & (ai_pred[mask] != y_true[mask]))
            ),
            "ai_beats_expert_count": int(
                np.sum((ai_pred[mask] != y_true[mask]) & (expert_pred[mask] == y_true[mask]))
            ),
        }

    examples = {
        "ai_correct_expert_wrong": [],
        "ai_wrong_expert_correct": [],
    }
    for name, mask_builder in [
        ("ai_correct_expert_wrong", lambda: ai_correct & ~expert_correct),
        ("ai_wrong_expert_correct", lambda: ~ai_correct & expert_correct),
    ]:
        indices = np.where(mask_builder())[0][:5]
        for idx in indices:
            examples[name].append(
                {
                    "index": int(idx),
                    "true_label": class_names[int(y_true[idx])],
                    "ai_prediction": class_names[int(ai_pred[idx])],
                    "expert_prediction": class_names[int(expert_pred[idx])],
                }
            )

    return {
        "counts": categories,
        "per_class_comparison": per_class,
        "examples": examples,
    }


def _plot_confusion_matrix(cm, class_names, output_path, title):
    figure, axis = plt.subplots(figsize=(7, 6))
    im = axis.imshow(cm, cmap="Greens")
    axis.set_title(title)
    axis.set_xticks(range(len(class_names)))
    axis.set_yticks(range(len(class_names)))
    axis.set_xticklabels(class_names, rotation=30, ha="right")
    axis.set_yticklabels(class_names)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            axis.text(j, i, cm[i, j], ha="center", va="center", color="black")
    figure.colorbar(im, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def _plot_per_class_accuracy(per_class, class_names, output_path, title):
    figure, axis = plt.subplots(figsize=(8, 5))
    values = [per_class[str(i)]["recall"] for i in range(len(class_names))]
    axis.bar(class_names, values, color="#2a9d8f")
    axis.set_ylim(0, 1.05)
    axis.set_title(title)
    axis.set_ylabel("Recall (class accuracy)")
    figure.tight_layout()
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def _plot_ai_vs_expert(per_class_comparison, output_path, title):
    classes = list(per_class_comparison.keys())
    ai_scores = [per_class_comparison[c]["ai_accuracy"] for c in classes]
    expert_scores = [per_class_comparison[c]["expert_accuracy"] for c in classes]
    x = np.arange(len(classes))
    width = 0.35
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.bar(x - width / 2, ai_scores, width, label="AI baseline", color="#457b9d")
    axis.bar(x + width / 2, expert_scores, width, label="Simulated expert", color="#2a9d8f")
    axis.set_xticks(x)
    axis.set_xticklabels(classes, rotation=20, ha="right")
    axis.set_ylim(0, 1.05)
    axis.set_title(title)
    axis.set_ylabel("Accuracy")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def run_expert_analysis(
    splits,
    calibration_result,
    test_expert_pred,
    ai_test_pred,
    output_dir=TASK2_DIR,
):
    os.makedirs(output_dir, exist_ok=True)
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    class_names = splits["class_names"]

    val_metrics = evaluate_predictions(splits["y_val"], calibration_result["validation_predictions"])
    test_metrics = evaluate_predictions(splits["y_test"], test_expert_pred)
    comparison = compare_ai_and_expert(
        splits["y_test"],
        ai_test_pred,
        test_expert_pred,
        class_names,
    )

    plot_paths = {
        "validation_confusion_matrix": os.path.join(plots_dir, "validation_confusion_matrix.png"),
        "test_confusion_matrix": os.path.join(plots_dir, "test_confusion_matrix.png"),
        "test_per_class_recall": os.path.join(plots_dir, "test_per_class_recall.png"),
        "test_ai_vs_expert": os.path.join(plots_dir, "test_ai_vs_expert.png"),
    }

    _plot_confusion_matrix(
        np.array(val_metrics["confusion_matrix"]),
        class_names,
        plot_paths["validation_confusion_matrix"],
        "Expert validation confusion matrix",
    )
    _plot_confusion_matrix(
        np.array(test_metrics["confusion_matrix"]),
        class_names,
        plot_paths["test_confusion_matrix"],
        "Expert official test confusion matrix",
    )
    _plot_per_class_accuracy(
        test_metrics["per_class"],
        class_names,
        plot_paths["test_per_class_recall"],
        "Expert per-class recall (test)",
    )
    _plot_ai_vs_expert(
        comparison["per_class_comparison"],
        plot_paths["test_ai_vs_expert"],
        "AI vs expert class accuracy (test)",
    )

    analysis = {
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "ai_expert_comparison": comparison,
        "competence_profile": calibration_result["expert"].competence_profile(),
        "calibrated_reliability": calibration_result["calibrated_reliability"],
        "target_profile": {
            CLASS_NAMES[int(label)]: value
            for label, value in calibration_result["target_profile"].items()
        },
        "validation_per_class_accuracy": {
            CLASS_NAMES[int(label)]: value
            for label, value in calibration_result["validation_per_class_accuracy"].items()
        },
        "plot_paths": plot_paths,
    }

    analysis_path = os.path.join(output_dir, "task2_analysis.json")
    with open(analysis_path, "w", encoding="utf-8") as handle:
        json.dump(analysis, handle, indent=2)

    return analysis_path, analysis


def save_task2_artifacts(calibration_result, analysis, splits, output_dir=TASK2_DIR):
    os.makedirs(output_dir, exist_ok=True)
    expert_path = os.path.join(output_dir, "task2_expert.joblib")
    joblib.dump(calibration_result["expert"], expert_path)

    config = {
        "random_state": splits["random_state"],
        "expert_name": calibration_result["expert"].name,
        "description": calibration_result["expert"].description,
        "calibrated_reliability": calibration_result["calibrated_reliability"],
        "target_profile": calibration_result["target_profile"],
        "error_distribution": calibration_result["error_distribution"],
        "design_rationale": calibration_result["design_rationale"],
    }
    config_path = os.path.join(output_dir, "task2_config.json")
    with open(config_path, "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)

    metrics = {
        "validation": analysis["validation_metrics"],
        "test_official": analysis["test_metrics"],
        "validation_per_class_accuracy": analysis["validation_per_class_accuracy"],
        "competence_profile": analysis["competence_profile"],
    }
    metrics_path = os.path.join(output_dir, "task2_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    comparison_path = os.path.join(output_dir, "task2_ai_comparison.json")
    with open(comparison_path, "w", encoding="utf-8") as handle:
        json.dump(analysis["ai_expert_comparison"], handle, indent=2)

    return {
        "expert_path": expert_path,
        "config_path": config_path,
        "metrics_path": metrics_path,
        "comparison_path": comparison_path,
        "analysis_path": os.path.join(output_dir, "task2_analysis.json"),
    }
