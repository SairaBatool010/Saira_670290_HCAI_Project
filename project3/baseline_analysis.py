import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .config import TASK1_DIR


def _confidence_entropy_margin(probabilities):
    confidence = probabilities.max(axis=1)
    entropy = -np.sum(probabilities * np.log(probabilities + 1e-12), axis=1)
    sorted_probs = np.sort(probabilities, axis=1)
    margin = sorted_probs[:, -1] - sorted_probs[:, -2]
    return confidence, entropy, margin


def _confidence_bin_accuracy(confidence, correct, bins=10):
    bin_edges = np.linspace(0.0, 1.0, bins + 1)
    rows = []
    for start, end in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (confidence >= start) & (confidence < end if end < 1.0 else confidence <= end)
        if mask.any():
            rows.append(
                {
                    "bin_start": round(float(start), 2),
                    "bin_end": round(float(end), 2),
                    "accuracy": round(float(correct[mask].mean()), 4),
                    "count": int(mask.sum()),
                }
            )
    return rows


def analyze_predictions(y_true, y_pred, probabilities, texts, class_names, split_name):
    confidence, entropy, margin = _confidence_entropy_margin(probabilities)
    correct = (y_pred == y_true)

    high_conf_wrong = []
    low_conf_correct = []
    wrong_indices = np.where(~correct)[0]
    correct_indices = np.where(correct)[0]

    if len(wrong_indices):
        wrong_conf_order = wrong_indices[np.argsort(-confidence[wrong_indices])]
        for idx in wrong_conf_order[:5]:
            high_conf_wrong.append(
                {
                    "index": int(idx),
                    "confidence": round(float(confidence[idx]), 4),
                    "predicted": class_names[int(y_pred[idx])],
                    "true_label": class_names[int(y_true[idx])],
                    "text_preview": texts[int(idx)][:240],
                }
            )

    if len(correct_indices):
        correct_conf_order = correct_indices[np.argsort(confidence[correct_indices])]
        for idx in correct_conf_order[:5]:
            low_conf_correct.append(
                {
                    "index": int(idx),
                    "confidence": round(float(confidence[idx]), 4),
                    "predicted": class_names[int(y_pred[idx])],
                    "true_label": class_names[int(y_true[idx])],
                    "text_preview": texts[int(idx)][:240],
                }
            )

    cm = np.zeros((len(class_names), len(class_names)), dtype=int)
    for true_label, pred_label in zip(y_true, y_pred):
        cm[int(true_label), int(pred_label)] += 1

    most_confused_pairs = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j and cm[i, j] > 0:
                most_confused_pairs.append(
                    {
                        "true_class": class_names[i],
                        "predicted_class": class_names[j],
                        "count": int(cm[i, j]),
                    }
                )
    most_confused_pairs.sort(key=lambda row: row["count"], reverse=True)

    return {
        "split": split_name,
        "confidence_stats": {
            "mean": round(float(confidence.mean()), 4),
            "std": round(float(confidence.std()), 4),
            "min": round(float(confidence.min()), 4),
            "max": round(float(confidence.max()), 4),
        },
        "entropy_stats": {
            "mean": round(float(entropy.mean()), 4),
            "std": round(float(entropy.std()), 4),
        },
        "margin_stats": {
            "mean": round(float(margin.mean()), 4),
            "std": round(float(margin.std()), 4),
        },
        "accuracy_vs_confidence_bins": _confidence_bin_accuracy(confidence, correct),
        "high_confidence_errors": high_conf_wrong,
        "low_confidence_correct": low_conf_correct,
        "most_confused_pairs": most_confused_pairs[:8],
    }


def _plot_confusion_matrix(cm, class_names, output_path, title):
    figure, axis = plt.subplots(figsize=(7, 6))
    im = axis.imshow(cm, cmap="Blues")
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


def _plot_confidence_distribution(confidence, correct, output_path, title):
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.hist(confidence[correct], bins=20, alpha=0.6, label="Correct")
    axis.hist(confidence[~correct], bins=20, alpha=0.6, label="Incorrect")
    axis.set_title(title)
    axis.set_xlabel("Confidence")
    axis.set_ylabel("Count")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def _plot_accuracy_vs_confidence(bin_rows, output_path, title):
    figure, axis = plt.subplots(figsize=(8, 5))
    centers = [(row["bin_start"] + row["bin_end"]) / 2 for row in bin_rows]
    accuracies = [row["accuracy"] for row in bin_rows]
    axis.plot(centers, accuracies, marker="o")
    axis.set_ylim(0, 1.05)
    axis.set_title(title)
    axis.set_xlabel("Confidence bin center")
    axis.set_ylabel("Accuracy")
    figure.tight_layout()
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def _plot_per_class_f1(per_class_metrics, class_names, output_path, title):
    figure, axis = plt.subplots(figsize=(8, 5))
    f1_scores = [per_class_metrics[str(i)]["f1"] for i in range(len(class_names))]
    axis.bar(class_names, f1_scores)
    axis.set_ylim(0, 1.05)
    axis.set_title(title)
    axis.set_ylabel("F1 score")
    figure.tight_layout()
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def run_baseline_analysis(task1_result, splits, output_dir=TASK1_DIR):
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    class_names = splits["class_names"]

    val_analysis = analyze_predictions(
        splits["y_val"],
        task1_result["validation_predictions"],
        task1_result["validation_probabilities"],
        splits["X_val"],
        class_names,
        "validation",
    )
    test_analysis = analyze_predictions(
        splits["y_test"],
        task1_result["test_predictions"],
        task1_result["test_probabilities"],
        splits["X_test"],
        class_names,
        "test_official",
    )

    val_conf, _, _ = _confidence_entropy_margin(task1_result["validation_probabilities"])
    test_conf, _, _ = _confidence_entropy_margin(task1_result["test_probabilities"])
    val_correct = task1_result["validation_predictions"] == splits["y_val"]
    test_correct = task1_result["test_predictions"] == splits["y_test"]

    plot_paths = {
        "validation_confusion_matrix": os.path.join(plots_dir, "validation_confusion_matrix.png"),
        "test_confusion_matrix": os.path.join(plots_dir, "test_confusion_matrix.png"),
        "validation_confidence_distribution": os.path.join(
            plots_dir, "validation_confidence_distribution.png"
        ),
        "test_confidence_distribution": os.path.join(plots_dir, "test_confidence_distribution.png"),
        "validation_accuracy_vs_confidence": os.path.join(
            plots_dir, "validation_accuracy_vs_confidence.png"
        ),
        "test_accuracy_vs_confidence": os.path.join(plots_dir, "test_accuracy_vs_confidence.png"),
        "test_per_class_f1": os.path.join(plots_dir, "test_per_class_f1.png"),
    }

    _plot_confusion_matrix(
        np.array(task1_result["validation_metrics"]["confusion_matrix"]),
        class_names,
        plot_paths["validation_confusion_matrix"],
        "Validation confusion matrix",
    )
    _plot_confusion_matrix(
        np.array(task1_result["test_metrics"]["confusion_matrix"]),
        class_names,
        plot_paths["test_confusion_matrix"],
        "Official test confusion matrix",
    )
    _plot_confidence_distribution(
        val_conf,
        val_correct,
        plot_paths["validation_confidence_distribution"],
        "Validation confidence distribution",
    )
    _plot_confidence_distribution(
        test_conf,
        test_correct,
        plot_paths["test_confidence_distribution"],
        "Test confidence distribution",
    )
    _plot_accuracy_vs_confidence(
        val_analysis["accuracy_vs_confidence_bins"],
        plot_paths["validation_accuracy_vs_confidence"],
        "Validation accuracy vs confidence",
    )
    _plot_accuracy_vs_confidence(
        test_analysis["accuracy_vs_confidence_bins"],
        plot_paths["test_accuracy_vs_confidence"],
        "Test accuracy vs confidence",
    )
    _plot_per_class_f1(
        task1_result["test_metrics"]["per_class"],
        class_names,
        plot_paths["test_per_class_f1"],
        "Official test per-class F1",
    )

    analysis = {
        "validation": val_analysis,
        "test_official": test_analysis,
        "plot_paths": plot_paths,
    }

    analysis_path = os.path.join(output_dir, "task1_analysis.json")
    with open(analysis_path, "w", encoding="utf-8") as handle:
        json.dump(analysis, handle, indent=2)

    return analysis_path, analysis
