import json
import os
from functools import lru_cache

from django.conf import settings
from django.http import FileResponse, Http404
from django.shortcuts import render

from .artifact_loaders import load_task1_model, load_task3_deferral_model
from .config import (
    CLASS_NAMES,
    TASK1_DIR,
    TASK2_DIR,
    TASK2B_DIR,
    TASK3_DIR,
    TASK3B_DIR,
    TASK4_DIR,
    TASK4B_DIR,
)
from .data import load_ag_news_splits
from .deferral_experiments import predict_learned_deferral
from .forms import HumanExpertForm
from .prediction_utils import extract_prediction_features


def _load_task1_summary():
    summary_path = os.path.join(TASK1_DIR, "task1_summary.json")
    metrics_path = os.path.join(TASK1_DIR, "task1_metrics.json")
    dataset_path = os.path.join(TASK1_DIR, "phase1_dataset_report.json")
    analysis_path = os.path.join(TASK1_DIR, "task1_analysis.json")

    if not os.path.exists(metrics_path):
        return None

    if os.path.exists(summary_path):
        with open(summary_path, encoding="utf-8") as handle:
            summary = json.load(handle)
    else:
        with open(metrics_path, encoding="utf-8") as handle:
            metrics = json.load(handle)
        summary = {
            "validation_metrics": metrics["validation"],
            "test_metrics": metrics["test_official"],
        }

    if "dataset_report" not in summary and os.path.exists(dataset_path):
        with open(dataset_path, encoding="utf-8") as handle:
            summary["dataset_report"] = json.load(handle)

    if "analysis" not in summary and os.path.exists(analysis_path):
        with open(analysis_path, encoding="utf-8") as handle:
            summary["analysis"] = json.load(handle)

    if "best_params" not in summary:
        config_path = os.path.join(TASK1_DIR, "task1_config.json")
        if os.path.exists(config_path):
            with open(config_path, encoding="utf-8") as handle:
                summary["best_params"] = json.load(handle).get("best_params")

    summary["plot_urls"] = {
        "test_confusion_matrix": settings.MEDIA_URL + "project3/experiments/task1/plots/test_confusion_matrix.png",
        "test_confidence_distribution": settings.MEDIA_URL
        + "project3/experiments/task1/plots/test_confidence_distribution.png",
        "test_accuracy_vs_confidence": settings.MEDIA_URL
        + "project3/experiments/task1/plots/test_accuracy_vs_confidence.png",
        "test_per_class_f1": settings.MEDIA_URL + "project3/experiments/task1/plots/test_per_class_f1.png",
    }
    return summary


def _load_task2_summary(task_dir=TASK2_DIR, summary_filename="task2_summary.json", plots_subdir="task2"):
    summary_path = os.path.join(task_dir, summary_filename)
    metrics_path = os.path.join(task_dir, "task2_metrics.json")

    if not os.path.exists(summary_path) and not os.path.exists(metrics_path):
        return None

    if os.path.exists(summary_path):
        with open(summary_path, encoding="utf-8") as handle:
            summary = json.load(handle)
    else:
        with open(metrics_path, encoding="utf-8") as handle:
            metrics = json.load(handle)
        summary = {
            "validation_metrics": metrics["validation"],
            "test_metrics": metrics["test_official"],
            "competence_profile": metrics.get("competence_profile"),
        }

    if "ai_expert_comparison" not in summary:
        comparison_path = os.path.join(task_dir, "task2_ai_comparison.json")
        if os.path.exists(comparison_path):
            with open(comparison_path, encoding="utf-8") as handle:
                summary["ai_expert_comparison"] = json.load(handle)

    summary["plot_urls"] = {
        "test_confusion_matrix": settings.MEDIA_URL + f"project3/experiments/{plots_subdir}/plots/test_confusion_matrix.png",
        "test_per_class_recall": settings.MEDIA_URL + f"project3/experiments/{plots_subdir}/plots/test_per_class_recall.png",
        "test_ai_vs_expert": settings.MEDIA_URL + f"project3/experiments/{plots_subdir}/plots/test_ai_vs_expert.png",
    }
    return summary


def _load_task3_summary(task_dir=TASK3_DIR, summary_filename="task3_summary.json", plots_subdir="task3"):
    summary_path = os.path.join(task_dir, summary_filename)
    metrics_path = os.path.join(task_dir, "task3_metrics.json")

    if not os.path.exists(metrics_path) and not os.path.exists(summary_path):
        return None

    if os.path.exists(summary_path):
        with open(summary_path, encoding="utf-8") as handle:
            summary = json.load(handle)
    else:
        with open(metrics_path, encoding="utf-8") as handle:
            data = json.load(handle)
        summary = {"strategies": data["strategies"], "best_threshold": data.get("best_threshold")}

    best = 0.0
    for data in summary.get("strategies", {}).values():
        best = max(best, data["test_official"]["team_accuracy"])
    summary["best_team_accuracy"] = best
    summary["plot_urls"] = {
        "strategy_comparison": settings.MEDIA_URL
        + f"project3/experiments/{plots_subdir}/plots/strategy_comparison_test.png",
        "threshold_search": settings.MEDIA_URL
        + f"project3/experiments/{plots_subdir}/plots/threshold_search_validation.png",
    }
    return summary


def _load_task4_summary(task_dir=TASK4_DIR, summary_filename="task4_summary.json", plots_subdir="task4"):
    summary_path = os.path.join(task_dir, summary_filename)
    metrics_path = os.path.join(task_dir, "task4_metrics.json")

    if not os.path.exists(metrics_path) and not os.path.exists(summary_path):
        return None

    if os.path.exists(summary_path):
        with open(summary_path, encoding="utf-8") as handle:
            summary = json.load(handle)
    else:
        with open(metrics_path, encoding="utf-8") as handle:
            data = json.load(handle)
        summary = {
            "query_budgets": data["query_budgets"],
            "strategies": data["strategies"],
        }

    best = 0.0
    for rows in summary.get("strategies", {}).values():
        for row in rows:
            best = max(best, row["test_metrics"]["team_accuracy"])
    summary["best_al_accuracy"] = best
    summary["plot_urls"] = {
        "learning_curves": settings.MEDIA_URL
        + f"project3/experiments/{plots_subdir}/plots/learning_curves_team_accuracy.png",
        "competence_mae": settings.MEDIA_URL
        + f"project3/experiments/{plots_subdir}/plots/learning_curves_competence_mae.png",
    }
    return summary


@lru_cache(maxsize=1)
def _get_test_corpus():
    splits = load_ag_news_splits()
    return splits["X_test"], splits["y_test"]


@lru_cache(maxsize=1)
def _get_task3_deferral_config():
    config_path = os.path.join(TASK3_DIR, "task3_config.json")
    if not os.path.exists(config_path):
        return {}, 0.5
    with open(config_path, encoding="utf-8") as handle:
        config = json.load(handle)
    competence_by_class = {int(k): v for k, v in config.get("expert_competence_by_class", {}).items()}
    threshold = config.get("learned_deferral_threshold", 0.5)
    return competence_by_class, threshold


def _build_human_demo(article_index):
    try:
        classifier = load_task1_model()
        defer_model = load_task3_deferral_model()
    except FileNotFoundError:
        return None

    X_test, y_test = _get_test_corpus()
    index = max(0, min(int(article_index), len(X_test) - 1))
    text = X_test[index]

    probs = classifier.predict_proba([text])[0]
    features = extract_prediction_features(probs.reshape(1, -1))
    competence_by_class, threshold = _get_task3_deferral_config()
    defer = bool(predict_learned_deferral(defer_model, features, competence_by_class, threshold)[0])

    ai_probs = {CLASS_NAMES[i]: float(probs[i] * 100) for i in range(len(probs))}
    top_class = CLASS_NAMES[int(probs.argmax())]

    return {
        "index": index,
        "text_preview": text[:500] + ("..." if len(text) > 500 else ""),
        "true_label": CLASS_NAMES[int(y_test[index])],
        "ai_probs": ai_probs,
        "top_prediction": top_class,
        "defer_message": (
            "AI recommends asking the expert."
            if defer
            else f"AI handles this article (predicted: {top_class})."
        ),
        "should_defer": defer,
    }


def index(request):
    task1_summary = _load_task1_summary()
    task2_summary = _load_task2_summary()
    task3_summary = _load_task3_summary()
    task4_summary = _load_task4_summary()

    task2b_summary = _load_task2_summary(TASK2B_DIR, "task2b_summary.json", "task2b")
    task3b_summary = _load_task3_summary(TASK3B_DIR, "task3b_summary.json", "task3b")
    task4b_summary = _load_task4_summary(TASK4B_DIR, "task4b_summary.json", "task4b")

    human_form = HumanExpertForm(request.POST or None)
    human_feedback = None
    human_demo = None

    if task1_summary and task3_summary:
        article_index = 0
        if request.method == "POST" and human_form.is_valid():
            article_index = human_form.cleaned_data["article_index"]
            human_label = int(human_form.cleaned_data["expert_label"])
            human_demo = _build_human_demo(article_index)
            if human_demo:
                label_name = CLASS_NAMES[human_label]
                human_feedback = (
                    f"You labeled this article as {label_name}. "
                    f"True label: {human_demo['true_label']}. "
                    f"AI top prediction: {human_demo['top_prediction']}."
                )
        else:
            if human_form.is_bound and human_form.is_valid():
                article_index = human_form.cleaned_data.get("article_index", 0)
            human_demo = _build_human_demo(article_index)

    report_path = os.path.join(settings.MEDIA_ROOT, "reports", "project3_report.pdf")
    context = {
        "task1_summary": task1_summary,
        "task1_ready": task1_summary is not None,
        "task2_summary": task2_summary,
        "task2_ready": task2_summary is not None,
        "task3_summary": task3_summary,
        "task3_ready": task3_summary is not None,
        "task4_summary": task4_summary,
        "task4_ready": task4_summary is not None,
        "task2b_summary": task2b_summary,
        "task2b_ready": task2b_summary is not None,
        "task3b_summary": task3b_summary,
        "task3b_ready": task3b_summary is not None,
        "task4b_summary": task4b_summary,
        "task4b_ready": task4b_summary is not None,
        "all_ready": all([task1_summary, task2_summary, task3_summary, task4_summary]),
        "human_form": human_form,
        "human_demo": human_demo,
        "human_feedback": human_feedback,
        "report_ready": os.path.exists(report_path),
    }
    return render(request, "project3/index.html", context)


def download_report(request):
    report_path = os.path.join(settings.MEDIA_ROOT, "reports", "project3_report.pdf")
    if not os.path.exists(report_path):
        raise Http404("Report not found. Run: python manage.py run_project3_phases5_10")
    return FileResponse(open(report_path, "rb"), as_attachment=True, filename="project3_report.pdf")
