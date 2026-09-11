import json
import os

from .artifact_loaders import load_task1_model, load_task2_expert
from .config import TASK2B_DIR, TASK3_DIR, TASK3B_DIR
from .data import load_ag_news_splits
from .deferral_experiments import run_deferral_experiments


def _run_task3_pipeline_generic(expert, output_dir, summary_filename, random_state=42, val_size=0.1, splits=None, classifier=None):
    if splits is None:
        splits = load_ag_news_splits(random_state=random_state, val_size=val_size)
    if classifier is None:
        classifier = load_task1_model()

    experiment = run_deferral_experiments(classifier, expert, splits, output_dir=output_dir)

    summary = {
        "artifact_paths": experiment["artifact_paths"],
        "best_threshold": experiment["best_threshold"],
        "learned_deferral_threshold": experiment["results"].get("learned_deferral_threshold"),
        "strategies": experiment["results"]["strategies"],
    }
    summary_path = os.path.join(output_dir, summary_filename)
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    return summary


def run_task3_pipeline(random_state=42, val_size=0.1):
    """Task 3 against Expert A (uniformly weaker than the AI)."""
    splits = load_ag_news_splits(random_state=random_state, val_size=val_size)
    classifier = load_task1_model()
    expert = load_task2_expert()
    return _run_task3_pipeline_generic(
        expert, TASK3_DIR, "task3_summary.json",
        random_state=random_state, val_size=val_size, splits=splits, classifier=classifier,
    )


def run_task3b_pipeline(random_state=42, val_size=0.1):
    """Task 3 against Expert B (Business specialist, complementary to the AI)."""
    splits = load_ag_news_splits(random_state=random_state, val_size=val_size)
    classifier = load_task1_model()
    expert = load_task2_expert(output_dir=TASK2B_DIR)
    return _run_task3_pipeline_generic(
        expert, TASK3B_DIR, "task3b_summary.json",
        random_state=random_state, val_size=val_size, splits=splits, classifier=classifier,
    )
