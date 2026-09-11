import json
import os

import joblib

from .config import TASK1_DIR, TASK2_DIR
from .simulated_expert import SimulatedExpert, build_confusion_matrix


def load_task1_model():
    path = os.path.join(TASK1_DIR, "task1_model.joblib")
    if not os.path.exists(path):
        raise FileNotFoundError("Run `python manage.py run_project3_task1` first.")
    return joblib.load(path)


def load_task2_expert(output_dir=None):
    output_dir = output_dir or TASK2_DIR
    command_hint = "run_project3_task2" if output_dir == TASK2_DIR else "run_project3_expertB"
    expert_path = os.path.join(output_dir, "task2_expert.joblib")
    config_path = os.path.join(output_dir, "task2_config.json")
    if not os.path.exists(expert_path):
        raise FileNotFoundError(f"Run `python manage.py {command_hint}` first.")

    expert = joblib.load(expert_path)
    if hasattr(expert, "confusion_matrix"):
        return expert

    with open(config_path, encoding="utf-8") as handle:
        config = json.load(handle)

    diagonal = config.get("calibrated_reliability", [0.74, 0.90, 0.86, 0.58])
    targets = {i: float(diagonal[i]) for i in range(len(diagonal))}
    matrix = build_confusion_matrix(targets)
    rebuilt = SimulatedExpert(matrix, random_state=config.get("random_state", 42))
    if hasattr(expert, "pipeline"):
        rebuilt.pipeline = expert.pipeline
    return rebuilt


def load_task3_deferral_model():
    from .config import TASK3_DIR

    path = os.path.join(TASK3_DIR, "task3_deferral_model.joblib")
    if not os.path.exists(path):
        raise FileNotFoundError("Run `python manage.py run_project3_task3` first.")
    return joblib.load(path)
