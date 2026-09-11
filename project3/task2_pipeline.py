import json
import os

import joblib

from .config import (
    CLASS_NAMES,
    EXPERT_B_TARGET_CLASS_ACCURACY,
    EXPERT_TARGET_CLASS_ACCURACY,
    TASK1_DIR,
    TASK2_DIR,
    TASK2B_DIR,
)
from .data import load_ag_news_splits
from .expert_analysis import run_expert_analysis, save_task2_artifacts
from .simulated_expert import (
    DOMAIN_KEYWORDS_EXPERT_B,
    EXPERT_B_CONFUSION_OFF_DIAGONAL,
    SimulatedExpert,
    build_confusion_matrix,
    calibrate_expert,
)


def _load_task1_model():
    model_path = os.path.join(TASK1_DIR, "task1_model.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            "Task 1 model not found. Run `python manage.py run_project3_task1` first."
        )
    return joblib.load(model_path)


def _run_task2_pipeline_generic(
    expert_seed,
    targets,
    output_dir,
    summary_filename,
    random_state=42,
    val_size=0.1,
    calibrate_kwargs=None,
    task1_model=None,
    splits=None,
):
    if splits is None:
        splits = load_ag_news_splits(random_state=random_state, val_size=val_size)
    if task1_model is None:
        task1_model = _load_task1_model()

    calibration_result = calibrate_expert(
        expert_seed,
        splits["X_train_fit"],
        splits["y_train_fit"],
        splits["X_val"],
        splits["y_val"],
        targets=targets,
        **(calibrate_kwargs or {}),
    )

    expert = calibration_result["expert"]
    test_expert_pred = expert.predict(splits["X_test"])
    ai_test_pred = task1_model.predict(splits["X_test"])

    analysis_path, analysis = run_expert_analysis(
        splits,
        calibration_result,
        test_expert_pred,
        ai_test_pred,
        output_dir=output_dir,
    )
    artifact_paths = save_task2_artifacts(calibration_result, analysis, splits, output_dir=output_dir)

    summary = {
        "artifact_paths": artifact_paths,
        "analysis_path": analysis_path,
        "validation_metrics": analysis["validation_metrics"],
        "test_metrics": analysis["test_metrics"],
        "ai_expert_comparison": analysis["ai_expert_comparison"],
        "competence_profile": analysis["competence_profile"],
        "calibrated_reliability": calibration_result["calibrated_reliability"],
        "target_profile": analysis["target_profile"],
        "validation_per_class_accuracy": analysis["validation_per_class_accuracy"],
        "pre_model_accuracy": {
            "overall": calibration_result["pre_model_accuracy"]["overall"],
            "per_class": {
                CLASS_NAMES[int(label)]: value
                for label, value in calibration_result["pre_model_accuracy"]["per_class"].items()
            },
        },
        "expert_name": expert.name,
        "expert_description": expert.description,
        "design_rationale": calibration_result["design_rationale"],
    }

    summary_path = os.path.join(output_dir, summary_filename)
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    return summary


def run_task2_pipeline(random_state=42, val_size=0.1):
    """Expert A: a generalist journalist, strong on Sports/Business, weak on Sci/Tech."""
    expert_seed = SimulatedExpert(
        build_confusion_matrix(EXPERT_TARGET_CLASS_ACCURACY),
        random_state=random_state,
    )
    return _run_task2_pipeline_generic(
        expert_seed,
        targets=EXPERT_TARGET_CLASS_ACCURACY,
        output_dir=TASK2_DIR,
        summary_filename="task2_summary.json",
        random_state=random_state,
        val_size=val_size,
    )


def run_task2b_pipeline(random_state=42, val_size=0.1):
    """
    Expert B: a Business specialist. Weak overall (target ~35% on World/Sports/Sci-Tech),
    but deliberately targeted to exceed the AI classifier's own Business accuracy, so
    Tasks 3-4 can also demonstrate the case where deferral has genuine, concentrated value.
    """
    expert_seed = SimulatedExpert(
        None,
        random_state=random_state,
        domain_keywords=DOMAIN_KEYWORDS_EXPERT_B,
        keyword_weight=0.92,
        sharpness=10.0,
        adaptive_keyword_weight=True,
        name="Business specialist",
        description=(
            "A simulated financial-news specialist: near-expert on Business articles "
            "(recognized via a large finance/markets vocabulary), but weak and close to "
            "random on World, Sports, and Sci/Tech, where it has no comparable domain "
            "vocabulary to draw on."
        ),
    )
    return _run_task2_pipeline_generic(
        expert_seed,
        targets=EXPERT_B_TARGET_CLASS_ACCURACY,
        output_dir=TASK2B_DIR,
        summary_filename="task2b_summary.json",
        random_state=random_state,
        val_size=val_size,
        calibrate_kwargs={
            "off_diagonal_variants": [EXPERT_B_CONFUSION_OFF_DIAGONAL],
            "clip_range": (0.20, 0.995),
            "scales": (0.9, 0.95, 1.0, 1.02, 1.05),
        },
    )
