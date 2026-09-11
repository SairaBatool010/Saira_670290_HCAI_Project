from .baseline_analysis import run_baseline_analysis
from .baseline_experiments import run_task1_baseline, save_task1_artifacts
from .data import load_ag_news_splits
from .dataset_analysis import analyze_dataset, save_dataset_report


def run_task1_pipeline(random_state=42, val_size=0.1):
    splits = load_ag_news_splits(random_state=random_state, val_size=val_size)

    dataset_report = analyze_dataset(splits)
    dataset_report_path = save_dataset_report(dataset_report)

    task1_result = run_task1_baseline(splits)
    artifact_paths = save_task1_artifacts(task1_result, splits)
    analysis_path, analysis = run_baseline_analysis(task1_result, splits)

    summary = {
        "dataset_report_path": dataset_report_path,
        "artifact_paths": artifact_paths,
        "analysis_path": analysis_path,
        "dataset_report": dataset_report,
        "test_metrics": task1_result["test_metrics"],
        "validation_metrics": task1_result["validation_metrics"],
        "best_params": task1_result["best_params"],
        "analysis": analysis,
    }
    return summary
