import json

from django.core.management.base import BaseCommand

from project3.config import TASK1_DIR
from project3.task1_pipeline import run_task1_pipeline


class Command(BaseCommand):
    help = "Run rigorous Project 3 Task 1 baseline pipeline (Phases 1-3)."

    def add_arguments(self, parser):
        parser.add_argument("--random-state", type=int, default=42)
        parser.add_argument("--val-size", type=float, default=0.1)

    def handle(self, *args, **options):
        self.stdout.write("Running Project 3 Task 1 pipeline...")
        summary = run_task1_pipeline(
            random_state=options["random_state"],
            val_size=options["val_size"],
        )

        summary_path = f"{TASK1_DIR}/task1_summary.json"
        serializable = {
            "dataset_report_path": summary["dataset_report_path"],
            "artifact_paths": summary["artifact_paths"],
            "analysis_path": summary["analysis_path"],
            "best_params": summary["best_params"],
            "validation_metrics": summary["validation_metrics"],
            "test_metrics": summary["test_metrics"],
            "dataset_report": summary["dataset_report"],
        }
        with open(summary_path, "w", encoding="utf-8") as handle:
            json.dump(serializable, handle, indent=2)

        test = summary["test_metrics"]
        self.stdout.write(self.style.SUCCESS("Task 1 complete."))
        self.stdout.write(f"Validation macro F1: {summary['validation_metrics']['macro_f1']:.4f}")
        self.stdout.write(f"Official test accuracy: {test['accuracy']:.4f}")
        self.stdout.write(f"Official test macro F1: {test['macro_f1']:.4f}")
        self.stdout.write(f"Artifacts saved to: {TASK1_DIR}")
