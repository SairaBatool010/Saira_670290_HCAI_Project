from django.core.management.base import BaseCommand

from project3.config import TASK3_DIR
from project3.task3_pipeline import run_task3_pipeline


class Command(BaseCommand):
    help = "Run Project 3 Task 3 learning-to-defer pipeline (Phase 5)."

    def add_arguments(self, parser):
        parser.add_argument("--random-state", type=int, default=42)
        parser.add_argument("--val-size", type=float, default=0.1)

    def handle(self, *args, **options):
        self.stdout.write("Running Project 3 Task 3 pipeline...")
        summary = run_task3_pipeline(
            random_state=options["random_state"],
            val_size=options["val_size"],
        )
        learned = summary["strategies"]["learned_deferral"]["test_official"]
        threshold = summary["strategies"]["confidence_threshold"]["test_official"]
        self.stdout.write(self.style.SUCCESS("Task 3 complete."))
        self.stdout.write(
            f"Learned deferral test accuracy: {learned['team_accuracy']:.4f} "
            f"(defer rate {learned['defer_rate']:.4f})"
        )
        self.stdout.write(
            f"Confidence threshold test accuracy: {threshold['team_accuracy']:.4f} "
            f"(defer rate {threshold['defer_rate']:.4f})"
        )
        self.stdout.write(f"Artifacts saved to: {TASK3_DIR}")
