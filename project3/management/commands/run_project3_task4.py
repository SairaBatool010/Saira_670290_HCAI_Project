from django.core.management.base import BaseCommand

from project3.config import TASK4_DIR
from project3.task4_pipeline import run_task4_pipeline


class Command(BaseCommand):
    help = "Run Project 3 Task 4 active learning pipeline (Phases 6-7)."

    def add_arguments(self, parser):
        parser.add_argument("--random-state", type=int, default=42)
        parser.add_argument("--val-size", type=float, default=0.1)

    def handle(self, *args, **options):
        self.stdout.write("Running Project 3 Task 4 pipeline (this may take several minutes)...")
        summary = run_task4_pipeline(
            random_state=options["random_state"],
            val_size=options["val_size"],
        )
        self.stdout.write(self.style.SUCCESS("Task 4 complete."))
        for strategy, rows in summary["strategies"].items():
            best = max(rows, key=lambda row: row["test_metrics"]["team_accuracy"])
            self.stdout.write(
                f"{strategy}: best team accuracy {best['test_metrics']['team_accuracy']:.4f} "
                f"at {best['budget']} queries"
            )
        self.stdout.write(f"Artifacts saved to: {TASK4_DIR}")
