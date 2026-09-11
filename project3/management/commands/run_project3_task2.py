from django.core.management.base import BaseCommand

from project3.config import TASK2_DIR
from project3.task2_pipeline import run_task2_pipeline


class Command(BaseCommand):
    help = "Run rigorous Project 3 Task 2 simulated expert pipeline (Phase 4)."

    def add_arguments(self, parser):
        parser.add_argument("--random-state", type=int, default=42)
        parser.add_argument("--val-size", type=float, default=0.1)

    def handle(self, *args, **options):
        self.stdout.write("Running Project 3 Task 2 pipeline...")
        summary = run_task2_pipeline(
            random_state=options["random_state"],
            val_size=options["val_size"],
        )

        test = summary["test_metrics"]
        comparison = summary["ai_expert_comparison"]["counts"]
        self.stdout.write(self.style.SUCCESS("Task 2 complete."))
        self.stdout.write(f"Validation expert accuracy: {summary['validation_metrics']['accuracy']:.4f}")
        self.stdout.write(f"Official test expert accuracy: {test['accuracy']:.4f}")
        self.stdout.write(f"Official test expert macro F1: {test['macro_f1']:.4f}")
        self.stdout.write(
            "AI vs expert (test): "
            f"both correct={comparison['both_correct']}, "
            f"AI only={comparison['ai_correct_expert_wrong']}, "
            f"expert only={comparison['ai_wrong_expert_correct']}, "
            f"both wrong={comparison['both_wrong']}"
        )
        self.stdout.write(f"Artifacts saved to: {TASK2_DIR}")
