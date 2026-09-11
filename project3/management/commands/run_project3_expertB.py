from django.core.management.base import BaseCommand

from project3.config import TASK2B_DIR, TASK3B_DIR, TASK4B_DIR
from project3.report_generator import generate_full_report
from project3.task2_pipeline import run_task2b_pipeline
from project3.task3_pipeline import run_task3b_pipeline
from project3.task4_pipeline import run_task4b_pipeline


class Command(BaseCommand):
    help = (
        "Run the complementary Expert B pipeline (Tasks 2-4): a Business specialist, "
        "weak overall but stronger than the AI in Business, to demonstrate the case "
        "where learning-to-defer and active learning have genuine value, alongside "
        "Expert A's case where they do not."
    )

    def add_arguments(self, parser):
        parser.add_argument("--random-state", type=int, default=42)
        parser.add_argument("--val-size", type=float, default=0.1)
        parser.add_argument("--skip-report", action="store_true")

    def handle(self, *args, **options):
        kwargs = {"random_state": options["random_state"], "val_size": options["val_size"]}

        self.stdout.write("Running Task 2 (Expert B)...")
        task2b = run_task2b_pipeline(**kwargs)
        test = task2b["test_metrics"]
        self.stdout.write(f"Expert B official test accuracy: {test['accuracy']:.4f}")
        self.stdout.write(f"Expert B per-class accuracy: {task2b['competence_profile']}")
        self.stdout.write(f"Artifacts saved to: {TASK2B_DIR}")

        self.stdout.write("Running Task 3 (Expert B)...")
        task3b = run_task3b_pipeline(**kwargs)
        best_team_acc = max(s["test_official"]["team_accuracy"] for s in task3b["strategies"].values())
        self.stdout.write(f"Best team accuracy (Expert B, test): {best_team_acc:.4f}")
        self.stdout.write(f"Artifacts saved to: {TASK3B_DIR}")

        self.stdout.write("Running Task 4 (Expert B)...")
        task4b = run_task4b_pipeline(**kwargs)
        self.stdout.write(f"Artifacts saved to: {TASK4B_DIR}")

        if not options["skip_report"]:
            self.stdout.write("Regenerating PDF report...")
            report_url, report_path = generate_full_report()
            self.stdout.write(self.style.SUCCESS(f"Report saved to: {report_path}"))

        self.stdout.write(self.style.SUCCESS("Expert B pipeline complete."))
