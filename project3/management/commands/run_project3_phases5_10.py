from django.core.management.base import BaseCommand

from project3.config import TASK3_DIR, TASK4_DIR
from project3.report_generator import generate_full_report
from project3.task3_pipeline import run_task3_pipeline
from project3.task4_pipeline import run_task4_pipeline


class Command(BaseCommand):
    help = "Run Project 3 Tasks 3-4 and generate the full PDF report (Phases 5-10)."

    def add_arguments(self, parser):
        parser.add_argument("--random-state", type=int, default=42)
        parser.add_argument("--val-size", type=float, default=0.1)
        parser.add_argument("--skip-task3", action="store_true")
        parser.add_argument("--skip-task4", action="store_true")
        parser.add_argument("--skip-report", action="store_true")

    def handle(self, *args, **options):
        kwargs = {"random_state": options["random_state"], "val_size": options["val_size"]}

        if not options["skip_task3"]:
            self.stdout.write("Running Task 3...")
            run_task3_pipeline(**kwargs)

        if not options["skip_task4"]:
            self.stdout.write("Running Task 4...")
            run_task4_pipeline(**kwargs)

        if not options["skip_report"]:
            self.stdout.write("Generating PDF report...")
            report_url, report_path = generate_full_report()
            self.stdout.write(self.style.SUCCESS(f"Report saved to: {report_path}"))

        self.stdout.write(self.style.SUCCESS("Phases 5-10 pipeline complete."))
        self.stdout.write(f"Task 3 artifacts: {TASK3_DIR}")
        self.stdout.write(f"Task 4 artifacts: {TASK4_DIR}")
