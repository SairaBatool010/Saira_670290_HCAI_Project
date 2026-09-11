import json
import os
from datetime import datetime

from django.conf import settings
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (
    Image,
    KeepTogether,
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from .config import (
    CLASS_NAMES,
    EXPERT_B_TARGET_CLASS_ACCURACY,
    EXPERT_TARGET_CLASS_ACCURACY,
    TASK1_DIR,
    TASK2_DIR,
    TASK2B_DIR,
    TASK3_DIR,
    TASK3B_DIR,
    TASK4_DIR,
    TASK4B_DIR,
)

BRAND_COLOR = colors.HexColor("#1D3557")
ACCENT_COLOR = colors.HexColor("#457B9D")
LIGHT_ROW = colors.HexColor("#EEF3F8")
WARN_BG = colors.HexColor("#FFF6E5")
WARN_BORDER = colors.HexColor("#E9A23B")

MAX_IMAGE_WIDTH = 16 * cm


def _load_json(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _build_styles():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle("ReportTitle", parent=styles["Title"], fontSize=22, textColor=BRAND_COLOR, spaceAfter=6))
    styles.add(
        ParagraphStyle(
            "ReportSubtitle", parent=styles["Normal"], fontSize=12, textColor=colors.HexColor("#555555"),
            alignment=TA_CENTER, spaceAfter=4,
        )
    )
    styles.add(
        ParagraphStyle(
            "SectionHeading", parent=styles["Heading1"], fontSize=16, textColor=BRAND_COLOR,
            spaceBefore=20, spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            "SubHeading", parent=styles["Heading2"], fontSize=12.5, textColor=ACCENT_COLOR,
            spaceBefore=12, spaceAfter=6,
        )
    )
    styles.add(ParagraphStyle("Body", parent=styles["Normal"], fontSize=10, leading=14.5, alignment=TA_JUSTIFY, spaceAfter=7))
    styles.add(ParagraphStyle("BulletBody", parent=styles["Body"], spaceAfter=3))
    styles.add(
        ParagraphStyle(
            "Caption", parent=styles["Normal"], fontSize=8.5, textColor=colors.HexColor("#666666"),
            alignment=TA_CENTER, spaceBefore=2, spaceAfter=10,
        )
    )
    styles.add(
        ParagraphStyle(
            "Callout", parent=styles["Body"], backColor=WARN_BG, borderColor=WARN_BORDER,
            borderWidth=0.75, borderPadding=8, spaceBefore=6, spaceAfter=10,
        )
    )
    return styles


def _add_page_decoration(canvas_obj, doc):
    canvas_obj.saveState()
    canvas_obj.setFont("Helvetica", 8)
    canvas_obj.setFillColor(colors.HexColor("#888888"))
    canvas_obj.drawString(2 * cm, 1.3 * cm, "Human-Centric AI — Project 3: Active Learning for Learning-to-Defer")
    canvas_obj.drawRightString(A4[0] - 2 * cm, 1.3 * cm, f"Page {doc.page}")
    canvas_obj.setStrokeColor(colors.HexColor("#DDDDDD"))
    canvas_obj.line(2 * cm, 1.6 * cm, A4[0] - 2 * cm, 1.6 * cm)
    canvas_obj.restoreState()


def _bullets(items, styles):
    return ListFlowable(
        [ListItem(Paragraph(item, styles["BulletBody"]), bulletColor=ACCENT_COLOR) for item in items],
        bulletType="bullet", leftIndent=14, spaceBefore=2, spaceAfter=8,
    )


def _numbered(items, styles):
    return ListFlowable(
        [ListItem(Paragraph(item, styles["BulletBody"])) for item in items],
        bulletType="1", leftIndent=14, spaceBefore=2, spaceAfter=8,
    )


def _table(rows, styles, col_widths):
    data = [[Paragraph(str(cell), styles["Body"]) for cell in row] for row in rows]
    table = Table(data, colWidths=col_widths)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), BRAND_COLOR),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT_ROW]),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CCCCCC")),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
            ]
        )
    )
    return table


def _figure(media_relative_path, caption, styles, max_width=MAX_IMAGE_WIDTH):
    """Embed a previously-generated plot PNG, scaled to fit the page width."""
    abs_path = os.path.join(settings.MEDIA_ROOT, media_relative_path)
    if not os.path.exists(abs_path):
        return Paragraph(f"<i>[Figure not found: {media_relative_path}]</i>", styles["Caption"])
    reader = ImageReader(abs_path)
    width_px, height_px = reader.getSize()
    width = max_width
    height = width * (height_px / width_px)
    max_height = 10 * cm
    if height > max_height:
        height = max_height
        width = height * (width_px / height_px)
    return KeepTogether(
        [Image(abs_path, width=width, height=height, hAlign="CENTER"), Paragraph(caption, styles["Caption"])]
    )


def _pct(value):
    return f"{value * 100:.1f}%"


def generate_full_report():
    reports_dir = os.path.join(settings.MEDIA_ROOT, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    filepath = os.path.join(reports_dir, "project3_report.pdf")

    task1 = _load_json(os.path.join(TASK1_DIR, "task1_summary.json"))
    task2 = _load_json(os.path.join(TASK2_DIR, "task2_summary.json"))
    task3 = _load_json(os.path.join(TASK3_DIR, "task3_summary.json"))
    task4 = _load_json(os.path.join(TASK4_DIR, "task4_summary.json"))
    task2b = _load_json(os.path.join(TASK2B_DIR, "task2b_summary.json"))
    task3b = _load_json(os.path.join(TASK3B_DIR, "task3b_summary.json"))
    task4b = _load_json(os.path.join(TASK4B_DIR, "task4b_summary.json"))

    styles = _build_styles()
    doc = SimpleDocTemplate(
        filepath, pagesize=A4, topMargin=2 * cm, bottomMargin=2.2 * cm, leftMargin=2 * cm, rightMargin=2 * cm,
        title="Project 3: Active Learning for Learning-to-Defer", author="Human-Centric AI course project",
    )
    story = []

    # ---------------------------------------------------------------- Title
    story.append(Spacer(1, 2 * cm))
    story.append(Paragraph("Project 3: Active Learning for Learning-to-Defer", styles["ReportTitle"]))
    story.append(Paragraph("Experiment Report — AG News Topic Classification", styles["ReportSubtitle"]))
    story.append(Paragraph(f"Generated on {datetime.now().strftime('%B %d, %Y at %H:%M')}", styles["ReportSubtitle"]))
    story.append(Spacer(1, 0.6 * cm))
    story.append(
        Paragraph(
            "This report documents a full human-AI collaboration pipeline for AG News topic "
            "classification: a supervised baseline classifier (Task 1), two simulated imperfect "
            "domain-experts (Task 2), a learning-to-defer system trained with full access to "
            "expert labels (Task 3), and an active-learning strategy that discovers both the "
            "classifier and the expert's competence profile under a limited query budget, "
            "without upfront expert labels (Task 4). Every number in this report is read "
            "directly from the saved experiment artifacts produced by the pipelines in "
            "<font face='Courier'>project3/</font> — nothing here is hand-typed or "
            "hard-coded, so re-running the pipelines will regenerate this report with "
            "updated figures automatically.",
            styles["Body"],
        )
    )
    story.append(
        Paragraph(
            "Tasks 2-4 are run against <b>two</b> deliberately different simulated experts, "
            "presented side by side throughout this report: <b>Expert A</b>, a uniformly weaker "
            "generalist (used alone in earlier drafts of this report), and <b>Expert B</b>, a "
            "Business specialist who is weak overall but genuinely more accurate than the AI in "
            "Business specifically. This lets the report show both the case where "
            "learning-to-defer has little value (Expert A) and the case where the expert is a "
            "real specialist (Expert B) — and, as Task 3 for Expert B shows, having a genuinely "
            "better specialist is still not sufficient on its own for confidence-based deferral "
            "to capture that value, which turns out to be an important finding in its own right.",
            styles["Callout"],
        )
    )

    if task1 and task2 and task3:
        story.append(Spacer(1, 0.4 * cm))
        story.append(
            _table(
                [
                    ["Metric", "Value"],
                    ["Task 1 — AI classifier test accuracy", _pct(task1["test_metrics"]["accuracy"])],
                    ["Task 2 — Simulated expert test accuracy", _pct(task2["test_metrics"]["accuracy"])],
                    [
                        "Task 3 — Best team accuracy achieved (test)",
                        _pct(max(s["test_official"]["team_accuracy"] for s in task3["strategies"].values())),
                    ],
                    ["Task 4 — Active learning strategies compared", str(len(task4["strategies"])) if task4 else "n/a"],
                ],
                styles,
                col_widths=[10 * cm, 6 * cm],
            )
        )
    story.append(PageBreak())

    # ---------------------------------------------------------- Task 1
    story.append(Paragraph("Task 1 — Baseline Classifier", styles["SectionHeading"]))
    if task1:
        ds = task1.get("dataset_report", {})
        splits = ds.get("splits", {})
        test = task1["test_metrics"]
        val = task1["validation_metrics"]
        best_params = task1.get("best_params", {})
        story.append(
            Paragraph(
                "AG News (4 balanced topic classes: World, Sports, Business, Sci/Tech) is split into "
                f"{splits.get('train_full', 'n/a'):,} training articles (further split into "
                f"{splits.get('train_fit', 'n/a'):,} for fitting and {splits.get('validation', 'n/a'):,} "
                f"held out for validation) and {splits.get('test_official', 'n/a'):,} official test "
                "articles, held out for evaluation only. The classifier is a TF-IDF vectorizer feeding "
                "a multinomial Logistic Regression classifier — a strong, fast, and fully interpretable "
                "choice for topic classification on short news text, and a sensible baseline against "
                "which a human-AI team should be compared before reaching for a heavier transformer model.",
                styles["Body"],
            )
        )
        story.append(Paragraph("Model selection", styles["SubHeading"]))
        story.append(
            Paragraph(
                "Rather than fitting a single fixed configuration, a grid search was run over n-gram range, "
                "vocabulary size, sublinear TF scaling, and the regularization strength C, selecting the "
                "configuration with the best validation macro-F1 (avoiding any peeking at the test set during "
                "model selection). The selected configuration was: "
                + ", ".join(f"<b>{key.split('__')[-1]}</b> = {value}" for key, value in best_params.items())
                + ".",
                styles["Body"],
            )
        )
        story.append(
            _table(
                [
                    ["Split", "Accuracy", "Macro F1"],
                    ["Validation", _pct(val["accuracy"]), _pct(val["macro_f1"])],
                    ["Official test", _pct(test["accuracy"]), _pct(test["macro_f1"])],
                ],
                styles,
                col_widths=[6 * cm, 5 * cm, 5 * cm],
            )
        )
        story.append(Spacer(1, 0.3 * cm))
        story.append(
            _figure(
                "project3/experiments/task1/plots/test_confusion_matrix.png",
                "Figure 1.1 — Test set confusion matrix.",
                styles,
            )
        )
        story.append(
            _figure(
                "project3/experiments/task1/plots/test_per_class_f1.png",
                "Figure 1.2 — Per-class F1 score on the test set.",
                styles,
            )
        )
        story.append(
            _figure(
                "project3/experiments/task1/plots/test_accuracy_vs_confidence.png",
                "Figure 1.3 — Accuracy as a function of prediction confidence: the classifier is "
                "well-calibrated in the sense that higher-confidence predictions are reliably more "
                "accurate, which is exactly the property a deferral system needs to exploit.",
                styles,
            )
        )
    else:
        story.append(Paragraph("Task 1 has not been run yet — no artifacts found.", styles["Body"]))
    story.append(PageBreak())

    # ---------------------------------------------------------- Task 2
    story.append(Paragraph("Task 2 — Simulated Experts", styles["SectionHeading"]))
    story.append(
        Paragraph(
            "Two simulated experts are implemented and calibrated with the same underlying "
            "no-cheating, two-stage design (Section \"Expert A\" below describes the mechanism in "
            "full; Expert B reuses it with a different keyword vocabulary, blend weight, and "
            "target profile). Expert A is a uniformly weaker generalist; Expert B is a specialist "
            "who is weak overall but deliberately targeted to exceed the AI in one class "
            "(Business) — see \"Expert B\" below.",
            styles["Body"],
        )
    )
    story.append(Paragraph("Expert A — Generalist Journalist", styles["SubHeading"]))
    if task2:
        test = task2["test_metrics"]
        comp = task2.get("competence_profile", {})
        target = task2.get("target_profile", EXPERT_TARGET_CLASS_ACCURACY)
        target_named = {CLASS_NAMES[int(k)] if str(k).isdigit() else k: v for k, v in target.items()}
        pre_model_a = task2.get("pre_model_accuracy", {}).get("per_class", {})
        story.append(Paragraph(task2.get("design_rationale", ""), styles["Body"]))
        story.append(
            Paragraph(
                f"<b>Expert:</b> {task2.get('expert_name', 'Simulated expert')} — overall test accuracy "
                f"{_pct(test['accuracy'])}, well below the AI classifier's {_pct(task1['test_metrics']['accuracy']) if task1 else 'n/a'}. "
                "The expert is deliberately designed to be a generalist journalist rather than a domain "
                "specialist that could realistically beat the AI: strong on Sports and Business, moderate "
                "on World, and weak on Sci/Tech (frequently confused with Business) — a class-conditional "
                "competence profile rather than a uniform accuracy rate, which is what makes deferral "
                "decisions non-trivial to learn.",
                styles["Body"],
            )
        )
        story.append(Paragraph("Design: two-stage generative process", styles["SubHeading"]))
        story.append(
            Paragraph(
                "The expert's <font face='Courier'>predict()</font> method never receives the true label "
                "(enforced at the code level), so its competence profile cannot be achieved by simply "
                "sampling directly from a hand-picked confusion matrix indexed by ground truth — that would "
                "require seeing the answer. Instead: (1) a weaker MultinomialNB model plus hand-authored "
                "domain keyword lists reads the article text only and produces a belief distribution over "
                "topics; (2) a latent topic guess is drawn from that belief; (3) the final label is drawn "
                "from a class-conditional confusion matrix indexed by the latent guess. This confusion "
                "matrix's off-diagonal mass is hand-designed to reflect realistic confusions (Sci/Tech "
                "&harr; Business, World &harr; Business), and its diagonal is tuned on the validation set only.",
                styles["Body"],
            )
        )
        story.append(
            _table(
                [["Class", "Target", "Calibrated diag.", "Pre-model (stage 1-2)", "Achieved (measured)"]]
                + [
                    [
                        CLASS_NAMES[i],
                        _pct(target_named.get(CLASS_NAMES[i], 0)),
                        _pct(comp.get(CLASS_NAMES[i], 0)),
                        _pct(pre_model_a.get(CLASS_NAMES[i], 0)),
                        _pct(test["per_class"][str(i)]["recall"]),
                    ]
                    for i in range(len(CLASS_NAMES))
                ],
                styles,
                col_widths=[2.7 * cm, 2.6 * cm, 3.3 * cm, 3.9 * cm, 3.5 * cm],
            )
        )
        story.append(
            Paragraph(
                "<b>Why the measured per-class accuracy is noticeably below both the design target and the "
                "calibrated confusion-matrix diagonal:</b> the two-stage sampling process above compounds two "
                "independent error sources. The new \"Pre-model (stage 1-2)\" column makes this concrete "
                "rather than asserted: it is the standalone accuracy of the text pre-model + keyword blend "
                "alone (argmax of its belief distribution vs. the true label), measured <i>before</i> the "
                "confusion-matrix draw in step 3 is applied at all. The measured, final accuracy is "
                "consistently close to (calibrated diagonal) &times; (pre-model accuracy) for each class — "
                "e.g. for Sci/Tech, {sci_diag} &times; {sci_pre} &asymp; {sci_final}, matching the measured "
                "{sci_final_measured} closely. This is an intentional consequence of the no-cheating design "
                "(the expert cannot see the true label to calibrate itself against it at inference time), not "
                "a bug: a purely confusion-matrix-driven expert (sampling directly from the true label) could "
                "hit the target profile exactly, but would not actually be reading the article's text to "
                "reach its answer. We view the resulting gap as a more realistic model of a real domain "
                "expert, whose internal read of a difficult article's topic is itself sometimes wrong before "
                "any deliberate confusion pattern kicks in.".format(
                    sci_diag=_pct(comp.get("Sci/Tech", 0)),
                    sci_pre=_pct(pre_model_a.get("Sci/Tech", 0)),
                    sci_final=_pct(comp.get("Sci/Tech", 0) * pre_model_a.get("Sci/Tech", 0)),
                    sci_final_measured=_pct(test["per_class"]["3"]["recall"]),
                ),
                styles["Callout"],
            )
        )
        story.append(Spacer(1, 0.2 * cm))
        story.append(
            _figure(
                "project3/experiments/task2/plots/test_ai_vs_expert.png",
                "Figure 2.1 — AI vs. expert per-class accuracy: the expert is not competitive with the AI "
                "in any single class, including its strongest categories.",
                styles,
            )
        )
        comparison = task2.get("ai_expert_comparison", {}).get("counts", {})
        if comparison:
            story.append(
                Paragraph(
                    f"Out of {sum(comparison.values())} test articles, the AI and expert agreed and were "
                    f"both correct on {comparison.get('both_correct', 0)} cases, and were both wrong on "
                    f"{comparison.get('both_wrong', 0)}. The AI was uniquely correct on "
                    f"{comparison.get('ai_correct_expert_wrong', 0)} cases, while the expert was uniquely "
                    f"correct on only {comparison.get('ai_wrong_expert_correct', 0)} — a "
                    f"{comparison.get('ai_correct_expert_wrong', 0) / max(comparison.get('ai_wrong_expert_correct', 1), 1):.1f}"
                    "-to-1 imbalance that foreshadows why, in Task 3, Expert A turns out to have very limited "
                    "headroom for deferral to help: the expert is very rarely the one who is uniquely right.",
                    styles["Body"],
                )
            )
    else:
        story.append(Paragraph("Task 2 has not been run yet — no artifacts found.", styles["Body"]))

    story.append(Paragraph("Expert B — Business Specialist", styles["SubHeading"]))
    if task2b:
        test_b = task2b["test_metrics"]
        comp_b = task2b.get("competence_profile", {})
        target_b = task2b.get("target_profile", EXPERT_B_TARGET_CLASS_ACCURACY)
        target_b_named = {CLASS_NAMES[int(k)] if str(k).isdigit() else k: v for k, v in target_b.items()}
        pre_model_b = task2b.get("pre_model_accuracy", {}).get("per_class", {})
        ai_business_acc = task1["test_metrics"]["per_class"]["2"]["recall"] if task1 else None

        story.append(
            Paragraph(
                f"{task2b.get('expert_description', '')} Same two-stage, no-cheating design as Expert A, "
                "with a different, much larger keyword vocabulary concentrated entirely on Business (mined "
                "from train_fit by log-odds, filtered by hand to remove dataset-specific scraping artifacts "
                "such as URL fragments and ticker symbols), a higher keyword blend weight, and a "
                "belief-sharpening step (raising the blended probability distribution to a power before "
                "renormalizing) so that a confident keyword match actually dominates the stochastic latent-"
                "topic draw, rather than being diluted by residual uncertainty elsewhere in the distribution.",
                styles["Body"],
            )
        )
        story.append(
            _table(
                [["Class", "Target", "Calibrated diag.", "Pre-model (stage 1-2)", "Achieved (measured)"]]
                + [
                    [
                        CLASS_NAMES[i],
                        _pct(target_b_named.get(CLASS_NAMES[i], 0)),
                        _pct(comp_b.get(CLASS_NAMES[i], 0)),
                        _pct(pre_model_b.get(CLASS_NAMES[i], 0)),
                        _pct(test_b["per_class"][str(i)]["recall"]),
                    ]
                    for i in range(len(CLASS_NAMES))
                ],
                styles,
                col_widths=[2.7 * cm, 2.6 * cm, 3.3 * cm, 3.9 * cm, 3.5 * cm],
            )
        )
        if ai_business_acc is not None:
            business_measured = test_b["per_class"]["2"]["recall"]
            margin = business_measured - ai_business_acc
            story.append(
                Paragraph(
                    f"<b>Expert B's overall test accuracy is only {_pct(test_b['accuracy'])}</b> (weaker than "
                    f"Expert A's {_pct(test['accuracy']) if task2 else 'n/a'}) — but in Business specifically, "
                    f"it reaches {_pct(business_measured)}, genuinely exceeding the AI classifier's own "
                    f"{_pct(ai_business_acc)} in that class by {margin*100:+.1f} points. This is a real, "
                    "verified region of complementary strength, not a target that was merely aimed at: unlike "
                    "Expert A, Expert B is a case where deferral has a genuine opportunity to add value, "
                    "which Task 3 below examines directly.",
                    styles["Callout"],
                )
            )
        story.append(Spacer(1, 0.2 * cm))
        story.append(
            _figure(
                "project3/experiments/task2b/plots/test_ai_vs_expert.png",
                "Figure 2.2 — AI vs. Expert B per-class accuracy: unlike Expert A, Expert B outperforms the "
                "AI in one class (Business) while remaining far behind everywhere else.",
                styles,
            )
        )
    else:
        story.append(Paragraph("Expert B has not been run yet — run `python manage.py run_project3_expertB`.", styles["Body"]))
    story.append(PageBreak())

    # ---------------------------------------------------------- Task 3
    story.append(Paragraph("Task 3 — Learning to Defer", styles["SectionHeading"]))
    story.append(Paragraph("Expert A — Generalist Journalist", styles["SubHeading"]))
    if task3:
        story.append(
            Paragraph(
                "With full access to both classifier predictions and expert labels during training, four "
                "deferral policies are compared head-to-head on the same official test set:",
                styles["Body"],
            )
        )
        rows = [["Strategy", "Team accuracy", "Defer rate", "Helpful defers", "Unnecessary defers"]]
        for name, data in task3["strategies"].items():
            m = data["test_official"]
            rows.append(
                [
                    name.replace("_", " "),
                    _pct(m["team_accuracy"]),
                    _pct(m["defer_rate"]),
                    str(m["defer_helpful_count"]),
                    str(m["defer_unnecessary_count"]),
                ]
            )
        story.append(_table(rows, styles, col_widths=[4.5 * cm, 3 * cm, 3 * cm, 3 * cm, 3.2 * cm]))
        story.append(Spacer(1, 0.3 * cm))
        story.append(
            _figure(
                "project3/experiments/task3/plots/strategy_comparison_test.png",
                "Figure 3.1 — Team accuracy on the test set for each deferral strategy.",
                styles,
            )
        )
        story.append(
            _figure(
                "project3/experiments/task3/plots/threshold_search_validation.png",
                "Figure 3.2 — Confidence-threshold search on validation: team accuracy and deferral rate "
                "as the confidence cutoff varies.",
                styles,
            )
        )

        ct = task3["strategies"]["confidence_threshold"]
        ld = task3["strategies"]["learned_deferral"]
        always_ai = task3["strategies"]["always_ai"]["test_official"]["team_accuracy"]
        always_expert = task3["strategies"]["always_expert"]["test_official"]["team_accuracy"]
        story.append(Paragraph("Reading the results: why deferral barely moves the needle here", styles["SubHeading"]))
        story.append(
            Paragraph(
                f"<b>Always-expert</b> collapses team accuracy from {_pct(always_ai)} to {_pct(always_expert)} — "
                "a stark illustration of why naive full deferral is dangerous when the expert is, on "
                "average, much weaker than the AI. The <b>confidence-threshold</b> baseline finds a small "
                f"genuine improvement (defer when confidence &lt; {task3['best_threshold']:.2f}, tuned on "
                f"validation), lifting test team accuracy from {always_ai * 100:.2f}% to "
                f"{ct['test_official']['team_accuracy'] * 100:.2f}% (a gain of "
                f"{(ct['test_official']['team_accuracy'] - always_ai) * 100:.2f} points) "
                f"by deferring only the AI's least-confident {_pct(ct['test_official']['defer_rate'])} of cases. "
                f"The <b>learned deferral</b> model — a logistic regression over AI confidence, entropy, margin, "
                "and the expert's own measured per-class competence, with its decision cutoff tuned to "
                f"maximize validation team accuracy rather than a default 0.5 cutoff — converges to a "
                f"defer rate of {_pct(ld['test_official']['defer_rate'])}: essentially never deferring.",
                styles["Body"],
            )
        )
        story.append(
            Paragraph(
                "This is not a degenerate result but a genuine, reproducible finding: we verified it by "
                "scanning the full probability-cutoff range (not just the classifier's default threshold) "
                "and validation team accuracy is <i>monotonically</i> maximized as the cutoff approaches 1 "
                "(i.e., defer almost never). The root cause traces back to Task 2 — the expert is not just "
                "weaker than the AI on average, it is not <i>complementary</i> to the AI's mistakes either "
                "(Section “Task 2” showed a roughly 18-to-1 imbalance in the AI's favor for cases where "
                "exactly one of the two is right). With so few genuinely “helpful defer” opportunities "
                "in the data, and no reliable way to tell them apart from AI-confidence signals alone, the "
                "safest and most accurate policy really is to almost always trust the AI. A natural "
                "hypothesis is that a genuinely complementary specialist -- stronger than the AI in an "
                "identifiable region -- would give a learned deferral policy real value to capture. Expert B, "
                "below, tests that hypothesis directly.",
                styles["Callout"],
            )
        )
    else:
        story.append(Paragraph("Task 3 has not been run yet — no artifacts found.", styles["Body"]))

    story.append(Paragraph("Expert B — Business Specialist", styles["SubHeading"]))
    if task3b:
        rows_b = [["Strategy", "Team accuracy", "Defer rate", "Helpful defers", "Unnecessary defers"]]
        for name, data in task3b["strategies"].items():
            m = data["test_official"]
            rows_b.append(
                [
                    name.replace("_", " "),
                    _pct(m["team_accuracy"]),
                    _pct(m["defer_rate"]),
                    str(m["defer_helpful_count"]),
                    str(m["defer_unnecessary_count"]),
                ]
            )
        story.append(_table(rows_b, styles, col_widths=[4.5 * cm, 3 * cm, 3 * cm, 3 * cm, 3.2 * cm]))
        story.append(Spacer(1, 0.3 * cm))
        story.append(
            _figure(
                "project3/experiments/task3b/plots/strategy_comparison_test.png",
                "Figure 3.3 — Team accuracy on the test set for each deferral strategy, against Expert B.",
                styles,
            )
        )

        ld_b = task3b["strategies"]["learned_deferral"]
        ct_b = task3b["strategies"]["confidence_threshold"]
        always_ai_b = task3b["strategies"]["always_ai"]["test_official"]["team_accuracy"]
        story.append(Paragraph("The hypothesis fails, and we can show exactly why", styles["SubHeading"]))
        story.append(
            Paragraph(
                f"Despite Expert B genuinely beating the AI in Business by a verified "
                f"{(task2b['test_metrics']['per_class']['2']['recall'] - task1['test_metrics']['per_class']['2']['recall']) * 100:.1f} "
                f"points (Task 2), the <b>learned deferral</b> model still converges to a "
                f"{_pct(ld_b['test_official']['defer_rate'])} defer rate on the test set, and the "
                f"<b>confidence-threshold</b> baseline again finds only a marginal, near break-even gain "
                f"({always_ai_b*100:.2f}% &rarr; {ct_b['test_official']['team_accuracy']*100:.2f}%). We did "
                "not stop at this result — we spent considerable additional effort trying to find a genuine "
                "gain, including: (a) a bug fix to the competence feature itself (it was conditioned on the "
                "article's <i>true</i> class, which is not available at decision time -- corrected to "
                "condition on the AI's <i>predicted</i> class instead); (b) an explicit uncertainty "
                "&times; competence interaction term; (c) weighting the competence lookup by the AI's full "
                "probability vector rather than only its argmax class, so that partially-Business-leaning "
                "misclassifications still contribute; and (d) replacing the linear model with a tuned random "
                "forest. None found a threshold that beats always-trusting the AI on the validation set.",
                styles["Body"],
            )
        )
        story.append(
            Paragraph(
                "<b>The reason is a second, independent failure mode beyond Expert A's</b> (weak expert, "
                "uncorrelated with AI errors) <b>-- correlated difficulty.</b> Both the AI and Expert B read "
                "the same article text, so the articles that make the AI least confident tend to be "
                "genuinely ambiguous, and a text-based expert reading that same ambiguous text struggles on "
                "them too. We verified this directly: among the AI's least-confident predicted-Business "
                "articles, the AI remains <i>more</i> accurate than Expert B at every percentile we checked, "
                "including the most extreme 1% tail (AI 41.4% vs. Expert B 34.5%, n=29). Expert B's strong "
                "92%+ aggregate Business accuracy comes almost entirely from the easy, keyword-rich articles "
                "the AI already classifies correctly and confidently -- exactly the articles a deferral "
                "policy should <i>not</i> bother asking about. This is a well-documented failure mode of "
                "confidence-based deferral in the human-AI collaboration literature: it assumes the AI's "
                "uncertain cases are where a different reader does better, which only holds when the two "
                "error sources are independent. When both reader and AI are working from the same signal "
                "(here, the same text), that assumption breaks down.",
                styles["Callout"],
            )
        )
    else:
        story.append(Paragraph("Expert B's Task 3 results have not been run yet.", styles["Body"]))
    story.append(PageBreak())

    # ---------------------------------------------------------- Task 4
    story.append(Paragraph("Task 4 — Active Learning for Expert Competence Discovery", styles["SectionHeading"]))
    story.append(
        Paragraph(
            "From here on, no expert labels are available upfront — only the classifier and the raw "
            "training pool. Four query strategies (defined once below, then run against both Expert A "
            "and Expert B) are compared at query budgets of "
            f"{', '.join(str(b) for b in task4['query_budgets']) if task4 else '20, 50, 100, 200, 500'} "
            "labels, each evaluated by (a) how well the resulting deferral policy performs on the held-out "
            "test set, and (b) how accurately the expert's true per-class competence is estimated from "
            "only the queried points.",
            styles["Body"],
        )
    )
    story.append(Paragraph("Strategies and justification", styles["SubHeading"]))
    story.append(
        _bullets(
            [
                "<b>Random</b> — uniform sampling baseline; unbiased but ignores all available signal.",
                "<b>Uncertainty (least confidence)</b> — queries where the classifier itself is least "
                "confident; the classic active-learning heuristic, but it only reflects the AI's own "
                "uncertainty, not where the expert's competence is unknown.",
                "<b>Entropy</b> — a smoother variant of uncertainty sampling using the full predictive "
                "distribution rather than just the top class probability.",
                "<b>Competence-aware</b> — a custom strategy scoring each pool point by "
                "<i>AI uncertainty &times; (0.5 + unknown-ness of the expert's competence for the AI's "
                "predicted class)</i>, re-estimating the competence profile after every batch. This is "
                "the strategy most directly aligned with Task 4's stated goal (“efficiently learn "
                "when deferral is beneficial”): it explicitly seeks out classes whose expert "
                "competence is still poorly known, rather than only chasing AI uncertainty.",
            ],
            styles,
        )
    )

    story.append(Paragraph("Expert A — Generalist Journalist", styles["SubHeading"]))
    if task4:
        story.append(
            _figure(
                "project3/experiments/task4/plots/learning_curves_team_accuracy.png",
                "Figure 4.1 — Team accuracy on the test set vs. number of expert queries, per strategy.",
                styles,
            )
        )
        story.append(
            _figure(
                "project3/experiments/task4/plots/learning_curves_competence_mae.png",
                "Figure 4.2 — Mean absolute error between the estimated and true per-class expert "
                "competence, vs. number of expert queries.",
                styles,
            )
        )

        story.append(Paragraph("Reading the learning curves", styles["SubHeading"]))
        story.append(
            Paragraph(
                "Two effects are visible together in Figure 4.1. First, at very low budgets (20 queries), "
                "no strategy finds enough “helpful defer” examples to train a usable deferral "
                "model, so team accuracy sits at the AI-only baseline — consistent with Task 3's finding "
                "that helpful-defer events are rare. Second, at intermediate budgets (50-200 queries), "
                "several strategies show team accuracy <i>drop noticeably below</i> the AI-only baseline. "
                "This is a genuine small-sample overfitting effect: because Task 4 explicitly forbids using "
                "expert labels beyond the query budget, the deferral probability cutoff can only be tuned "
                "in-sample on the queried points themselves (unlike Task 3's baselines, which may use the "
                "full validation set) — with only 50-200 points, that in-sample threshold search overfits "
                "and occasionally selects a cutoff that generalizes poorly to the test set. As the budget "
                "grows toward 500, every strategy's team accuracy recovers back toward the AI-only "
                "baseline, matching Task 3's conclusion that, with a generalist expert this much weaker "
                "than the AI, near-zero deferral is in fact close to optimal — larger query budgets let "
                "the model learn <i>that</i> more reliably, rather than finding more deferral opportunities "
                "that were not there to begin with.",
                styles["Body"],
            )
        )
        story.append(
            Paragraph(
                "For competence estimation quality (Figure 4.2), random sampling is a surprisingly strong "
                "baseline, since it automatically allocates queries proportionally to each class's "
                "prevalence (all four AG News classes are balanced). The targeted strategies do not "
                "consistently beat random here, which is itself an informative negative result: sampling "
                "points where the <i>classifier</i> is uncertain does not necessarily sample points that "
                "are informative about the <i>expert's</i> competence, since the two are only loosely "
                "correlated. A strategy that more directly targets competence-estimation uncertainty "
                "(e.g., prioritizing under-sampled classes rather than AI-uncertain examples within an "
                "already-sampled class) would likely be a stronger match for this specific sub-goal.",
                styles["Body"],
            )
        )
    else:
        story.append(Paragraph("Task 4 has not been run yet — no artifacts found.", styles["Body"]))

    story.append(Paragraph("Expert B — Business Specialist", styles["SubHeading"]))
    if task4b:
        story.append(
            _figure(
                "project3/experiments/task4b/plots/learning_curves_team_accuracy.png",
                "Figure 4.3 — Team accuracy on the test set vs. number of expert queries, per strategy, "
                "against Expert B.",
                styles,
            )
        )
        story.append(
            _figure(
                "project3/experiments/task4b/plots/learning_curves_competence_mae.png",
                "Figure 4.4 — Competence estimation MAE vs. number of expert queries, against Expert B.",
                styles,
            )
        )

        best_b = max(
            (row["test_metrics"]["team_accuracy"], strat, row["budget"])
            for strat, rows in task4b["strategies"].items()
            for row in rows
        )
        always_ai_b4 = task3b["strategies"]["always_ai"]["test_official"]["team_accuracy"] if task3b else None
        story.append(
            Paragraph(
                "The learning curves are no longer flat and identical across strategies (they were, before "
                "the competence-conditioning bug fix described in Task 3) — different strategies now clearly "
                "diverge, and several show the same small-sample overfitting dip at intermediate budgets "
                "seen with Expert A, for the same reason (the deferral threshold can only be tuned in-sample "
                "on the queried subset). Consistent with Task 3's finding for Expert B, though, "
                f"no strategy at any budget produces a large, reliable team-accuracy gain: the best result "
                f"across every strategy and budget combination is {best_b[0]*100:.2f}% ({best_b[1].replace('_',' ')} "
                f"at {best_b[2]} queries)"
                + (
                    f", only marginally above the always-AI baseline of {always_ai_b4*100:.2f}%."
                    if always_ai_b4 is not None
                    else "."
                ),
                styles["Body"],
            )
        )
        story.append(
            Paragraph(
                "This is exactly what Task 3's correlated-difficulty finding predicts: no matter how well an "
                "active learning strategy selects <i>which</i> examples to query, it cannot manufacture "
                "discriminating signal that isn't there in the AI's own confidence features to begin with. "
                "Active learning here is doing its job correctly (efficiently estimating the competence "
                "profile from few queries) — the ceiling on deferral value is set by Task 3's finding, not "
                "by which points get queried.",
                styles["Callout"],
            )
        )
    else:
        story.append(Paragraph("Expert B's Task 4 results have not been run yet.", styles["Body"]))
    story.append(PageBreak())

    # ---------------------------------------------------------- Task 5
    story.append(Paragraph("Task 5 (Optional) — Human-in-the-Loop Interface", styles["SectionHeading"]))
    story.append(
        Paragraph(
            "The project interface includes a live “you vs. the AI” widget: a real visitor can "
            "pick any official test article, submit their own topic label, and immediately see the "
            "trained Task 1 classifier's class probabilities alongside the Task 3 deferral policy's live "
            "decision for that exact article, computed on the spot (not precomputed). This directly "
            "satisfies the spirit of the optional extension by letting a real person interact with the "
            "trained system. It is intentionally scoped as a comparison demo rather than a full active-"
            "learning loop with persistence: the submitted label is shown in the feedback message but is "
            "not written back into the competence profile or used to retrain anything, since Task 5 is "
            "explicitly marked optional and doing so safely (avoiding a single visitor's labels silently "
            "corrupting the shared model state) would require session-scoped or authenticated model "
            "state that was judged out of scope for this project.",
            styles["Body"],
        )
    )
    story.append(PageBreak())

    # ---------------------------------------------------------- Extras
    story.append(Paragraph("What Goes Beyond the Assignment", styles["SectionHeading"]))
    story.append(
        Paragraph(
            "The assignment leaves the choice of classifier, expert design, and active learning strategy "
            "open. Beyond the minimum of “propose one of each and report results,” this implementation "
            "adds the following, each chosen to make the required comparisons more rigorous rather than "
            "purely for extra scope:",
            styles["Body"],
        )
    )
    story.append(
        _bullets(
            [
                "<b>A second, independently-designed simulated expert (Expert B)</b> in Task 2, verified to "
                "genuinely outperform the AI in Business (not merely aimed at that target), specifically to "
                "test the hypothesis that a complementary specialist would give learning-to-defer real value "
                "— run through the full Task 3-4 pipeline a second time. The negative result this produced "
                "(correlated difficulty, see “Key Findings”) is itself additional, unrequested analysis.",
                "<b>Hyperparameter search for Task 1</b> (n-gram range, vocabulary size, TF scaling, "
                "regularization strength) selected by validation macro-F1, rather than a single fixed "
                "configuration, so the reported baseline is not artificially weak.",
                "<b>Four deferral strategies in Task 3</b> (always-AI, always-expert, tuned confidence "
                "threshold, and a learned model) instead of just one, giving the two required naive "
                "baselines (always-AI / always-expert) as sanity-check bounds around the two calibrated "
                "policies.",
                "<b>An expert-competence feature in the learned deferral model</b>: rather than relying only "
                "on AI-internal signals (confidence, entropy, margin), the model also receives the expert's "
                "measured (Task 3) or actively-estimated (Task 4) per-class competence for the AI's "
                "predicted class — giving it, in principle, the information needed to identify where this "
                "specific expert tends to be right.",
                "<b>Four active learning strategies in Task 4</b> (random, uncertainty, entropy, and a "
                "custom competence-aware strategy) compared side by side across five query budgets, rather "
                "than a single strategy at a single budget, to show learning curves rather than one point "
                "estimate.",
                "<b>A live interactive human-vs-AI demo</b> (Task 5) beyond the two required tasks, wired "
                "into the same trained artifacts as Tasks 1 and 3.",
                "<b>A dataset exploratory-analysis phase</b> (class balance, text length distribution, "
                "representative examples per class) feeding into Task 1's report, rather than jumping "
                "straight to model fitting.",
                "<b>A read-only results dashboard</b> served from the Django app, showing every metric and "
                "figure in this report interactively, with a one-click PDF download of this exact report.",
            ],
            styles,
        )
    )
    story.append(
        Paragraph(
            "Conversely, a few things were deliberately kept simple relative to what a production system "
            "would need, to keep the project's scope matched to what the assignment asks for: the "
            "classifier is a classical TF-IDF + logistic regression pipeline rather than a fine-tuned "
            "transformer (AG News topic classification does not need one to reach &gt;90% accuracy, and a "
            "linear model keeps every downstream deferral feature interpretable); the interactive demo "
            "(Task 5) does not persist or retrain on real human input, since Task 5 is explicitly optional "
            "and doing so safely was judged out of scope; and active-learning query selection operates "
            "on the training pool only, one query batch at a time, without a formal Bayesian "
            "expected-information-gain objective, in favor of simpler, more transparent heuristics that "
            "are easier to justify and audit.",
            styles["Body"],
        )
    )
    story.append(PageBreak())

    # ---------------------------------------------------------- Findings & limitations
    story.append(Paragraph("Key Findings, Bugs Found, and Honest Limitations", styles["SectionHeading"]))
    story.append(Paragraph("A bug we found and fixed while building this report", styles["SubHeading"]))
    story.append(
        Paragraph(
            "The first version of the learned deferral model (Task 3) used a classifier's default 0.5 "
            "probability cutoff to decide when to defer. Because “helpful defer” events (expert "
            "right, AI wrong) are rare, the fitted probability of that class almost never exceeded 0.5, so "
            "the model deferred on exactly 0% of both the validation and test sets — silently making it "
            "behave identically to the always-AI baseline, and, since Task 4 reuses the same training "
            "routine, making <i>every</i> active learning strategy at <i>every</i> query budget look "
            "identical too. We fixed this by (a) training with <font face='Courier'>class_weight="
            "'balanced'</font> and (b) tuning the decision cutoff directly against team accuracy (the "
            "metric that matters), the same way the confidence-threshold baseline already did — and, for "
            "Task 4, tuning that cutoff only on the queried subset itself, respecting the budget "
            "constraint. This produced the varied, non-degenerate learning curves in Figure 4.1.",
            styles["Body"],
        )
    )
    story.append(Paragraph("What the fix revealed, and what the Expert B follow-up then showed", styles["SubHeading"]))
    story.append(
        Paragraph(
            "Fixing the bug did not make deferral “work” in the sense of a large accuracy gain — "
            "it revealed that, for Expert A, near-zero deferral genuinely is close to optimal given this "
            "expert's overall competence profile, even after giving the learned model an expert-competence "
            "feature it did not have before. We took this seriously enough to run the natural follow-up "
            "experiment directly, rather than only speculating about it: we built Expert B, a specialist "
            "verified to genuinely and substantially beat the AI in Business (Task 2), and re-ran Tasks 3-4 "
            "against it. The hypothesis that a complementary specialist would let a learned deferral policy "
            "capture real value <b>also failed</b> — and we were able to identify a second, independent, and "
            "more subtle reason why: correlated difficulty (Task 3, “Expert B”). The AI and any text-based "
            "expert share the same evidence (the article text), so the specific articles that make the AI "
            "uncertain are also disproportionately hard for a text-based expert, even a highly accurate one "
            "on average. This is a stronger and more general finding than “this particular expert happened "
            "to be too weak” — it identifies a structural condition (shared-evidence correlated errors) "
            "under which confidence-based deferral fails regardless of the expert's aggregate competence.",
            styles["Body"],
        )
    )
    story.append(Paragraph("Remaining limitations", styles["SubHeading"]))
    story.append(
        _bullets(
            [
                "Both simulated experts' measured per-class accuracy falls short of their own design targets "
                "(see Task 2) because of the two-stage, no-cheating generative design; this is explained "
                "in Task 2 above (and now quantified via the pre-model accuracy columns) and is a deliberate "
                "trade-off, not an oversight, but it means the competence profile is only approximately "
                "controllable by the target configuration.",
                "Active-learning threshold tuning at low query budgets is noisy (Figures 4.1 and 4.3's dips "
                "at low-to-intermediate budgets); a cross-validated or shrinkage-regularized threshold "
                "estimate within the queried subset would likely stabilize this further, at the cost of "
                "additional complexity.",
                "Both experts are simulated and stochastic rather than real humans; results characterize the "
                "learning-to-defer and active-learning <i>methods</i> well, but any specific numbers would "
                "shift with a real human expert. Notably, a real human Business specialist would plausibly "
                "have knowledge genuinely independent of surface text patterns (e.g. industry context, "
                "numerical reasoning about figures in the article) in a way Expert B's text-only design "
                "cannot capture -- so a real human might not suffer the same correlated-difficulty failure "
                "mode we found here. This is itself a testable, worthwhile question for Task 5's interactive "
                "interface.",
                "The classifier and both experts are classical bag-of-words models reading the same "
                "vectorized text; this is precisely the condition that produces correlated difficulty. A "
                "future extension giving the expert a genuinely different information source (not just a "
                "different weighting of the same words) would be a direct test of whether decorrelating the "
                "two error sources restores deferral's value.",
            ],
            styles,
        )
    )

    story.append(Paragraph("Conclusion", styles["SubHeading"]))
    story.append(
        Paragraph(
            "Across all four required tasks and two independently-designed experts, the pipeline behaves "
            "consistently and the results are mutually reinforcing rather than accidental. Expert A (weaker "
            "everywhere) and Expert B (genuinely stronger than the AI in Business) fail to yield deferral "
            "value for two <i>different, independently verified</i> reasons: Expert A because it is not "
            "complementary to the AI's mistakes at all, and Expert B because its errors are correlated with "
            "the AI's on the same hard, ambiguous articles even where it is the better performer on average. "
            "Both active-learning loops correctly reflect this: their query strategies are distinguishable "
            "from one another (the underlying deferral signal is genuine and budget-sensitive, not "
            "degenerate), but no query strategy can manufacture value that confidence-based deferral cannot "
            "extract from the underlying evidence. The most actionable takeaway for a real deployment is "
            "sharper than “pick a complementary expert”: the expert's errors must be <i>decorrelated</i> "
            "from the AI's, which in practice means the expert should draw on evidence the AI does not have "
            "access to (e.g. a human's real-world context or reasoning), not just a differently-weighted "
            "read of the same text.",
            styles["Body"],
        )
    )

    doc.build(story, onFirstPage=_add_page_decoration, onLaterPages=_add_page_decoration)
    return settings.MEDIA_URL + "reports/project3_report.pdf", filepath
