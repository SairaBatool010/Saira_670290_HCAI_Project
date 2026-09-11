import os
from datetime import datetime

from django.conf import settings
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from .data import FEATURE_DESCRIPTIONS, TOP_GENRES
from .study_design import (
    FEATURE_JUSTIFICATION,
    MODELING_NOTES,
    RANKING_MODEL_TEXT,
    STUDY_DESIGN,
)

BRAND_COLOR = colors.HexColor("#275CB2")
LIGHT_ROW = colors.HexColor("#EEF2FA")


def _build_styles():
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            "ReportTitle",
            parent=styles["Title"],
            fontSize=22,
            textColor=BRAND_COLOR,
            spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle(
            "ReportSubtitle",
            parent=styles["Normal"],
            fontSize=12,
            textColor=colors.HexColor("#555555"),
            alignment=TA_CENTER,
            spaceAfter=4,
        )
    )
    styles.add(
        ParagraphStyle(
            "SectionHeading",
            parent=styles["Heading1"],
            fontSize=15,
            textColor=BRAND_COLOR,
            spaceBefore=18,
            spaceAfter=8,
            borderPadding=0,
        )
    )
    styles.add(
        ParagraphStyle(
            "SubHeading",
            parent=styles["Heading2"],
            fontSize=12,
            textColor=colors.HexColor("#333333"),
            spaceBefore=10,
            spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle(
            "Body",
            parent=styles["Normal"],
            fontSize=10.5,
            leading=15,
            alignment=TA_JUSTIFY,
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            "BulletBody",
            parent=styles["Body"],
            spaceAfter=2,
        )
    )
    return styles


def _add_page_decoration(canvas_obj, doc):
    canvas_obj.saveState()
    canvas_obj.setFont("Helvetica", 8)
    canvas_obj.setFillColor(colors.HexColor("#888888"))
    canvas_obj.drawString(2 * cm, 1.3 * cm, "Human-Centric AI — Project 4: Preference Elicitation")
    canvas_obj.drawRightString(A4[0] - 2 * cm, 1.3 * cm, f"Page {doc.page}")
    canvas_obj.setStrokeColor(colors.HexColor("#DDDDDD"))
    canvas_obj.line(2 * cm, 1.6 * cm, A4[0] - 2 * cm, 1.6 * cm)
    canvas_obj.restoreState()


def _bullet_list(items, styles):
    return ListFlowable(
        [ListItem(Paragraph(item, styles["BulletBody"]), bulletColor=BRAND_COLOR) for item in items],
        bulletType="bullet",
        leftIndent=14,
        spaceBefore=2,
        spaceAfter=8,
    )


def _numbered_list(items, styles):
    return ListFlowable(
        [ListItem(Paragraph(item, styles["BulletBody"])) for item in items],
        bulletType="1",
        leftIndent=14,
        spaceBefore=2,
        spaceAfter=8,
    )


def _key_value_table(rows, styles, col_widths=(4.5 * cm, 11.5 * cm)):
    data = [
        [Paragraph(f"<b>{label}</b>", styles["Body"]), Paragraph(value, styles["Body"])]
        for label, value in rows
    ]
    table = Table(data, colWidths=list(col_widths))
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), LIGHT_ROW),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CCCCCC")),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    return table


def generate_report_pdf():
    reports_dir = os.path.join(settings.MEDIA_ROOT, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    filename = "project4_report.pdf"
    filepath = os.path.join(reports_dir, filename)

    styles = _build_styles()
    doc = SimpleDocTemplate(
        filepath,
        pagesize=A4,
        topMargin=2 * cm,
        bottomMargin=2.2 * cm,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        title="Project 4: Preference Elicitation",
        author="Human-Centric AI course project",
    )

    story = []

    # --- Title block -----------------------------------------------------
    story.append(Spacer(1, 2 * cm))
    story.append(Paragraph("Project 4: Preference Elicitation", styles["ReportTitle"]))
    story.append(Paragraph("Design and Implementation Report", styles["ReportSubtitle"]))
    story.append(
        Paragraph(
            f"Generated on {datetime.now().strftime('%B %d, %Y at %H:%M')}",
            styles["ReportSubtitle"],
        )
    )
    story.append(Spacer(1, 0.6 * cm))
    story.append(
        Paragraph(
            "This report documents the feature representation and preference model used by the "
            "interactive interface (Tasks 1 and 2), and the design of a user study comparing two "
            "elicitation interfaces (Task 3). The interactive elicitation interface itself "
            "(Task 4) is available from the project landing page.",
            styles["Body"],
        )
    )
    story.append(PageBreak())

    # --- Task 1: Feature representation -----------------------------------
    story.append(Paragraph("Task 1 — Feature Representation", styles["SectionHeading"]))
    story.append(Paragraph(FEATURE_JUSTIFICATION, styles["Body"]))
    story.append(Paragraph("Feature vector components", styles["SubHeading"]))
    story.append(_bullet_list(FEATURE_DESCRIPTIONS, styles))
    story.append(
        Paragraph(
            f"<b>Genre indicators</b> (one binary feature per genre): {', '.join(TOP_GENRES)}.",
            styles["Body"],
        )
    )
    story.append(
        Paragraph(
            "Each movie is represented by a standardized numeric feature vector x. "
            "User utility is modeled as a linear function of these features: "
            "U(x) = w<sup>T</sup>x, where w is the user's latent preference vector estimated "
            "from their responses.",
            styles["Body"],
        )
    )

    # --- Task 2: Ranking extension ----------------------------------------
    story.append(Paragraph("Task 2 — Ranking Extension of Bradley-Terry", styles["SectionHeading"]))
    story.append(Paragraph(RANKING_MODEL_TEXT, styles["Body"]))
    story.append(
        Paragraph(
            "In this project, Design 1 (pairwise comparisons) uses the standard two-item "
            "Bradley-Terry likelihood, and Design 2 (ranking of ten movies) uses the "
            "Plackett-Luce likelihood for full orderings described above. Both interfaces "
            "ultimately estimate the same latent vector w, which is what makes the two designs "
            "directly comparable.",
            styles["Body"],
        )
    )
    story.append(Paragraph("Parameter estimation", styles["SubHeading"]))
    story.append(Paragraph(MODELING_NOTES, styles["Body"]))
    story.append(PageBreak())

    # --- Task 3: User study --------------------------------------------
    story.append(Paragraph("Task 3 — User Study Protocol", styles["SectionHeading"]))
    story.append(Paragraph(f"<b>{STUDY_DESIGN['title']}</b>", styles["Body"]))
    story.append(Paragraph(STUDY_DESIGN["hypothesis"], styles["Body"]))

    story.append(Paragraph("Study overview", styles["SubHeading"]))
    story.append(
        _key_value_table(
            [
                ("Design type", STUDY_DESIGN["design_type"]),
                ("Design 1 (condition A)", STUDY_DESIGN["conditions"]["design1"]),
                ("Design 2 (condition B)", STUDY_DESIGN["conditions"]["design2"]),
                ("Target sample size", str(STUDY_DESIGN["participants"]["target_n"])),
                ("Inclusion criteria", STUDY_DESIGN["participants"]["inclusion"]),
                ("Recruitment", STUDY_DESIGN["participants"]["recruitment"]),
            ],
            styles,
        )
    )

    story.append(Paragraph("Procedure", styles["SubHeading"]))
    story.append(_numbered_list(STUDY_DESIGN["procedure"], styles))

    story.append(Paragraph("Outcome metrics", styles["SubHeading"]))
    story.append(_bullet_list(STUDY_DESIGN["metrics"], styles))

    story.append(Paragraph("Analysis plan", styles["SubHeading"]))
    story.append(_bullet_list(STUDY_DESIGN["analysis_plan"], styles))

    story.append(Paragraph("Known limitations and confounds", styles["SubHeading"]))
    story.append(_bullet_list(STUDY_DESIGN["confounds"], styles))

    doc.build(story, onFirstPage=_add_page_decoration, onLaterPages=_add_page_decoration)
    return settings.MEDIA_URL + "reports/" + filename, filepath
