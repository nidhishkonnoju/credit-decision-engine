"""Generate a downloadable PDF Credit Appraisal Memorandum from 5C CAM data.

Uses ReportLab Platypus for structured page layout with professional styling.
"""
from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm, mm
from reportlab.platypus import (
    Flowable,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

# ── Color palette ──────────────────────────────────────────────────────────────
COLOR_PRIMARY = colors.HexColor("#1a237e")        # Deep indigo
COLOR_HEADER_BG = colors.HexColor("#e8eaf6")      # Light indigo tint
COLOR_APPROVE = colors.HexColor("#2e7d32")         # Green
COLOR_REVIEW = colors.HexColor("#f57f17")          # Amber
COLOR_REJECT = colors.HexColor("#c62828")          # Red
COLOR_SECTION_BG = colors.HexColor("#f5f5f5")      # Light grey
COLOR_BORDER = colors.HexColor("#bdbdbd")          # Medium grey
COLOR_TEXT_DARK = colors.HexColor("#212121")        # Near-black
COLOR_TEXT_SECONDARY = colors.HexColor("#616161")   # Dark grey
COLOR_POSITIVE = colors.HexColor("#c62828")         # Red for risk-increasing
COLOR_NEGATIVE = colors.HexColor("#2e7d32")         # Green for risk-reducing

DECISION_COLORS = {
    "APPROVE": COLOR_APPROVE,
    "REVIEW": COLOR_REVIEW,
    "REJECT": COLOR_REJECT,
}

FIVE_C_DESCRIPTIONS = {
    "Character": "Repayment history, credit behavior, and account management track record.",
    "Capacity": "Ability to repay based on income, employment stability, and cash flow.",
    "Capital": "Financial reserves, credit utilization, and balance management.",
    "Collateral": "Asset backing, secured vs. unsecured exposure, and pledged security.",
    "Conditions": "External context such as age, education, and recent borrowing activity.",
}

DEFAULT_OUTPUT_DIR = Path("output")


# ── Custom Flowables ──────────────────────────────────────────────────────────


class HorizontalRule(Flowable):
    """A simple horizontal line separator."""

    def __init__(self, width: float, thickness: float = 0.5, color: colors.Color = COLOR_BORDER):
        super().__init__()
        self.width = width
        self.thickness = thickness
        self.color = color

    def wrap(self, available_width: float, available_height: float) -> tuple[float, float]:
        self.width = min(self.width, available_width)
        return self.width, self.thickness + 2 * mm

    def draw(self) -> None:
        self.canv.setStrokeColor(self.color)
        self.canv.setLineWidth(self.thickness)
        self.canv.line(0, mm, self.width, mm)


class DecisionBadge(Flowable):
    """A colored rounded-rectangle badge showing the decision."""

    def __init__(self, decision: str, width: float = 4 * cm, height: float = 1.2 * cm):
        super().__init__()
        self.decision = decision
        self.badge_width = width
        self.badge_height = height

    def wrap(self, available_width: float, available_height: float) -> tuple[float, float]:
        return self.badge_width, self.badge_height

    def draw(self) -> None:
        badge_color = DECISION_COLORS.get(self.decision, COLOR_TEXT_DARK)
        self.canv.setFillColor(badge_color)
        self.canv.roundRect(0, 0, self.badge_width, self.badge_height, 4 * mm, fill=1, stroke=0)
        self.canv.setFillColor(colors.white)
        self.canv.setFont("Helvetica-Bold", 14)
        text_width = self.canv.stringWidth(self.decision, "Helvetica-Bold", 14)
        x = (self.badge_width - text_width) / 2
        y = (self.badge_height - 14) / 2 + 2
        self.canv.drawString(x, y, self.decision)


# ── Styles ────────────────────────────────────────────────────────────────────


def _build_styles() -> dict[str, ParagraphStyle]:
    """Build a custom style dictionary for the CAM PDF."""
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "CAMTitle",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=20,
            textColor=COLOR_PRIMARY,
            spaceAfter=4 * mm,
            alignment=TA_CENTER,
        ),
        "subtitle": ParagraphStyle(
            "CAMSubtitle",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=10,
            textColor=COLOR_TEXT_SECONDARY,
            alignment=TA_CENTER,
            spaceAfter=6 * mm,
        ),
        "section_header": ParagraphStyle(
            "SectionHeader",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=13,
            textColor=COLOR_PRIMARY,
            spaceBefore=6 * mm,
            spaceAfter=3 * mm,
        ),
        "five_c_header": ParagraphStyle(
            "FiveCHeader",
            parent=base["Heading3"],
            fontName="Helvetica-Bold",
            fontSize=11,
            textColor=COLOR_TEXT_DARK,
            spaceBefore=4 * mm,
            spaceAfter=2 * mm,
        ),
        "five_c_desc": ParagraphStyle(
            "FiveCDesc",
            parent=base["Normal"],
            fontName="Helvetica-Oblique",
            fontSize=8,
            textColor=COLOR_TEXT_SECONDARY,
            spaceAfter=2 * mm,
        ),
        "body": ParagraphStyle(
            "CAMBody",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=10,
            textColor=COLOR_TEXT_DARK,
            leading=14,
            spaceAfter=2 * mm,
        ),
        "reason_positive": ParagraphStyle(
            "ReasonPositive",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=9,
            textColor=COLOR_POSITIVE,
            leftIndent=8 * mm,
            spaceAfter=1.5 * mm,
            bulletFontName="Helvetica",
            bulletFontSize=9,
        ),
        "reason_negative": ParagraphStyle(
            "ReasonNegative",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=9,
            textColor=COLOR_NEGATIVE,
            leftIndent=8 * mm,
            spaceAfter=1.5 * mm,
            bulletFontName="Helvetica",
            bulletFontSize=9,
        ),
        "empty_section": ParagraphStyle(
            "EmptySection",
            parent=base["Normal"],
            fontName="Helvetica-Oblique",
            fontSize=9,
            textColor=COLOR_TEXT_SECONDARY,
            leftIndent=8 * mm,
            spaceAfter=2 * mm,
        ),
        "label": ParagraphStyle(
            "MetricLabel",
            parent=base["Normal"],
            fontName="Helvetica-Bold",
            fontSize=9,
            textColor=COLOR_TEXT_DARK,
        ),
        "value": ParagraphStyle(
            "MetricValue",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=9,
            textColor=COLOR_TEXT_DARK,
        ),
        "footer": ParagraphStyle(
            "Footer",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=7,
            textColor=COLOR_TEXT_SECONDARY,
            alignment=TA_CENTER,
        ),
        "summary": ParagraphStyle(
            "Summary",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=10,
            textColor=COLOR_TEXT_DARK,
            leading=15,
            spaceAfter=4 * mm,
            borderWidth=1,
            borderColor=COLOR_BORDER,
            borderPadding=8,
        ),
    }


# ── PDF Content Builders ─────────────────────────────────────────────────────


def _build_header(
    styles: dict[str, ParagraphStyle],
    cam: dict[str, Any],
    page_width: float,
) -> list[Flowable]:
    """Build the report title and decision summary header."""
    elements: list[Flowable] = []

    elements.append(Paragraph("Credit Appraisal Memorandum", styles["title"]))
    now = datetime.datetime.now().strftime("%B %d, %Y at %H:%M")
    elements.append(Paragraph(f"Generated on {now}", styles["subtitle"]))
    elements.append(HorizontalRule(page_width, thickness=1.0, color=COLOR_PRIMARY))
    elements.append(Spacer(1, 4 * mm))

    # Decision badge row
    decision = cam.get("decision", "UNKNOWN")
    badge = DecisionBadge(decision)

    probability = cam.get("probability", 0.0)
    threshold = cam.get("threshold", 0.5)
    review_threshold = cam.get("review_threshold")

    score_lines = [
        f"<b>Rejection Risk Score:</b> {probability:.4f}",
        f"<b>Decision Threshold:</b> {threshold:.2f}",
    ]
    if review_threshold is not None:
        score_lines.append(f"<b>Review Threshold:</b> {review_threshold:.2f}")

    score_text = "<br/>".join(score_lines)
    score_para = Paragraph(score_text, styles["body"])

    badge_table = Table(
        [[badge, score_para]],
        colWidths=[4.5 * cm, page_width - 5 * cm],
    )
    badge_table.setStyle(
        TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (0, 0), 0),
            ("LEFTPADDING", (1, 0), (1, 0), 8 * mm),
        ])
    )
    elements.append(badge_table)
    elements.append(Spacer(1, 4 * mm))

    # Summary text
    summary = cam.get("summary", "")
    if summary:
        elements.append(Paragraph(summary, styles["summary"]))

    return elements


def _build_five_c_sections(
    styles: dict[str, ParagraphStyle],
    cam: dict[str, Any],
    page_width: float,
) -> list[Flowable]:
    """Build the structured 5C analysis sections."""
    elements: list[Flowable] = []

    elements.append(Paragraph("5C Credit Analysis", styles["section_header"]))
    elements.append(HorizontalRule(page_width, thickness=0.5))
    elements.append(Spacer(1, 2 * mm))

    sections = cam.get("sections", {})
    for category in ["Character", "Capacity", "Capital", "Collateral", "Conditions"]:
        section = sections.get(category, {"reasons": [], "empty_message": f"No data for {category}."})

        # Category header with description
        elements.append(Paragraph(f"▎ {category}", styles["five_c_header"]))
        desc = FIVE_C_DESCRIPTIONS.get(category, "")
        if desc:
            elements.append(Paragraph(desc, styles["five_c_desc"]))

        reasons = section.get("reasons", [])
        if reasons:
            for reason in reasons:
                text = reason.get("text", "")
                direction = reason.get("direction", "")
                contribution = reason.get("contribution", 0)
                abs_contribution = reason.get("absolute_contribution", abs(contribution))

                if "increases" in direction:
                    style = styles["reason_positive"]
                    arrow = "▲"
                else:
                    style = styles["reason_negative"]
                    arrow = "▼"

                reason_text = f"{arrow} {text} (impact: {abs_contribution:.4f})"
                elements.append(Paragraph(reason_text, style, bulletText="•"))
        else:
            empty_msg = section.get("empty_message", f"No high-confidence factors identified for {category}.")
            elements.append(Paragraph(empty_msg, styles["empty_section"]))

        elements.append(Spacer(1, 2 * mm))

    return elements


def _build_metrics_table(
    styles: dict[str, ParagraphStyle],
    model_result: dict[str, Any],
    page_width: float,
) -> list[Flowable]:
    """Build a formatted table of model performance metrics."""
    elements: list[Flowable] = []

    elements.append(Paragraph("Model Performance Metrics", styles["section_header"]))
    elements.append(HorizontalRule(page_width, thickness=0.5))
    elements.append(Spacer(1, 2 * mm))

    metrics = model_result.get("metrics", {})
    metric_display = [
        ("Accuracy", metrics.get("accuracy", "N/A")),
        ("Precision", metrics.get("precision", "N/A")),
        ("Recall", metrics.get("recall", "N/A")),
        ("F1 Score", metrics.get("f1", "N/A")),
        ("F2 Score", metrics.get("f2", "N/A")),
        ("ROC AUC", metrics.get("roc_auc", "N/A")),
        ("PR AUC", metrics.get("pr_auc", "N/A")),
        ("KS Statistic", metrics.get("ks", "N/A")),
        ("Predicted Reject Rate", metrics.get("predicted_reject_rate", "N/A")),
    ]

    # Build 2-column table
    table_data = [
        [
            Paragraph("<b>Metric</b>", styles["label"]),
            Paragraph("<b>Value</b>", styles["value"]),
            Paragraph("<b>Metric</b>", styles["label"]),
            Paragraph("<b>Value</b>", styles["value"]),
        ]
    ]

    for i in range(0, len(metric_display), 2):
        row = []
        name1, val1 = metric_display[i]
        row.append(Paragraph(name1, styles["label"]))
        row.append(Paragraph(_format_metric(val1), styles["value"]))
        if i + 1 < len(metric_display):
            name2, val2 = metric_display[i + 1]
            row.append(Paragraph(name2, styles["label"]))
            row.append(Paragraph(_format_metric(val2), styles["value"]))
        else:
            row.extend([Paragraph("", styles["label"]), Paragraph("", styles["value"])])
        table_data.append(row)

    col_width = page_width / 4
    table = Table(table_data, colWidths=[col_width * 1.3, col_width * 0.7, col_width * 1.3, col_width * 0.7])
    table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), COLOR_HEADER_BG),
            ("TEXTCOLOR", (0, 0), (-1, 0), COLOR_PRIMARY),
            ("GRID", (0, 0), (-1, -1), 0.5, COLOR_BORDER),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, COLOR_SECTION_BG]),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ])
    )
    elements.append(table)
    elements.append(Spacer(1, 4 * mm))

    return elements


def _build_stability_section(
    styles: dict[str, ParagraphStyle],
    model_result: dict[str, Any],
    cam: dict[str, Any],
    page_width: float,
) -> list[Flowable]:
    """Build the SHAP stability information section."""
    elements: list[Flowable] = []

    stability = model_result.get("stability_result")
    if not stability:
        return elements

    elements.append(Paragraph("SHAP Stability Analysis", styles["section_header"]))
    elements.append(HorizontalRule(page_width, thickness=0.5))
    elements.append(Spacer(1, 2 * mm))

    elements.append(
        Paragraph(
            "Explanations are restricted to features whose SHAP importance rankings remain "
            "stable across 100 model seeds, following Lin &amp; Wang (2025). "
            "Only the top 10 mean-ranked features with a rank range ≤ 3 are eligible for the memo.",
            styles["body"],
        )
    )

    stability_data = [
        [
            Paragraph("<b>Metric</b>", styles["label"]),
            Paragraph("<b>Value</b>", styles["value"]),
        ],
        [
            Paragraph("Number of Seeds", styles["label"]),
            Paragraph(str(stability.get("n_seeds", "N/A")), styles["value"]),
        ],
        [
            Paragraph("Overall Kendall's W", styles["label"]),
            Paragraph(_format_metric(stability.get("overall_w", "N/A")), styles["value"]),
        ],
        [
            Paragraph("Top-5 Feature W", styles["label"]),
            Paragraph(_format_metric(stability.get("top_5_w", "N/A")), styles["value"]),
        ],
        [
            Paragraph("Diversity Band W", styles["label"]),
            Paragraph(_format_metric(stability.get("diversity_band_w", "N/A")), styles["value"]),
        ],
        [
            Paragraph("Stable Features Used", styles["label"]),
            Paragraph(str(len(stability.get("stable_features", []))), styles["value"]),
        ],
        [
            Paragraph("Features Filtered Out", styles["label"]),
            Paragraph(str(len(stability.get("filtered_out", []))), styles["value"]),
        ],
    ]

    half_width = page_width / 2
    stability_table = Table(stability_data, colWidths=[half_width * 1.2, half_width * 0.8])
    stability_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), COLOR_HEADER_BG),
            ("TEXTCOLOR", (0, 0), (-1, 0), COLOR_PRIMARY),
            ("GRID", (0, 0), (-1, -1), 0.5, COLOR_BORDER),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, COLOR_SECTION_BG]),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ])
    )
    elements.append(stability_table)
    elements.append(Spacer(1, 3 * mm))

    # List the stable features
    stable_features = stability.get("stable_features", [])
    if stable_features:
        feature_list = ", ".join(stable_features)
        elements.append(
            Paragraph(
                f"<b>Stable features eligible for memo:</b> {feature_list}",
                styles["body"],
            )
        )

    return elements


def _build_footer(styles: dict[str, ParagraphStyle], page_width: float) -> list[Flowable]:
    """Build the report footer with disclaimer."""
    elements: list[Flowable] = []

    elements.append(Spacer(1, 8 * mm))
    elements.append(HorizontalRule(page_width, thickness=0.5, color=COLOR_TEXT_SECONDARY))
    elements.append(Spacer(1, 2 * mm))
    elements.append(
        Paragraph(
            "DISCLAIMER: This Credit Appraisal Memorandum is generated by an automated credit decision "
            "support system for evaluation purposes. It does not constitute a binding credit decision. "
            "All decisions should be reviewed by a qualified credit officer before final approval.",
            styles["footer"],
        )
    )
    elements.append(Spacer(1, 2 * mm))
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    elements.append(
        Paragraph(
            f"Report generated at {now} • Credit Decision Engine • Stability-Filtered 5C Analysis",
            styles["footer"],
        )
    )

    return elements


# ── Helpers ───────────────────────────────────────────────────────────────────


def _format_metric(value: Any) -> str:
    """Format a metric value for display."""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _resolve_output_path(output_path: str | None) -> Path:
    """Determine the output file path for the PDF."""
    if output_path is None or output_path == "auto":
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = DEFAULT_OUTPUT_DIR / f"cam_report_{timestamp}.pdf"
    else:
        path = Path(output_path)
        if path.suffix.lower() != ".pdf":
            path = path.with_suffix(".pdf")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


# ── Main Entry Point ─────────────────────────────────────────────────────────


def generate_cam_pdf(
    cam: dict[str, Any],
    model_result: dict[str, Any],
    output_path: str | None = None,
) -> Path:
    """Generate a professional PDF Credit Appraisal Memorandum.

    Parameters
    ----------
    cam : dict
        The structured 5C CAM output from ``generate_cam()``.
    model_result : dict
        The full model result dictionary containing metrics, stability data, etc.
    output_path : str or None
        Where to save the PDF. If ``None`` or ``"auto"``, generates a timestamped
        filename in the ``output/`` directory.

    Returns
    -------
    Path
        The absolute path to the generated PDF file.
    """
    pdf_path = _resolve_output_path(output_path)
    styles = _build_styles()

    page_width = A4[0] - 2 * 2 * cm  # A4 width minus left+right margins

    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
        title="Credit Appraisal Memorandum",
        author="Credit Decision Engine",
    )

    elements: list[Flowable] = []
    elements.extend(_build_header(styles, cam, page_width))
    elements.extend(_build_five_c_sections(styles, cam, page_width))
    elements.extend(_build_metrics_table(styles, model_result, page_width))
    elements.extend(_build_stability_section(styles, model_result, cam, page_width))
    elements.extend(_build_footer(styles, page_width))

    doc.build(elements)
    return pdf_path.resolve()
