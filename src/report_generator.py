"""
Upgrade 2: PDF Threat Report Generation
File: src/report_generator.py
"""

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table,
    TableStyle, HRFlowable
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from datetime import datetime
import os

REPORTS_DIR = "logs"


def generate_threat_report(detections: list, session_start: str, session_end: str = None):
    """
    Generate a PDF threat report after a monitoring session.

    Args:
        detections   : List of dicts with keys:
                         process_name, pid, confidence, threat_level, timestamp, action_taken
        session_start: Session start time string
        session_end  : Session end time string (defaults to now)

    Returns:
        str: Path to the generated PDF file
    """
    if session_end is None:
        session_end = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    os.makedirs(REPORTS_DIR, exist_ok=True)
    timestamp_str  = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path    = os.path.join(REPORTS_DIR, f"threat_report_{timestamp_str}.pdf")

    doc    = SimpleDocTemplate(output_path, pagesize=A4,
                               rightMargin=0.75*inch, leftMargin=0.75*inch,
                               topMargin=0.75*inch, bottomMargin=0.75*inch)
    styles = getSampleStyleSheet()
    story  = []

    # ── Title ──────────────────────────────────────────────────────────────
    title_style = ParagraphStyle(
        "Title", parent=styles["Title"],
        fontSize=16, textColor=colors.HexColor("#1a1a2e"),
        spaceAfter=4, alignment=TA_CENTER
    )
    sub_style = ParagraphStyle(
        "Sub", parent=styles["Normal"],
        fontSize=10, textColor=colors.HexColor("#555555"),
        alignment=TA_CENTER, spaceAfter=2
    )

    story.append(Paragraph("AI-Based Ransomware Pre-Attack Prediction System", title_style))
    story.append(Paragraph("Threat Detection Report", title_style))
    story.append(Spacer(1, 4))
    story.append(Paragraph("Paavai Engineering College, Namakkal — Batch 2026", sub_style))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#e94560")))
    story.append(Spacer(1, 12))

    # ── Session Summary ─────────────────────────────────────────────────────
    section_style = ParagraphStyle(
        "Section", parent=styles["Heading2"],
        fontSize=12, textColor=colors.HexColor("#1a1a2e"),
        spaceAfter=6
    )
    normal_style = ParagraphStyle(
        "Normal2", parent=styles["Normal"],
        fontSize=10, spaceAfter=4
    )

    story.append(Paragraph("📋 Session Summary", section_style))

    total      = len(detections)
    critical   = sum(1 for d in detections if d.get("threat_level") == "CRITICAL")
    high       = sum(1 for d in detections if d.get("threat_level") == "HIGH")
    quarantine = sum(1 for d in detections if "Quarantine" in d.get("action_taken", ""))

    summary_data = [
        ["Field", "Value"],
        ["Session Start",        session_start],
        ["Session End",          session_end],
        ["Total Threats Detected", str(total)],
        ["Critical Threats",     str(critical)],
        ["High Threats",         str(high)],
        ["Processes Quarantined",str(quarantine)],
        ["Report Generated",     datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
    ]

    summary_table = Table(summary_data, colWidths=[2.5*inch, 4*inch])
    summary_table.setStyle(TableStyle([
        ("BACKGROUND",  (0, 0), (-1, 0), colors.HexColor("#1a1a2e")),
        ("TEXTCOLOR",   (0, 0), (-1, 0), colors.white),
        ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",    (0, 0), (-1, -1), 9),
        ("BACKGROUND",  (0, 1), (-1, -1), colors.HexColor("#f5f5f5")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.HexColor("#f0f0f0"), colors.white]),
        ("GRID",        (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
        ("PADDING",     (0, 0), (-1, -1), 6),
        ("FONTNAME",    (0, 1), (0, -1), "Helvetica-Bold"),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 16))

    # ── Detection Details ───────────────────────────────────────────────────
    story.append(Paragraph("🚨 Detection Details", section_style))

    if detections:
        detail_headers = ["Process", "PID", "Confidence", "Threat Level", "Time", "Action"]
        detail_data    = [detail_headers]

        for d in detections:
            conf_pct = f"{float(d.get('confidence', 0)) * 100:.1f}%"
            detail_data.append([
                d.get("process_name", "Unknown"),
                str(d.get("pid", "-")),
                conf_pct,
                d.get("threat_level", "-"),
                d.get("timestamp", "-"),
                d.get("action_taken", "Monitored"),
            ])

        detail_table = Table(
            detail_data,
            colWidths=[1.5*inch, 0.6*inch, 0.9*inch, 1.0*inch, 1.5*inch, 1.2*inch]
        )

        # Color-code threat level column
        ts = [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e94560")),
            ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
            ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",   (0, 0), (-1, -1), 8),
            ("GRID",       (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
            ("PADDING",    (0, 0), (-1, -1), 5),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
             [colors.HexColor("#fff0f0"), colors.white]),
        ]
        # Highlight CRITICAL rows in red
        for i, d in enumerate(detections, start=1):
            if d.get("threat_level") == "CRITICAL":
                ts.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#ffe0e0")))
            elif d.get("threat_level") == "HIGH":
                ts.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#fff3e0")))

        detail_table.setStyle(TableStyle(ts))
        story.append(detail_table)
    else:
        story.append(Paragraph("✅ No threats detected during this session.", normal_style))

    story.append(Spacer(1, 16))

    # ── Model Info ──────────────────────────────────────────────────────────
    story.append(Paragraph("🤖 Ensemble Model Configuration", section_style))

    model_data = [
        ["Model",           "Weight", "Role"],
        ["XGBoost",         "30%",    "Primary classifier (gradient boosting)"],
        ["Random Forest",   "25%",    "Ensemble tree-based detection"],
        ["SVM (Calibrated)","15%",    "Support vector classification"],
        ["DNN",             "15%",    "Deep neural network pattern learning"],
        ["LSTM",            "15%",    "Sequential behavioral analysis"],
    ]
    model_table = Table(model_data, colWidths=[1.8*inch, 0.8*inch, 4.1*inch])
    model_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a1a2e")),
        ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
        ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",   (0, 0), (-1, -1), 9),
        ("GRID",       (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
        ("PADDING",    (0, 0), (-1, -1), 5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.HexColor("#f0f0f0"), colors.white]),
    ]))
    story.append(model_table)
    story.append(Spacer(1, 16))

    # ── Footer ──────────────────────────────────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#cccccc")))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "This report was auto-generated by the AI-Based Ransomware Pre-Attack Prediction System. "
        "Student ID: 622122111036 | Supervisor: Mrs. M. Kanagavalli, M.E. | "
        "HOD: Dr. P. Muthusamy, M.E., Ph.D.",
        ParagraphStyle("Footer", parent=styles["Normal"],
                       fontSize=7, textColor=colors.grey, alignment=TA_CENTER)
    ))

    doc.build(story)
    print(f"[PDF REPORT] Generated: {output_path}")
    return output_path


# ── Quick test ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sample_detections = [
        {
            "process_name": "suspicious_encrypt.exe",
            "pid": 4521,
            "confidence": 0.93,
            "threat_level": "CRITICAL",
            "timestamp": "2026-04-03 10:23:11",
            "action_taken": "Quarantine + Network Block"
        },
        {
            "process_name": "unknown_loader.exe",
            "pid": 3812,
            "confidence": 0.74,
            "threat_level": "HIGH",
            "timestamp": "2026-04-03 10:25:44",
            "action_taken": "Monitored"
        },
    ]

    path = generate_threat_report(
        detections=sample_detections,
        session_start="2026-04-03 10:00:00"
    )
    print(f"Report saved at: {path}")