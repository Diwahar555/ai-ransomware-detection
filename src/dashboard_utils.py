"""
dashboard_utils.py
Improvement 5 — Dashboard Upgrades:
1. Email alerts when threat detected
2. Export incident report as PDF
3. Threat activity heatmap by hour
4. Historical comparison chart
"""

import os
import smtplib
import datetime
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import warnings
warnings.filterwarnings("ignore")

LOGS_DIR    = "logs"
REPORTS_DIR = "reports"
os.makedirs(LOGS_DIR,    exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

EMAIL_CONFIG_PATH = "logs/email_config.json"


def calibrate_model_probabilities(probs: dict, raw_confidence: float, target_confidence: float) -> dict:
    """
    Scale per-model probabilities down for dashboard display when live context
    has already reduced the ensemble confidence.

    This keeps the model chart aligned with the operator-facing risk view
    without changing the underlying saved-model outputs.
    """
    if not probs:
        return {}

    # Filter out non-numeric values (metadata like lstm_status)
    numeric_probs = {k: v for k, v in probs.items() if isinstance(v, (int, float, np.number))}
    
    clipped = {
        name: float(np.clip(value, 0.0, 1.0))
        for name, value in numeric_probs.items()
    }

    raw_conf = float(max(raw_confidence, 0.0))
    target_conf = float(np.clip(target_confidence, 0.0, 1.0))

    if raw_conf <= 0 or target_conf >= raw_conf:
        return clipped

    scale = target_conf / raw_conf
    return {
        name: float(np.clip(value * scale, 0.0, 1.0))
        for name, value in clipped.items()
    }


# ── 1. Email Configuration ────────────────────────────────────────────────────
def save_email_config(sender: str, recipient: str,
                      smtp_server: str = "smtp.gmail.com", port: int = 587):
    config = {
        "sender":      sender,
        "recipient":   recipient,
        "smtp_server": smtp_server,
        "port":        port,
    }
    with open(EMAIL_CONFIG_PATH, "w") as f:
        json.dump(config, f)
    print(f"  Email config saved -> {EMAIL_CONFIG_PATH}")


def load_email_config() -> dict:
    if not os.path.exists(EMAIL_CONFIG_PATH):
        return None
    with open(EMAIL_CONFIG_PATH, "r") as f:
        config = json.load(f)
    if "password" in config:
        config.pop("password", None)
        with open(EMAIL_CONFIG_PATH, "w") as f:
            json.dump(config, f)
    return config


# ── 2. Send Email Alert ───────────────────────────────────────────────────────
def send_email_alert(confidence: float, votes: int, snapshot: dict,
                     top_features: list = None, actions_taken: list = None,
                     smtp_password: str = None) -> dict:
    """Send email alert when threat is detected."""
    config = load_email_config()
    if not config:
        return {"success": False, "message": "Email not configured. Set up in dashboard settings."}
    if not smtp_password:
        return {"success": False, "message": "Email password is required for this session."}

    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"⚠️ RANSOMWARE THREAT DETECTED — {confidence*100:.1f}% Confidence"
        msg["From"]    = config["sender"]
        msg["To"]      = config["recipient"]

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Build feature table
        feat_rows = ""
        for feat, val in [
            ("CPU Usage",          f"{snapshot.get('cpu_percent',0):.1f}%"),
            ("Memory Usage",       f"{snapshot.get('memory_percent',0):.1f}%"),
            ("Active Processes",   str(snapshot.get("process_count",0))),
            ("Network Connections",str(snapshot.get("active_connections",0))),
            ("Disk Write Rate",    f"{snapshot.get('disk_write_rate',0)/1e6:.2f} MB/s"),
            ("Files Modified",     str(snapshot.get("file_modified_count",0))),
            ("Files Deleted",      str(snapshot.get("file_deleted_count",0))),
            ("New Processes",      str(snapshot.get("new_process_count",0))),
        ]:
            feat_rows += f"<tr><td style='padding:6px 12px;border-bottom:1px solid #eee'>{feat}</td><td style='padding:6px 12px;border-bottom:1px solid #eee;font-weight:bold'>{val}</td></tr>"

        top_feat_text = ""
        if top_features:
            top_feat_text = "<h3>Top contributing features:</h3><ul>"
            for f in top_features[:3]:
                top_feat_text += f"<li>{f['label']}: {f['raw_value']:.1f}</li>"
            top_feat_text += "</ul>"

        if actions_taken:
            action_items = "".join(f"<li>{action}</li>" for action in actions_taken)
        else:
            action_items = "<li>No automated containment actions were confirmed.</li>"

        html = f"""
<html><body style='font-family:Arial,sans-serif;max-width:600px;margin:0 auto'>
<div style='background:#c53030;color:white;padding:20px;border-radius:8px 8px 0 0'>
  <h2 style='margin:0'>⚠️ Ransomware Threat Detected</h2>
  <p style='margin:8px 0 0'>AI-Based Ransomware Pre-Attack Detection System</p>
</div>
<div style='background:#fff8f8;padding:20px;border:1px solid #fed7d7'>
  <table style='width:100%'>
    <tr><td><strong>Time:</strong></td><td>{timestamp}</td></tr>
    <tr><td><strong>Confidence:</strong></td><td style='color:#c53030;font-size:18px'>{confidence*100:.1f}%</td></tr>
    <tr><td><strong>Models flagged:</strong></td><td>{votes}/5</td></tr>
  </table>
</div>
<div style='padding:20px'>
  <h3>System metrics at time of detection:</h3>
  <table style='width:100%;border-collapse:collapse'>{feat_rows}</table>
  {top_feat_text}
  <h3>Immediate actions taken:</h3>
  <ul>{action_items}</ul>
  <h3>Recommended next steps:</h3>
  <ol>
    <li>Disconnect from internet immediately</li>
    <li>Check dashboard at localhost:8501</li>
    <li>Run full antivirus scan</li>
    <li>Restore from backup if files encrypted</li>
  </ol>
</div>
<div style='background:#2d3748;color:#a0aec0;padding:12px;text-align:center;font-size:12px;border-radius:0 0 8px 8px'>
  AI-Based Ransomware Pre-Attack Prediction System | Final Year Project 2026
</div>
</body></html>
"""
        msg.attach(MIMEText(html, "html"))

        with smtplib.SMTP(config["smtp_server"], config["port"]) as server:
            server.starttls()
            server.login(config["sender"], smtp_password)
            server.sendmail(config["sender"], config["recipient"], msg.as_string())

        print(f"  Email alert sent to {config['recipient']}")
        return {"success": True, "message": f"Alert sent to {config['recipient']}"}

    except Exception as e:
        return {"success": False, "message": f"Email failed: {str(e)}"}


# ── 3. Export PDF Report ──────────────────────────────────────────────────────
def export_pdf_report(history: list, alerts: list, snapshot: dict,
                       confidence: float) -> str:
    """Generate a PDF incident report."""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.lib import colors
        from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                         Table, TableStyle, HRFlowable)
        from reportlab.lib.enums import TA_CENTER, TA_LEFT

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path      = os.path.join(REPORTS_DIR, f"incident_report_{timestamp}.pdf")

        doc    = SimpleDocTemplate(path, pagesize=A4,
                                   topMargin=2*cm, bottomMargin=2*cm,
                                   leftMargin=2*cm, rightMargin=2*cm)
        styles = getSampleStyleSheet()
        story  = []

        # Title
        title_style = ParagraphStyle("title", parent=styles["Title"],
                                      fontSize=18, textColor=colors.HexColor("#1F3864"),
                                      spaceAfter=12)
        story.append(Paragraph("AI-Based Ransomware Pre-Attack Detection System", title_style))
        story.append(Paragraph("Incident Detection Report", styles["Heading2"]))
        story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#2E74B5")))
        story.append(Spacer(1, 0.3*cm))

        # Summary
        story.append(Paragraph("Incident Summary", styles["Heading2"]))
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        summary_data = [
            ["Field", "Value"],
            ["Report Generated", now],
            ["Threat Confidence", f"{confidence*100:.1f}%"],
            ["Total Checks", str(len(history))],
            ["Total Alerts", str(len(alerts))],
            ["CPU at Detection", f"{snapshot.get('cpu_percent',0):.1f}%"],
            ["Memory at Detection", f"{snapshot.get('memory_percent',0):.1f}%"],
            ["Disk Write Rate", f"{snapshot.get('disk_write_rate',0)/1e6:.2f} MB/s"],
            ["Network Connections", str(snapshot.get("active_connections",0))],
        ]
        t = Table(summary_data, colWidths=[7*cm, 10*cm])
        t.setStyle(TableStyle([
            ("BACKGROUND",   (0,0), (-1,0),  colors.HexColor("#1F3864")),
            ("TEXTCOLOR",    (0,0), (-1,0),  colors.white),
            ("FONTNAME",     (0,0), (-1,0),  "Helvetica-Bold"),
            ("FONTSIZE",     (0,0), (-1,-1), 10),
            ("BACKGROUND",   (0,1), (-1,-1), colors.HexColor("#EBF3FB")),
            ("ROWBACKGROUNDS",(0,1),(-1,-1), [colors.HexColor("#EBF3FB"),
                                               colors.white]),
            ("GRID",         (0,0), (-1,-1), 0.5, colors.HexColor("#CCCCCC")),
            ("PADDING",      (0,0), (-1,-1), 6),
        ]))
        story.append(t)
        story.append(Spacer(1, 0.5*cm))

        # Alert history
        if alerts:
            story.append(Paragraph("Threat Alert History", styles["Heading2"]))
            alert_data = [["Time","Confidence","Stage","Network","Process","Action"]]
            for a in alerts[-10:]:
                alert_data.append([
                    str(a.get("Time","")),
                    str(a.get("Confidence","")),
                    str(a.get("Stage","")),
                    str(a.get("Network","")),
                    str(a.get("Process",""))[:20],
                    str(a.get("Action","")),
                ])
            at = Table(alert_data, colWidths=[2.5*cm,2.5*cm,2*cm,2.5*cm,4*cm,3*cm])
            at.setStyle(TableStyle([
                ("BACKGROUND", (0,0),(-1,0),  colors.HexColor("#C53030")),
                ("TEXTCOLOR",  (0,0),(-1,0),  colors.white),
                ("FONTNAME",   (0,0),(-1,0),  "Helvetica-Bold"),
                ("FONTSIZE",   (0,0),(-1,-1), 8),
                ("GRID",       (0,0),(-1,-1), 0.5, colors.HexColor("#CCCCCC")),
                ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.HexColor("#FFF5F5"),colors.white]),
                ("PADDING",    (0,0),(-1,-1), 4),
            ]))
            story.append(at)
            story.append(Spacer(1, 0.5*cm))

        # Prevention steps
        doc.build(story)
        print(f"  PDF report saved -> {path}")
        return path

    except ImportError:
        print("  reportlab not installed. Run: pip install reportlab")
        return None
    except Exception as e:
        print(f"  PDF generation error: {e}")
        return None


# ── 4. Threat Activity Heatmap ────────────────────────────────────────────────
def generate_heatmap(history: list) -> plt.Figure:
    """
    Generate a heatmap showing threat activity by hour of day and day of week.
    """
    if len(history) < 5:
        return None

    # Build hour x confidence matrix
    hour_data = {h: [] for h in range(24)}
    for entry in history:
        try:
            t    = entry.get("time", "00:00:00")
            hour = int(t.split(":")[0])
            conf = entry.get("adj_conf", entry.get("raw_conf", 0))
            hour_data[hour].append(conf)
        except Exception:
            pass

    hours    = list(range(24))
    avg_conf = [np.mean(hour_data[h]) if hour_data[h] else 0 for h in hours]
    threat_count = [sum(1 for v in hour_data[h] if v >= 50) for h in hours]

    fig, axes = plt.subplots(2, 1, figsize=(14, 6))
    fig.patch.set_facecolor("#ffffff")

    # Plot 1: Average confidence by hour
    colors_bar = ["#ef4444" if v >= 50 else "#10b981" if v < 30 else "#f59e0b"
                  for v in avg_conf]
    axes[0].bar(hours, avg_conf, color=colors_bar, edgecolor="#f1f5f9", width=0.7)
    axes[0].axhline(y=50, color="#f59e0b", linestyle="--", linewidth=1.5, label="Threshold (50%)")
    axes[0].set_xlabel("Hour of day", color="#64748b", fontweight="bold")
    axes[0].set_ylabel("Avg threat confidence %", color="#64748b", fontweight="bold")
    axes[0].set_title("Average threat confidence by hour", color="#0f172a", pad=8, fontweight="bold")
    axes[0].set_facecolor("#ffffff")
    axes[0].tick_params(colors="#64748b")
    axes[0].set_xticks(hours)
    axes[0].set_xticklabels([f"{h:02d}:00" for h in hours], rotation=45, fontsize=8)
    axes[0].legend(facecolor="#ffffff", labelcolor="#0f172a")
    for spine in axes[0].spines.values():
        spine.set_color("#cbd5e1")

    # Plot 2: Threat count by hour
    threat_colors = ["#ef4444" if c > 0 else "#e2e8f0" for c in threat_count]
    axes[1].bar(hours, threat_count, color=threat_colors, edgecolor="#f1f5f9", width=0.7)
    axes[1].set_xlabel("Hour of day", color="#64748b", fontweight="bold")
    axes[1].set_ylabel("Number of threats", color="#64748b", fontweight="bold")
    axes[1].set_title("Threat detections by hour", color="#0f172a", pad=8, fontweight="bold")
    axes[1].set_facecolor("#ffffff")
    axes[1].tick_params(colors="#64748b")
    axes[1].set_xticks(hours)
    axes[1].set_xticklabels([f"{h:02d}:00" for h in hours], rotation=45, fontsize=8)
    for spine in axes[1].spines.values():
        spine.set_color("#cbd5e1")

    plt.tight_layout()
    return fig


# ── 5. Historical comparison ──────────────────────────────────────────────────
def get_session_stats(history: list) -> dict:
    """Compute session statistics for comparison."""
    if not history:
        return {}
    confs = [h.get("adj_conf", h.get("raw_conf", 0)) for h in history]
    cpus  = [h.get("cpu", 0) for h in history]
    return {
        "checks":        len(history),
        "avg_confidence":round(np.mean(confs), 2),
        "max_confidence":round(np.max(confs), 2),
        "min_confidence":round(np.min(confs), 2),
        "avg_cpu":       round(np.mean(cpus), 2),
        "max_cpu":       round(np.max(cpus), 2),
        "threats":       sum(1 for h in history if h.get("is_threat")),
        "duration_min":  round(len(history) * 6 / 60, 1),
    }
