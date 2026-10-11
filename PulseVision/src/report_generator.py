"""
report_generator.py — CSV and PDF Report Generation Engine for PulseVision.
Generates downloadable tabular CSV data and professionally styled ReportLab PDF reports
with executive summary metrics, user demographics, embedded visual trend charts, and non-diagnostic disclaimers.
"""

from io import BytesIO
import os
import tempfile
from typing import Any, Dict, List, Optional, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    HRFlowable,
    Image,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from src.config import get_heart_rate_zone, get_max_heart_rate


class ReportGenerator:
    """Produces CSV dataframes and styled PDF reports for PulseVision sessions."""

    @staticmethod
    def generate_csv_data(
        stats: Dict[str, Any],
        bpm_records: List[Any],
        sqi_records: List[Any]
    ) -> Tuple[str, str]:
        """
        Generates two CSV strings:
        1. detailed_points_csv: time-series of heart rate and SQI estimates
        2. summary_csv: high-level metrics for the session including demographics
        """
        age = stats.get("user_age", 25)
        points = []
        for i, rec in enumerate(bpm_records):
            t = rec[0] if isinstance(rec, (list, tuple)) else rec.get("t", 0.0)
            bpm = rec[1] if isinstance(rec, (list, tuple)) else rec.get("bpm", 0.0)
            sqi = rec[2] if isinstance(rec, (list, tuple)) and len(rec) > 2 else rec.get("sqi", 0.0)
            zone = get_heart_rate_zone(bpm, age)
            points.append({
                "elapsed_seconds": round(t, 2),
                "estimated_bpm": round(bpm, 1),
                "cardiac_zone": zone["zone_name"],
                "signal_quality_pct": round(sqi, 1),
                "algorithm": stats.get("algorithm", "ADAPTIVE"),
                "status": "VALID"
            })

        df_points = pd.DataFrame(points)
        detailed_csv = df_points.to_csv(index=False) if not df_points.empty else "elapsed_seconds,estimated_bpm,cardiac_zone,signal_quality_pct,algorithm,status\n"

        summary_rows = [{
            "session_id": stats.get("session_id", "N/A"),
            "user_name": stats.get("label", "Guest"),
            "user_age": stats.get("user_age", 25),
            "user_gender": stats.get("user_gender", "Prefer not to say"),
            "activity_state": stats.get("activity_state", "Resting"),
            "max_heart_rate_target": stats.get("max_hr", get_max_heart_rate(age)),
            "start_time": stats.get("start_time", ""),
            "duration_seconds": stats.get("duration_sec", 0.0),
            "algorithm": stats.get("algorithm", "ADAPTIVE"),
            "average_bpm": stats.get("avg_bpm", "N/A"),
            "min_bpm": stats.get("min_bpm", "N/A"),
            "max_bpm": stats.get("max_bpm", "N/A"),
            "valid_estimates_count": stats.get("valid_count", 0),
            "signal_usability_pct": stats.get("usable_pct", 0.0)
        }]
        df_summary = pd.DataFrame(summary_rows)
        summary_csv = df_summary.to_csv(index=False)

        return detailed_csv, summary_csv

    @classmethod
    def generate_pdf_report(
        cls,
        stats: Dict[str, Any],
        bpm_records: List[Any],
        pulse_waveform: Optional[Any] = None
    ) -> bytes:
        """
        Generates a publication-grade PDF analytics report using ReportLab.
        Returns PDF bytes.
        """
        buffer = BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            rightMargin=36,
            leftMargin=36,
            topMargin=36,
            bottomMargin=36
        )

        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            "DocTitle",
            parent=styles["Heading1"],
            fontSize=22,
            leading=26,
            textColor=colors.HexColor("#0B132B"),
            spaceAfter=3
        )
        subtitle_style = ParagraphStyle(
            "DocSubtitle",
            parent=styles["Normal"],
            fontSize=10.5,
            leading=13,
            textColor=colors.HexColor("#3A506B"),
            spaceAfter=12
        )
        section_style = ParagraphStyle(
            "SectionTitle",
            parent=styles["Heading2"],
            fontSize=12,
            leading=15,
            textColor=colors.HexColor("#1C2541"),
            spaceBefore=10,
            spaceAfter=5
        )
        body_style = ParagraphStyle(
            "Body",
            parent=styles["Normal"],
            fontSize=9,
            leading=12,
            textColor=colors.HexColor("#1E293B")
        )
        disclaimer_style = ParagraphStyle(
            "Disclaimer",
            parent=styles["Normal"],
            fontSize=7.5,
            leading=10.5,
            textColor=colors.HexColor("#64748B")
        )

        elements = []

        # 1. Header Banner
        elements.append(Paragraph("PulseVision — Physiological Analytics Report", title_style))
        elements.append(Paragraph("Contactless Remote Photoplethysmography (rPPG) System", subtitle_style))
        elements.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#00F2FE"), spaceAfter=10))

        # 2. User Demographics & Session Metadata Grid
        age = stats.get("user_age", 25)
        max_hr = get_max_heart_rate(age)
        meta_data = [
            [
                Paragraph("<b>Subject Name:</b> " + str(stats.get("label", "Guest")), body_style),
                Paragraph("<b>Session ID:</b> " + str(stats.get("session_id", "N/A")), body_style)
            ],
            [
                Paragraph("<b>Age:</b> " + f"{age} years", body_style),
                Paragraph("<b>Gender:</b> " + str(stats.get("user_gender", "Unspecified")), body_style)
            ],
            [
                Paragraph("<b>Activity State:</b> " + str(stats.get("activity_state", "Resting")), body_style),
                Paragraph("<b>Max HR Target (220-Age):</b> " + f"{max_hr} BPM", body_style)
            ],
            [
                Paragraph("<b>Acquisition Date:</b> " + str(stats.get("start_time", "N/A"))[:19].replace("T", " "), body_style),
                Paragraph("<b>Monitored Duration:</b> " + f"{stats.get('duration_sec', 0.0):.1f}s", body_style)
            ]
        ]
        meta_table = Table(meta_data, colWidths=[265, 265])
        meta_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
            ("PADDING", (0, 0), (-1, -1), 4.5),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ]))
        elements.append(meta_table)
        elements.append(Spacer(1, 10))

        # 3. Executive Metrics Table
        elements.append(Paragraph("Physiological Summary Metrics & Target Zone", section_style))
        avg_str = f"{stats['avg_bpm']:.1f} BPM" if stats.get("avg_bpm") is not None else "N/A"
        min_str = f"{stats['min_bpm']:.1f} BPM" if stats.get("min_bpm") is not None else "N/A"
        max_str = f"{stats['max_bpm']:.1f} BPM" if stats.get("max_bpm") is not None else "N/A"
        count_str = str(stats.get("valid_count", 0))

        zone = get_heart_rate_zone(stats.get("avg_bpm"), age)
        zone_str = f"{zone['zone_name']} ({zone['intensity_pct']}%)" if stats.get("avg_bpm") else "N/A"

        stat_data = [
            ["Metric", "Observed Value", "Clinical & Physiological Context"],
            ["Average Heart Rate", avg_str, f"Mean cardiac frequency. Zone: {zone_str}"],
            ["Observed Min - Max", f"{min_str} - {max_str}", "Full span of detected physiological rates"],
            ["Signal Usability Ratio", f"{stats.get('usable_pct', 0.0):.1f}%", "Fraction of session with high-confidence tracking"],
            ["Valid Estimate Count", count_str, "Number of accepted spectral FFT estimates"]
        ]
        stat_table = Table(stat_data, colWidths=[140, 110, 280])
        stat_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1C2541")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 8.5),
            ("PADDING", (0, 0), (-1, -1), 5),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
            ("ALIGN", (1, 0), (1, -1), "CENTER"),
        ]))
        elements.append(stat_table)
        elements.append(Spacer(1, 10))

        # 4. Render and Embed Charts using Matplotlib
        chart_img_path = cls._render_report_chart_image(bpm_records, pulse_waveform)
        if chart_img_path and os.path.exists(chart_img_path):
            elements.append(Paragraph("Heart Rate Trend & Optical Pulse Waveform", section_style))
            elements.append(Image(chart_img_path, width=7.1 * inch, height=3.0 * inch))
            elements.append(Spacer(1, 10))

        # 5. Non-Diagnostic Medical Disclaimer
        elements.append(HRFlowable(width="100%", thickness=0.8, color=colors.HexColor("#CBD5E1"), spaceAfter=6))
        disclaimer_text = (
            "<b>IMPORTANT REGULATORY & MEDICAL DISCLAIMER:</b><br/>"
            "PulseVision is an experimental computer-vision wellness and research tool. It is <b>NOT a medical diagnostic device</b> "
            "and is not certified by the FDA, CE, or health authorities. Measurements are derived from optical webcam reflectance "
            "and may be affected by lighting, facial motion, and camera compression. This document does not constitute clinical advice. "
            "Never use PulseVision readings to diagnose, treat, or monitor cardiovascular conditions. Consult a licensed physician for clinical evaluation."
        )
        elements.append(Paragraph(disclaimer_text, disclaimer_style))

        # Build Document
        doc.build(elements)
        pdf_bytes = buffer.getvalue()
        buffer.close()

        if chart_img_path and os.path.exists(chart_img_path):
            try:
                os.remove(chart_img_path)
            except Exception:
                pass

        return pdf_bytes

    @staticmethod
    def _render_report_chart_image(bpm_records: List[Any], pulse_waveform: Optional[Any]) -> Optional[str]:
        """Renders twin matplotlib charts to temporary image file for PDF embedding."""
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8), dpi=150)
            fig.patch.set_facecolor("#FFFFFF")

            # Chart 1: BPM Trend
            if bpm_records and len(bpm_records) > 0:
                times = [rec[0] if isinstance(rec, (list, tuple)) else rec["t"] for rec in bpm_records]
                bpms = [rec[1] if isinstance(rec, (list, tuple)) else rec["bpm"] for rec in bpm_records]
                ax1.plot(times, bpms, color="#00C49F", marker="o", markersize=3, linewidth=2, label="Estimated BPM")
                ax1.axhline(np.mean(bpms), color="#0088FE", linestyle="--", linewidth=1.2,
                            label=f"Avg ({np.mean(bpms):.1f} BPM)")
                ax1.set_title("Heart Rate Trend Over Time", fontsize=9.5, fontweight="bold", color="#1C2541")
                ax1.set_xlabel("Elapsed Time (s)", fontsize=8, color="#3A506B")
                ax1.set_ylabel("Heart Rate (BPM)", fontsize=8, color="#3A506B")
                ax1.grid(True, linestyle=":", alpha=0.6)
                ax1.legend(fontsize=7, loc="upper right")
            else:
                ax1.text(0.5, 0.5, "Insufficient Valid BPM Data", ha="center", va="center", color="#94A3B8")
                ax1.set_title("Heart Rate Trend", fontsize=9.5)

            # Chart 2: Filtered Pulse Waveform
            if pulse_waveform is not None and len(pulse_waveform) > 10:
                n = len(pulse_waveform)
                t_wave = np.linspace(0, n / 30.0, n)
                ax2.plot(t_wave, pulse_waveform, color="#8884D8", linewidth=1.5, label="Filtered rPPG")
                ax2.set_title("Filtered Pulse Waveform (rPPG)", fontsize=9.5, fontweight="bold", color="#1C2541")
                ax2.set_xlabel("Window Time (s)", fontsize=8, color="#3A506B")
                ax2.set_ylabel("Normalized Amplitude", fontsize=8, color="#3A506B")
                ax2.grid(True, linestyle=":", alpha=0.6)
                ax2.legend(fontsize=7, loc="upper right")
            else:
                ax2.text(0.5, 0.5, "Pulse Waveform Unavailable", ha="center", va="center", color="#94A3B8")
                ax2.set_title("Filtered Pulse Waveform", fontsize=9.5)

            plt.tight_layout()
            temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            temp_path = temp_file.name
            temp_file.close()
            plt.savefig(temp_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            return temp_path
        except Exception as e:
            print(f"[ReportGenerator] Chart render error: {e}")
            return None
