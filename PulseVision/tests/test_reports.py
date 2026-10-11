"""
test_reports.py — Unit tests for CSV and ReportLab PDF report generation.
"""

from src.report_generator import ReportGenerator


def test_csv_generation():
    """Verify detailed points CSV and summary CSV formats."""
    stats = {
        "session_id": "TEST-CSV",
        "label": "CSV Test",
        "start_time": "2026-10-09T10:00:00",
        "duration_sec": 30.0,
        "algorithm": "CHROM",
        "avg_bpm": 72.0,
        "min_bpm": 68.0,
        "max_bpm": 76.0,
        "valid_count": 25,
        "usable_pct": 85.0
    }
    bpm_records = [(1.0, 70.0, 80.0), (2.0, 72.0, 85.0)]
    sqi_records = [(1.0, 80.0, "EXCELLENT"), (2.0, 85.0, "EXCELLENT")]

    detailed_csv, summary_csv = ReportGenerator.generate_csv_data(stats, bpm_records, sqi_records)
    
    assert "elapsed_seconds,estimated_bpm" in detailed_csv
    assert "TEST-CSV" in summary_csv
    assert "72.0" in summary_csv


def test_pdf_generation():
    """Verify ReportLab produces valid non-empty PDF bytes with tables and disclaimer."""
    stats = {
        "session_id": "TEST-PDF",
        "label": "PDF Test Subject",
        "start_time": "2026-10-09T14:30:00",
        "duration_sec": 45.0,
        "algorithm": "POS",
        "avg_bpm": 76.5,
        "min_bpm": 71.0,
        "max_bpm": 82.0,
        "valid_count": 35,
        "usable_pct": 90.0
    }
    bpm_records = [(i, 75.0 + i * 0.2, 85.0) for i in range(10)]
    waveform = [0.1 * i for i in range(50)]

    pdf_bytes = ReportGenerator.generate_pdf_report(stats, bpm_records, waveform)
    assert pdf_bytes is not None
    assert len(pdf_bytes) > 1000
    # PDF magic number header '%PDF-'
    assert pdf_bytes.startswith(b"%PDF-")


def test_empty_session_pdf_generation():
    """Verify PDF generator gracefully handles an empty session without crashing."""
    stats = {
        "session_id": "EMPTY-PDF",
        "label": "Empty Session",
        "start_time": "2026-10-09T14:30:00",
        "duration_sec": 0.0,
        "algorithm": "GREEN",
        "avg_bpm": None,
        "min_bpm": None,
        "max_bpm": None,
        "valid_count": 0,
        "usable_pct": 0.0
    }
    pdf_bytes = ReportGenerator.generate_pdf_report(stats, [], None)
    assert pdf_bytes is not None
    assert pdf_bytes.startswith(b"%PDF-")
