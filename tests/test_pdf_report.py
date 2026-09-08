"""Smoke tests for the PDF Credit Appraisal Memorandum generator."""
from __future__ import annotations

from pathlib import Path

from app.cam import generate_cam
from app.pdf_report import generate_cam_pdf


class TestPDFReportSmoke:
    """PDF generation tests using the session-scoped fixtures from conftest.py."""

    def test_generate_cam_pdf_creates_valid_pdf_file(self, credit_data, model_result, tmp_path):
        sample_row = credit_data["X_test"].iloc[[0]].copy()
        cam = generate_cam(sample_row, model_result)

        output_path = tmp_path / "test_cam_report.pdf"
        result_path = generate_cam_pdf(cam, model_result, output_path=str(output_path))

        assert result_path.exists(), "PDF file was not created"
        assert result_path.stat().st_size > 0, "PDF file is empty"

        with open(result_path, "rb") as f:
            header = f.read(5)
        assert header == b"%PDF-", "File does not start with PDF magic bytes"

    def test_generate_cam_pdf_auto_path_creates_output_dir(self, credit_data, model_result, tmp_path):
        sample_row = credit_data["X_test"].iloc[[0]].copy()
        cam = generate_cam(sample_row, model_result)

        output_path = tmp_path / "subdir" / "test_cam_report.pdf"
        result_path = generate_cam_pdf(cam, model_result, output_path=str(output_path))

        assert result_path.exists()
        assert result_path.parent.exists()
