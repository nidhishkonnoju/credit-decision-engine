"""Smoke tests for the PDF Credit Appraisal Memorandum generator."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from app.cam import generate_cam
from app.modeling import fit_credit_model
from app.pdf_report import generate_cam_pdf
from app.preprocessing import build_preprocessed_data


class PDFReportSmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = build_preprocessed_data()
        cls.model_result = fit_credit_model(cls.data)

    def test_generate_cam_pdf_creates_valid_pdf_file(self):
        sample_row = self.data["X_test"].iloc[[0]].copy()
        cam = generate_cam(sample_row, self.model_result)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_cam_report.pdf"
            result_path = generate_cam_pdf(cam, self.model_result, output_path=str(output_path))

            self.assertTrue(result_path.exists(), "PDF file was not created")
            self.assertGreater(result_path.stat().st_size, 0, "PDF file is empty")

            with open(result_path, "rb") as f:
                header = f.read(5)
            self.assertEqual(header, b"%PDF-", "File does not start with PDF magic bytes")

    def test_generate_cam_pdf_auto_path_creates_output_dir(self):
        sample_row = self.data["X_test"].iloc[[0]].copy()
        cam = generate_cam(sample_row, self.model_result)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "subdir" / "test_cam_report.pdf"
            result_path = generate_cam_pdf(cam, self.model_result, output_path=str(output_path))

            self.assertTrue(result_path.exists())
            self.assertTrue(result_path.parent.exists())


if __name__ == "__main__":
    unittest.main()
