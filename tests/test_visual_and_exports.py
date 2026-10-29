"""
Tests for Visual Builder (Pillow, matplotlib) and Export Manager (PDF, DOCX, MD, TXT, CSV, PNG).
"""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from PIL import Image

from src.visual_builder import (
    generate_poster_image,
    generate_brochure_image,
    generate_chart_image,
    parse_sections_from_markdown,
    parse_panels_from_markdown,
    parse_chart_data,
)
from src.export_manager import (
    export_to_pdf,
    export_to_docx,
    export_to_markdown,
    export_to_txt,
    export_to_csv,
    export_image_to_png,
)


class TestVisualAndExports(unittest.TestCase):

    def test_parse_sections_from_markdown(self):
        sample_md = """
# Main Title
## Background
This is the problem description.

## Methodology
This is the methodology section.
"""
        sections = parse_sections_from_markdown(sample_md)
        self.assertGreaterEqual(len(sections), 2)
        headers = [s["header"] for s in sections]
        self.assertIn("Background", headers)
        self.assertIn("Methodology", headers)

    def test_parse_panels_from_markdown(self):
        brochure_md = """
### Panel 1: Cover
Welcome to the studio.
### Panel 2: Features
Key features.
### Panel 3: Contact
Get in touch.
"""
        panels = parse_panels_from_markdown(brochure_md)
        self.assertEqual(len(panels), 3)

    def test_parse_chart_data(self):
        chart_md = """
# Data Analysis
```json
{
  "chart_type": "bar",
  "labels": ["Alpha", "Beta", "Gamma"],
  "values": [12, 24, 36]
}
```
"""
        chart_dict = parse_chart_data(chart_md)
        self.assertIsNotNone(chart_dict)
        self.assertEqual(chart_dict["labels"], ["Alpha", "Beta", "Gamma"])
        self.assertEqual(chart_dict["values"], [12, 24, 36])

    def test_generate_poster_image(self):
        img = generate_poster_image(
            title="Academic Research Poster",
            subtitle="Testing CPU Rendering",
            palette_name="modern_blue",
        )
        self.assertIsInstance(img, Image.Image)
        self.assertEqual(img.size, (1400, 1900))

    def test_generate_brochure_image(self):
        img = generate_brochure_image(
            title="Tri-Fold Promotional Brochure",
            palette_name="emerald",
        )
        self.assertIsInstance(img, Image.Image)
        self.assertEqual(img.size, (1800, 1200))

    def test_generate_chart_image(self):
        img = generate_chart_image(
            chart_type="bar",
            labels=["Unit 1", "Unit 2", "Unit 3"],
            values=[15.5, 30.2, 45.8],
            title="Performance Metrics",
        )
        self.assertIsInstance(img, Image.Image)
        self.assertGreater(img.size[0], 500)

    def test_export_to_pdf(self):
        md = "# Sample Title\n## Section 1\nThis is paragraph content.\n- Bullet point 1\n- Bullet point 2\n\n| Item | Val |\n|---|---|\n| A | 10 |"
        pdf_bytes = export_to_pdf("Sample Title", md)
        self.assertTrue(pdf_bytes.startswith(b"%PDF"))
        self.assertGreater(len(pdf_bytes), 1000)

    def test_export_to_docx(self):
        md = "# Sample Word Doc\n## Overview\nThis is a test paragraph.\n- Item A\n- Item B\n\n| Param | Spec |\n|---|---|\n| Pin | 14 |"
        docx_bytes = export_to_docx("Sample Word Doc", md)
        # DOCX is a zip file, starts with 'PK'
        self.assertTrue(docx_bytes.startswith(b"PK"))
        self.assertGreater(len(docx_bytes), 2000)

    def test_export_to_markdown_and_txt(self):
        md_res = export_to_markdown("My Report", "## Content\nSome text.")
        self.assertIn("---", md_res)
        self.assertIn("My Report", md_res)

        txt_res = export_to_txt("My Report", "## Content\nSome text.")
        self.assertIn("MY REPORT", txt_res)
        self.assertIn("Some text.", txt_res)

    def test_export_to_csv(self):
        md_table = "| Name | Value | Category |\n|---|---|---|\n| Voltage | 5V | Hardware |\n| Flash | 32KB | Memory |"
        csv_str = export_to_csv(md_table)
        self.assertIn("Voltage,5V,Hardware", csv_str)
        self.assertIn("Flash,32KB,Memory", csv_str)

    def test_export_image_to_png(self):
        img = Image.new("RGB", (100, 100), color="blue")
        png_bytes = export_image_to_png(img)
        # PNG signature: \x89PNG\r\n\x1a\n
        self.assertTrue(png_bytes.startswith(b"\x89PNG"))


if __name__ == "__main__":
    unittest.main()

