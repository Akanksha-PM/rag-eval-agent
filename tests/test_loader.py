"""Tests for src.ingestion.loader."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.ingestion import loader


class LoadFromFileTestCase(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmpdir.name)

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_txt_file(self):
        file_path = self.tmp_path / "sample.txt"
        file_path.write_text("Hello from a plain text file.")
        self.assertEqual(
            loader.load_from_file(str(file_path)), "Hello from a plain text file."
        )

    def test_md_file(self):
        file_path = self.tmp_path / "sample.md"
        file_path.write_text("# Title\n\nSome **markdown** content.")
        text = loader.load_from_file(str(file_path))
        self.assertIn("# Title", text)
        self.assertIn("Some **markdown** content.", text)

    def test_html_file_extracts_visible_text_only(self):
        file_path = self.tmp_path / "sample.html"
        file_path.write_text(
            "<html><head><style>.a{}</style><script>var x=1;</script></head>"
            "<body><nav>Nav link</nav><p>Real content.</p></body></html>"
        )
        text = loader.load_from_file(str(file_path))
        self.assertIn("Real content.", text)
        self.assertNotIn("var x=1", text)
        self.assertNotIn("Nav link", text)

    def test_unsupported_extension_raises_value_error(self):
        file_path = self.tmp_path / "sample.exe"
        file_path.write_text("binary-ish content")
        with self.assertRaises(ValueError):
            loader.load_from_file(str(file_path))


class LoadFromUrlTestCase(unittest.TestCase):
    def test_strips_script_and_nav_tags(self):
        html = (
            "<html><head><script>evil()</script></head>"
            "<body><nav>Site nav</nav><header>Header</header>"
            "<p>Actual documentation content.</p>"
            "<footer>Footer</footer></body></html>"
        )
        mock_response = MagicMock()
        mock_response.text = html
        mock_response.raise_for_status = MagicMock()

        with patch(
            "src.ingestion.loader.requests.get", return_value=mock_response
        ) as mock_get:
            text = loader.load_from_url("https://example.com/docs")

        mock_get.assert_called_once_with("https://example.com/docs", timeout=10)
        self.assertIn("Actual documentation content.", text)
        self.assertNotIn("evil()", text)
        self.assertNotIn("Site nav", text)
        self.assertNotIn("Header", text)
        self.assertNotIn("Footer", text)

    def test_raises_on_http_error(self):
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = Exception("boom")

        with patch("src.ingestion.loader.requests.get", return_value=mock_response):
            with self.assertRaises(Exception):
                loader.load_from_url("https://example.com/docs")


if __name__ == "__main__":
    unittest.main()
