"""
Unit tests for utils.py functions.
Tests text splitting and file loading with mock data.
"""

import unittest
from unittest.mock import patch, mock_open, MagicMock
from pathlib import Path
import sys
import os

# Add parent directory to path to import utils
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils import split_text, load_text_from_file


class TestSplitText(unittest.TestCase):
    """Test cases for the split_text function."""

    def test_basic_splitting(self):
        """Test basic text splitting without overlap."""
        text = "A" * 100
        chunks = split_text(text, chunk_size=25, overlap=0)
        self.assertEqual(len(chunks), 4)
        self.assertEqual(len(chunks[0]), 25)

    def test_splitting_with_overlap(self):
        """Test text splitting with overlap."""
        text = "ABCDEFGHIJ" * 10  # 100 characters
        chunks = split_text(text, chunk_size=30, overlap=10)

        # Should have overlap between chunks
        self.assertTrue(len(chunks) > 1)
        # Last 10 chars of first chunk should appear in second chunk
        if len(chunks) > 1:
            self.assertTrue(chunks[0][-10:] in chunks[1])

    def test_empty_text(self):
        """Test splitting empty text."""
        chunks = split_text("", chunk_size=100, overlap=10)
        self.assertEqual(chunks, [])

    def test_text_smaller_than_chunk_size(self):
        """Test text smaller than chunk size."""
        text = "Short text"
        chunks = split_text(text, chunk_size=100, overlap=10)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0], text)

    def test_overlap_larger_than_chunk_size(self):
        """Test that overlap is adjusted if too large."""
        text = "A" * 200
        # This should automatically adjust overlap
        chunks = split_text(text, chunk_size=50, overlap=60)
        # Should still work without crashing
        self.assertTrue(len(chunks) > 0)

    def test_whitespace_handling(self):
        """Test that empty chunks are filtered out."""
        text = "   \n\n   "
        chunks = split_text(text, chunk_size=10, overlap=2)
        # Should not include empty/whitespace-only chunks
        self.assertTrue(all(chunk.strip() for chunk in chunks) or len(chunks) == 0)


class TestLoadTextFromFile(unittest.TestCase):
    """Test cases for file loading functions."""

    @patch("utils.Path")
    def test_txt_file_loading(self, mock_path):
        """Test loading a plain text file."""
        mock_file_content = "This is a test file."

        with patch("builtins.open", mock_open(read_data=mock_file_content)):
            # Mock the path
            mock_path_instance = MagicMock()
            mock_path_instance.suffix = ".txt"
            mock_path.return_value = mock_path_instance

            result = load_text_from_file("test.txt")
            self.assertEqual(result, mock_file_content)

    @patch("utils.pdfplumber.open")
    @patch("utils.Path")
    def test_pdf_file_loading(self, mock_path, mock_pdf_open):
        """Test loading a PDF file."""
        # Mock PDF structure
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "PDF content"

        mock_pdf = MagicMock()
        mock_pdf.pages = [mock_page]
        mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
        mock_pdf.__exit__ = MagicMock(return_value=False)

        mock_pdf_open.return_value = mock_pdf

        # Mock the path
        mock_path_instance = MagicMock()
        mock_path_instance.suffix = ".pdf"
        mock_path.return_value = mock_path_instance

        result = load_text_from_file("test.pdf")
        self.assertEqual(result, "PDF content")

    @patch("utils.Document")
    @patch("utils.Path")
    def test_docx_file_loading(self, mock_path, mock_document):
        """Test loading a DOCX file."""
        # Mock DOCX structure
        mock_para1 = MagicMock()
        mock_para1.text = "First paragraph"
        mock_para2 = MagicMock()
        mock_para2.text = "Second paragraph"

        mock_doc = MagicMock()
        mock_doc.paragraphs = [mock_para1, mock_para2]
        mock_document.return_value = mock_doc

        # Mock the path
        mock_path_instance = MagicMock()
        mock_path_instance.suffix = ".docx"
        mock_path.return_value = mock_path_instance

        result = load_text_from_file("test.docx")
        self.assertEqual(result, "First paragraph\nSecond paragraph")

    @patch("utils.Path")
    def test_unsupported_file_type(self, mock_path):
        """Test that unsupported file types raise ValueError."""
        mock_path_instance = MagicMock()
        mock_path_instance.suffix = ".xyz"
        mock_path.return_value = mock_path_instance

        with self.assertRaises(ValueError) as context:
            load_text_from_file("test.xyz")

        self.assertIn("Unsupported file type", str(context.exception))


class TestTextChunkingEdgeCases(unittest.TestCase):
    """Test edge cases for text chunking."""

    def test_exact_chunk_boundary(self):
        """Test text that's exactly multiple of chunk size."""
        text = "A" * 200
        chunks = split_text(text, chunk_size=50, overlap=0)
        self.assertEqual(len(chunks), 4)
        self.assertEqual(sum(len(c) for c in chunks), 200)

    def test_unicode_text(self):
        """Test splitting text with unicode characters."""
        text = "Hello 世界 " * 50
        chunks = split_text(text, chunk_size=50, overlap=10)
        self.assertTrue(len(chunks) > 0)
        # Verify all chunks are valid strings
        for chunk in chunks:
            self.assertIsInstance(chunk, str)

    def test_newline_preservation(self):
        """Test that newlines are preserved in chunks."""
        text = "Line 1\nLine 2\nLine 3\n" * 10
        chunks = split_text(text, chunk_size=50, overlap=10)
        # At least some chunks should contain newlines
        self.assertTrue(any("\n" in chunk for chunk in chunks))


if __name__ == "__main__":
    unittest.main()
