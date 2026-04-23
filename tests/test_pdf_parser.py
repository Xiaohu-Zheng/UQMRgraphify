"""Tests for PDF parsing and academic paper extraction."""
import json
import pytest
from pathlib import Path

from graphify.pdf_parser import (
    parse_paper_from_markdown,
    _extract_arxiv_id,
    _extract_doi,
    _parse_metadata_from_text,
    get_pdf_converter_info,
    PaperMetadata,
    ParsedPaper,
)


class TestArxivIdExtraction:
    """Test arXiv ID extraction from text."""

    def test_extract_new_format(self):
        assert _extract_arxiv_id("arXiv:2301.12345") == "2301.12345"
        assert _extract_arxiv_id("arxiv 2301.12345") == "2301.12345"
        assert _extract_arxiv_id("See arXiv:1706.03762 for details") == "1706.03762"

    def test_extract_old_format(self):
        assert _extract_arxiv_id("arXiv:cs.LG/0701001") == "cs.LG/0701001"
        assert _extract_arxiv_id("arxiv cs.AI/9901001") == "cs.AI/9901001"

    def test_no_arxiv_id(self):
        assert _extract_arxiv_id("This is a regular paper") is None
        assert _extract_arxiv_id("") is None


class TestDoiExtraction:
    """Test DOI extraction from text."""

    def test_extract_doi(self):
        assert _extract_doi("DOI: 10.1234/test.2023.001") == "10.1234/test.2023.001"
        assert _extract_doi("https://doi.org/10.5678/paper.2024") == "10.5678/paper.2024"

    def test_clean_trailing_punctuation(self):
        assert _extract_doi("See 10.1234/test.2023.001.") == "10.1234/test.2023.001"
        assert _extract_doi("(10.1234/test.2023.001)") == "10.1234/test.2023.001"

    def test_no_doi(self):
        assert _extract_doi("This has no DOI") is None


class TestMetadataExtraction:
    """Test paper metadata extraction from text."""

    def test_extract_title(self):
        text = "Attention Is All You Need\n\nAbstract\n..."
        metadata = _parse_metadata_from_text(text, "paper.pdf")
        assert "Attention" in metadata.title

    def test_extract_abstract(self):
        text = "Title\n\nAbstract\n\nThis paper proposes a novel approach...\n\nKeywords: deep learning, attention"
        metadata = _parse_metadata_from_text(text, "paper.pdf")
        assert "novel approach" in metadata.abstract

    def test_extract_keywords(self):
        text = "Title\n\nAbstract\n...\n\nKeywords: deep learning, attention, transformer"
        metadata = _parse_metadata_from_text(text, "paper.pdf")
        assert "deep learning" in metadata.keywords
        assert "attention" in metadata.keywords

    def test_extract_arxiv_from_text(self):
        text = "Title\n\narXiv:2301.12345\n\nAbstract\n..."
        metadata = _parse_metadata_from_text(text, "paper.pdf")
        assert metadata.arxiv_id == "2301.12345"


class TestPaperParsing:
    """Test full paper parsing from Markdown."""

    def test_parse_sections(self):
        markdown = """# Title

# Abstract

This is the abstract.

# 1. Introduction

The introduction text.

## 1.1 Background

Background content.

# 2. Methods

Method details.
"""
        parsed = parse_paper_from_markdown(markdown, "paper.md")
        assert len(parsed.sections) >= 3
        section_titles = [s.title for s in parsed.sections]
        assert any("Abstract" in t for t in section_titles)
        assert any("Introduction" in t for t in section_titles)

    def test_parse_figures(self):
        markdown = """# Paper

<!-- FIGURE: Figure 1 -->
![Network Architecture](./images/fig1.png)

Some text.

![Another figure](./images/fig2.png)
"""
        parsed = parse_paper_from_markdown(markdown, "paper.md")
        assert len(parsed.figures) >= 1
        assert any("Network" in f.caption for f in parsed.figures)

    def test_parse_tables(self):
        markdown = """# Paper

<!-- TABLE: Table 1 -->
| Column A | Column B |
|----------|----------|
| Value 1  | Value 2  |
| Value 3  | Value 4  |
"""
        parsed = parse_paper_from_markdown(markdown, "paper.md")
        assert len(parsed.tables) >= 1
        assert parsed.tables[0].headers == ["Column A", "Column B"]

    def test_parse_equations(self):
        markdown = """# Paper

The loss function is:

$$L = -\\sum_{i} y_i \\log(\\hat{y}_i)$$

And the gradient:

$\\nabla L = \\frac{\\partial L}{\\partial W}$
"""
        parsed = parse_paper_from_markdown(markdown, "paper.md")
        assert len(parsed.equations) >= 1
        assert any("sum" in eq.latex.lower() for eq in parsed.equations)

    def test_parse_references(self):
        markdown = """# References

[1] Vaswani et al., Attention Is All You Need, NeurIPS, 2017

[2] He et al., Deep Residual Learning, CVPR, 2016
"""
        parsed = parse_paper_from_markdown(markdown, "paper.md")
        assert len(parsed.references) >= 1
        assert any("Vaswani" in r.authors for r in parsed.references)


class TestConverterInfo:
    """Test PDF converter availability check."""

    def test_get_converter_info(self):
        info = get_pdf_converter_info()
        assert "pypdf" in info
        assert info["pypdf"]["available"] is True  # Always available in pdf extra
        assert "marker" in info
        assert "pymupdf" in info
        assert "pdfplumber" in info


class TestDetectIntegration:
    """Test detect.py integration with PDF conversion."""

    def test_convert_pdf_file_not_found(self, tmp_path):
        from graphify.detect import convert_pdf_file
        result = convert_pdf_file(Path("/nonexistent/file.pdf"), tmp_path)
        assert result is None

    def test_pdf_classification(self):
        from graphify.detect import classify_file, FileType
        assert classify_file(Path("paper.pdf")) == FileType.PAPER
        assert classify_file(Path("document.PDF")) == FileType.PAPER
