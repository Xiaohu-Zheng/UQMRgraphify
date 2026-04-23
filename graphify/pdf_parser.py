# PDF to Markdown conversion with academic paper support
# Supports: marker (best quality), pymupdf+pdfplumber, pypdf (fallback)
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class PaperMetadata:
    """Structured metadata extracted from an academic paper."""
    title: str = ""
    authors: list[str] = field(default_factory=list)
    abstract: str = ""
    keywords: list[str] = field(default_factory=list)
    arxiv_id: Optional[str] = None
    doi: Optional[str] = None
    source_file: str = ""


@dataclass
class PaperSection:
    """A section from an academic paper."""
    title: str
    level: int  # 1 for #, 2 for ##, etc.
    content: str
    section_id: str = ""


@dataclass
class PaperFigure:
    """A figure from an academic paper."""
    figure_id: str
    caption: str
    image_path: Optional[str] = None


@dataclass
class PaperTable:
    """A table from an academic paper."""
    table_id: str
    caption: str
    headers: list[str] = field(default_factory=list)
    rows: list[list[str]] = field(default_factory=list)


@dataclass
class PaperEquation:
    """An equation from an academic paper."""
    equation_id: str
    latex: str
    number: Optional[str] = None


@dataclass
class PaperReference:
    """A reference/citation from an academic paper."""
    ref_id: str
    authors: str = ""
    title: str = ""
    venue: str = ""
    year: str = ""
    doi: Optional[str] = None


@dataclass
class ParsedPaper:
    """Complete parsed academic paper."""
    metadata: PaperMetadata
    sections: list[PaperSection] = field(default_factory=list)
    figures: list[PaperFigure] = field(default_factory=list)
    tables: list[PaperTable] = field(default_factory=list)
    equations: list[PaperEquation] = field(default_factory=list)
    references: list[PaperReference] = field(default_factory=list)
    full_markdown: str = ""


def _has_marker() -> bool:
    """Check if marker-pdf is available."""
    try:
        import marker  # noqa: F401
        return True
    except ImportError:
        return False


def _has_pymupdf() -> bool:
    """Check if pymupdf is available."""
    try:
        import fitz  # pymupdf imports as fitz  # noqa: F401
        return True
    except ImportError:
        return False


def _has_pdfplumber() -> bool:
    """Check if pdfplumber is available."""
    try:
        import pdfplumber  # noqa: F401
        return True
    except ImportError:
        return False


def _extract_arxiv_id(text: str) -> Optional[str]:
    """Extract arXiv ID from text."""
    # New format: YYMM.NNNNN or YYMM.NNNNNN
    match = re.search(r'arxiv[:\s]*(\d{4}\.\d{4,5})', text, re.IGNORECASE)
    if match:
        return match.group(1)
    # Old format: subject-class/YYMMNNN (e.g., cs.LG/0701001)
    match = re.search(r'arxiv[:\s]*([a-z-]+\.?[a-z-]*/\d{7})', text, re.IGNORECASE)
    if match:
        return match.group(1)
    return None


def _extract_doi(text: str) -> Optional[str]:
    """Extract DOI from text."""
    match = re.search(r'10\.\d{4,}/[^\s]+', text)
    if match:
        doi = match.group(0)
        # Clean trailing punctuation
        doi = re.sub(r'[.,;)\]]+$', '', doi)
        return doi
    return None


def _parse_metadata_from_text(text: str, filename: str) -> PaperMetadata:
    """Parse paper metadata from text content."""
    metadata = PaperMetadata(source_file=filename)

    # Extract arXiv ID
    metadata.arxiv_id = _extract_arxiv_id(text)

    # Extract DOI
    metadata.doi = _extract_doi(text)

    # Extract title (first non-empty line or look for title markers)
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    if lines:
        # Skip common headers
        for line in lines[:5]:
            if line and not any(skip in line.lower() for skip in ['arxiv', 'proceedings', 'conference', 'journal']):
                metadata.title = line[:200]  # Limit title length
                break

    # Extract abstract
    abstract_match = re.search(
        r'(?:abstract|summary)[:\s]*\n(.*?)(?=\n\s*(?:keywords|1\.|introduction|section))',
        text,
        re.IGNORECASE | re.DOTALL
    )
    if abstract_match:
        metadata.abstract = abstract_match.group(1).strip()[:2000]

    # Extract keywords
    keywords_match = re.search(
        r'keywords?[:\s]*([^\n]+)',
        text,
        re.IGNORECASE
    )
    if keywords_match:
        keywords_str = keywords_match.group(1)
        # Split by common delimiters
        keywords = re.split(r'[,;·]', keywords_str)
        metadata.keywords = [k.strip() for k in keywords if k.strip() and len(k.strip()) > 2][:10]

    # Extract authors (heuristic: look for email or affiliation patterns)
    # This is a best-effort extraction
    author_section = re.search(
        r'^([A-Z][a-z]+ [A-Z][a-z]+(?:,?\s+(?:and|&)?\s*[A-Z][a-z]+ [A-Z][a-z]+)*)',
        text[:1000],
        re.MULTILINE
    )
    if author_section:
        authors_str = author_section.group(1)
        authors_str = re.sub(r'\s+and\s+', ', ', authors_str)
        authors_str = re.sub(r'\s*&\s*', ', ', authors_str)
        metadata.authors = [a.strip() for a in authors_str.split(',') if a.strip()][:10]

    return metadata


def _convert_with_marker(pdf_path: Path, out_dir: Path) -> tuple[str, Path | None]:
    """Convert PDF using marker library (best quality for academic papers)."""
    try:
        from marker.converters.pdf import PdfConverter
        from marker.models import create_model_dict
        from marker.output import text_from_rendered

        # Create output directory
        out_dir.mkdir(parents=True, exist_ok=True)

        # Initialize models (this may download on first run)
        model_dict = create_model_dict()

        # Convert
        converter = PdfConverter(artifact_dict=model_dict)
        rendered = converter(str(pdf_path))
        markdown_text, _, images = text_from_rendered(rendered)

        # Save images if any
        images_dir = out_dir / f"{pdf_path.stem}_images"
        if images:
            images_dir.mkdir(exist_ok=True)
            for img_name, img_data in images.items():
                img_path = images_dir / img_name
                img_path.write_bytes(img_data)

        # Save markdown
        out_path = out_dir / f"{pdf_path.stem}.md"
        out_path.write_text(markdown_text, encoding="utf-8")

        return markdown_text, out_path

    except Exception as e:
        print(f"[graphify] marker conversion failed: {e}")
        return "", None


def _convert_with_pymupdf(pdf_path: Path, out_dir: Path) -> tuple[str, Path | None]:
    """Convert PDF using pymupdf (good balance of speed and quality)."""
    try:
        import fitz  # pymupdf

        doc = fitz.open(str(pdf_path))
        out_dir.mkdir(parents=True, exist_ok=True)

        # Create images directory
        images_dir = out_dir / f"{pdf_path.stem}_images"
        images_dir.mkdir(exist_ok=True)

        markdown_parts = []
        figure_counter = 1
        table_counter = 1

        for page_num, page in enumerate(doc):
            # Extract text
            text = page.get_text()
            markdown_parts.append(text)
            markdown_parts.append("\n\n")

            # Extract images
            images = page.get_images()
            for img_index, img in enumerate(images):
                xref = img[0]
                try:
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]
                    img_filename = f"figure_{figure_counter}.{image_ext}"
                    img_path = images_dir / img_filename
                    img_path.write_bytes(image_bytes)
                    markdown_parts.append(f"\n<!-- FIGURE: Figure {figure_counter} -->\n")
                    markdown_parts.append(f"![Figure {figure_counter}](./{pdf_path.stem}_images/{img_filename})\n\n")
                    figure_counter += 1
                except Exception:
                    pass

        doc.close()

        markdown_text = "".join(markdown_parts)
        out_path = out_dir / f"{pdf_path.stem}.md"
        out_path.write_text(markdown_text, encoding="utf-8")

        return markdown_text, out_path

    except Exception as e:
        print(f"[graphify] pymupdf conversion failed: {e}")
        return "", None


def _convert_with_pdfplumber(pdf_path: Path, out_dir: Path) -> tuple[str, Path | None]:
    """Convert PDF using pdfplumber (good for tables)."""
    try:
        import pdfplumber

        out_dir.mkdir(parents=True, exist_ok=True)
        markdown_parts = []
        table_counter = 1

        with pdfplumber.open(str(pdf_path)) as pdf:
            for page in pdf.pages:
                # Extract text
                text = page.extract_text() or ""
                markdown_parts.append(text)
                markdown_parts.append("\n\n")

                # Extract tables
                tables = page.extract_tables()
                for table in tables:
                    if table and len(table) > 1:
                        markdown_parts.append(f"\n<!-- TABLE: Table {table_counter} -->\n")
                        # Header row
                        if table[0]:
                            headers = [str(cell or "") for cell in table[0]]
                            markdown_parts.append("| " + " | ".join(headers) + " |\n")
                            markdown_parts.append("| " + " | ".join(["---"] * len(headers)) + " |\n")
                        # Data rows
                        for row in table[1:]:
                            cells = [str(cell or "") for cell in row]
                            markdown_parts.append("| " + " | ".join(cells) + " |\n")
                        markdown_parts.append("\n")
                        table_counter += 1

        markdown_text = "".join(markdown_parts)
        out_path = out_dir / f"{pdf_path.stem}.md"
        out_path.write_text(markdown_text, encoding="utf-8")

        return markdown_text, out_path

    except Exception as e:
        print(f"[graphify] pdfplumber conversion failed: {e}")
        return "", None


def _convert_with_pypdf(pdf_path: Path, out_dir: Path) -> tuple[str, Path | None]:
    """Fallback: Convert PDF using pypdf (basic text extraction)."""
    try:
        from pypdf import PdfReader

        reader = PdfReader(str(pdf_path))
        out_dir.mkdir(parents=True, exist_ok=True)

        pages = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages.append(text)

        markdown_text = "\n\n".join(pages)
        out_path = out_dir / f"{pdf_path.stem}.md"
        out_path.write_text(markdown_text, encoding="utf-8")

        return markdown_text, out_path

    except Exception as e:
        print(f"[graphify] pypdf conversion failed: {e}")
        return "", None


def convert_pdf_to_markdown(
    pdf_path: Path,
    out_dir: Path,
    prefer_marker: bool = True,
) -> tuple[str, Path | None]:
    """
    Convert a PDF file to structured Markdown.

    Tries converters in order of quality:
    1. marker (best for academic papers, handles tables/equations)
    2. pymupdf (good balance, extracts images)
    3. pdfplumber (good for tables)
    4. pypdf (fallback, text only)

    Args:
        pdf_path: Path to the PDF file
        out_dir: Directory to save the output markdown and images
        prefer_marker: If True, try marker first (requires more dependencies)

    Returns:
        Tuple of (markdown_text, output_path) or ("", None) on failure
    """
    pdf_path = Path(pdf_path)
    out_dir = Path(out_dir)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    markdown_text = ""
    out_path = None

    # Try marker first (best quality for academic papers)
    if prefer_marker and _has_marker():
        markdown_text, out_path = _convert_with_marker(pdf_path, out_dir)
        if markdown_text:
            return markdown_text, out_path

    # Try pymupdf (good balance of speed and quality)
    if _has_pymupdf():
        markdown_text, out_path = _convert_with_pymupdf(pdf_path, out_dir)
        if markdown_text:
            return markdown_text, out_path

    # Try pdfplumber (good for tables)
    if _has_pdfplumber():
        markdown_text, out_path = _convert_with_pdfplumber(pdf_path, out_dir)
        if markdown_text:
            return markdown_text, out_path

    # Fallback to pypdf (always available as it's in the pdf extra)
    markdown_text, out_path = _convert_with_pypdf(pdf_path, out_dir)

    return markdown_text, out_path


def parse_paper_from_markdown(markdown_text: str, filename: str) -> ParsedPaper:
    """
    Parse an academic paper from its Markdown representation.

    Extracts structured information including metadata, sections,
    figures, tables, equations, and references.

    Args:
        markdown_text: The Markdown content of the paper
        filename: Original filename for reference

    Returns:
        ParsedPaper with all extracted components
    """
    metadata = _parse_metadata_from_text(markdown_text, filename)

    sections = []
    figures = []
    tables = []
    equations = []
    references = []

    # Parse sections (headers)
    section_pattern = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)
    matches = list(section_pattern.finditer(markdown_text))

    for i, match in enumerate(matches):
        level = len(match.group(1))
        title = match.group(2).strip()
        start = match.end()

        # Get content until next section or end
        if i + 1 < len(matches):
            end = matches[i + 1].start()
        else:
            end = len(markdown_text)

        content = markdown_text[start:end].strip()

        section_id = re.sub(r'[^a-z0-9]+', '_', title.lower()).strip('_')
        sections.append(PaperSection(
            title=title,
            level=level,
            content=content,
            section_id=section_id,
        ))

    # Parse figures
    figure_pattern = re.compile(
        r'<!--\s*FIGURE[:\s]+([^>]+)\s*-->\s*!\[([^\]]*)\]\(([^)]+)\)',
        re.IGNORECASE
    )
    for i, match in enumerate(figure_pattern.finditer(markdown_text)):
        figure_id = match.group(1).strip() or f"Figure {i + 1}"
        caption = match.group(2).strip()
        image_path = match.group(3).strip()
        figures.append(PaperFigure(
            figure_id=figure_id,
            caption=caption,
            image_path=image_path,
        ))

    # Also find standalone images
    img_pattern = re.compile(r'!\[([^\]]*)\]\(([^)]+)\)')
    for match in img_pattern.finditer(markdown_text):
        # Skip if already captured as figure
        if "<!-- FIGURE" in markdown_text[max(0, match.start() - 50):match.start()]:
            continue
        figures.append(PaperFigure(
            figure_id=f"Figure {len(figures) + 1}",
            caption=match.group(1).strip(),
            image_path=match.group(2).strip(),
        ))

    # Parse tables
    table_pattern = re.compile(
        r'<!--\s*TABLE[:\s]+([^>]+)\s*-->\s*\|([^\n]+)\|\s*\|[-|\s]+\|\s*((?:\|[^\n]+\|\s*)+)',
        re.IGNORECASE
    )
    for match in table_pattern.finditer(markdown_text):
        table_id = match.group(1).strip()
        header_line = match.group(2).strip()
        body_lines = match.group(3).strip()

        headers = [h.strip() for h in header_line.split('|') if h.strip()]
        rows = []
        for line in body_lines.split('\n'):
            if line.strip():
                cells = [c.strip() for c in line.split('|') if c.strip()]
                if cells:
                    rows.append(cells)

        tables.append(PaperTable(
            table_id=table_id,
            caption=table_id,
            headers=headers,
            rows=rows,
        ))

    # Parse equations (LaTeX)
    eq_pattern = re.compile(
        r'\$\$(.+?)\$\$|\$(.+?)\$|\\\[(.+?)\\\]|\\\((.+?)\\\)',
        re.DOTALL
    )
    for i, match in enumerate(eq_pattern.finditer(markdown_text)):
        latex = match.group(1) or match.group(2) or match.group(3) or match.group(4)
        if latex:
            latex = latex.strip()
            equations.append(PaperEquation(
                equation_id=f"Eq_{i + 1}",
                latex=latex,
            ))

    # Parse references
    # Pattern 1: Numbered references [1] Author, Title, Venue, Year
    ref_pattern = re.compile(
        r'\[(\d+)\]\s*(.+?)(?=\n\s*\[\d+\]|\n\s*$|\Z)',
        re.DOTALL
    )
    for match in ref_pattern.finditer(markdown_text):
        ref_id = match.group(1)
        ref_text = match.group(2).strip()

        # Try to parse the reference
        # Common format: Authors, Title, Venue, Year
        parts = re.split(r',\s*', ref_text)

        authors = parts[0] if parts else ""
        title = parts[1] if len(parts) > 1 else ""
        venue = parts[2] if len(parts) > 2 else ""

        year_match = re.search(r'(19|20)\d{2}', ref_text)
        year = year_match.group(0) if year_match else ""

        doi = _extract_doi(ref_text)

        references.append(PaperReference(
            ref_id=ref_id,
            authors=authors,
            title=title,
            venue=venue,
            year=year,
            doi=doi,
        ))

    return ParsedPaper(
        metadata=metadata,
        sections=sections,
        figures=figures,
        tables=tables,
        equations=equations,
        references=references,
        full_markdown=markdown_text,
    )


def convert_pdf_file(
    pdf_path: Path,
    out_dir: Path,
    include_metadata: bool = True,
) -> Path | None:
    """
    Convert a PDF file to Markdown with optional YAML frontmatter.

    This is the main entry point for PDF conversion, following the same
    pattern as convert_office_file().

    Args:
        pdf_path: Path to the PDF file
        out_dir: Directory to save the output
        include_metadata: If True, add YAML frontmatter with extracted metadata

    Returns:
        Path to the generated Markdown file, or None on failure
    """
    pdf_path = Path(pdf_path)
    out_dir = Path(out_dir)

    if not pdf_path.exists():
        return None

    # Convert PDF to markdown
    markdown_text, md_path = convert_pdf_to_markdown(pdf_path, out_dir)

    if not markdown_text or not md_path:
        return None

    if include_metadata:
        # Parse the paper and add metadata
        parsed = parse_paper_from_markdown(markdown_text, pdf_path.name)

        # Build YAML frontmatter
        frontmatter_lines = ["---"]

        if parsed.metadata.title:
            frontmatter_lines.append(f'title: "{parsed.metadata.title}"')

        if parsed.metadata.authors:
            authors_str = ", ".join(f'"{a}"' for a in parsed.metadata.authors)
            frontmatter_lines.append(f"authors: [{authors_str}]")

        if parsed.metadata.abstract:
            # Escape quotes and newlines
            abstract = parsed.metadata.abstract.replace('"', '\\"').replace('\n', ' ')
            frontmatter_lines.append(f'abstract: "{abstract[:500]}"')

        if parsed.metadata.keywords:
            keywords_str = ", ".join(f'"{k}"' for k in parsed.metadata.keywords)
            frontmatter_lines.append(f"keywords: [{keywords_str}]")

        if parsed.metadata.arxiv_id:
            frontmatter_lines.append(f'arxiv_id: "{parsed.metadata.arxiv_id}"')

        if parsed.metadata.doi:
            frontmatter_lines.append(f'doi: "{parsed.metadata.doi}"')

        frontmatter_lines.append(f'source_file: "{pdf_path.name}"')
        frontmatter_lines.append(f'type: paper')
        frontmatter_lines.append("---")
        frontmatter_lines.append("")

        # Add section count summary
        summary_lines = [
            f"<!-- Converted from {pdf_path.name} -->",
            f"<!-- Sections: {len(parsed.sections)} | Figures: {len(parsed.figures)} | Tables: {len(parsed.tables)} | Equations: {len(parsed.equations)} | References: {len(parsed.references)} -->",
            "",
        ]

        # Combine frontmatter and content
        full_content = "\n".join(frontmatter_lines) + "\n".join(summary_lines) + markdown_text

        # Use a stable filename with hash
        name_hash = hashlib.sha256(str(pdf_path.resolve()).encode()).hexdigest()[:8]
        final_path = out_dir / f"{pdf_path.stem}_{name_hash}.md"
        final_path.write_text(full_content, encoding="utf-8")

        # Remove the intermediate file if it's different
        if md_path != final_path and md_path.exists():
            md_path.unlink()

        return final_path

    return md_path


def get_pdf_converter_info() -> dict:
    """
    Get information about available PDF converters.

    Returns:
        Dict with converter availability and recommendations
    """
    return {
        "marker": {
            "available": _has_marker(),
            "quality": "best",
            "features": ["text", "tables", "equations", "figures"],
            "description": "Best for academic papers, handles complex layouts",
        },
        "pymupdf": {
            "available": _has_pymupdf(),
            "quality": "good",
            "features": ["text", "figures"],
            "description": "Fast and reliable, extracts images",
        },
        "pdfplumber": {
            "available": _has_pdfplumber(),
            "quality": "good",
            "features": ["text", "tables"],
            "description": "Good for table extraction",
        },
        "pypdf": {
            "available": True,  # Always available in pdf extra
            "quality": "basic",
            "features": ["text"],
            "description": "Basic text extraction fallback",
        },
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m graphify.pdf_parser <pdf_file> [output_dir]")
        sys.exit(1)

    pdf_file = Path(sys.argv[1])
    output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("graphify-out/converted")

    print(f"Converting {pdf_file}...")
    print(f"Available converters: {get_pdf_converter_info()}")

    result = convert_pdf_file(pdf_file, output_dir)

    if result:
        print(f"Output: {result}")
    else:
        print("Conversion failed")
        sys.exit(1)
