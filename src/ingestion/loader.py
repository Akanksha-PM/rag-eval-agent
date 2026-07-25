"""Document loading for ingestion.

Both entry points return clean plain text ready for chunking:

- ``load_from_file(path)``: dispatches by file extension (pdf, docx, txt,
  md, html) to the appropriate parser and returns extracted plain text.
- ``load_from_url(url)``: fetches the page over HTTP and strips
  boilerplate (script, style, nav, header, footer) with beautifulsoup4,
  returning the cleaned plain text.
"""

from pathlib import Path

import requests
from bs4 import BeautifulSoup
from docx import Document
from pypdf import PdfReader


def _extract_visible_text(soup: BeautifulSoup) -> str:
    for tag in soup(["script", "style", "nav", "header", "footer"]):
        tag.decompose()
    return soup.get_text(separator="\n")


def _collapse_whitespace(text: str) -> str:
    lines = [line.strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def load_from_file(path: str) -> str:
    """Load and return clean plain text from a local file (pdf/docx/txt/md/html)."""
    file_path = Path(path)
    extension = file_path.suffix.lower()

    if extension == ".pdf":
        reader = PdfReader(str(file_path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    if extension == ".docx":
        document = Document(str(file_path))
        return "\n".join(p.text for p in document.paragraphs)

    if extension in (".txt", ".md"):
        return file_path.read_text()

    if extension in (".html", ".htm"):
        soup = BeautifulSoup(file_path.read_text(), "html.parser")
        return _collapse_whitespace(_extract_visible_text(soup))

    raise ValueError(f"Unsupported file extension: {extension}")


def load_from_url(url: str) -> str:
    """Fetch a URL and return clean plain text with boilerplate stripped."""
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    return _collapse_whitespace(_extract_visible_text(soup))
