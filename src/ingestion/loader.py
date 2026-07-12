"""Document loading for ingestion.

Both entry points return clean plain text ready for chunking:

- ``load_from_file(path)``: dispatches by file extension (pdf, docx, txt,
  md, html) to the appropriate parser and returns extracted plain text.
- ``load_from_url(url)``: fetches the page over HTTP and strips
  boilerplate (nav, scripts, styles, ads, etc.) with beautifulsoup4,
  returning the cleaned plain text.
"""


def load_from_file(path):
    """Load and return clean plain text from a local file (pdf/docx/txt/md/html)."""
    raise NotImplementedError


def load_from_url(url):
    """Fetch a URL and return clean plain text with boilerplate stripped."""
    raise NotImplementedError
