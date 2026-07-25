"""Splits plain text into overlapping chunks for embedding and retrieval."""

from config import Config


def _split_into_paragraphs(text: str) -> list[str]:
    paragraphs = [p.strip() for p in text.split("\n\n")]
    return [p for p in paragraphs if p]


def chunk_text(text: str, product: str, source: str) -> list[dict]:
    """Split text into overlapping, product/source-tagged chunks.

    CHUNK_SIZE and CHUNK_OVERLAP (config.py) are word counts, not token
    counts -- a simplification. A real tokenizer (matching whatever model
    or embedder ultimately consumes these chunks) would size them more
    accurately and would be a natural upgrade over this word-count
    approximation.
    """
    if not text or not text.strip():
        return []

    chunk_size = Config.CHUNK_SIZE
    chunk_overlap = Config.CHUNK_OVERLAP

    paragraphs = _split_into_paragraphs(text)

    # Greedily merge paragraphs into chunks of up to chunk_size words each.
    # A paragraph that alone exceeds chunk_size is hard-split into
    # chunk_size-word pieces so no chunk ever exceeds the limit. Overlap is
    # layered on top of these raw chunks afterwards.
    raw_chunks: list[list[str]] = []
    current_words: list[str] = []

    for paragraph in paragraphs:
        words = paragraph.split()

        if len(words) > chunk_size:
            if current_words:
                raw_chunks.append(current_words)
                current_words = []
            for i in range(0, len(words), chunk_size):
                raw_chunks.append(words[i : i + chunk_size])
            continue

        if not current_words:
            current_words = words
        elif len(current_words) + len(words) <= chunk_size:
            current_words = current_words + words
        else:
            raw_chunks.append(current_words)
            current_words = words

    if current_words:
        raw_chunks.append(current_words)

    product_slug = product.lower().replace(" ", "_")

    chunks = []
    for index, words in enumerate(raw_chunks):
        if index == 0 or chunk_overlap <= 0:
            chunk_words = words
        else:
            previous_words = raw_chunks[index - 1]
            # Only prepend as much overlap as fits without pushing this
            # chunk past chunk_size (matters for chunks already at the cap,
            # e.g. hard-split pieces).
            headroom = chunk_size - len(words)
            overlap_amount = min(chunk_overlap, headroom)
            overlap_words = previous_words[-overlap_amount:] if overlap_amount > 0 else []
            chunk_words = overlap_words + words

        chunks.append(
            {
                "id": f"{product_slug}_{index}",
                "product": product,
                "source": source,
                "chunk_index": index,
                "text": " ".join(chunk_words),
                "word_count": len(chunk_words),
            }
        )

    return chunks
