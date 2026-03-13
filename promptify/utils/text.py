"""Text processing utilities."""

from __future__ import annotations

import re
from typing import List


def chunk_text(text: str, max_chars: int = 4000, overlap: int = 200) -> List[str]:
    """Split text into overlapping chunks."""
    if len(text) <= max_chars:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + max_chars
        if end < len(text):
            # Try to break at sentence boundary
            last_period = text.rfind(".", start, end)
            if last_period > start + max_chars // 2:
                end = last_period + 1
        chunks.append(text[start:end].strip())
        start = end - overlap
    return chunks


def normalize_whitespace(text: str) -> str:
    """Collapse multiple whitespace into single spaces."""
    return re.sub(r"\s+", " ", text).strip()
