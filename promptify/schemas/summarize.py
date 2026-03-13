"""Summarization output schemas."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel


class Summary(BaseModel):
    """A summarization result."""

    summary: str
    key_points: Optional[List[str]] = None
