"""QA output schemas."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class Answer(BaseModel):
    """A question-answering result."""

    answer: str
    evidence: Optional[str] = None
    confidence: Optional[float] = None
