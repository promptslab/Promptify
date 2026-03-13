"""Generation output schemas."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class GeneratedQuestion(BaseModel):
    """A generated question with optional answer."""

    question: str
    answer: Optional[str] = None


class SQLQuery(BaseModel):
    """A generated SQL query."""

    query: str
    explanation: Optional[str] = None
