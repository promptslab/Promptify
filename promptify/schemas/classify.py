"""Classification output schemas."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel


class Classification(BaseModel):
    """A single classification result."""

    label: str
    confidence: Optional[float] = None


class MultiLabelResult(BaseModel):
    """Multi-label classification result."""

    labels: List[Classification]
