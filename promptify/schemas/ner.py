"""NER output schemas."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel


class Entity(BaseModel):
    """A single named entity."""

    text: str
    label: str
    start: Optional[int] = None
    end: Optional[int] = None


class NERResult(BaseModel):
    """Collection of extracted entities."""

    entities: List[Entity]
