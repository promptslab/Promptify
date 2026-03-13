"""Extraction output schemas."""

from __future__ import annotations

from typing import Dict, List

from pydantic import BaseModel


class Relation(BaseModel):
    """A subject-predicate-object triple."""

    subject: str
    predicate: str
    object: str


class TableRow(BaseModel):
    """A single row of tabular data."""

    data: Dict[str, str]


class ExtractionResult(BaseModel):
    """Collection of extracted relations or table rows."""

    relations: List[Relation] = []
    rows: List[TableRow] = []
