"""Extraction tasks — relations and tabular data."""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

from pydantic import BaseModel

from promptify.schemas.extract import ExtractionResult
from promptify.tasks.base import BaseTask


class ExtractRelations(BaseTask):
    """Extract semantic relations (subject-predicate-object triples) from text.

    Example
    -------
    >>> extractor = ExtractRelations(model="gpt-4o-mini")
    >>> result = extractor("Einstein was born in Ulm in 1879.")
    >>> result.relations[0].subject
    'Einstein'
    """

    def __init__(
        self,
        model: str,
        domain: Optional[str] = None,
        examples: Optional[List[Tuple[str, str]]] = None,
        instruction: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model=model,
            output_schema=ExtractionResult,
            instruction=instruction
            or "Extract semantic relations from the text and return structured JSON.",
            template="relation_extraction",
            domain=domain,
            examples=examples,
            **kwargs,
        )


class ExtractTable(BaseTask):
    """Extract tabular data from unstructured text.

    Example
    -------
    >>> extractor = ExtractTable(model="gpt-4o-mini")
    >>> result = extractor("John is 30, lives in NYC. Jane is 25, lives in LA.")
    """

    def __init__(
        self,
        model: str,
        examples: Optional[List[Tuple[str, str]]] = None,
        instruction: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model=model,
            output_schema=ExtractionResult,
            instruction=instruction
            or "Extract structured tabular data from the text and return JSON.",
            template="tabular_extraction",
            examples=examples,
            **kwargs,
        )
