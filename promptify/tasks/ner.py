"""Named Entity Recognition task."""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

from promptify.schemas.ner import NERResult
from promptify.tasks.base import BaseTask

_DEFAULT_INSTRUCTION = (
    "You are a Named Entity Recognition (NER) system. "
    "Extract entities from the given text and return structured JSON."
)


class NER(BaseTask):
    """Named Entity Recognition.

    Example
    -------
    >>> ner = NER(model="gpt-4o-mini", domain="medical")
    >>> result = ner("Patient has chronic hip pain and osteoporosis")
    >>> result.entities[0].text
    'chronic hip pain'
    """

    def __init__(
        self,
        model: str,
        domain: Optional[str] = None,
        labels: Optional[List[str]] = None,
        examples: Optional[List[Tuple[str, str]]] = None,
        instruction: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model=model,
            output_schema=NERResult,
            instruction=instruction or _DEFAULT_INSTRUCTION,
            template="ner",
            domain=domain,
            labels=labels,
            examples=examples,
            **kwargs,
        )
