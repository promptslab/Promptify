"""Summarization task."""

from __future__ import annotations

from typing import Any, Optional

from promptify.schemas.summarize import Summary
from promptify.tasks.base import BaseTask

_DEFAULT_INSTRUCTION = (
    "You are a text summarization system. "
    "Summarize the given text and return structured JSON."
)


class Summarize(BaseTask):
    """Text summarization.

    Example
    -------
    >>> summarizer = Summarize(model="gpt-4o-mini")
    >>> result = summarizer("Long article text here...")
    >>> result.summary
    'Concise summary...'
    """

    def __init__(
        self,
        model: str,
        max_length: Optional[int] = None,
        key_points: bool = False,
        domain: Optional[str] = None,
        instruction: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model=model,
            output_schema=Summary,
            instruction=instruction or _DEFAULT_INSTRUCTION,
            template="summarize",
            domain=domain,
            max_length=max_length,
            key_points=key_points,
            **kwargs,
        )
