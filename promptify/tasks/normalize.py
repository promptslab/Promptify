"""Normalization tasks — text normalization and topic modelling."""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

from pydantic import BaseModel

from promptify.tasks.base import BaseTask


class _NormalizationResult(BaseModel):
    normalized_text: str


class NormalizeText(BaseTask):
    """Normalize text according to specified rules.

    Example
    -------
    >>> norm = NormalizeText(model="gpt-4o-mini", rules=["lowercase", "remove punctuation"])
    >>> result = norm("Hello, World!")
    """

    def __init__(
        self,
        model: str,
        rules: Optional[List[str]] = None,
        examples: Optional[List[Tuple[str, str]]] = None,
        instruction: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model=model,
            output_schema=_NormalizationResult,
            instruction=instruction
            or "Normalize the text according to the rules and return JSON.",
            template="text_normalization",
            examples=examples,
            rules=rules,
            **kwargs,
        )


class _Topic(BaseModel):
    topic: str
    words: List[str]


class _TopicResult(BaseModel):
    topics: List[_Topic]


class ExtractTopics(BaseTask):
    """Extract topics from text.

    Example
    -------
    >>> topics = ExtractTopics(model="gpt-4o-mini", num_topics=3)
    >>> result = topics("Long article about technology and science...")
    """

    def __init__(
        self,
        model: str,
        num_topics: int = 5,
        domain: Optional[str] = None,
        instruction: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model=model,
            output_schema=_TopicResult,
            instruction=instruction
            or "Extract coherent topics from the text and return JSON.",
            template="topic_modelling",
            domain=domain,
            num_topics=num_topics,
            **kwargs,
        )
