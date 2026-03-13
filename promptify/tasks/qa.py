"""Question Answering task."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel

from promptify.schemas.qa import Answer
from promptify.tasks.base import BaseTask

_DEFAULT_INSTRUCTION = (
    "You are a question answering system. "
    "Answer the question based on the provided context and return structured JSON."
)


class QA(BaseTask):
    """Extractive / generative QA.

    Example
    -------
    >>> qa = QA(model="gpt-4o-mini")
    >>> answer = qa("Einstein was born in Ulm.", question="Where was Einstein born?")
    >>> answer.answer
    'Ulm'
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
            output_schema=Answer,
            instruction=instruction or _DEFAULT_INSTRUCTION,
            template="qa",
            domain=domain,
            examples=examples,
            **kwargs,
        )

    def __call__(self, text: str, question: str = "", **kwargs: Any) -> Answer:  # type: ignore[override]
        """Run QA — pass question as kwarg for template rendering."""
        return super().__call__(text, question=question, **kwargs)  # type: ignore[return-value]

    async def acall(self, text: str, question: str = "", **kwargs: Any) -> Answer:  # type: ignore[override]
        return await super().acall(text, question=question, **kwargs)  # type: ignore[return-value]
