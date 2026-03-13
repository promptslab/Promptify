"""Generation tasks — questions and SQL."""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

from pydantic import BaseModel

from promptify.schemas.generate import GeneratedQuestion, SQLQuery
from promptify.tasks.base import BaseTask


class _QuestionGenResult(BaseModel):
    questions: List[GeneratedQuestion]


class GenerateQuestions(BaseTask):
    """Generate question-answer pairs from text.

    Example
    -------
    >>> gen = GenerateQuestions(model="gpt-4o-mini", num_questions=3)
    >>> result = gen("Einstein was born in Ulm in 1879.")
    """

    def __init__(
        self,
        model: str,
        num_questions: int = 3,
        domain: Optional[str] = None,
        instruction: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model=model,
            output_schema=_QuestionGenResult,
            instruction=instruction
            or "Generate question-answer pairs from the text and return JSON.",
            template="question_generation",
            domain=domain,
            num_questions=num_questions,
            **kwargs,
        )


class GenerateSQL(BaseTask):
    """Convert natural language to SQL queries.

    Example
    -------
    >>> gen = GenerateSQL(model="gpt-4o-mini", schema="CREATE TABLE users (id INT, name TEXT)")
    >>> result = gen("Get all users named John")
    >>> result.query
    "SELECT * FROM users WHERE name = 'John'"
    """

    def __init__(
        self,
        model: str,
        schema: Optional[str] = None,
        examples: Optional[List[Tuple[str, str]]] = None,
        instruction: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model=model,
            output_schema=SQLQuery,
            instruction=instruction
            or "Convert the natural language query to SQL and return JSON.",
            template="sql_writer",
            examples=examples,
            schema=schema,
            **kwargs,
        )
