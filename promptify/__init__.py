"""Promptify v3 — Task-based NLP engine with structured outputs."""

from promptify._version import __version__
from promptify.core.config import ModelConfig
from promptify.core.logging import setup_logging
from promptify.engine.cost import get_cost_summary
from promptify.tasks import (
    NER,
    QA,
    Classify,
    ExtractRelations,
    ExtractTable,
    ExtractTopics,
    GenerateQuestions,
    GenerateSQL,
    NormalizeText,
    Summarize,
    Task,
)

__all__ = [
    "__version__",
    "NER",
    "Classify",
    "QA",
    "Summarize",
    "Task",
    "ExtractRelations",
    "ExtractTable",
    "GenerateQuestions",
    "GenerateSQL",
    "NormalizeText",
    "ExtractTopics",
    "ModelConfig",
    "setup_logging",
    "get_cost_summary",
]
