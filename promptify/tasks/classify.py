"""Classification task — binary, multiclass, and multilabel."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Type

from pydantic import BaseModel

from promptify.schemas.classify import Classification, MultiLabelResult
from promptify.tasks.base import BaseTask

_DEFAULT_INSTRUCTION = (
    "You are a text classification system. "
    "Classify the given text and return structured JSON."
)


class Classify(BaseTask):
    """Text classification (binary, multiclass, or multilabel).

    Example
    -------
    >>> clf = Classify(model="gpt-4o-mini", labels=["positive", "negative", "neutral"])
    >>> result = clf("Amazing product!")
    >>> result.label
    'positive'
    """

    def __init__(
        self,
        model: str,
        labels: List[str],
        multi_label: bool = False,
        domain: Optional[str] = None,
        examples: Optional[List[Tuple[str, str]]] = None,
        instruction: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        self.multi_label = multi_label

        # Pick the right template and schema
        if multi_label:
            template = "classify_multilabel"
            schema: Type[BaseModel] = MultiLabelResult
        elif len(labels) == 2:
            template = "classify_binary"
            kwargs["label_0"] = labels[0]
            kwargs["label_1"] = labels[1]
            schema = Classification
        else:
            template = "classify_multiclass"
            schema = Classification

        super().__init__(
            model=model,
            output_schema=schema,
            instruction=instruction or _DEFAULT_INSTRUCTION,
            template=template,
            domain=domain,
            labels=labels,
            examples=examples,
            **kwargs,
        )
