"""Shared test fixtures."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

from promptify.engine.llm import LLMEngine, LLMResponse
from promptify.schemas.classify import Classification
from promptify.schemas.ner import Entity, NERResult


class MockLLMEngine(LLMEngine):
    """Mock engine that returns pre-configured responses."""

    def __init__(self, response_text: str = "", parsed: Optional[BaseModel] = None):
        # Don't call super().__init__ — we don't need a real config
        self._response_text = response_text
        self._parsed = parsed

    def complete(self, messages, output_schema=None, **kwargs):
        parsed = self._parsed
        if parsed is None and output_schema and self._response_text:
            try:
                data = json.loads(self._response_text)
                parsed = output_schema.model_validate(data)
            except Exception:
                pass
        return LLMResponse(
            text=self._response_text,
            parsed=parsed,
            raw_response=None,
            usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
            model="mock-model",
            cost=0.001,
        )

    async def acomplete(self, messages, output_schema=None, **kwargs):
        return self.complete(messages, output_schema, **kwargs)


@pytest.fixture
def mock_ner_response():
    """Pre-built NER response."""
    return json.dumps({
        "entities": [
            {"text": "chronic hip pain", "label": "CONDITION"},
            {"text": "osteoporosis", "label": "CONDITION"},
        ]
    })


@pytest.fixture
def mock_classify_response():
    """Pre-built classification response."""
    return json.dumps({"label": "positive", "confidence": 0.95})


@pytest.fixture
def sample_medical_text():
    return "The patient has chronic hip pain and osteoporosis"


@pytest.fixture
def sample_sentiment_text():
    return "Amazing product! Best purchase I've ever made."
