"""Tests for QA task."""

from __future__ import annotations

import json

import pytest

from promptify.schemas.qa import Answer
from promptify.tasks.qa import QA
from tests.conftest import MockLLMEngine


class TestQA:
    def test_qa_basic(self):
        response = json.dumps({
            "answer": "Ulm",
            "evidence": "Einstein was born in Ulm",
            "confidence": 0.95,
        })
        qa = QA(model="gpt-4o-mini")
        qa.engine = MockLLMEngine(response_text=response)

        result = qa("Einstein was born in Ulm in 1879.", question="Where was Einstein born?")
        assert isinstance(result, Answer)
        assert result.answer == "Ulm"
        assert result.confidence == 0.95

    def test_qa_with_domain(self):
        response = json.dumps({"answer": "Ibuprofen"})
        qa = QA(model="gpt-4o-mini", domain="medical")
        qa.engine = MockLLMEngine(response_text=response)

        result = qa("Take ibuprofen for pain.", question="What medication?")
        assert result.answer == "Ibuprofen"

    @pytest.mark.asyncio
    async def test_qa_async(self):
        response = json.dumps({"answer": "42"})
        qa = QA(model="gpt-4o-mini")
        qa.engine = MockLLMEngine(response_text=response)

        result = await qa.acall("The answer is 42.", question="What is the answer?")
        assert result.answer == "42"
