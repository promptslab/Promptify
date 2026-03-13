"""Tests for NER task."""

from __future__ import annotations

import json

import pytest

from promptify.schemas.ner import Entity, NERResult
from promptify.tasks.ner import NER
from tests.conftest import MockLLMEngine


class TestNER:
    def test_ner_with_mock_engine(self, mock_ner_response, sample_medical_text):
        ner = NER(model="gpt-4o-mini", domain="medical")
        ner.engine = MockLLMEngine(response_text=mock_ner_response)

        result = ner(sample_medical_text)
        assert isinstance(result, NERResult)
        assert len(result.entities) == 2
        assert result.entities[0].text == "chronic hip pain"
        assert result.entities[0].label == "CONDITION"
        assert result.entities[1].text == "osteoporosis"

    def test_ner_with_labels(self, mock_ner_response):
        ner = NER(model="gpt-4o-mini", labels=["PERSON", "LOCATION", "ORG"])
        ner.engine = MockLLMEngine(response_text=mock_ner_response)

        result = ner("John works at Google in NYC")
        assert isinstance(result, NERResult)

    def test_ner_with_examples(self, mock_ner_response):
        examples = [
            ("John lives in NYC", '[{"text": "John", "label": "PERSON"}]'),
        ]
        ner = NER(model="gpt-4o-mini", examples=examples)
        ner.engine = MockLLMEngine(response_text=mock_ner_response)

        result = ner("Test text")
        assert isinstance(result, NERResult)

    def test_ner_parsed_response(self):
        parsed = NERResult(entities=[Entity(text="Python", label="LANGUAGE")])
        ner = NER(model="gpt-4o-mini")
        ner.engine = MockLLMEngine(response_text="", parsed=parsed)

        result = ner("Python is great")
        assert result.entities[0].text == "Python"
        assert result.entities[0].label == "LANGUAGE"

    @pytest.mark.asyncio
    async def test_ner_async(self, mock_ner_response, sample_medical_text):
        ner = NER(model="gpt-4o-mini")
        ner.engine = MockLLMEngine(response_text=mock_ner_response)

        result = await ner.acall(sample_medical_text)
        assert isinstance(result, NERResult)
        assert len(result.entities) == 2

    def test_ner_batch(self, mock_ner_response):
        ner = NER(model="gpt-4o-mini")
        ner.engine = MockLLMEngine(response_text=mock_ner_response)

        results = ner.batch(["text1", "text2", "text3"], max_concurrent=2)
        assert len(results) == 3
        assert all(isinstance(r, NERResult) for r in results)
