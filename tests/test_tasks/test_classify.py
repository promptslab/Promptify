"""Tests for classification task."""

from __future__ import annotations

import json

import pytest

from promptify.schemas.classify import Classification, MultiLabelResult
from promptify.tasks.classify import Classify
from tests.conftest import MockLLMEngine


class TestClassify:
    def test_multiclass(self, mock_classify_response, sample_sentiment_text):
        clf = Classify(model="gpt-4o-mini", labels=["positive", "negative", "neutral"])
        clf.engine = MockLLMEngine(response_text=mock_classify_response)

        result = clf(sample_sentiment_text)
        assert isinstance(result, Classification)
        assert result.label == "positive"
        assert result.confidence == 0.95

    def test_binary(self, mock_classify_response):
        clf = Classify(model="gpt-4o-mini", labels=["spam", "not_spam"])
        clf.engine = MockLLMEngine(response_text=mock_classify_response)

        result = clf("Buy now! Limited time offer!")
        assert isinstance(result, Classification)

    def test_multilabel(self):
        response = json.dumps({
            "labels": [
                {"label": "tech", "confidence": 0.9},
                {"label": "science", "confidence": 0.8},
            ]
        })
        clf = Classify(
            model="gpt-4o-mini",
            labels=["tech", "science", "sports"],
            multi_label=True,
        )
        clf.engine = MockLLMEngine(response_text=response)

        result = clf("AI breakthrough in quantum computing")
        assert isinstance(result, MultiLabelResult)
        assert len(result.labels) == 2
        assert result.labels[0].label == "tech"

    @pytest.mark.asyncio
    async def test_classify_async(self, mock_classify_response):
        clf = Classify(model="gpt-4o-mini", labels=["pos", "neg"])
        clf.engine = MockLLMEngine(response_text=mock_classify_response)

        result = await clf.acall("Great!")
        assert isinstance(result, Classification)
