"""Tests for custom Task factory."""

from __future__ import annotations

import json
from typing import List

import pytest
from pydantic import BaseModel

from promptify.tasks.base import Task
from tests.conftest import MockLLMEngine


class MovieReview(BaseModel):
    sentiment: str
    rating: float
    key_themes: List[str]


class TestCustomTask:
    def test_custom_schema(self):
        response = json.dumps({
            "sentiment": "mostly positive",
            "rating": 7.5,
            "key_themes": ["visuals", "pacing"],
        })
        task = Task(
            model="gpt-4o",
            output_schema=MovieReview,
            instruction="Analyze this movie review.",
        )
        task.engine = MockLLMEngine(response_text=response)

        result = task("Nolan's best work. Stunning visuals but the plot drags.")
        assert isinstance(result, MovieReview)
        assert result.sentiment == "mostly positive"
        assert result.rating == 7.5
        assert "visuals" in result.key_themes

    def test_custom_task_with_examples(self):
        response = json.dumps({
            "sentiment": "negative",
            "rating": 3.0,
            "key_themes": ["acting"],
        })
        examples = [
            ("Great movie!", '{"sentiment": "positive", "rating": 9.0, "key_themes": ["all"]}'),
        ]
        task = Task(
            model="gpt-4o",
            output_schema=MovieReview,
            instruction="Analyze this movie review.",
            examples=examples,
        )
        task.engine = MockLLMEngine(response_text=response)

        result = task("Terrible acting.")
        assert isinstance(result, MovieReview)
        assert result.rating == 3.0

    @pytest.mark.asyncio
    async def test_custom_task_async(self):
        response = json.dumps({
            "sentiment": "positive",
            "rating": 8.0,
            "key_themes": ["story"],
        })
        task = Task(
            model="gpt-4o",
            output_schema=MovieReview,
            instruction="Analyze this movie review.",
        )
        task.engine = MockLLMEngine(response_text=response)

        result = await task.acall("Great story!")
        assert isinstance(result, MovieReview)

    def test_custom_task_batch(self):
        response = json.dumps({
            "sentiment": "neutral",
            "rating": 5.0,
            "key_themes": ["average"],
        })
        task = Task(
            model="gpt-4o",
            output_schema=MovieReview,
            instruction="Analyze this movie review.",
        )
        task.engine = MockLLMEngine(response_text=response)

        results = task.batch(["text1", "text2"], max_concurrent=2)
        assert len(results) == 2
        assert all(isinstance(r, MovieReview) for r in results)
