"""Tests for prompt builder."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from promptify.core.exceptions import TemplateNotFoundError
from promptify.prompts.builder import PromptBuilder


class SampleSchema(BaseModel):
    answer: str
    confidence: float


class TestPromptBuilder:
    def test_no_template_basic(self):
        builder = PromptBuilder()
        messages = builder.build(
            instruction="You are a helpful assistant.",
            text_input="What is 2+2?",
        )
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert "helpful assistant" in messages[0]["content"]
        assert messages[1]["role"] == "user"
        assert "2+2" in messages[1]["content"]

    def test_no_template_with_domain(self):
        builder = PromptBuilder()
        messages = builder.build(
            instruction="You are an expert.",
            text_input="Analyze this.",
            domain="medical",
        )
        assert "Domain: medical" in messages[1]["content"]

    def test_no_template_with_labels(self):
        builder = PromptBuilder()
        messages = builder.build(
            instruction="Classify.",
            text_input="Great!",
            labels=["positive", "negative"],
        )
        assert "positive" in messages[1]["content"]

    def test_no_template_with_examples(self):
        builder = PromptBuilder()
        examples = [("input1", "output1"), ("input2", "output2")]
        messages = builder.build(
            instruction="Do task.",
            text_input="test",
            examples=examples,
        )
        # system + 2 example pairs + user = 6 messages
        assert len(messages) == 6
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "input1"
        assert messages[2]["role"] == "assistant"
        assert messages[2]["content"] == "output1"

    def test_no_template_with_schema(self):
        builder = PromptBuilder()
        messages = builder.build(
            instruction="Answer questions.",
            text_input="What?",
            output_schema=SampleSchema,
        )
        assert "json" in messages[0]["content"].lower() or "JSON" in messages[0]["content"]

    def test_template_ner(self):
        builder = PromptBuilder(template="ner")
        messages = builder.build(
            instruction="Extract entities.",
            text_input="John works at Google.",
            domain="general",
            labels=["PERSON", "ORG"],
        )
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert "John works at Google" in messages[1]["content"]

    def test_template_classify_multiclass(self):
        builder = PromptBuilder(template="classify_multiclass")
        messages = builder.build(
            instruction="Classify text.",
            text_input="Great product!",
            labels=["positive", "negative", "neutral"],
        )
        assert "positive" in messages[1]["content"]

    def test_template_not_found(self):
        with pytest.raises(TemplateNotFoundError):
            PromptBuilder(template="nonexistent_template_xyz")

    def test_template_qa(self):
        builder = PromptBuilder(template="qa")
        messages = builder.build(
            instruction="Answer the question.",
            text_input="Einstein was born in Ulm.",
            question="Where was Einstein born?",
        )
        assert "Einstein" in messages[1]["content"]
        assert "Ulm" in messages[1]["content"]
