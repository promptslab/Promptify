"""Tests for LLM engine."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from promptify.core.config import ModelConfig
from promptify.engine.llm import LLMEngine, LLMResponse


class SampleOutput(BaseModel):
    answer: str
    confidence: float


class TestLLMResponse:
    def test_defaults(self):
        r = LLMResponse(text="hello")
        assert r.text == "hello"
        assert r.parsed is None
        assert r.usage == {}
        assert r.cost == 0.0

    def test_with_parsed(self):
        parsed = SampleOutput(answer="yes", confidence=0.9)
        r = LLMResponse(text='{"answer": "yes"}', parsed=parsed)
        assert r.parsed is not None
        assert r.parsed.answer == "yes"


class TestLLMEngine:
    def test_build_params_basic(self):
        config = ModelConfig(model="gpt-4o-mini")
        engine = LLMEngine(config)
        params = engine._build_params(
            messages=[{"role": "user", "content": "hi"}]
        )
        assert params["model"] == "gpt-4o-mini"
        assert params["temperature"] == 0.0
        assert "api_key" not in params

    def test_build_params_with_schema(self):
        config = ModelConfig(model="gpt-4o-mini")
        engine = LLMEngine(config)
        params = engine._build_params(
            messages=[{"role": "user", "content": "hi"}],
            output_schema=SampleOutput,
        )
        assert params["response_format"]["type"] == "json_schema"
        assert params["response_format"]["json_schema"]["name"] == "SampleOutput"

    def test_build_params_with_api_key(self):
        config = ModelConfig(model="gpt-4o", api_key="sk-test123")
        engine = LLMEngine(config)
        params = engine._build_params(
            messages=[{"role": "user", "content": "hi"}]
        )
        assert params["api_key"] == "sk-test123"

    def test_parse_response(self):
        config = ModelConfig(model="gpt-4o-mini")
        engine = LLMEngine(config)

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"answer": "yes", "confidence": 0.9}'
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_response.usage.total_tokens = 15
        mock_response.model = "gpt-4o-mini"

        result = engine._parse_response(mock_response, SampleOutput)
        assert result.parsed is not None
        assert result.parsed.answer == "yes"
        assert result.usage["total_tokens"] == 15

    def test_map_exception_auth(self):
        config = ModelConfig(model="gpt-4o-mini")
        engine = LLMEngine(config)
        from promptify.core.exceptions import ModelAuthenticationError

        exc = engine._map_exception(Exception("Invalid API key"))
        assert isinstance(exc, ModelAuthenticationError)

    def test_map_exception_rate_limit(self):
        config = ModelConfig(model="gpt-4o-mini")
        engine = LLMEngine(config)
        from promptify.core.exceptions import ModelRateLimitError

        exc = engine._map_exception(Exception("Rate limit exceeded 429"))
        assert isinstance(exc, ModelRateLimitError)
