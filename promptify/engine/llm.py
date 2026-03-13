"""LLM engine wrapping LiteLLM for universal model access."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type

import litellm
from pydantic import BaseModel
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from promptify.core.config import ModelConfig
from promptify.core.exceptions import (
    ModelAuthenticationError,
    ModelConnectionError,
    ModelRateLimitError,
    ModelResponseError,
)

logger = logging.getLogger("promptify")


@dataclass
class LLMResponse:
    """Response from LLM engine."""

    text: str
    parsed: Optional[BaseModel] = None
    raw_response: Any = None
    usage: Dict[str, int] = field(default_factory=dict)
    model: str = ""
    cost: float = 0.0


class LLMEngine:
    """Universal LLM engine backed by LiteLLM."""

    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        litellm.drop_params = True

    def _build_params(
        self,
        messages: List[Dict[str, str]],
        output_schema: Optional[Type[BaseModel]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "presence_penalty": self.config.presence_penalty,
            "frequency_penalty": self.config.frequency_penalty,
        }
        if self.config.api_key:
            params["api_key"] = self.config.api_key
        if self.config.max_tokens:
            params["max_tokens"] = self.config.max_tokens
        if self.config.stop:
            params["stop"] = self.config.stop
        if self.config.timeout:
            params["timeout"] = self.config.timeout

        if output_schema is not None:
            params["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": output_schema.__name__,
                    "schema": output_schema.model_json_schema(),
                    "strict": False,
                },
            }

        params.update(self.config.extra_params)
        params.update(kwargs)
        return params

    def _map_exception(self, exc: Exception) -> Exception:
        exc_name = type(exc).__name__
        exc_str = str(exc).lower()
        if "auth" in exc_str or "api key" in exc_str or "401" in exc_str:
            return ModelAuthenticationError(str(exc))
        if "rate" in exc_str or "429" in exc_str:
            return ModelRateLimitError(str(exc))
        if "connect" in exc_str or "timeout" in exc_str:
            return ModelConnectionError(str(exc))
        return ModelResponseError(str(exc))

    def _parse_response(
        self,
        response: Any,
        output_schema: Optional[Type[BaseModel]] = None,
    ) -> LLMResponse:
        choice = response.choices[0]
        text = choice.message.content or ""

        usage = {}
        if hasattr(response, "usage") and response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens or 0,
                "completion_tokens": response.usage.completion_tokens or 0,
                "total_tokens": response.usage.total_tokens or 0,
            }

        cost = 0.0
        try:
            cost = litellm.completion_cost(completion_response=response)
        except Exception:
            pass

        parsed = None
        if output_schema and text:
            try:
                data = json.loads(text)
                parsed = output_schema.model_validate(data)
            except Exception:
                logger.debug("Structured parse failed, raw text available in response")

        return LLMResponse(
            text=text,
            parsed=parsed,
            raw_response=response,
            usage=usage,
            model=response.model or self.config.model,
            cost=cost,
        )

    def complete(
        self,
        messages: List[Dict[str, str]],
        output_schema: Optional[Type[BaseModel]] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Synchronous completion."""
        params = self._build_params(messages, output_schema, **kwargs)

        @retry(
            retry=retry_if_exception_type(
                (ModelConnectionError, ModelRateLimitError)
            ),
            wait=wait_exponential(multiplier=1, min=1, max=60),
            stop=stop_after_attempt(self.config.max_retries),
            reraise=True,
        )
        def _call() -> LLMResponse:
            try:
                response = litellm.completion(**params)
                return self._parse_response(response, output_schema)
            except Exception as exc:
                mapped = self._map_exception(exc)
                if isinstance(mapped, (ModelConnectionError, ModelRateLimitError)):
                    raise mapped from exc
                raise mapped from exc

        return _call()

    async def acomplete(
        self,
        messages: List[Dict[str, str]],
        output_schema: Optional[Type[BaseModel]] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Async completion."""
        params = self._build_params(messages, output_schema, **kwargs)
        try:
            response = await litellm.acompletion(**params)
            return self._parse_response(response, output_schema)
        except Exception as exc:
            raise self._map_exception(exc) from exc
