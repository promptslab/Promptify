"""Base task abstraction — the core of Promptify v3."""

from __future__ import annotations

import asyncio
import logging
from abc import ABC
from typing import Any, Dict, List, Optional, Tuple, Type

from pydantic import BaseModel

from promptify.core.config import ModelConfig
from promptify.engine.cost import track_cost
from promptify.engine.llm import LLMEngine
from promptify.parser.parser import Parser
from promptify.prompts.builder import PromptBuilder

logger = logging.getLogger("promptify")


class BaseTask(ABC):
    """Abstract base for all NLP tasks.

    Subclasses set default output_schema, instruction, and template.
    """

    def __init__(
        self,
        model: str,
        output_schema: Type[BaseModel],
        instruction: str,
        template: Optional[str] = None,
        domain: Optional[str] = None,
        labels: Optional[List[str]] = None,
        examples: Optional[List[Tuple[str, str]]] = None,
        api_key: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        model_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k
            in {
                "temperature",
                "top_p",
                "max_tokens",
                "stop",
                "presence_penalty",
                "frequency_penalty",
                "timeout",
                "max_retries",
            }
        }
        self.engine = LLMEngine(ModelConfig(model=model, api_key=api_key, **model_kwargs))
        self.output_schema = output_schema
        self.instruction = instruction
        self.domain = domain
        self.labels = labels
        self.examples = examples
        self.prompt_builder = PromptBuilder(template=template)
        self.parser = Parser()
        self._extra_kwargs = {
            k: v for k, v in kwargs.items() if k not in model_kwargs
        }

    def _build_messages(self, text: str, **kwargs: Any) -> List[Dict[str, str]]:
        """Build prompt messages for this task."""
        merged = {**self._extra_kwargs, **kwargs}
        return self.prompt_builder.build(
            instruction=self.instruction,
            text_input=text,
            domain=self.domain,
            labels=self.labels,
            examples=self.examples,
            output_schema=self.output_schema,
            **merged,
        )

    def __call__(self, text: str, **kwargs: Any) -> BaseModel:
        """Synchronous execution."""
        messages = self._build_messages(text, **kwargs)
        response = self.engine.complete(messages, output_schema=self.output_schema)
        track_cost(response.cost, response.usage)
        if response.parsed:
            return response.parsed
        return self.parser.parse(response.text, self.output_schema)

    async def acall(self, text: str, **kwargs: Any) -> BaseModel:
        """Async execution."""
        messages = self._build_messages(text, **kwargs)
        response = await self.engine.acomplete(messages, output_schema=self.output_schema)
        track_cost(response.cost, response.usage)
        if response.parsed:
            return response.parsed
        return self.parser.parse(response.text, self.output_schema)

    def batch(
        self, texts: List[str], max_concurrent: int = 5, **kwargs: Any
    ) -> List[BaseModel]:
        """Batch processing with async concurrency under the hood."""

        async def _run() -> List[BaseModel]:
            semaphore = asyncio.Semaphore(max_concurrent)

            async def _process(text: str) -> BaseModel:
                async with semaphore:
                    return await self.acall(text, **kwargs)

            return await asyncio.gather(*[_process(t) for t in texts])

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # Already in an async context — create a new thread
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(lambda: asyncio.run(_run())).result()
        else:
            return asyncio.run(_run())


class Task(BaseTask):
    """Generic custom task factory — use any Pydantic output schema.

    Example
    -------
    >>> from pydantic import BaseModel
    >>> class MovieReview(BaseModel):
    ...     sentiment: str
    ...     rating: float
    >>> task = Task(model="gpt-4o", output_schema=MovieReview,
    ...             instruction="Analyze this movie review.")
    >>> review = task("Great movie!")
    """

    def __init__(
        self,
        model: str,
        output_schema: Type[BaseModel],
        instruction: str,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model=model,
            output_schema=output_schema,
            instruction=instruction,
            **kwargs,
        )
