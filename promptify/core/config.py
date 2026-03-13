"""Pydantic configuration models."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator


class ModelConfig(BaseModel):
    """Configuration for the LLM engine."""

    model: str = Field(description="Model identifier (e.g. 'gpt-4o-mini', 'claude-sonnet-4-20250514')")
    api_key: Optional[str] = Field(default=None, description="API key (falls back to env var)")
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    max_tokens: Optional[int] = Field(default=None, gt=0)
    stop: Optional[Union[str, List[str]]] = None
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    timeout: Optional[float] = Field(default=None, gt=0)
    max_retries: int = Field(default=3, ge=0, le=20)
    extra_params: Dict[str, Any] = Field(default_factory=dict)

    model_config = {"frozen": False}


class CacheConfig(BaseModel):
    """Configuration for response caching."""

    enabled: bool = True
    backend: Literal["memory", "disk", "redis"] = "memory"
    maxsize: int = Field(default=128, gt=0)
    ttl: Optional[int] = Field(default=3600, gt=0, description="TTL in seconds")
    redis_url: Optional[str] = None

    @field_validator("redis_url")
    @classmethod
    def validate_redis_url(cls, v: Optional[str], info: Any) -> Optional[str]:
        if info.data.get("backend") == "redis" and not v:
            raise ValueError("redis_url is required when backend is 'redis'")
        return v
