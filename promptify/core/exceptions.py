"""Promptify exception hierarchy."""


class PromptifyError(Exception):
    """Base exception for all Promptify errors."""


class ConfigurationError(PromptifyError):
    """Invalid configuration."""


class ModelError(PromptifyError):
    """Base for model-related errors."""


class ModelConnectionError(ModelError):
    """Failed to connect to model provider."""


class ModelAuthenticationError(ModelError):
    """Invalid API key or authentication failure."""


class ModelRateLimitError(ModelError):
    """Rate limit exceeded."""


class ModelResponseError(ModelError):
    """Invalid or unexpected model response."""


class TemplateError(PromptifyError):
    """Base for template-related errors."""


class TemplateNotFoundError(TemplateError):
    """Template file not found."""


class TemplateMissingVariableError(TemplateError):
    """Required template variable not provided."""


class ParserError(PromptifyError):
    """Failed to parse model output."""


class PipelineError(PromptifyError):
    """Error in task pipeline execution."""


class EvaluationError(PromptifyError):
    """Error during evaluation."""
