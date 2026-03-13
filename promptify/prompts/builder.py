"""Prompt builder — constructs OpenAI-format messages from Jinja2 templates."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple, Type

from jinja2 import Environment, FileSystemLoader, Template
from pydantic import BaseModel

from promptify.core.exceptions import TemplateNotFoundError, TemplateMissingVariableError

TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), "templates")


class PromptBuilder:
    """Build chat-format message lists from Jinja2 templates.

    Parameters
    ----------
    template : str or None
        Template name (e.g. "ner") resolved from templates/ directory,
        OR a full path to a custom .jinja file.
        If None, uses a generic instruction-based prompt.
    """

    def __init__(self, template: Optional[str] = None) -> None:
        self._template_name = template
        self._jinja_template: Optional[Template] = None
        self._env: Optional[Environment] = None

        if template is not None:
            self._load_template(template)

    def _load_template(self, template: str) -> None:
        """Load a Jinja2 template by name or path."""
        # Check built-in templates first
        jinja_name = template if template.endswith(".jinja") else f"{template}.jinja"
        builtin_path = os.path.join(TEMPLATES_DIR, jinja_name)

        if os.path.isfile(builtin_path):
            self._env = Environment(loader=FileSystemLoader(TEMPLATES_DIR))
            self._jinja_template = self._env.get_template(jinja_name)
        elif os.path.isfile(template):
            # Custom path
            template_dir, template_file = os.path.split(template)
            self._env = Environment(loader=FileSystemLoader(template_dir))
            self._jinja_template = self._env.get_template(template_file)
        else:
            raise TemplateNotFoundError(f"Template not found: {template}")

    def _render_template(self, **kwargs: Any) -> str:
        """Render the Jinja2 template with given variables."""
        if self._jinja_template is None:
            raise TemplateNotFoundError("No template loaded")
        return self._jinja_template.render(**kwargs).strip()

    def _build_schema_instruction(self, output_schema: Optional[Type[BaseModel]]) -> str:
        """Generate output format instructions from a Pydantic schema."""
        if output_schema is None:
            return ""
        schema = output_schema.model_json_schema()
        fields = []
        for name, prop in schema.get("properties", {}).items():
            field_type = prop.get("type", "any")
            desc = prop.get("description", "")
            fields.append(f"  - {name}: {field_type}" + (f" ({desc})" if desc else ""))
        field_str = "\n".join(fields)
        return (
            f"\n\nRespond with valid JSON matching this schema:\n"
            f"{{\n{field_str}\n}}"
        )

    def build(
        self,
        instruction: str,
        text_input: str,
        domain: Optional[str] = None,
        labels: Optional[List[str]] = None,
        examples: Optional[List[Tuple[str, str]]] = None,
        output_schema: Optional[Type[BaseModel]] = None,
        **kwargs: Any,
    ) -> List[Dict[str, str]]:
        """Build OpenAI-format messages.

        Returns
        -------
        list of dict
            Messages in [{"role": "system", "content": ...}, {"role": "user", "content": ...}] format.
        """
        messages: List[Dict[str, str]] = []

        if self._jinja_template is not None:
            # Template-based: render template as user message
            template_vars: Dict[str, Any] = {
                "text_input": text_input,
                "domain": domain,
                "labels": labels,
                "examples": examples,
                "description": kwargs.get("description"),
                **kwargs,
            }
            rendered = self._render_template(**template_vars)

            system_content = instruction
            schema_hint = self._build_schema_instruction(output_schema)
            if schema_hint:
                system_content += schema_hint

            messages.append({"role": "system", "content": system_content})
            messages.append({"role": "user", "content": rendered})
        else:
            # No template: instruction-based prompt
            system_content = instruction
            schema_hint = self._build_schema_instruction(output_schema)
            if schema_hint:
                system_content += schema_hint

            messages.append({"role": "system", "content": system_content})

            # Inject few-shot examples
            if examples:
                for ex_input, ex_output in examples:
                    messages.append({"role": "user", "content": ex_input})
                    messages.append({"role": "assistant", "content": ex_output})

            # Build user message
            user_parts = []
            if domain:
                user_parts.append(f"Domain: {domain}")
            if labels:
                user_parts.append(f"Labels: {', '.join(labels)}")
            user_parts.append(text_input)
            messages.append({"role": "user", "content": "\n".join(user_parts)})

        return messages
