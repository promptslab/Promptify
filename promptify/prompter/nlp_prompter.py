import os
import uuid
from glob import glob
import datetime
from pathlib import Path

from promptify.utils.data_utils import *
from promptify.prompter.template_loader import TemplateLoader

from typing import List, Dict, Any, Optional
from jinja2 import Environment, FileSystemLoader, meta, Template


class Prompter:

    """
    A class to generate and manage prompts.

    """

    def __init__(
        self,
        template,
        from_string = False,
        allowed_missing_variables: Optional[List[str]] = None,
        default_variable_values: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize Prompter with default or user-specified settings.

        Parameters
        ----------
        template : str, optional
            A Jinja2 template to use for generating the prompt. Must be a valid file path.
        allowed_missing_variables : list of str, optional
            A list of variable names that are allowed to be missing from the template. Default is ['examples', 'description', 'output_format'].
        default_variable_values : dict of str: any, optional
            A dictionary mapping variable names to default values to be used in the template.
            If a variable is not found in the input dictionary or in the default values, it will be assumed to be required and an error will be raised. Default is an empty dictionary.
        """

        self.template = template
        self.template_loader = TemplateLoader()
        self.allowed_missing_variables = [
            "examples",
            "description",
            "output_format",
        ]
        self.allowed_missing_variables.extend(allowed_missing_variables or [])
        self.default_variable_values = default_variable_values or {}
        self.from_string = from_string


    def update_default_variable_values(self, new_defaults: Dict[str, Any]) -> None:
        self.default_variable_values.update(new_defaults)

    def generate(self, text_input, model_name, **kwargs) -> str:
        """
        Generates a prompt based on a template and input variables.

        Parameters
        ----------
        text_input : str
            The input text to use in the prompt.
        **kwargs : dict
            Additional variables to be used in the template.

        Returns
        -------
        str
            The generated prompt string.
        """

        loader = self.template_loader.load_template(
            self.template, model_name, self.from_string
        )

        kwargs["text_input"] = text_input

        if loader["environment"]:
            variables = self.template_loader.get_template_variables(
                loader["environment"], loader["template_name"]
            )
            variables_dict = {
                temp_variable_: kwargs.get(temp_variable_, None)
                for temp_variable_ in variables
            }

            variables_missing = [
                variable
                for variable in variables
                if variable not in kwargs
                and variable not in self.allowed_missing_variables
                and variable not in self.default_variable_values
            ]

            if variables_missing:
                raise ValueError(
                    f"Missing required variables in template {', '.join(variables_missing)}"
                )
        else:
            variables_dict = {"data": None}

        kwargs.update(self.default_variable_values)
        prompt = loader["template"].render(**kwargs).strip()

        if kwargs.get("verbose", False):
            print(prompt)

        return prompt, variables_dict
