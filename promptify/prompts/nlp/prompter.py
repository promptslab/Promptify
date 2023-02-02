import os
from typing import List

from jinja2 import Environment, FileSystemLoader, meta

dir_path = os.path.dirname(os.path.realpath(__file__))
templates_dir = os.path.join(dir_path, "templates")


class Prompter:
    def __init__(
        self,
        model,
        templates_path=templates_dir,
        allowed_missing_variables=["examples", "description", "output_format"],
    ) -> None:
        self.environment = Environment(loader=FileSystemLoader(templates_path))
        self.model = model
        self.allowed_missing_variables = allowed_missing_variables
        self.model_args_count = self.model.run.__code__.co_argcount
        self.model_variables = self.model.run.__code__.co_varnames[1 : self.model_args_count]

    def list_templates(self) -> List[str]:
        return self.environment.list_templates()

    def get_template_variables(self, template_name: str) -> List[str]:
        template_source = self.environment.loader.get_source(self.environment, template_name)
        parsed_content = self.environment.parse(template_source)
        undeclared_variables = meta.find_undeclared_variables(parsed_content)
        return undeclared_variables

    def generate_prompt(self, template_name, **kwargs) -> str:
        variables = self.get_template_variables(template_name)
        variables_missing = []
        for variable in variables:
            if variable not in kwargs and variable not in self.allowed_missing_variables:
                variables_missing.append(variable)
        assert len(variables_missing) == 0, f"Missing required variables in template {variables_missing}"
        template = self.environment.get_template(template_name)
        prompt = template.render(**kwargs).strip()
        return prompt

    def fit(self, template_name, **kwargs):
        prompt_variables = self.get_template_variables(template_name)
        prompt_kwargs = {}
        model_kwargs = {}
        for variable in kwargs:
            if variable in prompt_variables:
                prompt_kwargs[variable] = kwargs[variable]
            elif variable in self.model_variables:
                model_kwargs[variable] = kwargs[variable]
        prompt = self.generate_prompt(template_name, **prompt_kwargs)
        output = self.model.run(prompts=[prompt], **model_kwargs)
        return output[0]
