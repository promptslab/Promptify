import os
import glob
import uuid
from typing import List, Dict, Any, Optional
from jinja2 import Environment, FileSystemLoader, meta


class Prompter:
    def __init__(
        self,
        model,
        template: Optional[str] = None,
        raw_prompt: bool = False,
        allowed_missing_variables: Optional[List[str]] = None,
        default_variable_values: Optional[Dict[str, Any]] = None,
        max_completion_length: int = 20,
        cache_prompt: bool = False,
    ) -> None:
        self.template_path = template
        self.raw_prompt = raw_prompt
        self.model = model
        self.max_completion_length = max_completion_length
        self.cache_prompt = cache_prompt
        self.prompt_cache = {}
        self.loaded_templates = {}

        self.allowed_missing_variables = allowed_missing_variables or [
            "examples",
            "description",
            "output_format",
        ]

        self.default_variable_values = default_variable_values or {}
        self.model_args_count = self.model.run.__code__.co_argcount
        self.model_variables = self.model.run.__code__.co_varnames[
            1 : self.model_args_count
        ]
        self.prompt_variables_map = {}

    def get_available_templates(self, template_path: str) -> Dict[str, str]:
        all_templates = glob.glob(f"{template_path}/*.jinja")
        template_names = [os.path.basename(template) for template in all_templates]
        template_dict = dict(zip(template_names, all_templates))
        return template_dict

    def update_default_variable_values(self, new_defaults: Dict[str, Any]) -> None:
        """Updates the default variable values with the given dictionary."""
        self.default_variable_values.update(new_defaults)

    def load_multiple_templates(self, templates: List):
        template_dict = {}
        for template in templates:
            uuid_key = str(uuid.uuid4())
            name = os.path.basename(template) + "_" + uuid_key
            template_dict[name] = self.load_template(template)
        return template_dict

    def load_template(self, template: str):
        if template in self.loaded_templates:
            return self.loaded_templates[template]

        current_dir = os.path.dirname(os.path.realpath("."))
        templates_dir = os.path.join(current_dir, "codes", "templates")

        default_templates = self.get_available_templates(templates_dir)

        if template in default_templates:
            template_name = template
            template_dir = templates_dir
            environment = Environment(loader=FileSystemLoader(template_dir))
            template_instance = environment.get_template(template)

        else:
            self.verify_template_path(template)
            custom_template_dir, custom_template_name = os.path.split(template)

            template_name = custom_template_name
            template_dir = custom_template_dir
            environment = Environment(loader=FileSystemLoader(template_dir))
            template_instance = environment.get_template(custom_template_name)

        template_data = {
            "template_name": template_name,
            "template_dir": template_dir,
            "environment": environment,
            "template": template_instance,
        }

        self.loaded_templates[template] = template_data
        return self.loaded_templates[template]

    def verify_template_path(self, templates_path: str):
        if not os.path.isfile(templates_path):
            raise ValueError(f"Templates path {templates_path} does not exist")

    def list_templates(self, environment) -> List[str]:
        """Returns a list of available templates."""
        return environment.list_templates()

    def get_multiple_template_variables(self, dict_templates: dict):
        results = {}
        for key, value in dict_templates.items():
            results[key] = self.get_template_variables(
                value["environment"], value["template_name"]
            )
        return results

    def get_template_variables(self, environment, template_name) -> List[str]:
        if template_name in self.prompt_variables_map:
            return self.prompt_variables_map[template_name]
        template_source = environment.loader.get_source(environment, template_name)
        parsed_content = environment.parse(template_source)
        undeclared_variables = meta.find_undeclared_variables(parsed_content)
        self.prompt_variables_map[template_name] = undeclared_variables
        return undeclared_variables

    def generate_prompt(self, text_input, **kwargs) -> str:
        loader = self.load_template(self.template_path)
        variables = self.get_template_variables(
            loader["environment"], loader["template_name"]
        )

        kwargs["text_input"] = text_input
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

        kwargs.update(self.default_variable_values)
        prompt = loader["template"].render(**kwargs).strip()
        return prompt

    def raw_fit(self, prompt: str):
        outputs = [
            self.model.model_output_raw(output)
            for output in self.model.run(prompts=[prompt])
        ]
        return outputs

    def fit(self, text_input, **kwargs):
        if self.raw_prompt:
            return self.raw_fit(text_input)

        if not self.template_path:
            raise ValueError(
                "ReferenceError: template is not defined. Task template from existing templates such as ner.jinja, qa.jinja etc or provide custom jinja template with absolute path"
            )

        prompt = self.generate_prompt(text_input, **kwargs)

        if "verbose" in kwargs:
            if kwargs["verbose"]:
                print(prompt)

        if self.cache_prompt and prompt in self.prompt_cache:
            output = self.prompt_cache[prompt]
            return output
        else:
            response = self.model.execute_with_retry(prompts=[prompt])
            outputs = [
                self.model.model_output(
                    output, max_completion_length=self.max_completion_length
                )
                for output in response
            ]
            if self.cache_prompt:
                self.prompt_cache[prompt] = outputs

        return outputs