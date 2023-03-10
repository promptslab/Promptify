import os
from typing import List

from jinja2 import Environment, FileSystemLoader, meta

dir_path = os.path.dirname(os.path.realpath(__file__))
templates_dir = os.path.join(dir_path, "templates")


class Prompter:
    """
    A class to prompt user inputs by rendering a template with specified model and templates path.

    Parameters
    ----------
    model : object
        A model that has a run method taking variable inputs.
    templates_path : str, optional
        Path to the directory containing the templates.
    allowed_missing_variables : list, optional
        A list of allowed missing variables in the rendered template.

    Attributes
    ----------
    environment : jinja2.Environment
        An instance of Jinja2 Environment.
    model : object
        The specified model to be used for prompting.
    allowed_missing_variables : list
        A list of allowed missing variables in the rendered template.
    model_args_count : int
        Number of arguments of the model's run method.
    model_variables : list
        A list of variable names in the model's run method.
    """
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
        """
        Return a list of templates in the specified templates directory.

        Returns
        -------
        List[str]
            A list of template names.

        """
        return self.environment.list_templates()

    def get_template_variables(self, template_name: str) -> List[str]:
        """
        Return a list of undeclared variables in the specified template.

        Parameters
        ----------
        template_name : str
            Name of the template to analyze.

        Returns
        -------
        List[str]
            A list of variable names used in the specified template but not defined.

        """
        template_source = self.environment.loader.get_source(self.environment, template_name)
        parsed_content = self.environment.parse(template_source)
        undeclared_variables = meta.find_undeclared_variables(parsed_content)
        return undeclared_variables
    
    
    def generate_prompt(self, template_name: str, **kwargs) -> str:
        """
        Generate a prompt from the specified template.

        Parameters
        ----------
        template_name : str
            Name of the template to use.
        **kwargs : dict
            Keyword arguments to use as variables in the template.

        Returns
        -------
        str
            A string containing the generated prompt.

        Raises
        ------
        AssertionError
            If a required variable is missing in the template.

        """
        variables = self.get_template_variables(template_name)
        variables_missing = []
        for variable in variables:
            if variable not in kwargs and variable not in self.allowed_missing_variables:
                variables_missing.append(variable)
        assert len(variables_missing) == 0, f"Missing required variables in template {variables_missing}"
        template = self.environment.get_template(template_name)
        prompt = template.render(**kwargs).strip()
        return prompt

    def fit(self, template_name: str, **kwargs):
        """
        Train the model on a generated prompt.

        Parameters
        ----------
        template_name : str
            Name of the template to use for generating the prompt.
        **kwargs : dict
            Keyword arguments to use as variables in the template and model.

        Returns
        -------
        str
            The model's output for the generated prompt.

        """
        prompt_variables = []
        if template_name == "bypass":
            pass
        else:
            prompt_variables = self.get_template_variables(template_name)
        
        prompt_kwargs = {}
        model_kwargs = {}
        for variable in kwargs:
            if variable in prompt_variables:
                prompt_kwargs[variable] = kwargs[variable]
            elif variable in self.model_variables:
                model_kwargs[variable] = kwargs[variable]

        prompt= ""

        if "prompt" in kwargs:
            prompt =  kwargs['prompt']
        else:
            prompt = self.generate_prompt(template_name, **prompt_kwargs)

        output = self.model.run(prompts=[prompt], **model_kwargs)
        return output[0]
