import pytest
from promptify import Prompter
from promptify import OpenAI
from typing import List, Optional, Union, Dict
import os


class TestPrompter:
    @pytest.fixture
    def model(self):
        model = OpenAI(api_key="", api_wait=1, api_retry=1)
        return model

    def test_custom_template(self, model):

        # replace the template path with own path, this is just for testing
        prompter = Prompter(
            model=model, template="/Users/stoicbatman/Desktop/pytest_project/ner.jinja"
        )
        output = prompter.fit(
            "Elon Reeve Musk FRS is a business. He is the founder of SpaceX; Tesla, Inc.; Twitter, Inc.; Neuralink and OpenAI",
            domain="general",
            labels=None,
        )
        assert isinstance(output, list)
        assert isinstance(output[0]["parsed"], Dict)
        assert isinstance(output[0]["parsed"]["data"]["completion"][0]["T"], str)
        assert isinstance(output[0]["parsed"]["data"]["completion"][0]["E"], str)

    def test_generate_prompt(self, model):
        prompter = Prompter(model=model, template="ner.jinja")
        prompt = prompter.generate_prompt(
            "Elon Reeve Musk FRS is a business. He is the founder of SpaceX; Tesla, Inc.; Twitter, Inc.; Neuralink and OpenAI",
            domain="general",
            labels=None,
        )
        assert isinstance(prompt, str)

    def test_fit(self, model):
        prompter = Prompter(model=model, template="ner.jinja")
        output = prompter.fit(
            "Elon Reeve Musk FRS is a business. He is the founder of SpaceX; Tesla, Inc.; Twitter, Inc.; Neuralink and OpenAI",
            domain="general",
            labels=None,
        )
        assert isinstance(output, list)
        assert isinstance(output[0]["parsed"], Dict)
        assert isinstance(output[0]["parsed"]["data"]["completion"][0]["T"], str)
        assert isinstance(output[0]["parsed"]["data"]["completion"][0]["E"], str)

    def test_raw_fit(self, model):
        prompter = Prompter(model=model, raw_prompt=True)
        output = prompter.fit("quick brown fox jump over")
        assert isinstance(output, list)
        assert isinstance(output[0]["text"], str)

    def test_load_template(self, model):
        prompter = Prompter(model=model)
        template_data = prompter.load_template("ner.jinja")
        assert "template_name" in template_data
        assert "template_dir" in template_data
        assert "environment" in template_data
        assert "template" in template_data

    def test_get_available_templates(self, model):
        prompter = Prompter(model)
        templates_path = os.path.join(
            os.path.dirname(os.path.realpath(".")), "codes", "templates"
        )
        templates = prompter.get_available_templates(templates_path)
        assert isinstance(templates, dict)
        for key, value in templates.items():
            assert key.endswith(".jinja")
            assert value.endswith(".jinja")

    def test_list_templates(self, model):
        prompter = Prompter(model=model, template="ner.jinja")
        loader = prompter.load_template("ner.jinja")
        templates = loader["environment"].list_templates()
        assert isinstance(templates, list)
        assert len(templates) > 0

    def test_template_variables(self, model):
        prompter = Prompter(model=model, template="ner.jinja")
        loader = prompter.load_template("ner.jinja")
        variables = prompter.get_template_variables(
            loader["environment"], loader["template_name"]
        )
        assert isinstance(variables, set)
        assert len(variables) > 0

    def test_update_default_variable_values(self, model):
        prompter = Prompter(model=model, template="ner.jinja")
        new_defaults = {"description": "test description", "domain": "test domain"}
        prompter.update_default_variable_values(new_defaults)
        assert prompter.default_variable_values == new_defaults

    def test_missing_template_path_error(self, model):
        with pytest.raises(ValueError):
            prompter = Prompter(model=model)
            prompter.load_template("non_existent_template.jinja")
