from pathlib import Path
from typing import Any, Dict, Optional
from promptify.prompter.conversation_logger import *
from promptify.utils.data_utils import *
from promptify.prompter.prompt_cache import PromptCache


class Pipeline:
    def __init__(self, prompter, model, **kwargs):
        self.prompter = prompter
        self.model = model
        self.max_completion_length: int = kwargs.get("max_completion_length", 20)
        self.cache_prompt = kwargs.get("cache_prompt", True)
        self.cache_size = kwargs.get("cache_size", 200)
        self.prompt_cache = PromptCache(self.cache_size)
        self.conversation_path = Path.cwd()

        self.model_args_count = self.model.run.__code__.co_argcount
        self.model_variables = self.model.run.__code__.co_varnames[
            1 : self.model_args_count
        ]

        self.conversation_path = os.getcwd()
        self.model_dict = {
            key: value
            for key, value in model.__dict__.items()
            if is_string_or_digit(value)
        }
        self.logger = ConversationLogger(self.conversation_path, self.model_dict)

    def fit(self, text_input: str, **kwargs) -> Any:
        """
        Processes an input text through the pipeline: generates a prompt, gets a response from the model,
        caches the response, logs the conversation, and returns the output.
        """

        try:
            template, variables_dict = self.prompter.generate(text_input, **kwargs)
        except ValueError as e:
            print(f"Error in generating prompt: {e}")
            return None

        if kwargs.get("verbose", False):
            print(template)

        if self.cache_prompt:
            output = self.prompt_cache.get(template)
            if output:
                return output

        try:
            response = self.model.execute_with_retry(prompts=[template])
        except Exception as e:
            print(f"Error in model execution: {e}")
            return None

        outputs = [
            self.model.model_output(
                output, max_completion_length=self.max_completion_length
            )
            for output in response
        ]

        if self.cache_prompt:
            self.prompt_cache.add(template, outputs)

        message = create_message(
            template,
            variables_dict,
            outputs[0]["text"],
            outputs[0]["parsed"]["data"]["completion"],
        )
        self.logger.add_message(message)
        return outputs
