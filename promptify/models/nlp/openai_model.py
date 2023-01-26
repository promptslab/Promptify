from typing import List, Union

import openai

from .model import Model
from .utils import get_encoder


class OpenAI(Model):

    name = "OpenAI"
    description = "OpenAI API for text completion using various models"

    def __init__(self, api_key: str):
        self._api_key = api_key
        self._openai = openai
        self._openai.api_key = self._api_key
        self.supported_models = ["text-davinci-003", "text-curie-001", "text-babbage-001", "text-ada-001"]
        self.encoder = get_encoder()

    def list_models(self):
        ## get all models for OpenAI API
        list_of_models = [model.id for model in self._openai.Model.list()["data"]]
        ## compare with supported models and return model_list
        models = []
        for model in self.supported_models:
            if model in list_of_models:
                models.append(model)
        return models

    def run(
        self,
        prompts: List[str],
        model_name: str = "text-davinci-003",
        temperature: float = 0.7,
        max_tokens: int = 4000,
        top_p: float = 0.1,
        frequency_penalty: float = 0,
        presence_penalty: float = 0,
        stop: Union[str, None] = None,
    ):
        """
        prompts: The prompt(s) to generate completions for, encoded as a string, array of strings, array of tokens, or array of token arrays.
        model_name: ID of the model to use. You can use the `list_models` function to see all of your available models
        temperature: What sampling temperature to use. Higher values means the model will take more risks. Try 0.9 for more creative applications, and 0 (argmax sampling) for ones with a well-defined answer.
                    We generally recommend altering this or top_p but not both.
        max_tokens: The maximum number of tokens to generate in the completion.
                    The token count of your prompt plus max_tokens cannot exceed the model's context length. Most models have a context length of 2048 tokens (except for the newest models, which support 4096).
        top_p: An alternative to sampling with temperature, called nucleus sampling, where the model considers the results of the tokens with top_p probability mass. So 0.1 means only the tokens comprising the top 10% probability mass are considered.
                We generally recommend altering this or temperature but not both.
        frequency_penalty: Number between -2.0 and 2.0. Positive values penalize new tokens based on their existing frequency in the text so far, decreasing the model's likelihood to repeat the same line verbatim.
        presence_penalty: Number between -2.0 and 2.0. Positive values penalize new tokens based on whether they appear in the text so far, increasing the model's likelihood to talk about new topics.
        stop: Up to 4 sequences where the API will stop generating further tokens. The returned text will not contain the stop sequence.
        """
        assert model_name in self.list_models(), "model not supported"
        result = []
        for prompt in prompts:
            len_prompt_tokens = len(self.encoder.encode(prompt))
            max_tokens_prompt = max_tokens - len_prompt_tokens
            response = self._openai.Completion.create(
                model=model_name,
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens_prompt,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                stop=stop,
            )
            data = {}
            data.update(response["usage"])
            data["text"] = response["choices"][0]["text"]
            result.append(data)

        return result
