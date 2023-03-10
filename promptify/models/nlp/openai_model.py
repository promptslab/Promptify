import re
from typing import Iterable, List, Optional, Union

import openai

from .model import Model
from .utils import get_encoder

GPT_MODEL_TOKENS = {
    r'gpt-3.5-turbo((?=-).*)?': 4096,
    r'.*-davinci-003':          4000,
    r'.*-curie-001':            2048,
    r'.*-babbage-001':          2048,
    r'.*-ada-001':              2048,
}


class OpenAI(Model):
    name = "OpenAI"
    description = "OpenAI API for text completion using various models"

    def __init__(self, api_key: str, model: str = "text-davinci-003"):
        self._api_key = api_key
        self.model = model
        self._openai = openai
        self._openai.api_key = self._api_key
        self.supported_models = (
            "gpt-3.5-turbo",
            "text-davinci-003",
            "text-curie-001",
            "text-babbage-001",
            "text-ada-001",
        )
        self.encoder = get_encoder()
        assert self.model in self.list_models(), "model not supported"

    def list_models(self):
        ## get all models for OpenAI API
        list_of_models = [model_.id for model_ in self._openai.Model.list()["data"]]
        ## compare with supported models and return model_list
        models = []
        for model_ in self.supported_models:
            if model_ in list_of_models:
                models.append(model_)
        return models

    def run(
        self,
        prompts: List[str],
        model_name: str = "text-davinci-003",
        suffix: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 0,
        top_p: float = 1,
        stop: Union[str, Iterable[str], None] = None,
        presence_penalty: float = 0,
        frequency_penalty: float = 0,
    ):
        """
        prompts: The prompt(s) to generate completions for, encoded as a string, array of strings, array of tokens, or array of token arrays.
        max_tokens: The maximum number of tokens to generate in the completion.
                    The token count of your prompt plus max_tokens cannot exceed the model's context length. Most models have a context length of 2048 tokens (except for the newest models, which support 4096).
        temperature: What sampling temperature to use. Higher values means the model will take more risks. Try 0.9 for more creative applications, and 0 (argmax sampling) for ones with a well-defined answer.
                    We generally recommend altering this or top_p but not both.
        top_p: An alternative to sampling with temperature, called nucleus sampling, where the model considers the results of the tokens with top_p probability mass. So 0.1 means only the tokens comprising the top 10% probability mass are considered.
                We generally recommend altering this or temperature but not both.
        stop: Up to 4 sequences where the API will stop generating further tokens. The returned text will not contain the stop sequence.
        presence_penalty: Number between -2.0 and 2.0. Positive values penalize new tokens based on whether they appear in the text so far, increasing the model's likelihood to talk about new topics.
        frequency_penalty: Number between -2.0 and 2.0. Positive values penalize new tokens based on their existing frequency in the text so far, decreasing the model's likelihood to repeat the same line verbatim.
        """

        result = []

        if model_name:
            self.model = model_name

        for prompt in prompts:
            # Automatically calculate max output tokens if not specified
            if not max_tokens:
                prompt_tokens = 0
                if self.model == "gpt-3.5-turbo":
                    text_to_encode = ""
                    text_to_encode = str(prompt)
                    prompt_tokens = len(self.encoder.encode(text_to_encode))
                else:
                    prompt_tokens = len(self.encoder.encode(prompt))
                model_max_tokens = next((v for k, v in GPT_MODEL_TOKENS.items() if re.fullmatch(k, self.model.lower())))
                max_tokens = model_max_tokens - prompt_tokens

            data = {}
            if self.model.startswith("gpt-3.5-turbo"):
                response = self._openai.ChatCompletion.create(
                    model=self.model,
                    messages=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    stop=stop,
                    presence_penalty=presence_penalty,
                    frequency_penalty=frequency_penalty,
                )
                data["text"] = response["choices"][0]["message"]["content"]
                data["role"] = response["choices"][0]["message"]["role"]

            else:
                response = self._openai.Completion.create(
                    model=self.model,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    suffix=suffix,
                    temperature=temperature,
                    top_p=top_p,
                    stop=stop,
                    presence_penalty=presence_penalty,
                    frequency_penalty=frequency_penalty,
                )
                data["text"] = response["choices"][0]["text"]
            data.update(response["usage"])
            result.append(data)

        return result
