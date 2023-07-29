from typing import Dict, List, Optional, Tuple, Union
import openai
import json
import tiktoken
from promptify.parser.parser import Parser
from promptify.models.text2text.api.base_model import Model


class OpenAI(Model):
    name = "OpenAI"
    description = "OpenAI API for text completion using various models"

    SUPPORTED_MODELS = {
        "completion_models": set(
            [
                "text-davinci-003",
                "davinci",
                "text-davinci-001",
                "ada",
                "text-curie-001",
                "text-ada-001",
                "text-babbage-001",
                "curie",
                "text-davinci-002",
            ]
        ),
        "chat_models": set(
            [
                "gpt-4-0314",
                "gpt-3.5-turbo-16k-0613",
                "gpt-3.5-turbo-0301",
                "gpt-3.5-turbo-16k",
                "gpt-4",
                "gpt-3.5-turbo",
                "gpt-3.5-turbo-0613",
                "gpt-4-0613",
            ]
        ),
    }

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        top_p: float = 1,
        n: int = 1,
        stop: Optional[Union[str, List[str]]] = None,
        presence_penalty: float = 0,
        frequency_penalty: float = 0,
        logit_bias: Optional[Dict[str, int]] = None,
        request_timeout: Union[float, Tuple[float, float]] = None,
        api_wait=60,
        api_retry=6,
        json_depth_limit: int = 20,
    ):
        super().__init__(api_key, model, api_wait, api_retry)

        self.temperature = temperature
        self.top_p = top_p
        self.n = n
        self.stop = stop
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        self.logit_bias = logit_bias or {}
        self.request_timeout = request_timeout
        self.json_depth_limit = json_depth_limit
        self.set_key(api_key)
        self._verify_model()
        self._initialize_encoder()
        self._initialize_parser()
        self.parameters = self.get_parameters()

    def set_key(self, api_key: str):
        self._openai = openai
        self._openai.api_key = api_key

    def _verify_model(self):
        model_type = (
            "completion_models"
            if self.model in self.SUPPORTED_MODELS["completion_models"]
            else "chat_models"
        )
        if self.model not in self.SUPPORTED_MODELS[model_type]:
            raise ValueError(f"Unsupported model: {self.model}")
        self.model_type = model_type

    def _initialize_encoder(self):
        self.encoder = tiktoken.encoding_for_model(self.model)

    def _initialize_parser(self):
        self.parser = Parser()

    def set_model(self, model: str):
        self.model = model
        self._verify_model()

    def supported_models(self):
        return list(itertools.chain(*self.SUPPORTED_MODELS.values()))

    def get_parameters(self):
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "n": self.n,
            "stop": self.stop,
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
            "logit_bias": self.logit_bias,
            "request_timeout": self.request_timeout,
        }

    def get_description(self):
        return self.description

    def get_endpoint(self):
        model = self._openai.Model.retrieve(self.model)
        return model["id"]

    def run(self, prompt: str):
        if self.model_type == "chat_models":
            return self._chat_api(prompt)
        elif self.model_type == "completion_models":
            return self._completion_api(prompt)


    def _completion_api(self, prompt: str):
        self.parameters["prompt"] = prompt
        self.parameters["max_tokens"] = self._calculate_max_tokens(prompt)
        response = self._openai.Completion.create(
            model=self.model,
            **self.parameters,
        )
        return response

    def _chat_api(self, prompt: str):
        prompt_template = [
            {"role": "system", "content": "you are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]

        self.parameters["max_tokens"] = self._calculate_max_tokens(prompt_template)
        self.parameters["messages"] = prompt_template
        response = self._openai.ChatCompletion.create(
            model=self.model,
            **self.parameters,
        )
        return response

    def _calculate_max_tokens(self, prompt: str) -> int:
        prompt_tokens = len(self.encoder.encode(str(prompt)))
        max_tokens = self._default_max_tokens(self.model) - prompt_tokens
        return max_tokens

    def _default_max_tokens(self, model_name: str) -> int:
        token_dict = {
            "text-babbage-001": 2040,
            "text-ada-001": 2048,
            "ada": 2048,
            "babbage": 2048,
            "text-curie-001": 2048,
            "curie": 2048,
            "davinci": 2048,
            "code-cushman-002": 2048,
            "code-cushman-001": 2048,
            "text-davinci-003": 4000,
            "gpt-3.5-turbo": 4096,
            "gpt-3.5-turbo-0301": 4096,
            "gpt-3.5-turbo-0613": 4096,
            "text-davinci-002": 4096,
            "code-davinci-002": 8000,
            "code-davinci-001": 8000,
            "gpt-4": 8192,
            "gpt-4-0314": 8192,
            "gpt-4-0613": 8192,
            "gpt-3.5-turbo-16k": 16385,
            "gpt-3.5-turbo-16k-0613": 16385,
            "gpt-4-32k": 32768,
            "gpt-4-32k-0314": 32768,
            "gpt-4-32k-0613": 32768,
        }
        return token_dict[model_name]

    def model_output_raw(self, response: Dict) -> Dict:
        data = {}
        if self.model_type == "chat_models":
            data["text"] = response["choices"][0]["message"]["content"].strip(" \n")
        elif self.model_type == "completion_models":
            data["text"] = response["choices"][0]["text"]

        data["usage"] = dict(response["usage"])
        return data

    def model_output(self, response, json_depth_limit: int) -> Dict:
        data = self.model_output_raw(response)
        data["parsed"] = self.parser.fit(data["text"], json_depth_limit)

        return data
