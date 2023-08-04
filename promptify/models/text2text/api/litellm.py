from typing import Dict, List, Optional, Tuple, Union
import openai
import json
import tiktoken
from promptify.parser.parser import Parser
from promptify.models.text2text.api.base_model import Model
import litellm 
from litellm import completion

class LiteLLM(Model):
    name = "LiteLLM"
    description = "Using the LiteLLM I/O library to call LLM Providers - Replicate (Llama2), PaLM, Anthropic, etc."

    SUPPORTED_MODELS = litellm.model_list

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
        self.api_key = api_key

    def _verify_model(self):
        if self.model not in self.SUPPORTED_MODELS:
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
        return self.SUPPORTED_MODELS

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
        return self.model

    def run(self, prompt: str):
        return self._chat_api(prompt)

    def _chat_api(self, prompt: str):
        prompt_template = [
            {"role": "system", "content": "you are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]

        self.parameters["messages"] = prompt_template
        response = self._openai.ChatCompletion.create(
            model=self.model,
            api_key=self.api_key
            **self.parameters,
        )
        return response


    def model_output_raw(self, response: Dict) -> Dict:
        data = {}
        try:
            data["text"] = response["choices"][0]["message"]["content"].strip(" \n")
        except Exception as e:
            data["text"] = response[0]["choices"][0]["message"]["content"].strip(" \n")

        return data

    def model_output(self, response, json_depth_limit: int) -> Dict:
        data = self.model_output_raw(response)
        data["parsed"] = self.parser.fit(data["text"], json_depth_limit)

        return data
