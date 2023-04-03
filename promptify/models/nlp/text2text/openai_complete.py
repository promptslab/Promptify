from typing import Dict, List, Optional, Tuple, Union
import openai
import tiktoken
from parser import Parser
from .base_model import Model



class OpenAI_Complete(Model):
    name = "OpenAI"
    description = "OpenAI API for text completion using various models"

    def __init__(
        self,
        api_key: str,
        model: str = "text-davinci-003",
        temperature: float = 0.7,
        top_p: float = 1,
        n: int = 1,
        logprobs: Optional[int] = None,
        echo: bool = False,
        stop: Optional[Union[str, List[str]]] = None,
        presence_penalty: float = 0,
        frequency_penalty: float = 0,
        best_of: int = 1,
        logit_bias: Optional[Dict[str, int]] = None,
        request_timeout: Union[float, Tuple[float, float]] = None,
        api_wait=None,
        api_retry=None,
        max_completion_length: int = 20,
    ):
        super().__init__(api_key, model, api_wait, api_retry)

        self.temperature = temperature
        self.top_p = top_p
        self.n = n
        self.logprobs = logprobs
        self.echo = echo
        self.stop = stop
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        self.best_of = best_of
        self.logit_bias = logit_bias or {}
        self.request_timeout = request_timeout
        self.max_completion_length = max_completion_length
        self._verify_model()
        self.encoder    = tiktoken.encoding_for_model(self.model)
        self.max_tokens = self.default_max_tokens(self.model)

        self.parser = Parser()
        self.set_key(self.api_key)

    @classmethod
    def supported_models(cls) -> Dict[str, str]:
        return {
            "text-davinci-003": "text-davinci-003 can do any language task with better quality, longer output, and consistent instruction-following than the curie, babbage, or ada models. Also supports inserting completions within text.",
            "text-curie-001": "text-curie-001 is very capable, faster and lower cost than Davinci.",
            "text-babbage-001": "text-babbage-001 is capable of straightforward tasks, very fast, and lower cost.",
            "text-ada-001": "text-ada-001 is capable of very simple tasks, usually the fastest model in the GPT-3 series, and lowest cost.",
        }

    def default_max_tokens(self, model_name: str) -> int:
        token_dict = {
            "text-davinci-003": 4000,
            "text-curie-001": 2048,
            "text-babbage-001": 2048,
            "text-ada-001": 2048,
        }
        return token_dict[model_name]

    def _verify_model(self):
        if self.model not in self.supported_models():
            raise ValueError(f"Unsupported model: {self.model}")

    def set_key(self, api_key: str):
        self._openai = openai
        self._openai.api_key = api_key

    def set_model(self, model: str):
        self.model = model
        self._verify_model()

    def get_description(self) -> str:
        return self.supported_models()[self.model]

    def get_endpoint(self) -> str:
        model = openai.Model.retrieve(self.model)
        return model["id"]

    def calculate_max_tokens(self, prompt: str) -> int:
        prompt_tokens = len(self.encoder.encode(prompt))
        max_tokens = self.default_max_tokens(self.model) - prompt_tokens
        return max_tokens

    def model_output_raw(self, response: Dict) -> Dict:
        
        data = {}
        data["text"]  = response["choices"][0]["text"]
        data["usage"] = dict(response["usage"])
        return data

    def model_output(self, response: Dict, max_completion_length: int) -> Dict:
        
        data = {}
        data["text"]   = self.parser.escaped_(response["choices"][0]["text"])
        data["usage"]  = dict(response["usage"])
        data["parsed"] = self.parser.fit(data["text"], max_completion_length)
        return data
    

    def get_parameters(
        self,
    ) -> Dict[str, Union[str, int, float, List[str], Dict[str, int]]]:
        return {
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "n": self.n,
            "logprobs": self.logprobs,
            "echo": self.echo,
            "stop": self.stop,
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
            "best_of": self.best_of,
            "logit_bias": self.logit_bias,
            "request_timeout": self.request_timeout,
        }

    def run(self, prompts: List[str]) -> List[Optional[str]]:
        """
        prompts: The prompt(s) to generate completions for, encoded as a string, array of strings, array of tokens, or array of token arrays.
        """
        result = []

        for prompt in prompts:
            # Automatically calculate max output tokens if not specified

            max_tokens = self.calculate_max_tokens(prompt)
            response = self._openai.Completion.create(
                model=self.model,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                n=self.n,
                logprobs=self.logprobs,
                echo=self.echo,
                stop=self.stop,
                best_of=self.best_of,
                logit_bias=self.logit_bias,
                request_timeout=self.request_timeout,
                presence_penalty=self.presence_penalty,
                frequency_penalty=self.frequency_penalty,
            )
            result.append(response)
        return result