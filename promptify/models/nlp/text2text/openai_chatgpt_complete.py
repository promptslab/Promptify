from typing import Dict, List, Optional, Tuple, Union
import openai
import tiktoken
from parser import Parser
from base_model import Model


class OpenAI_ChatComplete(Model):
    name = "OpenAI"
    description = "OpenAI API for text completion using various models"

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-3.5-turbo",
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
        session_identifier: str = None,
        messages = [
        {"role": "system", "content": "you are a helpful assistant"},
        ],
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
        self.messages = messages.copy() if messages else []
        self.parser = Parser()
        self.set_key(self.api_key)
        self.session_identifier = session_identifier
        if session_identifier:
            self._load_session(session_identifier)

    @classmethod
    def supported_models(cls) -> Dict[str, str]:
        return {
            "gpt-4": "More capable than any GPT-3.5 model, able to do more complex tasks, and optimized for chat. Will be updated with our latest model iteration.",
            "gpt-3.5-turbo": "	Most capable GPT-3.5 model and optimized for chat at 1/10th the cost of text-davinci-003. Will be updated with our latest model iteration",
            }

    def default_max_tokens(self, model_name: str) -> int:
        token_dict = {
            "gpt-4": 8192,
            "gpt-3.5-turbo": 4096,
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
        prompt =  str(prompt)
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
        
    def _store_session(self, session_identifier: str):
        import json
        import os 
        
        if not os.path.exists("sessions"):
            os.mkdir("sessions")
        with open(f"sessions/{session_identifier}.json", "w") as f:
            json.dump(self.messages, f)
            
    def _load_session(self, session_identifier: str):
        import json
        import os
        
        if not os.path.exists(f"sessions/{session_identifier}.json") or os.path.getsize(f"sessions/{session_identifier}.json") == 0:
            print("No session found")
        else:
            with open(f"sessions/{session_identifier}.json", "r") as f:
                self.messages = json.load(f)

    def run(self, prompt: str) -> List[Optional[str]]:
        """
        prompt: str - The prompt to use for the completion
       """
       
        result = []
        
        self.messages.append({"role": "user", "content": prompt})

        max_tokens = self.calculate_max_tokens(self.messages)
        response = self._openai.ChatCompletion.create(
            model=self.model,
            messages= self.messages,
            max_tokens=max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            n=self.n,
            stop=self.stop,
            logit_bias=self.logit_bias,
            request_timeout=self.request_timeout,
            presence_penalty=self.presence_penalty,
            frequency_penalty=self.frequency_penalty,
        )

        self.messages.append({"role": "assistant", "content": response["choices"][0]["message"]["content"].strip(" \n")})

        result.append(response["choices"][0]["message"]["content"].strip(" \n"))
        
        if self.session_identifier:
            self._store_session(self.session_identifier)
        
        return result[-1], self.messages
