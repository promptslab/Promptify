from typing import Dict, List, Optional, Tuple, Union
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
from promptify.parser.parser import Parser
from promptify.models.text2text.api.base_model import Model
import re

class AnthropicModel(Model):
    name = "Anthropic"
    description = "Anthropic API for text completion using Claude model"

    SUPPORTED_MODELS = set(
        [
            "claude-instant-1",
            "claude-2"
        ]
    )

    POSSIBLE_PREFIX = ['here are', 'here is', 'following are']

    def __init__(
        self,
        api_key: str,
        model: str = "claude-instant-1",
        temperature: float = 1,
        top_p: float = 0.7,
        top_k: int = 5,
        max_tokens_to_sample: int = 300,
        stop_sequences: Optional[Union[str, List[str]]] = ["\n\nHuman:"],
        api_wait=60,
        api_retry=6,
        json_depth_limit: int = 20
    ):
        super().__init__(api_key, model, api_wait, api_retry)

        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.max_tokens_to_sample = max_tokens_to_sample
        self.json_depth_limit = json_depth_limit
        self.stop_sequences = stop_sequences
        self.set_key(api_key)
        self._verify_model()
        self.parameters = self.get_parameters()
        self._initialize_parser()

        

    def set_key(self, api_key: str):
        self._anthropic = Anthropic(api_key=api_key)

    def _verify_model(self):
        if self.model not in self.SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model: {self.model}")

    def set_model(self, model: str):
        self.model = model
        self._verify_model()

    def _initialize_parser(self):
        self.parser = Parser()

    def supported_models(self):
        return list(self.SUPPORTED_MODELS)

    def get_parameters(self):
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "max_tokens_to_sample": self.max_tokens_to_sample,
            "stop_sequences": self.stop_sequences,
        }

    def get_description(self):
        return self.description

    def get_endpoint(self):
        return self.model

    def run(self, prompt: str):
        self.parameters["prompt"] = f"{HUMAN_PROMPT} {prompt} {AI_PROMPT}"
        response = self._anthropic.completions.create(
            model=self.model,
            **self.parameters,
        )
        return response

    def extract_string_json(self, text):
        pattern = r'\[\s*(\{.*?\})\s*\]'
        match = re.search(pattern, text, flags=re.S)
        if match:
            try:
                return match.group()
            except json.JSONDecodeError as e:
                raise f"Error decoding JSON: {e}"

    def model_output_raw(self, response: Dict) -> Dict:
        data = {}
        raw_response = response.completion.strip(" \n")
        
        string_json = []
        for prefix in self.POSSIBLE_PREFIX:
            if prefix in raw_response.lower():
                string_json.append(self.extract_string_json(str(raw_response)))
                break
        
        if string_json:
            data['text'] = str(string_json)
        else:
            data['text'] = str(raw_response)
        return data

    def model_output(self, response, json_depth_limit) -> Dict:
        data = self.model_output_raw(response)
        data["parsed"] = self.parser.fit(data["text"], json_depth_limit)
        return data
