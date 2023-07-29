from promptify.parser.parser import Parser
from promptify.models.text2text.api.base_model import Model
from typing import Dict, List, Optional, Tuple, Union
import cohere

class CohereModel(Model):
    name = "Cohere"
    description = "Cohere API for text completion using Command model"

    SUPPORTED_MODELS = set(
        [
            "command", 
            "command-light", 
            "command-medium-beta", 
            "command-nightly", 
            "command-xlarge-beta"
        ]
    )
    
    def __init__(
        self,
        api_key: str,
        model: str = "command",
        temperature: float = 0.9,
        top_k: int = 0,
        max_tokens: int = 200,
        stop_sequences: Optional[Union[str, List[str]]] = [],
        return_likelihoods: str = 'NONE',
        stream: bool = False,
        truncate: str = 'END',
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        top_p: float = 0.75,
        num_generations: int = 1,
        api_wait=60,
        api_retry=6,
        json_depth_limit: int = 20
    ):
        super().__init__(api_key, model, api_wait, api_retry)

        self.temperature = temperature
        self.top_k = top_k
        self.max_tokens = max_tokens
        self.stop_sequences = stop_sequences
        self.return_likelihoods = return_likelihoods
        self.json_depth_limit = json_depth_limit
        self.stream = stream
        self.truncate = truncate
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        self.top_p = top_p
        self.num_generations = num_generations
        self.set_key(api_key)
        self._verify_model()
        self.parameters = self.get_parameters()
        self._initialize_parser()


    # https://dashboard.cohere.ai/api-keys

    def set_key(self, api_key: str):
        self._cohere = cohere.Client(api_key)

    def _verify_model(self):
        if self.model not in self.SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model: {self.model}")
            
    def _initialize_parser(self):
        self.parser = Parser()

    def set_model(self, model: str):
        self.model = model
        self._verify_model()

    def supported_models(self):
        return list(self.SUPPORTED_MODELS)

    def get_parameters(self):
        return {
            "temperature": self.temperature,
            "k": self.top_k,
            "max_tokens": self.max_tokens,
            "stop_sequences": self.stop_sequences,
            "return_likelihoods": self.return_likelihoods,
            "stream": self.stream,
            "truncate": self.truncate,
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
            "p": self.top_p,
            "num_generations": self.num_generations
        }

    def get_description(self):
        return self.description

    def get_endpoint(self):
        return self.model

    def run(self, prompt: str):
        self.parameters["prompt"] = prompt
        response = self._cohere.generate(
            model=self.model,
            **self.parameters,
        )
        return response

    def model_output_raw(self, response: Dict) -> Dict:
        data = {}
        raw_response = response.generations[0].text.strip(" \n")
        data['text'] = str(raw_response)
        return data
    
    def model_output(self, response, json_depth_limit) -> Dict:
        data = self.model_output_raw(response)
        data["parsed"] = self.parser.fit(data["text"], json_depth_limit)
        return data
