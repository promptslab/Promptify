from typing import Dict, List, Optional, Tuple, Union
from huggingface_hub import model_info
from promptify.models.text2text.api.base_model import Model
from huggingface_hub.utils import (
    RepositoryNotFoundError,
    build_hf_headers,
    hf_raise_for_status,
)

from promptify import __version__
import requests


class HubModel(Model):

    name = "HuggingFace"
    description = "HuggingFace API for text completion using various models"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "google/flan-t5-xl",
        wait_for_model: bool = True,
        use_cache: bool = True,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        temperature: float = 1.0,
        repetition_penalty: Optional[float] = None,
        max_new_tokens: Optional[int] = 20,
        max_time: Optional[float] = None,
        num_return_sequences: int = 1,
        do_sample: bool = False,
        api_wait= 60,
        api_retry= 6,
        json_depth_limit: int = 20,
    ):
        

        self.set_key(api_key)
        self.set_model(model)
        super().__init__(self.api_key, self.model, api_wait, api_retry)

        self.wait_for_model = wait_for_model
        self.use_cache = use_cache
        self.top_k = top_k
        self.top_p = top_p
        self.temperature = temperature
        self.repetition_penalty = repetition_penalty
        self.max_new_tokens = max_new_tokens
        self.max_time = max_time
        self.num_return_sequences = num_return_sequences
        self.do_sample = do_sample
        self.json_depth_limit = json_depth_limit

        self._verify_model()
        self.set_key(self.api_key)

    @classmethod
    def supported_models(cls) -> Dict[str, str]:

        response = requests.get(
            "https://huggingface.co/api/models?pipeline_tag=text2text-generation&sort=downloads&direction=-1"
        )

        hf_raise_for_status(response)
        return {
            model: f"check more details at https://huggingface.co/{model}"
            for model in sorted(item["id"] for item in response.json())
        }

    def _verify_model(self):

        if self.model not in self.supported_models():
            raise ValueError(f"Unsupported model: {self.model}")

    def set_key(self, api_key: Optional[str]):

        self.api_key = api_key
        self._headers = build_hf_headers(
            token=api_key, library_name="promptify", library_version=__version__
        )

    def set_model(self, model: str):

        self.model = model
        if self.model.startswith("https://") and "huggingface" in self.model:
            # User provided a URL pointing to a Inference Endpoint
            self._url = self.model
            self.model = self.model.split("models/")[-1]
        else:
            self._url = self.get_endpoint()

    def get_description(self) -> str:

        return self.supported_models()[self.model]

    def model_output(self, response: Dict, json_depth_limit = None) -> Dict:
        
        return [item["generated_text"] for item in response.json()]
    
    def model_output_raw(self, response):
      return response.text
    
    
    def get_endpoint(self) -> str:
        try:
            info = model_info(self.model, token=self.api_key)
        except RepositoryNotFoundError:
            raise ValueError(
                f"Model '{self.model}' does not exist on the Huggingface Hub. Please visit https://huggingface.co/models to"
                " find a suitable model."
            )

        if info.pipeline_tag not in ("text-generation", "text2text-generation"):
            raise ValueError(
                f"Cannot use model {self.model}. Pipeline is of type {info.pipeline_tag}. Expecting either"
                " 'text-generation' or 'text2text-generation'."
            )

        return f"https://api-inference.huggingface.co/pipeline/{info.pipeline_tag}/{self.model}"

    def get_parameters(
        self,
    ) -> Dict[str, Union[str, int, float, List[str], Dict[str, int]]]:

        return {
            "wait_for_model": self.wait_for_model,
            "use_cache": self.use_cache,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "temperature": self.temperature,
            "repetition_penalty": self.repetition_penalty,
            "max_new_tokens": self.max_new_tokens,
            "max_time": self.max_time,
            "num_return_sequences": self.num_return_sequences,
            "do_sample": self.do_sample,
        }

    def run(self, prompts: List[str]) -> List[Optional[str]]:
        result = []
        for prompt in prompts:
            response = requests.post(
                self._url,
                headers=self._headers,
                json={
                    "inputs": prompt,
                    "options": {
                        "wait_for_model": self.wait_for_model,
                        "use_cache": self.use_cache,
                    },
                    "parameters": {
                        "top_k": self.top_k,
                        "top_p": self.top_p,
                        "temperature": self.temperature,
                        "repetition_penalty": self.repetition_penalty,
                        "max_new_tokens": self.max_new_tokens,
                        "max_time": self.max_time,
                        "num_return_sequences": self.num_return_sequences,
                        "do_sample": self.do_sample,
                    },
                },
            )
            hf_raise_for_status(response)
            result.append(response)
        return result
