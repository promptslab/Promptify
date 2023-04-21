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

    """
    A class representing the HuggingFace API for text completion using various models.

    Attributes
    ----------
    name : str
        The name of the model, set to "HuggingFace".
    description : str
        A short description of the model, set to "HuggingFace API for text completion using various models".

    Methods
    -------
    __init__(self, api_key=None, model_id_or_url="google/flan-t5-xl", wait_for_model=True, use_cache=True, top_k=None,
            top_p=None, temperature=1.0, repetition_penalty=None, max_new_tokens=None, max_time=None,
            num_return_sequences=1, do_sample=True, api_wait=None, api_retry=None, max_completion_length=20):
        Initializes a new HuggingFace object with the specified parameters.

    supported_models() -> Dict[str, str]:
        Returns a dictionary of all the models supported by HuggingFace.

    _verify_model(self):
        Verifies if the given model is supported by HuggingFace.

    set_key(self, api_key: Optional[str]):
        Sets the API key to be used to connect to HuggingFace.

    set_model(self, model: str):
        Sets the HuggingFace model to be used for text completion.

    get_description(self) -> str:
        Returns a short description of the HuggingFace model being used for text completion.

    model_output(self, response: Dict) -> Dict:
        Extracts the generated text from the HuggingFace model response.

    get_endpoint(self) -> str:
        Gets the API endpoint of the HuggingFace model being used for text completion.

    get_parameters(self) -> Dict[str, Union[str, int, float, List[str], Dict[str, int]]]:
        Gets the parameters used to configure the HuggingFace model for text completion.

    run(self, prompts: List[str]) -> List[Optional[str]]:
        Runs the HuggingFace model for text completion on the specified prompts and returns a list of responses.
    """

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
        max_completion_length: int = 20,
    ):
        """
        Initializes a new HuggingFace object with the specified parameters.

        Parameters
        ----------
        api_key : Optional[str], default=None
            The API key to use to connect to HuggingFace.
        model_id_or_url : str, default="google/flan-t5-xl"
            The ID or URL of the HuggingFace model to use for text completion.
        wait_for_model : bool, default=True
            Whether to wait for the HuggingFace model to be ready before sending requests to it.
        use_cache : bool, default=True
            Whether to use the cache when sending requests to the HuggingFace model.
        top_k : Optional[int], default=None
            The maximum number of top-k candidates to consider when generating text.
        top_p : Optional[float], default=None
            The cumulative probability of top-p candidates to consider when generating text.
        temperature : float, default=1.0
            The temperature to use when generating text.
        repetition_penalty : Optional[float], default=None
            The repetition penalty to use when generating text.
        max_new_tokens : Optional[int], default=20
            The maximum number of new tokens to generate when completing text.
        max_time : Optional[float], default=None
            The maximum amount of time to spend generating text, in seconds.
        num_return_sequences : int, default=1
            The number of sequences to generate for each prompt.
        do_sample : bool, default=True
            Whether to use sampling when generating text.
        api_wait : 60
            Not used.
        api_retry : 6
            Not used.
        max_completion_length : int, default=20
            The maximum length of the completed text.

        Returns
        -------
        None
        """

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
        self.max_completion_length = max_completion_length

        self._verify_model()
        self.set_key(self.api_key)

    @classmethod
    def supported_models(cls) -> Dict[str, str]:
        """
        Returns a dictionary of all the models supported by HuggingFace.

        Returns
        -------
        Dict[str, str]
            A dictionary where the keys are the model IDs and the values are short descriptions of the models.
        """

        response = requests.get(
            "https://huggingface.co/api/models?pipeline_tag=text2text-generation&sort=downloads&direction=-1"
        )

        hf_raise_for_status(response)
        return {
            model: f"check more details at https://huggingface.co/{model}"
            for model in sorted(item["id"] for item in response.json())
        }

    def _verify_model(self):
        """
        Verifies if the given model is supported by HuggingFace.

        Raises
        ------
        ValueError
            If the model is not supported by HuggingFace.

        Returns
        -------
        None
        """

        if self.model not in self.supported_models():
            raise ValueError(f"Unsupported model: {self.model}")

    def set_key(self, api_key: Optional[str]):
        """
        Sets the API key to be used to connect to HuggingFace.

        Parameters
        ----------
        api_key : Optional[str]
            The API key to use to connect to HuggingFace.

        Returns
        -------
        None
        """

        self.api_key = api_key
        self._headers = build_hf_headers(
            token=api_key, library_name="promptify", library_version=__version__
        )

    def set_model(self, model: str):
        """
        Sets the HuggingFace model to be used for text completion.

        Parameters
        ----------
        model : str
            The ID or URL of the HuggingFace model to use for text completion.

        Returns
        -------
        None
        """

        self.model = model
        if self.model.startswith("https://") and "huggingface" in self.model:
            # User provided a URL pointing to a Inference Endpoint
            self._url = self.model
            self.model = self.model.split("models/")[-1]
        else:
            self._url = self.get_endpoint()

    def get_description(self) -> str:
        """
        Returns a short description of the HuggingFace model.

        Returns
        -------
        str
            A short description of the HuggingFace model.

        Notes
        -----
        This function calls the `supported_models` class method to get a dictionary of all the supported models,
        and returns the short description for the current model ID.
        """

        return self.supported_models()[self.model]

    def model_output(self, response: Dict, max_completion_length = None) -> Dict:
        """
        Returns the model output in a dictionary format.

        Parameters
        ----------
        response : Dict
            The response from the HuggingFace API.

        Returns
        -------
        Dict
            A dictionary containing the model output.

        Notes
        -----
        This function converts the API response into a dictionary format, where the "text" key maps to a list of
        strings containing the generated text for each input prompt.
        """
        
        return [item["generated_text"] for item in response.json()]
    
#         data = {}
#         data["text"] = [
#             [item["generated_text"] for item in each_result.json()]
#             for each_result in response
#         ]
#         return data
    
    def model_output_raw(self, response):
      return response.text
    
    
    def get_endpoint(self) -> str:
        """
        Returns the API endpoint URL for the HuggingFace model.

        Returns
        -------
        str
            The API endpoint URL for the HuggingFace model.

        Raises
        ------
        ValueError
            If the model does not exist on the Huggingface Hub, or if the model pipeline is not supported.
        """

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
        """
        Returns a dictionary of the model parameters.

        Returns
        -------
        Dict[str, Union[str, int, float, List[str], Dict[str, int]]]
            A dictionary containing the model parameters.
        """

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
        """
        Runs the HuggingFace model on the given prompts.

        Parameters
        ----------
        prompts : List[str]
            A list of text prompts to complete.

        Returns
        -------
        List[Optional[str]]
            A list of completed text strings, one for each input prompt.

        Notes
        -----
        This function sends a POST request to the HuggingFace API for each input prompt, with the appropriate model
        parameters and headers. It returns a list of completed text strings, one for each input prompt.
        """

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
