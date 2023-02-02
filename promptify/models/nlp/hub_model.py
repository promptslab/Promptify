from typing import List, Optional, Union

import requests
from huggingface_hub import model_info
from huggingface_hub.utils import RepositoryNotFoundError, build_hf_headers, hf_raise_for_status
from promptify import __version__

from .model import Model


class HubModel(Model):
    name = "HF Hub"
    description = "Leverage HF Inference for text completion using any model available on the Hub."

    def __init__(self, model_id_or_url: str = "google/flan-t5-xl", api_key: Optional[str] = None) -> None:
        """
        Initialize `HubModel` for a given model_id or URL on the Huggingface Hub.

        model_id_or_url:
            - either the ID of a model on the Huggingface Hub. You can visit https://huggingface.co/models to look for a
              model suiting your needs. Defaults to `google/flan-t5-xl`, a popular text-generation model fine-tuned on
              more than 1000 additional tasks covering 60 languages.
            - either the URL to an Inference Endpoint running on Huggingface. Inference Endpoints is a production-ready
              solution to easily deploy any models from the Hub on dedicated and autoscaling infrastructure managed by
              Huggingface. See documentation for model details: https://huggingface.co/docs/inference-endpoints/index.

        api_key:
            User token to authenticate on Huggingface. Visit https://huggingface.co/settings/tokens to generate a token.
            It is also possible to login by running `huggingface-cli login` in your terminal. Authentication is mandatory
            for Inference Endpoints urls and optional (but preferred) for the free Inference API.
        """
        if model_id_or_url.startswith("https://") and "huggingface" in model_id_or_url:
            # User provided a URL pointing to a Inference Endpoint
            self._url = model_id_or_url
        else:
            self._url = _get_url_from_model_id(model_id=model_id_or_url, api_key=api_key)
        self._headers = build_hf_headers(token=self._api_key, library_name="promptify", library_version=__version__)

    @classmethod
    def list_models(cls) -> List[str]:
        """
        Get a list of supported models on the Hub. Only a subset is returned.
        The best way to search for models is to visit https://huggingface.co/models and filter for your needs.
        """
        response = requests.get(
            "https://huggingface.co/api/models?pipeline_tag=text2text-generation&sort=downloads&direction=-1"
        )
        hf_raise_for_status(response)
        return sorted(item["id"] for item in response.json())

    def run(
        self,
        prompts: Union[str, List[str]],
        wait_for_model: bool = True,
        use_cache: bool = True,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        temperature: float = 1.0,
        repetition_penalty: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
        max_time: Optional[float] = None,
        return_full_text: bool = False,
        num_return_sequences: int = 1,
        do_sample: bool = True,
    ) -> List[str]:
        """
        Run Inference on a given set of prompts.

        Detailed parameters list can be found here: https://huggingface.co/docs/api-inference/detailed_parameters#text-generation-task

        prompts:
            The prompt(s) to generate completions for, encoded as a string or array of strings.
        wait_for_model:
            Either we should wait for the model to be loaded in the Inference API. Popular models are often
            already loaded but more specific models have to be pre-heated before being able to use them.
        use_cache:
            There is a cache layer on the inference API to speedup requests we have already seen. Most models can use
            those results as is as models are deterministic (meaning the results will be the same anyway). However if
            you use a non deterministic model, you can set this parameter to prevent the caching mechanism from being
            used resulting in a real new query.
        top_k:
            Integer to define the top tokens considered within the sample operation to create new text.
        top_p:
            Float to define the tokens that are within the sample operation of text generation. Add tokens in the sample
            for more probable to least probable until the sum of the probabilities is greater than top_p.
        temperature:
            Float (0.0-100.0). The temperature of the sampling operation. 1 means regular sampling, 0 means always take
            the highest score, 100.0 is getting closer to uniform probability.
        repetition_penalty:
            Float (0.0-100.0). The more a token is used within generation the more it is penalized to not be picked in
            successive generation passes.
        max_new_tokens:
            Int (0-250). The amount of new tokens to be generated, this does not include the input length it is a estimate
            of the size of generated text you want. Each new tokens slows down the request, so look for balance between
            response times and length of text generated.
        max_time:
            Float (0-120.0). The amount of time in seconds that the query should take maximum. Network can cause some
            overhead so it will be a soft limit. Use that in combination with max_new_tokens for best results.
        return_full_text:
            Bool. If set to False, the return results will not contain the original query making it easier for prompting.
        num_return_sequences:
            Integer. The number of proposition you want to be returned.
        do_sample:
            Bool. Whether or not to use sampling, use greedy decoding otherwise.
        """
        response = requests.post(
            self._url,
            headers=self._headers,
            json={
                "inputs": prompts,
                "options": {
                    "wait_for_model": wait_for_model,
                    "use_cache": use_cache,
                },
                "parameters": {
                    "top_k": top_k,
                    "top_p": top_p,
                    "temperature": temperature,
                    "repetition_penalty": repetition_penalty,
                    "max_new_tokens": max_new_tokens,
                    "max_time": max_time,
                    "return_full_text": return_full_text,
                    "num_return_sequences": num_return_sequences,
                    "do_sample": do_sample,
                },
            },
        )
        hf_raise_for_status(response)

        return [item["generated_text"] for item in response.json()]


def _get_url_from_model_id(model_id: str, api_key: Optional[str]) -> str:
    """Checks model id is valid and renders the Inference API URL for it."""
    # Check if model exists
    try:
        info = model_info(model_id, token=api_key)
    except RepositoryNotFoundError:
        raise ValueError(
            f"Model '{model_id}' does not exist on the Huggingface Hub. Please visit https://huggingface.co/models to"
            "find a suitable model."
        )

    # Check if it's a text-generation model
    if info.pipeline_tag not in ("text-generation", "text2text-generation"):
        raise ValueError(
            f"Cannot use model {model_id}. Pipeline is of type {info.pipeline_tag}. Expecting either"
            " 'text-generation' or 'text2text-generation'."
        )

    # Pre-build URL and headers for later
    return f"https://api-inference.huggingface.co/pipeline/{info.pipeline_tag}/{model_id}"
