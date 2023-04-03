from abc import ABCMeta, abstractmethod
from typing import List, Optional, Union, Dict
import tenacity


class Model(metaclass=ABCMeta):
    name = ""
    description = ""

    def __init__(
        self,
        api_key: str,
        model: str,
        api_wait: int = None,
        api_retry: int = None,
        **kwargs
    ):
        """
        Initializes the Model class with the required parameters and verifies the model is supported by the endpoint.
        :param api_key: str, Model API key if needed for the endpoint
        :param model: str, name of the LLM model to use for the endpoint
        :param api_wait: int, maximum wait time for an API request before retrying (in seconds)
        :param api_retry: int, number of times to retry an API request before failing
        :param **kwargs: additional arguments to be passed to the OpenAI API call
        """

        self.api_key = api_key
        self.model = model
        self.api_wait = api_wait
        self.api_retry = api_retry
        self._verify_model()
        self.set_key(api_key)

    @classmethod
    @abstractmethod
    def supported_models(cls) -> List[str]:
        """
        Get a list of supported models for the endpoint
        """
        raise NotImplementedError

    @abstractmethod
    def _verify_model(self):
        """
        Verify the model is supported by the endpoint
        """
        raise NotImplementedError

    @abstractmethod
    def set_key(self, api_key: str):
        """
        Set endpoint API key if needed
        """
        raise NotImplementedError

    @abstractmethod
    def set_model(self, model: str):
        """
        Set model name for the endpoint
        """
        raise NotImplementedError

    @abstractmethod
    def get_description(self) -> str:
        """
        Get model description
        """
        raise NotImplementedError

    @abstractmethod
    def get_endpoint(self) -> str:
        """
        Get model endpoint
        """
        raise NotImplementedError

    @abstractmethod
    def get_parameters(self) -> Dict[str, str]:
        """
        Get model parameters
        """
        raise NotImplementedError

    @abstractmethod
    def run(self, prompts: List[str]) -> List[str]:
        """
        Run the LLM on the given prompt list.
        :param prompts: List[str], list of prompts to run on the LLM
        :returns: List[str], list of responses from the LLM
        """

        raise NotImplementedError

    @abstractmethod
    def model_output(self, response):
        """
        Get the model output from the response
        """
        raise NotImplementedError

    def _retry_decorator(self):
        """
        Decorator function for retrying API requests if they fail
        """

        return tenacity.retry(
            wait=tenacity.wait_random_exponential(
                multiplier=0.3, exp_base=3, max=self.api_wait
            ),
            stop=tenacity.stop_after_attempt(self.api_retry),
        )

    def execute_with_retry(self, *args, **kwargs):
        """
        Decorated version of the `run` method with the retry logic
        """
        decorated_run = self._retry_decorator()(self.run)
        return decorated_run(*args, **kwargs)