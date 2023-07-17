from abc import ABCMeta, abstractmethod
from typing import List, Optional, Union, Dict
import tenacity


class Model(metaclass=ABCMeta):

    """
    Abstract base class for a large language model(llm).

    Attributes
    ----------
    name : str
        The name of the language model.
    description : str
        A brief description of the language model.

    Methods
    -------
    __init__(api_key, model, api_wait=None, api_retry=None, **kwargs) -> None:
        Initializes the Model class with the required parameters and verifies the model is supported by the endpoint.
    supported_models() -> List[str]:
        Abstract method to return a list of supported models for the endpoint.
    _verify_model() -> None:
        Abstract method to verify if the model is supported by the endpoint.
    set_key(api_key) -> None:
        Abstract method to set the endpoint API key.
    set_model(model) -> None:
        Abstract method to set the model name for the endpoint.
    get_description() -> str:
        Abstract method to get the model description.
    get_endpoint() -> str:
        Abstract method to get the model endpoint.
    get_parameters() -> Dict[str, str]:
        Abstract method to get the model parameters.
    run(prompts) -> List[str]:
        Abstract method to run the language model on the given list of prompts and return the list of responses.
    model_output(response) -> Any:
        Abstract method to get the model output from the response.
    _retry_decorator() -> tenacity.Retry:
        Decorator function for retrying API requests if they fail.
    execute_with_retry(*args, **kwargs) -> List[str]:
        Decorated version of the `run` method with the retry logic.


    Examples
    --------
    >>> class MyModel(Model):
    ...     def __init__(self, api_key, model, api_wait=None, api_retry=None, **kwargs):
    ...         super().__init__(api_key, model, api_wait, api_retry, **kwargs)
    ...
    ...     @classmethod
    ...     def supported_models(cls) -> List[str]:
    ...         return ['gpt', 'davinci']
    ...
    ...     def _verify_model(self):
    ...         assert self.model in self.supported_models(), f"{self.model} is not a supported model"
    ...
    ...     def set_key(self, api_key: str):
    ...         self.api_key = api_key
    ...
    ...     def set_model(self, model: str):
    ...         self.model = model
    ...
    ...     def get_description(self) -> str:
    ...         return self.description
    ...
    ...     def get_endpoint(self) -> str:
    ...         return "https://api.openai.com/v1/completions"
    ...
    ...     def get_parameters(self) -> Dict[str, str]:
    ...         return {"model": self.model, "prompt": "", "temperature": "0.7"}
    ...
    ...     def run(self, prompts: List[str]) -> List[str]:
    ...         # Send the request to OpenAI's API
    ...         response = requests.post(
    ...             self.get_endpoint(),
    ...             headers={
    ...                 "Content-Type": "application/json",
    ...                 "Authorization": f"Bearer {self.api_key}",
    ...             },
    ...             json=self.get_parameters(),
    ...         )
    ...
    ...         # Get the output from the response
    ...         output = self.model_output(response)
    ...
    ...         # Return the output
    ...         return output
    ...
    ...     def model_output(self, response):
    ...         return response.json()['choices'][0]['text']

    Notes
    -----
    This class is an abstract base class for creating large language model classes.
    It provides common methods and attributes that can be used by different llms
    classes to make calls to llms API more streamlined.

    """

    name = ""
    description = ""

    def __init__(
        self,
        api_key: str,
        model: str,
        api_wait: int = 60,
        api_retry: int = 6,
        **kwargs
    ):
        """
        Initializes the Model class with the required parameters and verifies the model is supported by the endpoint.

        Parameters
        ----------
        api_key : str
            The API key if needed for the endpoint.
        model : str
            The name of the LLM model to use for the endpoint.
        api_wait : int, optional
            Maximum wait time for an API request before retrying (in seconds), by default 60.
        api_retry : int, optional
            Number of times to retry an API request before failing, by default 6.
        **kwargs : dict
            Additional arguments to be passed to the Model API call.

        Notes
        -----
        This method initializes the Model class with the required parameters and verifies that the given model is supported by the endpoint. It sets the values of `api_key`, `model`, `api_wait`, and `api_retry` attributes of the class.

        Examples
        --------
        >>> my_model = MyModel(api_key="my_api_key", model="davinci")

        """

        self.api_key = api_key
        self.model = model
        self.api_wait = api_wait
        self.api_retry = api_retry
        self._verify_model()
        self.set_key(api_key)

    @abstractmethod
    def supported_models(self):
        """
        Get a list of supported models for the endpoint.

        Returns
        -------
        List[str]
            A list of supported models for the endpoint.

        Notes
        -----
        This method is an abstract method and must be implemented in the derived classes.

        Examples
        --------
        >>> class MyModel(Model):
        ...     def supported_models(self):
        ...         return ['gpt', 'davinci']
        """

        raise NotImplementedError

    @abstractmethod
    def _verify_model(self):
        """
        Verify the model is supported by the endpoint.

        Notes
        -----
        This method is an abstract method and must be implemented in the derived classes.

        Examples
        --------
        >>> class MyModel(Model):
        ...     def _verify_model(self):
        ...         assert self.model in self.supported_models(), f"{self.model} is not a supported model"
        """

        raise NotImplementedError

    @abstractmethod
    def set_key(self, api_key: str):
        """
        Set endpoint API key if needed.

        Parameters
        ----------
        api_key : str
            The API key to set.

        Notes
        -----
        This method is an abstract method and must be implemented in the derived classes.

        Examples
        --------
        >>> class MyModel(Model):
        ...     def set_key(self, api_key: str):
        ...         self.api_key = api_key
        """

        raise NotImplementedError

    @abstractmethod
    def set_model(self, model: str):
        """
        Set model name for the endpoint.

        Parameters
        ----------
        model : str
            The model name to set.

        Notes
        -----
        This method is an abstract method and must be implemented in the derived classes.

        Examples
        --------
        >>> class MyModel(Model):
        ...     def set_model(self, model: str):
        ...         self.model = model
        """

        raise NotImplementedError

    @abstractmethod
    def get_description(self) -> str:
        """
        Get model description.

        Returns
        -------
        str
            A string containing the model description.

        Notes
        -----
        This method is an abstract method and must be implemented in the derived classes.

        Examples
        --------
        >>> class MyModel(Model):
        ...     def get_description(self) -> str:
        ...         return self.description
        """

        raise NotImplementedError

    @abstractmethod
    def get_endpoint(self) -> str:
        """
        Get model endpoint.

        Returns
        -------
        str
            A string containing the model endpoint.

        Notes
        -----
        This method is an abstract method and must be implemented in the derived classes.

        Examples
        --------
        >>> class MyModel(Model):
        ...     def get_endpoint(self) -> str:
        ...         return "https://api.openai.com/v1/completions"
        """

        raise NotImplementedError

    @abstractmethod
    def get_parameters(self) -> Dict[str, str]:
        """
        Get model parameters.

        Returns
        -------
        Dict[str, str]
            A dictionary containing the model parameters.

        Notes
        -----
        This method is an abstract method and must be implemented in the derived classes.

        Examples
        --------
        >>> class MyModel(Model):
        ...     def get_parameters(self) -> Dict[str, str]:
        ...         return {"model": self.model, "prompt": "", "temperature": "0.7"}
        """

        raise NotImplementedError
    
    @abstractmethod
    def run(self, prompts: List[str]) -> List[str]:
        """
        Run the LLM on the given prompt list.

        Parameters
        ----------
        prompts : List[str]
            A list of prompts to run on the LLM.

        Returns
        -------
        List[str]
            A list of responses from the LLM.

        Notes
        -----
        This method is an abstract method and must be implemented in the derived classes.

        Examples
        --------
        >>> class MyModel(Model):
        ...     def run(self, prompts: List[str]) -> List[str]:
        ...         # Send the request to OpenAI's API
        ...         response = requests.post(
        ...             self.get_endpoint(),
        ...             headers={
        ...                 "Content-Type": "application/json",
        ...                 "Authorization": f"Bearer {self.api_key}",
        ...             },
        ...             json=self.get_parameters(),
        ...         )
        ...
        ...         # Get the output from the response
        ...         output = self.model_output(response)
        ...
        ...         # Return the output
        ...         return output
        """

        raise NotImplementedError

    @abstractmethod
    def model_output(self, response):
        """
        Get the model output from the response.

        Parameters
        ----------
        response : requests.Response
            The response from the API call.

        Notes
        -----
        This method is an abstract method and must be implemented in the derived classes.

        Examples
        --------
        >>> class MyModel(Model):
        ...     def model_output(self, response):
        ...         return response.json()['choices'][0]['text']
        """
        raise NotImplementedError

    def _retry_decorator(self):
        """
        Decorator function for retrying API requests if they fail.

        Returns
        -------
        tenacity.Retrying
            A decorator function for retrying API requests.

        Notes
        -----
        This method is a decorator function for retrying API requests using tenacity.
        """

        return tenacity.retry(
            wait=tenacity.wait_random_exponential(
                multiplier=0.3, exp_base=3, max=self.api_wait
            ),
            stop=tenacity.stop_after_attempt(self.api_retry),
        )

    def execute_with_retry(self, *args, **kwargs):
        """

        Decorated version of the run method with the retry logic.

        Parameters
        ----------
        *args : tuple
            A tuple of arguments to pass to the `run` method.
        **kwargs : dict
            A dictionary of keyword arguments to pass to the `run` method.

        Returns
        -------
        Any
            The output of the `run` method.

        Notes
        -----
        This method is a decorated version of the `run` method with the retry logic.
        """

        decorated_run = self._retry_decorator()(self.run)
        return decorated_run(*args, **kwargs)
