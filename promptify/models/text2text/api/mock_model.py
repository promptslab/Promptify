from typing import List, Dict
from promptify.models.text2text.api.base_model import Model


class MockModel(Model):

    """
    A mock model for testing purposes.

    This is a base class for models that provides a common interface for interacting with models.
    It defines methods for setting model parameters, running the model, and retrieving model output.

    Attributes
    ----------
    name : str
        The name of the model.
    description : str
        A description of the model.

    Methods
    -------
    supported_models() -> List[str]:
        Returns a list of supported models.

    set_key(api_key: str):
        Sets the API key for the model.

    set_model(model: str):
        Sets the model to use.

    get_description() -> str:
        Returns the description of the model.

    get_endpoint() -> str:
        Returns the endpoint for the model.

    get_parameters() -> Dict[str, str]:
        Returns the parameters for the model.

    run(prompts: List[str]) -> List[str]:
        Runs the model on the given prompts and returns a list of responses.

    model_output(response):
        Processes the model's response and returns it as output.

    """

    name = "mock_model"
    description = "Mock model for testing purposes"

    
    def supported_models(self) -> List[str]:
        """
        Returns a list of supported models.

        Returns
        -------
        List[str]
            A list of supported models.
        """
        return ['mock_model']

    def _verify_model(self):
        """
        Checks if the model is supported.

        Raises
        ------
        ValueError
            If the model is not supported.

        """
        if self.model not in self.supported_models():
            raise ValueError(f"Unsupported model: {self.model}")

    def set_key(self, api_key: str):
        """
        Sets the API key for the model.

        Parameters
        ----------
        api_key : str
            The API key for the model.

        """
        self.api_key = api_key

    def set_model(self, model: str):
        """
        Sets the model to use.

        Parameters
        ----------
        model : str
            The name of the model to use.

        Raises
        ------
        ValueError
            If the model is not supported.

        """

        self.model = model
        self._verify_model()

    def get_description(self) -> str:
        """
        Returns the description of the model.

        Returns
        -------
        str
            The description of the model.

        """
        return self.description

    def get_endpoint(self) -> str:
        """
        Returns the endpoint for the model.

        Returns
        -------
        str
            The endpoint for the model.

        """
        return "https://mock.endpoint/"

    def get_parameters(self) -> Dict[str, str]:
        """
        Returns the parameters for the model.

        Returns
        -------
        Dict[str, str]
            A dictionary of the model's parameters.

        """
        return {"param": "value"}

    def run(self, prompts: List[str]) -> List[str]:
        """
        Runs the model on the given prompts and returns a list of responses.

        Parameters
        ----------
        prompts : List[str]
            A list of prompts to run the model on.

        Returns
        -------
        List[str]
            A list of responses from the model.

        """
        return {'text': 'response', 'parsed': {'data': {"completion" : []}}}

    def model_output(self, response, max_completion_length = None):
        """
        Processes the model's response and returns it as output.

        Parameters
        ----------
        response :
            The response from the model.

        Returns
        -------
        Output of the model.

        """

        response = {'text': 'response', 'parsed': {'data': {"completion" : []}}}


        
        return response

    def model_output_raw(self, response, max_completion_length = None):
        """
        Processes the model's response and returns it as output.

        Parameters
        ----------
        response :
            The response from the model.

        Returns
        -------
        Output of the model.

        """

        response = {'text': 'response', 'parsed': {'data': {"completion" : []}}}
        return response
