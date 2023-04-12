from typing import Dict, List, Optional, Tuple, Union
import openai
import json
import tiktoken
from promptify.parser.parser import Parser
from promptify.models.nlp.text2text.base_model import Model


class OpenAI(Model):

    """
    A class representing the OpenAI API for text completion using various models.
    Attributes
    ----------
    name : str
        The name of the OpenAI model.
    description : str
        A brief description of the OpenAI model.
    api_key : str
        The API key required to access the OpenAI API.
    model : str
        The name of the OpenAI model to use.
    temperature : float
        The temperature to use when generating completions.
    top_p : float
        The value of the "top p" parameter to use when generating completions.
    n : int
        The number of completions to generate.
    logprobs : int, optional
        If not None, includes the log probabilities of the generated tokens in the response.
    echo : bool
        If True, echoes the prompt in the response.
    stop : Union[str, List[str]], optional
        The stop sequence(s) to use when generating completions.
    presence_penalty : float
        The presence penalty to use when generating completions.
    frequency_penalty : float
        The frequency penalty to use when generating completions.
    best_of : int
        The number of tries to get the best completion.
    logit_bias : Dict[str, int], optional
        A dictionary containing logit bias values for specific tokens.
    request_timeout : Union[float, Tuple[float, float]], optional
        The request timeout for the OpenAI API.
    max_completion_length : int
        The maximum length of the completion.
    Methods
    -------
    supported_models() -> Dict[str, str]:
        Returns a dictionary of the supported OpenAI models with their descriptions.
    default_max_tokens(model_name: str) -> int:
        Returns the default maximum number of tokens for the specified model.
    _verify_model():
        Verifies that the specified model is supported by OpenAI.
    set_key(api_key: str):
        Sets the API key required to access the OpenAI API.
    set_model(model: str):
        Sets the OpenAI model to use.
    get_description() -> str:
        Returns the description of the OpenAI model.
    get_endpoint() -> str:
        Returns the endpoint of the OpenAI model.
    calculate_max_tokens(prompt: str) -> int:
        Calculates the maximum number of tokens for the completion.
    model_output_raw(response: Dict) -> Dict:
        Returns the raw output of the OpenAI API.
    model_output(response: Dict, max_completion_length: int) -> Dict:
        Returns the parsed and completed output of the OpenAI API.
    get_parameters() -> Dict[str, Union[str, int, float, List[str], Dict[str, int]]]:
        Returns a dictionary of the current parameters for the OpenAI model.
    run(prompts: List[str]) -> List[Optional[str]]:
        Runs the OpenAI model on the specified prompts and returns the completions.
    """

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
        api_wait= 60,
        api_retry= 6,
        max_completion_length: int = 20,
        summary_threshold: int = 1500,
        session_identifier: str = None,
        messages = None
    ):
        """
        Constructs an instance of the OpenAI class.
        Parameters
        ----------
        api_key : str
            The API key required to access the OpenAI API.
        model : str, optional
            The name of the OpenAI model to use, by default "text-davinci-003".
        temperature : float, optional
            The temperature to use when generating completions, by default 0.7.
        top_p : float, optional
            The value of the "top p" parameter to use when generating completions, by default 1.
        n : int, optional
            The number of completions to generate, by default 1.
        logprobs : int, optional
            If not None, includes the log probabilities of the generated tokens in the response, by default None.
        echo : bool, optional
            If True, echoes the prompt in the response, by default False.
        stop : Union[str, List[str]], optional
            The stop sequence(s) to use when generating completions, by default None.
        presence_penalty : float, optional
            The presence penalty to use when generating completions, by default 0.
        frequency_penalty : float, optional
            The frequency penalty to use when generating completions, by default 0.
        best_of : int, optional
            The number of tries to get the best completion, by default 1.
        logit_bias : Dict[str, int], optional
            A dictionary containing logit bias values for specific tokens, by default None.
        request_timeout : Union[float, Tuple[float, float]], optional
            The request timeout for the OpenAI API, by default None.
        max_completion_length : int, optional
            The maximum length of the completion, by default 20.
        """

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
        self.encoder = tiktoken.encoding_for_model(self.model)
        self.max_tokens = self.default_max_tokens(self.model)

        if self.model in [ "gpt-3.5-turbo"]:
          self.messages = messages.copy() if messages else [{"role": "system", "content": "you are a helpful assistant"}]
        else:
          self.messages  = ""
        
        self.parser = Parser()
        self.set_key(self.api_key)
        self.session_identifier = session_identifier
        self.summary_threshold = summary_threshold
        if session_identifier:
            self._load_session(session_identifier)

    @classmethod
    def supported_models(cls) -> Dict[str, str]:
        """
        Returns a dictionary of the supported OpenAI models with their descriptions.
        Returns
        -------
        Dict[str, str]
            A dictionary of the supported OpenAI models with their descriptions.
        """
        return {
            "text-davinci-003": "text-davinci-003 can do any language task with better quality, longer output, and consistent instruction-following than the curie, babbage, or ada models. Also supports inserting completions within text.",
            "text-curie-001": "text-curie-001 is very capable, faster and lower cost than Davinci.",
            "text-babbage-001": "text-babbage-001 is capable of straightforward tasks, very fast, and lower cost.",
            "text-ada-001": "text-ada-001 is capable of very simple tasks, usually the fastest model in the GPT-3 series, and lowest cost.",
            "gpt-4": "More capable than any GPT-3.5 model, able to do more complex tasks, and optimized for chat. Will be updated with our latest model iteration.",
            "gpt-3.5-turbo": "	Most capable GPT-3.5 model and optimized for chat at 1/10th the cost of text-davinci-003. Will be updated with our latest model iteration",
        }
    


    def default_max_tokens(self, model_name: str) -> int:
        """
        Returns the default maximum number of tokens for a given OpenAI model.
        Parameters
        ----------
        model_name : str
            The name of the OpenAI model to retrieve the default maximum number of tokens for.
        Returns
        -------
        int
            The default maximum number of tokens for the given OpenAI model.
        """

        token_dict = {
            "text-davinci-003": 4000,
            "text-curie-001": 2048,
            "text-babbage-001": 2048,
            "text-ada-001": 2048,
            "gpt-4": 8192,
            "gpt-3.5-turbo": 4096,
        }
        return token_dict[model_name]

    def _verify_model(self):
        """
        Raises a ValueError if the current OpenAI model is not supported.
        """
        if self.model not in self.supported_models():
            raise ValueError(f"Unsupported model: {self.model}")

    def set_key(self, api_key: str):
        """
        Sets the OpenAI API key to be used for making API requests.
        Parameters
        ----------
        api_key : str
            The OpenAI API key to use for making API requests.
        """

        self._openai = openai
        self._openai.api_key = api_key

    def set_model(self, model: str):
        """
        Sets the current OpenAI model to be used for generating completions.
        Parameters
        ----------
        model : str
            The name of the OpenAI model to use for generating completions.
        """

        self.model = model
        self._verify_model()

    def get_description(self) -> str:
        """
        Returns the description of the current OpenAI model.
        Returns
        -------
        str
            The description of the current OpenAI model.
        """

        return self.supported_models()[self.model]

    def get_endpoint(self) -> str:
        """
        Returns the endpoint ID of the current OpenAI model.
        Returns
        -------
        str
            The endpoint ID of the current OpenAI model.
        """
        model = openai.Model.retrieve(self.model)
        return model["id"]

    def calculate_max_tokens(self, prompt: str) -> int:
        """
        Calculates the maximum number of tokens for the current OpenAI model given a prompt.
        Parameters
        ----------
        prompt : str
            The prompt to calculate the maximum number of tokens for.
        Returns
        -------
        int
            The maximum number of tokens for the current OpenAI model given the prompt.
        """

        prompt        =  str(prompt)
        prompt_tokens = len(self.encoder.encode(prompt))
        max_tokens    = self.default_max_tokens(self.model) - prompt_tokens

        print(prompt_tokens, max_tokens)
        return max_tokens

    def model_output_raw(self, response: Dict) -> Dict:
        """
        Returns the raw output data from an OpenAI API response.
        Parameters
        ----------
        response : Dict
            The OpenAI API response.
        Returns
        -------
        Dict
            The raw output data from the OpenAI API response.
        """

        data = {}

        if self.model in [ "gpt-3.5-turbo"]:
            data["text"] = response["choices"][0]["message"]['content'].strip(" \n")
        else:
            data["text"] = response["choices"][0]["text"].strip(" \n")
        
        data["usage"] = dict(response["usage"])
        return data

    def model_output(self, response: Dict, max_completion_length: int) -> Dict:
        """
        Returns a dictionary containing the parsed output from an OpenAI API response.
        Parameters
        ----------
        response : Dict
            The OpenAI API response.
        max_completion_length : int
            The maximum length of the completion.
        Returns
        -------
        Dict
            A dictionary containing the parsed output from the OpenAI API response.
        """

        data = {}

        if self.model in [ "gpt-3.5-turbo"]:
            data["text"] = response["choices"][0]["message"]['content'].strip(" \n")
        else:


            data["text"] = response["choices"][0]["text"]

        data["usage"] = dict(response["usage"])
        data["parsed"] = self.parser.fit(data["text"], max_completion_length)

        return data
    
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
    
    
    def get_parameters(
        self,
    ) -> Dict[str, Union[str, int, float, List[str], Dict[str, int]]]:
        """
        Returns a dictionary containing the current parameters for the OpenAI API request.
        Returns
        -------
        Dict[str, Union[str, int, float, List[str], Dict[str, int]]]
            A dictionary containing the current parameters for the OpenAI API request.
        """
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
        Generates completions for a list of prompts.
        Parameters
        ----------
        prompts : List[str]
            A list of prompts to generate completions for.
        Returns
        -------
        List[Optional[str]]
            A list of generated completions, or None if an error occurred.
        """

        result = []

        for prompt in prompts:
            # Automatically calculate max output tokens if not specified
  
            if self.model in [ "gpt-3.5-turbo"]:
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
            else:
                self.messages+=f'prompt: {prompt}'
                max_tokens    = self.calculate_max_tokens(self.messages)
                response      = self._openai.Completion.create(
                    model=self.model,
                    prompt=self.messages,
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
                self.messages+=f'{response["choices"][0]["text"]}'
            result.append(response)
            if self.session_identifier:
              self._store_session(self.session_identifier)
        return result