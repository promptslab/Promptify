from abc import ABCMeta, abstractmethod
from typing import List


class Model(metaclass=ABCMeta):
    name = ""
    description = ""

    @abstractmethod
    def list_models(self) -> List[str]:
        """
        Get list of supported models
        """
        raise NotImplementedError

    def verify_model(self, model):
        """
        Verify the model is supported
        """
        return model in self.list_models()

    @abstractmethod
    def run(self, *args, **kwargs):
        """
        run selected model to get completion from LLM
        """
        raise NotImplementedError
