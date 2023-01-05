from typing import List


class Model:

    name = ""
    description = ""

    def __init__(
        self,
    ):
        pass

    def list_models(self) -> List[str]:
        """
        Get list of supported models
        """
        raise NotImplementedError

    def verify_model(self, model):
        """
        Verify the model is supported
        """
        return model in self.get_models()

    def run(self, *args, **kwargs):
        """
        run selected model to get completion from LLM
        """
        raise NotImplementedError
