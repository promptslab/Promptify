from typing import List, Optional

from huggingface_hub import InferenceApi

from .model import Model


class HubModel(Model):
    def __init__(self, api_key: Optional[str] = None) -> None:
        super().__init__()
        self._api_key = api_key

    def list_models(self) -> List[str]:
        """
        Get list of supported models
        """
        return ["google/flan-t5-xl"]

    def run(
        self,
        prompts: List[str],
        model_id: str = "google/flan-t5-xl",
    ):
        """
        prompts: The prompt(s) to generate completions for, encoded as a string, array of strings, array of tokens, or array of token arrays.
        model_id: ID of the model on the Huggingface Hub. You can use the `list_models` function to see all of your available models.
        """
        if not self.verify_model(model_id):
            raise ValueError(f"Model not supported on Huggingface Hub: {model_id}")
        api = InferenceApi(model_id, task="text2text-generation", token=self._api_key)
        return [api(inputs=prompt)[0]["generated_text"] for prompt in prompts]
