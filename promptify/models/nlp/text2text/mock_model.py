from typing import List, Dict
from .base_model import Model

class MockModel(Model):
    
    name = "mock_model"
    description = "Mock model for testing purposes"

    @classmethod
    def supported_models(cls) -> List[str]:
        return [cls.name]

    def _verify_model(self):
        if self.model not in self.supported_models():
            raise ValueError(f"Unsupported model: {self.model}")

    def set_key(self, api_key: str):
        self.api_key = api_key

    def set_model(self, model: str):
        self.model = model
        self._verify_model()

    def get_description(self) -> str:
        return self.description

    def get_endpoint(self) -> str:
        return "https://mock.endpoint/"

    def get_parameters(self) -> Dict[str, str]:
        return {"param": "value"}

    def run(self, prompts: List[str]) -> List[str]:
        return ["response" for _ in prompts]

    def model_output(self, response):
        return response