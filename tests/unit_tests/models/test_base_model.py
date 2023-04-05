import pytest
from promptify import Model
from promptify import MockModel

@pytest.fixture
def mock_model():
    return MockModel("api_key", "mock_model", api_wait=5, api_retry=5)

def test_mock_model_is_instance_of_model(mock_model):
    assert isinstance(mock_model, Model)

def test_supported_models(mock_model):
    assert mock_model.supported_models() == ["mock_model"]

def test_init(mock_model):
    assert mock_model.api_key == "api_key"
    assert mock_model.model == "mock_model"

def test_get_description(mock_model):
    assert mock_model.get_description() == "Mock model for testing purposes"

def test_get_endpoint(mock_model):
    assert mock_model.get_endpoint() == "https://mock.endpoint/"

def test_get_parameters(mock_model):
    assert mock_model.get_parameters() == {"param": "value"}

def test_run(mock_model):
    prompts = ["prompt1", "prompt2"]
    responses = mock_model.run(prompts)
    assert responses == ["response", "response"]

def test_model_output(mock_model):
    response = "response"
    assert mock_model.model_output(response) == response

def test_execute_with_retry(mock_model):
    prompts = ["prompt1", "prompt2"]
    responses = mock_model.execute_with_retry(prompts)
    assert responses == ["response", "response"]
