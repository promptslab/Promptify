import pytest
from promptify import HubModel
from typing import List, Dict
import requests


@pytest.fixture
def huggingface_complete():
    # set api key for testing -> https://huggingface.co/docs/hub/security-tokens
    huggingface_complete = HubModel(api_key = "", model= "https://api-inference.huggingface.co/models/mrm8488/t5-base-finetuned-common_gen", api_wait=1, api_retry=1)
    return huggingface_complete


def test_supported_models(huggingface_complete):
    models = huggingface_complete.supported_models()
    assert "mrm8488/t5-base-finetuned-common_gen" in models


def test_invalid_model(huggingface_complete):
    with pytest.raises(ValueError):
        huggingface_complete.set_model("invalid-model")


def test_get_description(huggingface_complete):
    description = huggingface_complete.get_description()
    assert isinstance(description, str)


def test_get_endpoint(huggingface_complete):
    endpoint = huggingface_complete.get_endpoint()
    assert isinstance(endpoint, str)


def test_execute_with_query(huggingface_complete):
    prompts = ["Hello, my name is", "The quick brown fox jumps over the"]
    results = huggingface_complete.model_output(huggingface_complete.execute_with_retry(prompts))
    assert isinstance(results, dict)
    assert "text" in results
    assert len(results["text"]) == 2
    assert isinstance(results["text"], list)

    for result in results["text"]:
        assert isinstance(result, List)
        for text in result:
            assert isinstance(text, str)
            assert text != ""
