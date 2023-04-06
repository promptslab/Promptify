import pytest
from promptify import OpenAI
from typing import List, Dict
from promptify import Parser

@pytest.fixture
def openai_complete():
    openai_complete = OpenAI(api_key="", api_wait=5, api_retry=5)
    return openai_complete


def test_supported_models(openai_complete):
    models = openai_complete.supported_models()
    assert "text-davinci-003" in models
    assert "text-curie-001" in models
    assert "text-babbage-001" in models
    assert "text-ada-001" in models


def test_invalid_model(openai_complete):
    with pytest.raises(ValueError):
        openai_complete.set_model("invalid-model")


def test_get_description(openai_complete):
    description = openai_complete.get_description()
    assert isinstance(description, str)


def test_get_endpoint(openai_complete):
    endpoint = openai_complete.get_endpoint()
    assert isinstance(endpoint, str)


def test_run(openai_complete):
    prompts = ["Hello, my name is", "The quick brown fox jumps over the"]
    results = openai_complete.run(prompts)
    assert len(results) == 2
    for result in results:
        assert "text" in result["choices"][0]
        assert result["choices"][0]["text"] != ""
        assert isinstance(result["choices"][0]["text"], str)


def test_execute_with_retry(openai_complete):
    prompts = [
        """You are a highly intelligent and accurate medical domain Named-entity recognition(NER) system. You take Passage as input and your task is to recognize and extract specific types of medical domain named entities in that given passage and classify into a set of entity types. Your output valid json and format is only [{{'T': type of entity from predefined entity types, 'E': entity in the input text}},...,{{'branch' : Appropriate branch of the passage ,'group': Appropriate Group of the passage}}] form, no other form.
                Examples: Input: The patient had abdominal pain and 30-pound weight loss then developed jaundice. He had epigastric pain. A thin-slice CT scan was performed, which revealed a pancreatic mass with involved lymph nodes and ring enhancing lesions with liver metastases Output: [[{'T': 'SYMPTOM', 'E': 'abdominal pain'}, {'T': 'QUANTITY', 'E': '30-pound'}, {'T': 'SYMPTOM', 'E': 'jaundice'}, {'T': 'SYMPTOM', 'E': "epigastric pain"}, {'T': 'TEST', 'E': 'thin-slice CT scan'}, {'T': 'ANATOMY', 'E': "pancreatic mass"}, {'T': 'ANATOMY', 'E': 'ring enhancing lesions'}, {'T': 'ANATOMY', 'E': 'liver'}, {'T': 'DISEASE', 'E': 'metastases'}, {'branch': 'Health', 'group': 'Clinical medicine'}]] Input: Several important diseases of the nervous system are associated with dysfunctions of the dopamine system, and some of the key medications used to treat them work by altering the effects of dopamine. Parkinson's disease, a degenerative condition causing tremor and motor impairment, is caused by a loss of dopamine-secreting neurons in an area of the midbrain called the substantia nigra. Its metabolic precursor L-DOPA can be manufactured; Levodopa, a pure form of L-DOPA, is the most widely used treatment for Parkinson's. Output:"""
    ]
    results = openai_complete.execute_with_retry(prompts)
    for result in results:
        result_s = openai_complete.model_output(result, 20)["parsed"][
            "data"
        ]["completion"][0]

        assert isinstance(result_s, List)
        assert isinstance(result_s[0], Dict)


def test_model_output_with_parser(openai_complete):
    prompts = [
        """You are a highly intelligent and accurate medical domain Named-entity recognition(NER) system. You take Passage as input and your task is to recognize and extract specific types of medical domain named entities in that given passage and classify into a set of entity types. Your output valid json and format is only [{{'T': type of entity from predefined entity types, 'E': entity in the input text}},...,{{'branch' : Appropriate branch of the passage ,'group': Appropriate Group of the passage}}] form, no other form.
                Examples: Input: The patient had abdominal pain and 30-pound weight loss then developed jaundice. He had epigastric pain. A thin-slice CT scan was performed, which revealed a pancreatic mass with involved lymph nodes and ring enhancing lesions with liver metastases Output: [[{'T': 'SYMPTOM', 'E': 'abdominal pain'}, {'T': 'QUANTITY', 'E': '30-pound'}, {'T': 'SYMPTOM', 'E': 'jaundice'}, {'T': 'SYMPTOM', 'E': "epigastric pain"}, {'T': 'TEST', 'E': 'thin-slice CT scan'}, {'T': 'ANATOMY', 'E': "pancreatic mass"}, {'T': 'ANATOMY', 'E': 'ring enhancing lesions'}, {'T': 'ANATOMY', 'E': 'liver'}, {'T': 'DISEASE', 'E': 'metastases'}, {'branch': 'Health', 'group': 'Clinical medicine'}]] Input: Several important diseases of the nervous system are associated with dysfunctions of the dopamine system, and some of the key medications used to treat them work by altering the effects of dopamine. Parkinson's disease, a degenerative condition causing tremor and motor impairment, is caused by a loss of dopamine-secreting neurons in an area of the midbrain called the substantia nigra. Its metabolic precursor L-DOPA can be manufactured; Levodopa, a pure form of L-DOPA, is the most widely used treatment for Parkinson's. Output:"""
    ]

    results = openai_complete.run(prompts)
    for result in results:
        result_s = openai_complete.model_output(result, 20)["parsed"][
            "data"
        ]["completion"][0]

        assert isinstance(result_s, List)
        assert isinstance(result_s[0], Dict)


def test_single_double_quote(openai_complete):
    prompts = [
        """You are a highly intelligent and accurate medical domain Named-entity recognition(NER) system. You take Passage as input and your task is to recognize and extract specific types of medical domain named entities in that given passage and classify into a set of entity types. Your output valid json and format is only [{{'T': type of entity from predefined entity types, 'E': entity in the input text}},...,{{'branch' : Appropriate branch of the passage ,'group': Appropriate Group of the passage}}] form, no other form.
                Examples: Input: The patient had abdominal pain and 30-pound weight loss then developed jaundice. He had epigastric pain. A thin-slice CT scan was performed, which revealed a pancreatic mass with involved lymph nodes and ring enhancing lesions with liver metastases Output: [[{'T': 'SYMPTOM', 'E': 'abdominal pain'}, {'T': 'QUANTITY', 'E': '30-pound'}, {'T': 'SYMPTOM', 'E': 'jaundice'}, {'T': 'SYMPTOM', 'E': "epigastric pain"}, {'T': 'TEST', 'E': 'thin-slice CT scan'}, {'T': 'ANATOMY', 'E': "pancreatic mass"}, {'T': 'ANATOMY', 'E': 'ring enhancing lesions'}, {'T': 'ANATOMY', 'E': 'liver'}, {'T': 'DISEASE', 'E': 'metastases'}, {'branch': 'Health', 'group': 'Clinical medicine'}]] Input: Several important diseases of the nervous system are associated with dysfunctions of the dopamine system, and some of the key medications used to treat them work by altering the effects of dopamine. Parkinson's disease, a degenerative condition causing tremor and motor impairment, is caused by a loss of dopamine-secreting neurons in an area of the midbrain called the substantia nigra. Its metabolic precursor L-DOPA can be manufactured; Levodopa, a pure form of L-DOPA, is the most widely used treatment for Parkinson's. Output:"""
    ]

    parser = Parser()
    results = openai_complete.run(prompts)
    result_filterd = eval(parser.escaped_(results[0]["choices"][0]["text"]))
    assert isinstance(result_filterd[0], List)
    assert isinstance(result_filterd[0][0], Dict)
