from typing import List, Tuple


def ner(
    text_input: str,
    domain: str = "",
    labels: List[str] = [],
    description: str = "",
    one_shot: bool = True,
    examples: List[Tuple[str, List[str]]] = [],
):
    """
    Perform named-entity recognition on the given text input.
    Parameters:
    text_input (str): The input text to be classified.
    domain (str, optional): The domain of the classification task. Default is an empty string.
    labels (list, optional): A list of possible entity types for the classification. Default is an empty list.
    description (str, optional): A description of the classification task. Default is an empty string.
    one_shot: bool: Include one shot example or not
    examples (list, optional): A list of examples, where each example is a tuple of the form (input_text, labels). Default is an empty list.
    Returns:
    str: A string template describing the classification task and the input and output of the function.
    """

    # create the template string with the labels
    if labels:
        if domain:
            old_template = f"Perform {domain} domain Named-entity recognition, the output entity types must be from provided TAG LIST\n[TAG LIST]: {labels}"
            template = f"You are a highly intelligent and accurate {domain} domain Named-entity recognition(NER) system. You take Passage as input and your task is to recognize and extract specific types of {domain} domain named entities in that given passage and classify into a set of following predefined entity types:\n{labels}\nYour output format is only [{{'T': type of entity from predefined entity types, 'E': entity in the input text}},...,{{'branch' : Appropriate branch of the passage ,'group': Appropriate Group of the passage}}] form, no other form.\n"
        else:
            old_template = f"Perform Named-entity recognition, the output entity types must be from provided TAG LIST\n[TAG LIST]: {labels}"
            template = f"You are a highly intelligent and accurate Named-entity recognition(NER) system. You take Passage as input and your task is to recognize and extract specific types of named entities in that given passage and classify into a set of following predefined entity types:\n{labels}\nYour output format is only [{{'T': type of entity from predefined entity types, 'E': entity in the input text}},...,{{'branch' : Appropriate branch of the passage ,'group': Appropriate Group of the passage}}] form, no other form.\n"
    else:
        if domain:
            template = f"You are a highly intelligent and accurate {domain} domain Named-entity recognition(NER) system. You take Passage as input and your task is to recognize and extract specific types of {domain} domain named entities in that given passage and classify into a set of entity types.\nYour output format is only [{{'T': type of entity from {domain} domain, 'E': entity in the input text}},...,{{'branch' : Appropriate branch of the passage,'group': Appropriate Group of the passage}}] form, no other form.\n"
        else:
            template = f"You are a highly intelligent and accurate Named-entity recognition(NER) system. You take Passage as input and your task is to recognize and extract specific types of named entities in that given passage and classify into a set of entity types.\nYour output format is only [{{'T': type of entity, 'E': entity in the input text}},...,{{'branch' : Appropriate branch of the passage ,'group': Appropriate Group of the passage}}] form, no other form.\n"
    if description:
        template = f"{description}\n{template}"
    template += "\n"

    if one_shot:
        default_example = """Examples:\nInput: The patient had abdominal pain and 30-pound weight loss then developed jaundice. He had epigastric pain. A thin-slice CT scan was performed, which revealed a pancreatic mass with involved lymph nodes and ring enhancing lesions with liver metastases\nOutput: [{'T': 'SYMPTOM', 'E': 'abdominal pain'}, {'T': 'QUANTITY', 'E': '30-pound'}, {'T': 'SYMPTOM', 'E': 'jaundice'}, {'T': 'SYMPTOM', 'E': 'epigastric pain'}, {'T': 'TEST', 'E': 'thin-slice CT scan'}, {'T': 'ANATOMY', 'E': 'pancreatic mass'}, {'T': 'ANATOMY', 'E': 'ring enhancing lesions'}, {'T': 'ANATOMY', 'E': 'liver'}, {'T': 'DISEASE', 'E': 'metastases'},{"branch":"Health","group":"Clinical medicine"}]\n"""
        template = template + default_example

    # if examples are provided, add them to the template
    if examples:
        template += ""
        for example in examples:
            template += f"Input: {example[0]}\nOutput: {example[1]}\n"

    # add the input text to the template
    template += f"\nInput: {text_input}\nOutput:"
    return template
