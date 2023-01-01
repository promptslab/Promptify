from typing import List, Tuple

def ner(
    text_input: str,
    domain: str = "",
    labels: List[str] = [],
    description: str = "",
    examples: List[Tuple[str, List[str]]] = [],
):
    """
    Perform named-entity recognition on the given text input.

    Parameters:
    text_input (str): The input text to be classified.
    domain (str, optional): The domain of the classification task. Default is an empty string.
    labels (list, optional): A list of possible entity types for the classification. Default is an empty list.
    description (str, optional): A description of the classification task. Default is an empty string.
    examples (list, optional): A list of examples, where each example is a tuple of the form (input_text, labels). Default is an empty list.

    Returns:
    str: A string template describing the classification task and the input and output of the function.
    """

    # create the template string with the labels
    if labels:
        if domain:
            template = f"Perform {domain} domain Named-entity recognition, the output entity types must be from provided TAG LIST\n[TAG LIST]: {labels}"
        else:
            template = f"Perform Named-entity recognition, the output entity types must be from provided TAG LIST\n[TAG LIST]: {labels}"
    else:
        if domain:
            template = f"Perform {domain} domain Named-entity recognition"
        else:
            template = "Perform Named-entity recognition"
    if description:
        template = f"{description}\n{template}"
    template += "\n\n"

    default_example = """Examples:\nInput: The patient had abdominal pain and 30-pound weight loss then developed jaundice. He had epigastric pain. A thin-slice CT scan was performed, which revealed a pancreatic mass with involved lymph nodes and ring enhancing lesions with liver metastases\nOutput: [{'E': 'SYMPTOM', 'W': 'abdominal pain'}, {'E': 'QUANTITY', 'W': '30-pound'}, {'E': 'SYMPTOM', 'W': 'jaundice'}, {'E': 'SYMPTOM', 'W': 'epigastric pain'}, {'E': 'TEST', 'W': 'thin-slice CT scan'}, {'E': 'ANATOMY', 'W': 'pancreatic mass'}, {'E': 'ANATOMY', 'W': 'ring enhancing lesions'}, {'E': 'ANATOMY', 'W': 'liver'}, {'E': 'DISEASE', 'W': 'metastases'},{"branch":"Health","group":"Clinical medicine"}]\n"""
    template = template + default_example

    # if examples are provided, add them to the template
    if examples:
        template += ""
        for example in examples:
            template += f"Input: {example[0]}\nOutput: {example[1]}\n"

    # add the input text to the template
    template += f"\nInput: {text_input}\nOutput:"
    return template
