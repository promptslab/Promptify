def binary_classification(text_input, labels, description="", examples=[]):
    """
    Perform Binary Text Classification, the output will be either the first label or the second label in the provided list.
    
    Parameters:
    text_input (str): The input text to be classified.
    labels (list): A list of two possible labels for the classification.
    description (str, optional): A description of the classification task. Default is an empty string.
    examples (list, optional): A list of examples for the classification task. Default is an empty list.
    
    Returns:
    str: A string template describing the classification task and the input and output of the function.
    """
    # create the template string with the labels
    if description:
        template = f"{description}\nPerform Binary Text Classification, the output will be either {labels[0]} or {labels[1]}\n\n"
    else:
        template = f"Perform Binary Text Classification, the output will be either {labels[0]} or {labels[1]}\n\n"

    # if examples are provided, add them to the template
    if examples:
        examp = format_examples(examples)
        template = template + examp

    # add the input text to the template
    template = template + f"[P]:{text_input}\n[O]:"
    return template



def multiclass_classification(text_input, labels, description="", examples=[]):
    """
    Perform Multiclass Text Classification, the output class must be from provided Label List.
    
    Parameters:
    text_input (str): The input text to be classified.
    labels (list): A list of possible labels for the classification.
    description (str, optional): A description of the classification task. Default is an empty string.
    examples (list, optional): A list of examples for the classification task. Default is an empty list.
    
    Returns:
    str: A string template describing the classification task and the input and output of the function.
    """
    # create the template string with the list of labels
    template = f"Perform Multiclass Text Classification, the output class must be from provided Label List\n[Label List]: {labels}\n\n"

    # if examples are provided, add them to the template
    if examples:
        examp = format_examples(examples)
        template = template + examp

    # if a description is provided, add it to the beginning of the template
    if description:
        template = f"{description}\n{template}"

    # add the input text to the template
    template = template + f"[P]:{text_input}\n[O]:"
    return template


def multilabel_classification(text_input, domain="", labels=[], description="", examples=[]):
    """
    Perform Multi-Label Text Classification, the output classes must be from the provided Label List.
    
    Parameters:
    text_input (str): The input text to be classified.
    domain (str, optional): The domain of the classification task. Default is an empty string.
    labels (list, optional): A list of possible labels for the classification. Default is an empty list.
    description (str, optional): A description of the classification task. Default is an empty string.
    examples (list, optional): A list of examples for the classification task. Default is an empty list.
    
    Returns:
    str: A string template describing the classification task and the input and output of the function.
    """
    # create the template string with the labels
    if labels:
        if domain:
            template = f"Perform {domain} domain Multi-Label Text Classification, the output classes must be from provided Label List\n[Label List]: {labels}\n\n"
        else:
            template = f"Perform Multi-Label Text Classification, the output classes must be from provided Label List\n[Label List]: {labels}\n\n"
    else:
        if domain:
            template = f"Perform {domain} domain Multi-Label Text Classification\n\n"
        else:
            template = f"Perform Multi-Label Text Classification\n\n"

    # add the default example to the template
    default_example = """[P]:The patient is a 93-year-old female with a medical history of chronic right hip pain, osteoporosis, hypertension, depression, and chronic atrial fibrillation admitted for evaluation and management of severe nausea and vomiting and urinary tract infection\n[O]:{"main class":"Health","1":"Medicine","2":"Patient care","3":"Discharge Summary","4":"Geriatric medicine","5":"Chronic pain","6":"Osteoporosis","7":"Hypertension","8":"Depression","9":"Atrial fibrillation","10":"Nausea and vomiting","11":"Urinary tract infection","branch":"Health","group":"Clinical medicine"}\n"""
    template = template + default_example

    # if examples are provided, add them to the template
    if examples:
        examp = format_examples(examples)
        template = template + examp

    # if a description is provided, add it to the beginning of the template
    if description:
        template = f"{description}\n{template}"

    # add the input text to the template
    template = template + f"[P]:{text_input}\n[O]:"
    return template
