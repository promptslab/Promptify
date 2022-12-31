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
