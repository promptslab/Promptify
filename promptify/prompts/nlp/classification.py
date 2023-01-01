from typing import List, Dict, Tuple


class TextClassification:
    def __init__(self, text_input: str):
        self.text_input = text_input

    def binary(self, labels: List[str], description: str = "", examples: List[Tuple[str, str]] = []):
        """
        Perform binary text classification, classifying the input text as either the first label or the second label in the provided list.
        Parameters:
        labels (list): A list of two possible labels for the classification.
        description (str, optional): A description of the classification task. Default is an empty string.
        examples (list, optional): A list of examples, where each example is a tuple of the form (input_text, label). Default is an empty list.
        Returns:
        str: A string template describing the classification task and the input and output of the function.
        """
        # create the template string with the labels
        template = f"Perform binary text classification, classifying the input text as either {labels[0]} or {labels[1]}"
        if description:
            template = f"{description}\n{template}"
        template += "\n\n"

        # if examples are provided, add them to the template
        if examples:
            template += "Examples:\n"
            for example in examples:
                template += f"Input: {example[0]}\nOutput: {example[1]}\n"

        # add the input text to the template
        template += f"\nInput:\n{self.text_input}\nOutput:"
        return template

    def multiclass(self, labels: List[str], description: str = "", examples: List[Tuple[str, str]] = []):
        """
        Perform multiclass text classification, classifying the input text as one of the labels in the provided list.
        Parameters:
        labels (list): A list of possible labels for the classification.
        description (str, optional): A description of the classification task. Default is an empty string.
        examples (list, optional): A list of examples, where each example is a tuple of the form (input_text, label). Default is an empty list.
        Returns:
        str: A string template describing the classification task and the input and output of the function.
        """
        # create the template string with the list of labels
        template = f"Perform multiclass text classification, classifying the input text as one of the following labels:\n{labels}"
        if description:
            template = f"{description}\n{template}"
        template += "\n\n"

        # if examples are provided, add them to the template
        if examples:
            template += "Examples:\n"
            for example in examples:
                template += f"Input: {example[0]}\nOutput: {example[1]}\n"

        # add the input text to the template
        template += f"\nInput:\n{self.text_input}\nOutput:"
        return template
    
    
    def multilabel(self, domain: str = "", labels: List[str] = [], description: str = "", examples: List[Tuple[str, List[str]]] = []):
        """
        Perform multi-label text classification, classifying the input text as one or more of the labels in the provided list.
        Parameters:
        domain (str, optional): The domain of the classification task. Default is an empty string.
        labels (list, optional): A list of possible labels for the classification. Default is an empty list.
        description (str, optional): A description of the classification task. Default is an empty string.
        examples (list, optional): A list of examples, where each example is a tuple of the form (input_text, labels). Default is an empty list.
        Returns:
        str: A string template describing the classification task and the input and output of the function.
        """
        # create the template string with the labels
        if labels:
            if domain:
                template = f"Perform {domain} domain multi-label text classification, classifying the input text as one or more of the following labels:\n{labels}"
            else:
                template = f"Perform multi-label text classification, classifying the input text as one or more of the following labels:\n{labels}"
        else:
            if domain:
                template = f"Perform {domain} domain multi-label text classification"
            else:
                template = "Perform multi-label text classification"
        if description:
            template = f"{description}\n{template}"
        template += "\n\n"
        
        
        default_example = """Examples:\nInput:The patient is a 93-year-old female with a medical history of chronic right hip pain, osteoporosis, hypertension, depression, and chronic atrial fibrillation admitted for evaluation and management of severe nausea and vomiting and urinary tract infection\nOutput:{"main class":"Health","1":"Medicine","2":"Patient care","3":"Discharge Summary","4":"Geriatric medicine","5":"Chronic pain","6":"Osteoporosis","7":"Hypertension","8":"Depression","9":"Atrial fibrillation","10":"Nausea and vomiting","11":"Urinary tract infection","branch":"Health","group":"Clinical medicine"}\n"""
        template = template + default_example

        # if examples are provided, add them to the template
        if examples:
            template += ""
            for example in examples:
                template += f"Input: {example[0]}\nOutput: {example[1]}\n"

        # add the input text to the template
        template += f"\nInput:\n{self.text_input}\nOutput:"
        return template
