from typing import List, Dict, Tuple


class TextClassification:
    def __init__(self):
        pass

    def binary(
        self,
        text_input: str,
        labels: List[str],
        description: str = "",
        examples: List[Tuple[str, str]] = [],
    ):
        """
        Perform binary text classification, classifying the input text as either the first label or the second label in the provided list.
        Parameters:
        text_input (str): The input text to be classified.
        labels (list): A list of two possible labels for the classification.
        description (str, optional): A description of the classification task. Default is an empty string.
        examples (list, optional): A list of examples, where each example is a tuple of the form (input_text, label). Default is an empty list.
        Returns:
        str: A string template describing the classification task and the input and output of the function.
        """
        # create the template string with the labels
        old_template = f"Perform binary text classification, classifying the input text as either {labels[0]} or {labels[1]}"
        template = f"You are a highly intelligent and accurate Binary Classification system. You take Passage as input and classify that as either {labels[0]} or {labels[1]} Category. Your output format is only [{{'C': Category}}] form, no other form.\n"

        if description:
            template = f"{description}\n{template}"
        template += "\n"

        # if examples are provided, add them to the template
        if examples:
            template += "Examples:\n"
            for example in examples:
                template += f"Input: {example[0]}\nOutput: { [{'C': example[1]}] }\n"

        # add the input text to the template
        template += f"\nInput: {text_input}\nOutput:"
        return template

    def multiclass(
        self,
        text_input: str,
        labels: List[str],
        description: str = "",
        examples: List[Tuple[str, str]] = [],
    ):
        """
        Perform multiclass text classification, classifying the input text as one of the labels in the provided list.
        Parameters:
        text_input (str): The input text to be classified.
        labels (list): A list of possible labels for the classification.
        description (str, optional): A description of the classification task. Default is an empty string.
        examples (list, optional): A list of examples, where each example is a tuple of the form (input_text, label). Default is an empty list.
        Returns:
        str: A string template describing the classification task and the input and output of the function.
        """
        # create the template string with the list of labels
        old_template = f"Perform multiclass text classification, classifying the input text as one of the following labels:\n{labels}"
        template = f"You are a highly intelligent and accurate Multiclass Classification system. You take Passage as input and classify that as one of the following appropriate Categories:\n{labels}. Your output format is only [{{'C': Appropriate Category from the list of provided Categories}}] form, no other form.\n"

        if description:
            template = f"{description}\n{template}"
        template += "\n\n"

        # if examples are provided, add them to the template
        if examples:
            template += "Examples:\n"
            for example in examples:
                template += f"Input: {example[0]}\nOutput: { [{'C': example[1]}] }\n"

        # add the input text to the template
        template += f"\nInput: {text_input}\nOutput:"
        return template

    def multilabel(
        self,
        text_input: str,
        domain: str = "",
        labels: List[str] = [],
        n_output_labels: int = 5,
        one_shot: bool = True,
        description: str = "",
        examples: List[Tuple[str, List[str]]] = [],
    ):
        
        """
        Perform multi-label text classification on the input text, categorizing it as one or more of the provided labels.
        Parameters:
        text_input (str): The input text to be classified.
        domain (str, optional): The domain of the classification task. Default is an empty string.
        labels (list, optional): A list of possible labels for the classification. Default is an empty list.
        n_output_labels (int): The maximum number of labels to be output for multi-labels. Default is 5.
        one_shot (bool): Indicates whether to include one-shot examples. Default is True.
        description (str, optional): A description of the classification task. Default is an empty string.
        examples (list, optional): A list of examples, where each example is a tuple of the form (input_text, labels). Default is an empty list. Example: [(text_1, [label1, label2, label3, label4])]
        Returns:
        str: A string template describing the classification task and the input and output of the function.
        """
        
        
        # create the template string with the labels

        if labels:
            if domain:
                old_template = f"Perform {domain} domain multi-label text classification, classifying the input text as one or more of the following labels:\n{labels}"
                template = f"You are a highly intelligent and accurate {domain} domain multi-label classification system. You take Passage as input and classify that into {n_output_labels} appropriate {domain} domain Categories:\n{labels}. Your output format is only [{{'main class': Main Classification Category ,'1': 2nd level Classification Category, '2': 3rd level Classification Category,...,'branch' : Appropriate branch of the Passage ,'group': Appropriate Group of the Passage}}] form, no other form.\n"
            else:
                old_template = f"Perform multi-label text classification, classifying the input text as one or more of the following labels:\n{labels}"
                template = f"You are a highly intelligent and accurate multi-label classification system. You take Passage as input and classify that into {n_output_labels} appropriate Categories:\n{labels}. Your output format is only [{{'main class': Main Classification Category ,'1': 2nd level Classification Category, '2': 3rd level Classification Category,..., 'branch' : Appropriate branch of the Passage ,'group': Appropriate Group of the Passage}}] form, no other form.\n"
        else:
            if domain:
                old_template = (
                    f"Perform {domain} domain multi-label text classification"
                )
                template = f"You are a highly intelligent and accurate {domain} domain multi-label classification system. You take Passage as input and classify that into {n_output_labels} appropriate {domain} domain Categories.\nYour output format is only [{{'main class': Main Classification Category ,'1': 2nd level Classification Category, '2': 3rd level Classification Category,..., 'branch' : Appropriate branch of the Passage ,'group': Appropriate Group of the Passage}}] form, no other form.\n"
            else:
                template = f"You are a highly intelligent and accurate multi-label classification system. You take Passage as input and classify that into {n_output_labels} appropriate Categories.\nYour output format is only [{{'main class': Main Classification Category ,'1': 2nd level Classification Category, '2': 3rd level Classification Category,...,'branch' : Appropriate branch of the Passage ,'group': Appropriate Group of the Passage}}] form, no other form.\n"
        if description:
            template = f"{description}\n{template}"
        template += "\n"

        if one_shot:
            default_example = """Examples:\nInput: The patient is a 93-year-old female with a medical history of chronic right hip pain, osteoporosis, hypertension, depression, and chronic atrial fibrillation admitted for evaluation and management of severe nausea and vomiting and urinary tract infection\nOutput: {"main class":"Health","1":"Medicine","2":"Patient care","3":"Discharge Summary","4":"Geriatric medicine","5":"Chronic pain","6":"Osteoporosis","7":"Hypertension","8":"Depression","9":"Atrial fibrillation","10":"Nausea and vomiting","11":"Urinary tract infection","branch":"Health","group":"Clinical medicine"}\n"""
            template = template + default_example

        # if examples are provided, add them to the template
        if examples:
            print("a")
            template += ""
            for example in examples:
                ex_labels = {str(k): example[1][k] for k in range(len(example[1]))}

                template += f"Input: {example[0]}\nOutput: { ex_labels }\n"

        # add the input text to the template
        template += f"\nInput: {text_input}\nOutput:"
        return template
