from typing import List, Tuple

def QA(
    context: str,  # the context paragraph
    question: str,  # the question to be answered
    domain: str = "",  # the domain of the question, optional
    description: str = "",  # a description of the function, optional
    examples: List[Tuple[str, str, str]] = [],  # a list of example contexts, questions, and answers, optional
):
    """
    This function returns a template string with the input context and question, as well as any provided examples or
    description. The template also includes a description of the function's purpose. If a domain is provided, it is
    included in the function's description.
    """
    if domain:
        template = f"You are a highly intelligent {domain} domain question answering bot. You take Context and Question as input and return the answer from the Paragraph. Retain as much information as needed to answer the question at a later time."
    else:
        template = f"You are a highly intelligent question answering bot. You take Context and Question as input and return the answer from the Paragraph. Retain as much information as needed to answer the question at a later time."
    # if a description is provided, add it to the template
    if description:
        template = f"{description}\n{template}"
    template += "\n\n"
    # add a default example to the template
    default_example = """Examples:
Context: Beyoncé Giselle Knowles-Carter (/biːˈjɒnseɪ/ bee-YON-say) (born September 4, 1981) is an American singer, songwriter, record producer and actress. Born and raised in Houston, Texas, she performed in various singing and dancing competitions as a child, and rose to fame in the late 1990s as lead singer of R&B girl-group Destiny’s Child.
Question: When did Beyonce start becoming popular?
Answer: in the late 1990s
Question: What areas did Beyonce compete in when she was growing up?
Answer: singing and dancing
"""
    template += default_example
    # if examples are provided, add them to the template
    if examples:
        for example in examples:
            template += f"Context: {example[0]}\nQuestion: {example[1]}\nAnswer: {example[2]}\n"
    # add the input context and question to the template
    template += f"\nContext: {context}\nQuestion: {question}\nAnswer:"
    return template
