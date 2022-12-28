from promptify.models.nlp.models import GPT3
from promptify.models.nlp.model_list import top_model_list
import random

api_key = "<api-key>"
gpt3_model = GPT3(api_key)

model_name = random.choice(top_model_list)
prompt1 = """
Below is a paragraph from the Medical domain. Perform a multi-label classification on a given paragraph; the Output will be as follows.

{'classes' [Multi-label classification classes]}

"abdominal cavity, largest hollow space of the body. Its upper boundary is the diaphragm, a sheet of muscle and connective tissue that 
"""
results = gpt3_model.prompt_model_generation(model_name=model_name, prompts=prompt1, max_tokens=100, multiple=False)

print(results)
