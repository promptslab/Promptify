from promptify import OpenAI
from promptify import Prompter

sentence = """The patient is a 93-year-old female with a medical  				 
                history of chronic right hip pain, osteoporosis,					
                hypertension, depression, and chronic atrial						
                fibrillation admitted for evaluation and management				
                of severe nausea and vomiting and urinary tract				
                infection"""


model = OpenAI(api_key="")

prompter = Prompter(model=model, template="ner.jinja")

output = prompter.fit(text_input=sentence, domain="medical", labels=None)

print(output)
