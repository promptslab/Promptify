from promptify import Prompter,OpenAI, Pipeline

sentence     =  "The patient is a 93-year-old female with a medical  				 
                history of chronic right hip pain, osteoporosis,					
                hypertension, depression, and chronic atrial						
                fibrillation admitted for evaluation and management				
                of severe nausea and vomiting and urinary tract				
                infection"

model        = OpenAI(api_key) # or `HubModel()` for Huggingface-based inference or 'Azure' etc
prompter     = Prompter('ner.jinja') # select a template or provide custom template
pipe         = Pipeline(prompter , model)

output = pipe.fit(text_input=sentence, domain="medical", labels=None)
print(output)
