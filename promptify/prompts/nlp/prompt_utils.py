import json
import numpy as np

def read_json(file):
    with open(file, 'r') as f:
        data = json.load(f)
    return data

def get_examples(n, task, domain):
    
    task_dict = {'ner': 'ner.json', 'multilabel': 'ml.json'}
    if task in task_dict:
        examples  = read_json(task_dict[task])
        examples  = [k for k in examples if k['domain']==domain]
        examples  = np.random.choice(examples, n)
        
        examples  = [{"sentence": k['sentence'], "data": k["data"]} for k in examples]
        return examples
      
      
