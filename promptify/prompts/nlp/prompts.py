import json
import numpy as np

def read_json(file):
    with open(file, 'r') as f:
        data = json.load(f)
    return data


def get_examples(n, task, domain = None):
    task_dict = {'ner': 'ner.json'}
    if task in task_dict:
        examples  = np.random.choice(read_json(task_dict[task]), n)
        examples  = [{"sentence": k['sentence'], "tags": k["tags"]} for k in examples]
        return examples



def ner(text_input, config = None):
    
    main_config       = {'task'          :  'ner',
                         'description'   :  '',
                         'domain'        :  '',
                         'n_shots'       :  1,
                         'n_ner'         :  10,
                         'output_format' :  [{"entity_group":"",
                                                "score":"",
                                                "word":"",
                                                "start":"",
                                                "end":""
                                             },]}
    
    if config:
        main_config.update(config)
    
    examples_samples = get_examples(main_config['n_shots'], main_config['task'], main_config['domain'])


    shots_template   = "Following are the examples of Named Entity Recognition task \n\n[examples]: " + str(examples_samples)
    if main_config['n_ner']!='':
        template = shots_template + "\n\nPerform "  + str(main_config['n_ner']) + " Named Entity Recognition on the below paragraph, the output should be in this format\n\n" + str(main_config['output_format']) + '\n\n[paragraph]: ' + text_input
        return template
    else:
        template = shots_template + "\n\nPerform many Named Entity Recognition on the below paragraph, , the output should be in this format\n\n" + str(main_config['output_format']) + '\n\n[paragraph]: ' + text_input
        return template    
