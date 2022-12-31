import json
import numpy as np
from constants import task_type, task_dict


def read_json(file):
    with open(file, 'r') as f:
        data = json.load(f)
    return data


def format_examples(examples):
    text_template = ""
    for sample in examples:
        text_template+=f"[P]:{sample['text']}\n[O]:{sample['label']}\n"
    return text_template


def get_examples(n, task, domain):
    if task in task_dict:
        examples = read_json(task_dict[task])
        examples = [k for k in examples if k['domain'] == domain]
        examples = np.random.choice(examples, n)
        examples = [{"sentence": k['sentence'], "data": k["data"]} for k in examples]
        return examples


def get_default_config(task_name, config=None):
    default_args = {
        "description": "",
        "domain": "medical",
        "n_shots": 1,
        "n_level": 3,
    }

    default_config = {} if config is None else config

    if config:
        default_config.update(config)

    default_config['task'] = task_name

    if task_name == "binary":
        output_format = [
            {
                "label": "",
                "score": "",
                "complexity": ""
            }
        ]

        default_config["output_format"] = output_format

    elif task_name == "multiclass":
        output_format = [
            {
                "category": "",
                "confidence_score": "",
                "complexity": ""
            }
        ]

        default_config["output_format"] = output_format

    elif task_name == "multilabel":
        main_class = [{"main class": "", "confidence_score": ""}]
        last_info = {"branch": "", "group": ""}
        output_format = [
            {f"{levels[k + 1]} level class": "", "confidence_score": ""}
            for k in range(int(config["n_level"]))
        ]
        main_class.extend(output_format)
        main_class.append(last_info)
        default_config["output_format"] = main_class

    elif task_name == "ner":
        default_config['output_format'] = [
            {"entity_group": "", "score": "", "word": "", "start": "", "end": ""},
        ]

    elif task_name == "question_answer":
        output_format = [
            {
                "question": "",
                "extracted_answer": "",
                "offset": ""
            }
        ]
        default_config['output_format'] = output_format

    elif task_name == "question_answer_gen":
        output_format = [
            {
                "question": "",
                "extracted_answer": "",
                "reasoning_type": ""
            }
        ]
        default_config['output_format'] = output_format

    elif task_name == "summarization":
        output_format = [
            {
                "summary_text": ""
            }
        ]
        default_config['output_format'] = output_format

    elif task_name == "sentence_similarity":
        output_format = [
            {
                "similarity_score": ""
            }
        ]
        default_config['output_format'] = output_format

    default_config.update(default_args)
    return default_config


def get_template(shots_template, config, text_input, isNER=False):
    if isNER:
        return (
                shots_template
                + "\n\nPerform "
                + str(config["n_ner"])
                + " Named Entity Recognition on the below paragraph, The output must be in the below form\n\n"
                + str(config["output_format"])
                + "\n\n[paragraph]: "
                + text_input
        )
    template = (
            shots_template
            + "\n\nPerform Binary Text Classification on the below paragraph, The output must be in the below form\n\n"
            + str(config['output_format'])
            + "\n\n[paragraph]: "
            + text_input
    )
    return template


def get_shots_template(task_name, examples, description=""):
    if description != "":
        shots_template = (
                description
                + f"\nFollowing are the examples of {task_type[task_name]} task \n\n[examples]: "
                + str(examples)
        )
    else:
        shots_template = (
                f"Following are the examples of {task_type[task_name]} task \n\n[examples]: "
                + str(examples)
        )
    return shots_template
