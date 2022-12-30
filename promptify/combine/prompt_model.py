from promptify.prompts.nlp.prompts import binary, multiclass, multilabel, ner, question_answer, question_answer_gen, \
    summarization, sentence_similarity
from promptify.models.nlp.models import GPT3
from promptify.models.nlp.model_list import top_model_list
import random


class CombinePromptModel:
    def __init__(self, task_type, api_key):
        self._gpt3_model = GPT3(api_key)

    def _get_prompt(self, text_input, config, task_type):
        if task_type == "binary":
            return binary(text_input=text_input, config=config)
        elif task_type == "multiclass":
            return multiclass(text_input=text_input, config=config)
        elif task_type == "multilabel":
            return multilabel(text_input=text_input, config=config)
        elif task_type == "ner":
            return ner(text_input=text_input, config=config)
        elif task_type == "question_answer":
            return question_answer(text_input=text_input, config=config)
        elif task_type == "question_answer_gen":
            return question_answer_gen(text_input=text_input, config=config)
        elif task_type == "summarization":
            return summarization(text_input=text_input, config=config)
        elif task_type == "sentence_similarity":
            return sentence_similarity(text_input=text_input, config=config)

    def get_model_generation_output(self, text_input, config, task_type, max_tokens):
        model_name = random.choice(top_model_list)
        prompt = self._get_prompt(text_input=text_input, config=config, task_type=task_type)
        results = self._gpt3_model.prompt_model_generation(
            model_name=model_name,
            prompts=prompt,
            max_tokens=max_tokens,
            multiple=False
        )

        return results

