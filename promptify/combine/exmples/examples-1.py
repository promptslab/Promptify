from promptify.combine.prompt_model import CombinePromptModel

text_input = ""
config = None
task_type = ""
api_key = ""
max_tokens = 10
model = CombinePromptModel(
    api_key=api_key,
)

results = model.get_model_generation_output(
    text_input=text_input, config=config, task_type=task_type, max_tokens=max_tokens
)

print(results)
