import openai


class Openai_models:
    def __init__(self, api_key):
        self._api_key = api_key
        self._openai = openai
        self._openai.api_key = openai.api_key = self._api_key

    def get_model_list(self):
        list_of_models = [model.id for model in self._openai.Model.list()["data"]]
        return list_of_models

    def run(
        self,
        model_name=None,
        prompts=None,
        temperature=0.7,
        max_tokens=3646,
        top_p=0.1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None,
        multiple=False,
    ):
        result = []
        if multiple:
            for prompt in prompts:
                response = self._openai.Completion.create(
                    model=model_name,
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty,
                    stop=stop,
                )
                data = {}
                data.update(response["usage"])
                data["text"] = response["choices"][0]["text"]
                data["logprobs"] = response["choices"][0]["logprobs"]
                result.append(data)

        else:
            response = self._openai.Completion.create(
                model=model_name,
                prompt=prompts,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                stop=stop,
            )
            data = {}
            data.update(response["usage"])
            data["text"] = response["choices"][0]["text"]
            data["logprobs"] = response["choices"][0]["logprobs"]
            result.append(data)
        return result
