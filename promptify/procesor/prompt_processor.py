import tiktoken
from typing import Dict, List, Optional, Tuple, Union

class PromptProcessor:
    def __init__(self, model: str = "text-davinci-003", token_limit: int = 4, api_key: str = None):


        self._openai = openai
        self.api_key = api_key
        self._openai.api_key = api_key

        self.model       = model
        self.token_limit = token_limit
        self.encoder     = tiktoken.encoding_for_model(model)

    def shorten_prompt(self, prompt):
        return textwrap.shorten(prompt, self.token_limit)

    def get_token_count(self, text: str) -> int:
        return len(self.encoder.encode(text))

    def adjust_token_limit(self, token_limit: int):
        self.token_limit = token_limit

    def split_prompt(self, prompt, max_parts=2):
        tokens = self.encoder.encode(prompt)
        part_length = len(tokens) // max_parts
        parts = [self.encoder.decode(prompt)(tokens[i * part_length:(i + 1) * part_length]) for i in range(max_parts)]
        return parts

    def generate_summary(self, prompt, summary_length=100):
        response = self._openai.Completion.create(
            engine=self.model,
            prompt=f"Please summarize the following text in {summary_length} words: {prompt}",
            max_tokens=summary_length,
            n=1,
            stop=None,
            temperature=0.5,
        )
        return response.choices[0].text.strip()
    

    def batch_prompt(self, prompt: str, batch_size: int = 4) -> List[str]:
        tokens = self.encoder.encode(prompt)
        batches = []
        start_idx = 0
        while start_idx < len(tokens):
            end_idx = start_idx + batch_size - 1
            batches.append(self.encoder.decode(tokens[start_idx:end_idx]).strip())
            start_idx = end_idx
        return batches
    

    def trim_prompt(self, prompt: str, mode: str) -> str:
        if mode not in ["pre", "post"]:
            raise ValueError('Invalid mode. Must be one of: "pre", "post"')
        if not prompt:
            raise ValueError("Input text is empty.")
        tokens = self.encoder.encode(prompt)
        if len(tokens) > self.token_limit:
            if mode == "pre":
                tokens = tokens[-self.token_limit:]
            else:
                tokens = tokens[:self.token_limit]
        return self.encoder.decode(tokens)
