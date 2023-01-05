__version__ = "0.1.0"

from .models.nlp.openai_model import OpenAI
from .prompts.nlp import (classification, explain, ner, question_answering,
                          summary)
