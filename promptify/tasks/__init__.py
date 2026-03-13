from promptify.tasks.base import BaseTask, Task
from promptify.tasks.classify import Classify
from promptify.tasks.extract import ExtractRelations, ExtractTable
from promptify.tasks.generate import GenerateQuestions, GenerateSQL
from promptify.tasks.ner import NER
from promptify.tasks.normalize import ExtractTopics, NormalizeText
from promptify.tasks.qa import QA
from promptify.tasks.summarize import Summarize

__all__ = [
    "BaseTask",
    "Task",
    "NER",
    "Classify",
    "QA",
    "Summarize",
    "ExtractRelations",
    "ExtractTable",
    "GenerateQuestions",
    "GenerateSQL",
    "NormalizeText",
    "ExtractTopics",
]
