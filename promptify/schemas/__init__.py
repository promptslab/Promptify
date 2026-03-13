from promptify.schemas.classify import Classification, MultiLabelResult
from promptify.schemas.extract import ExtractionResult, Relation, TableRow
from promptify.schemas.generate import GeneratedQuestion, SQLQuery
from promptify.schemas.ner import Entity, NERResult
from promptify.schemas.qa import Answer
from promptify.schemas.summarize import Summary

__all__ = [
    "Entity",
    "NERResult",
    "Classification",
    "MultiLabelResult",
    "Answer",
    "Summary",
    "Relation",
    "TableRow",
    "ExtractionResult",
    "GeneratedQuestion",
    "SQLQuery",
]
