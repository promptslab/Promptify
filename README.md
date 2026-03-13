<div align="center">
<img width="110px" src="https://raw.githubusercontent.com/promptslab/Promptify/main/assets/logo.png">
<h1>Promptify</h1></div>
<!-- 
<h2 align="center">Promptify</h2> -->

<p align="center">
  <p align="center">Task-based NLP engine with Pydantic structured outputs, built-in evaluation, and LiteLLM as the universal LLM backend. Think "scikit-learn for LLM-powered NLP".
</p>
</p>

 <h4 align="center">
  <a href="https://github.com/promptslab/Promptify/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="Promptify is released under the Apache 2.0 license." />
  </a>
  <a href="https://pypi.org/project/promptify/">
    <img src="https://badge.fury.io/py/Promptify.svg" alt="PyPI version" />
  </a>
  <a href="http://makeapullrequest.com">
    <img src="https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square" alt="http://makeapullrequest.com" />
  </a>
  <a href="https://discord.gg/m88xfYMbK6">
    <img src="https://img.shields.io/badge/Discord-Community-orange" alt="Community" />
  </a>
  <a href="#">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="colab" />
  </a>
</h4>

<img width="910px" src="https://raw.githubusercontent.com/promptslab/Promptify/main/assets/dark.png">

## Installation

### With pip

Requires Python 3.9+.

```bash
pip install promptify
```

or

```bash
pip install git+https://github.com/promptslab/Promptify.git
```

For evaluation metrics support:
```bash
pip install promptify[eval]
```

## Quick Tour

### 3-Line NER

```python
from promptify import NER

ner = NER(model="gpt-4o-mini", domain="medical")
result = ner("The patient is a 93-year-old female with a medical history of chronic right hip pain, osteoporosis, hypertension, depression, and chronic atrial fibrillation admitted for evaluation and management of severe nausea and vomiting and urinary tract infection")
```

**Output:**
```python
NERResult(entities=[
    Entity(text="93-year-old", label="AGE"),
    Entity(text="chronic right hip pain", label="CONDITION"),
    Entity(text="osteoporosis", label="CONDITION"),
    Entity(text="hypertension", label="CONDITION"),
    Entity(text="depression", label="CONDITION"),
    Entity(text="chronic atrial fibrillation", label="CONDITION"),
    Entity(text="severe nausea and vomiting", label="SYMPTOM"),
    Entity(text="urinary tract infection", label="CONDITION"),
])
```

### Classification

```python
from promptify import Classify

clf = Classify(model="gpt-4o-mini", labels=["positive", "negative", "neutral"])
result = clf("Amazing product! Best purchase I've ever made.")
# Classification(label="positive", confidence=0.95)
```

### Question Answering

```python
from promptify import QA

qa = QA(model="gpt-4o-mini")
answer = qa("Einstein was born in Ulm in 1879.", question="Where was Einstein born?")
# Answer(answer="Ulm", evidence="Einstein was born in Ulm", confidence=0.98)
```

### Custom Task with Any Pydantic Schema

```python
from promptify import Task
from pydantic import BaseModel

class MovieReview(BaseModel):
    sentiment: str
    rating: float
    key_themes: list[str]

task = Task(model="gpt-4o", output_schema=MovieReview, instruction="Analyze this movie review.")
review = task("Nolan's best work. Stunning visuals but the plot drags.")
# MovieReview(sentiment="mostly positive", rating=7.5, key_themes=["visuals", "pacing"])
```

### Any Provider - Just Change the Model String

```python
ner_openai = NER(model="gpt-4o-mini")
ner_claude = NER(model="claude-sonnet-4-20250514")
ner_local  = NER(model="ollama/llama3")
```

### Batch Processing

```python
results = ner.batch(["text1", "text2", "text3"], max_concurrent=10)
```

### Async Support

```python
result = await ner.acall("Patient has diabetes")
```

### Built-in Evaluation

```python
from promptify.eval import evaluate

scores = evaluate(task=ner, dataset=labeled_data, metrics=["precision", "recall", "f1"])
# {"precision": 0.92, "recall": 0.88, "f1": 0.90}
```

## Features

- **2-3 lines of code** for any NLP task -no training data required
- **Pydantic structured outputs** -type-safe results, not raw strings
- **Any LLM provider** via LiteLLM -OpenAI, Anthropic, Google, Ollama, Azure, and 100+ more
- **Built-in tasks** -NER, Classification (binary/multiclass/multilabel), QA, Summarization, Relation Extraction, SQL Generation, and more
- **Custom tasks** -bring your own Pydantic schema for any structured output
- **Few-shot examples** -easily add examples to improve accuracy
- **Domain specialization** -pass `domain="medical"` or any domain for context-aware prompts
- **Batch processing** -async concurrency under the hood for processing multiple texts
- **Async support** -native `await` support with `acall()`
- **Evaluation framework** -precision, recall, F1, accuracy, exact match, ROUGE metrics
- **Safe parser** -fallback JSON completion for providers without native structured outputs (no `eval()`)
- **Cost tracking** -built-in token usage and cost monitoring via `get_cost_summary()`

### Supported NLP Tasks

| Task | Class | Output Schema |
|------|-------|---------------|
| Named Entity Recognition | `NER` | `NERResult` (list of `Entity`) |
| Binary Classification | `Classify` | `Classification` |
| Multiclass Classification | `Classify` | `Classification` |
| Multilabel Classification | `Classify(multi_label=True)` | `MultiLabelResult` |
| Question Answering | `QA` | `Answer` |
| Summarization | `Summarize` | `Summary` |
| Relation Extraction | `ExtractRelations` | `ExtractionResult` |
| Tabular Extraction | `ExtractTable` | `ExtractionResult` |
| Question Generation | `GenerateQuestions` | list of `GeneratedQuestion` |
| SQL Generation | `GenerateSQL` | `SQLQuery` |
| Text Normalization | `NormalizeText` | normalized text |
| Topic Modelling | `ExtractTopics` | list of topics |
| Custom Task | `Task` | any Pydantic `BaseModel` |

## Community
<div align="center">
If you are interested in Prompt-Engineering, LLMs, and NLP, please consider joining <a href="https://discord.gg/m88xfYMbK6">PromptsLab</a></div>
<div align="center">
<img alt="Join us on Discord" src="https://img.shields.io/discord/1069129502472556587?color=5865F2&logo=discord&logoColor=white">
</div>



```

@misc{Promptify2022,
  title = {Promptify: Structured Output from LLMs},
  author = {Pal, Ankit},
  year = {2022},
  howpublished = {\url{https://github.com/promptslab/Promptify}},
  note = {Prompt-Engineering components for NLP tasks in Python}
}

```

## 💁 Contributing

We welcome any contributions to our open source project, including new features, improvements to infrastructure, and more comprehensive documentation. 
Please see the [contributing guidelines](contribute.md)
