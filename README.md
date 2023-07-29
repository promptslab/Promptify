<div align="center">
<img width="110px" src="https://raw.githubusercontent.com/promptslab/Promptify/main/assets/logo.png">
<h1>Promptify</h1></div>
<!-- 
<h2 align="center">Promptify</h2> -->

<p align="center">
  <p align="center">Prompt Engineering, Solve NLP Problems with LLM's & Easily generate different NLP Task prompts for popular generative models like GPT, PaLM, and more with Promptify
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

This repository is tested on Python 3.7+, openai 0.25+.

You should install Promptify using Pip command

```bash
pip3 install promptify
```

or

```bash
pip3 install git+https://github.com/promptslab/Promptify.git
```

## Quick tour

To immediately use a LLM model for your NLP task, we provide the `Pipeline` API.

```python
from promptify import Prompter,OpenAI, Pipeline

sentence     =  """The patient is a 93-year-old female with a medical  				 
                history of chronic right hip pain, osteoporosis,					
                hypertension, depression, and chronic atrial						
                fibrillation admitted for evaluation and management				
                of severe nausea and vomiting and urinary tract				
                infection"""

model        = OpenAI(api_key) # or `HubModel()` for Huggingface-based inference or 'Azure' etc
prompter     = Prompter('ner.jinja') # select a template or provide custom template
pipe         = Pipeline(prompter , model)


result = pipe.fit(sentence, domain="medical", labels=None)


### Output

[
    {"E": "93-year-old", "T": "Age"},
    {"E": "chronic right hip pain", "T": "Medical Condition"},
    {"E": "osteoporosis", "T": "Medical Condition"},
    {"E": "hypertension", "T": "Medical Condition"},
    {"E": "depression", "T": "Medical Condition"},
    {"E": "chronic atrial fibrillation", "T": "Medical Condition"},
    {"E": "severe nausea and vomiting", "T": "Symptom"},
    {"E": "urinary tract infection", "T": "Medical Condition"},
    {"Branch": "Internal Medicine", "Group": "Geriatrics"},
]
 
```

<p float="left">
  <img src="https://raw.githubusercontent.com/promptslab/Promptify/main/assets/ner.png" width="250" />
  <img src="https://raw.githubusercontent.com/promptslab/Promptify/main/assets/multilabel.png" width="250" /> 
  <img src="https://raw.githubusercontent.com/promptslab/Promptify/main/assets/qa_gen.png" width="250" />
</p>
<h4 align="center">GPT-3 Example with NER, MultiLabel, Question Generation Task</h3>


<h2>Features 🎮 </h2>
<ul>
  <li> Perform NLP tasks (such as NER and classification) in just 2 lines of code, with no training data required</li>
  <li> Easily add one shot, two shot, or few shot examples to the prompt</li>
  <li> Handling out-of-bounds prediction from LLMS (GPT, t5, etc.)</li>
  <li> Output always provided as a Python object (e.g. list, dictionary) for easy parsing and filtering. This is a major advantage over LLMs generated output, whose unstructured and raw output makes it difficult to use in business or other applications.</li>
  <li> Custom examples and samples can be easily added to the prompt</li>
  <li> 🤗 Run inference on any model stored on the Huggingface Hub (see <a href="https://github.com/promptslab/Promptify/blob/main/notebooks/huggingface.ipynb">notebook guide</a>).</li>
  <li> Optimized prompts to reduce OpenAI token costs (coming soon)</li>
</ul>


### Supporting wide-range of Prompt-Based NLP tasks :

| Task Name | Colab Notebook | Status |
|-------------|-------|-------|
| Named Entity Recognition | [NER Examples with GPT-3](https://colab.research.google.com/drive/16DUUV72oQPxaZdGMH9xH1WbHYu6Jqk9Q?usp=sharing) | ✅  |
| Multi-Label Text Classification | [Classification Examples with GPT-3](https://colab.research.google.com/drive/1gNqDxNyMMUO67DxigzRAOa7C_Tcr2g6M?usp=sharing) | ✅    |
| Multi-Class Text Classification | [Classification Examples with GPT-3](https://colab.research.google.com/drive/1gNqDxNyMMUO67DxigzRAOa7C_Tcr2g6M?usp=sharing) | ✅    |
| Binary Text Classification  | [Classification Examples with GPT-3](https://colab.research.google.com/drive/1gNqDxNyMMUO67DxigzRAOa7C_Tcr2g6M?usp=sharing) | ✅    |
| Question-Answering | [QA Task Examples with GPT-3](https://colab.research.google.com/drive/1Yhl7iFb7JF0x89r1L3aDuufydVWX_VrL?usp=sharing) | ✅    |
| Question-Answer Generation | [QA Task Examples with GPT-3](https://colab.research.google.com/drive/1Yhl7iFb7JF0x89r1L3aDuufydVWX_VrL?usp=sharing) | ✅    |
| Relation-Extraction | [Relation-Extraction Examples with GPT-3](https://colab.research.google.com/drive/1iW4QNjllc8ktaQBWh3_04340V-tap1co?usp=sharing) | ✅    |
| Summarization  | [Summarization Task Examples with GPT-3](https://colab.research.google.com/drive/1PlXIAMDtrK-RyVdDhiSZy6ztcDWsNPNw?usp=sharing) | ✅    |
| Explanation    | [Explanation Task Examples with GPT-3](https://colab.research.google.com/drive/1PlXIAMDtrK-RyVdDhiSZy6ztcDWsNPNw?usp=sharing) | ✅    |
| SQL Writer    | [SQL Writer Example with GPT-3](https://colab.research.google.com/drive/1JNUYCTdqkdeIAxiX-NzR-4dngdmWj0rV?usp=sharing) | ✅    |
| Tabular Data | |    |
| Image Data | |     |
| More Prompts | |     |


## Docs

[Promptify Docs](https://promptify.readthedocs.io/)

## Community 
<div align="center">
If you are interested in Prompt-Engineering, LLMs, ChatGPT and other latest research discussions, please consider joining <a href="https://discord.gg/m88xfYMbK6">PromptsLab</a></div>
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
