<div align="center">
<img width="110px" src="https://raw.githubusercontent.com/promptslab/Promptify/main/logo/logo.png">
<h1>Promptify</h1></div>
<!-- 
<h2 align="center">Promptify</h2> -->
<h4 align="center"> Solve NLP Problems with LLM's & Easily generate different NLP Task prompts for popular generative models like GPT, PaLM, and more with Promptify</h3>

 

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![GitHub commit](https://img.shields.io/github/last-commit/monk1337/resp)](https://github.com/promptslab/Promptify/commits/main)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)
[![Open All Collab](https://colab.research.google.com/assets/colab-badge.svg)](#)


## Installation

### With pip

This repository is tested on Python 3.7+, openai 0.25+.

You should install Promptify using Pip command

```bash
pip3 install promptify
```

## Quick tour

To immediately use a LLM model for your NLP task, we provide the `Prompter` API.

```python
from promptify import OpenAI
from promptify import Prompter

sentence     =  "The patient is a 93-year-old female with a medical  				 
                history of chronic right hip pain, osteoporosis,					
                hypertension, depression, and chronic atrial						
                fibrillation admitted for evaluation and management				
                of severe nausea and vomiting and urinary tract				
                infection"

model        = OpenAI(api_key)
nlp_prompter = Prompter(model)


result       = nlp_prompter.fit('ner.jinja',
                          domain      = 'medical',
                          text_input  = sentence)
                          
                          
### Output

[{'E': '93-year-old', 'T': 'Age'},
 {'E': 'chronic right hip pain', 'T': 'Medical Condition'},
 {'E': 'osteoporosis', 'T': 'Medical Condition'},
 {'E': 'hypertension', 'T': 'Medical Condition'},
 {'E': 'depression', 'T': 'Medical Condition'},
 {'E': 'chronic atrial fibrillation', 'T': 'Medical Condition'},
 {'E': 'severe nausea and vomiting', 'T': 'Symptom'},
 {'E': 'urinary tract infection', 'T': 'Medical Condition'},
 {'Branch': 'Internal Medicine', 'Group': 'Geriatrics'}]
 
```

<p float="left">
  <img src="https://raw.githubusercontent.com/promptslab/Promptify/main/logo/ner.png" width="250" />
  <img src="https://raw.githubusercontent.com/promptslab/Promptify/main/logo/multilabel.png" width="250" /> 
  <img src="https://raw.githubusercontent.com/promptslab/Promptify/main/logo/qa_gen.png" width="250" />
</p>
<h4 align="center">GPT-3 Example with NER, MultiLabel, Question Generation Task</h3>


<h2>Features 🎮 </h2>
<ul>
  <li> Perform NLP tasks (such as NER and classification) in just 2 lines of code, with no training data required</li>
  <li> Easily add one shot, two shot, or few shot examples to the prompt</li>
  <li> Handling out-of-bounds prediction from LLMS (GPT, t5, etc.)
  <li> Output always provided as a Python object (e.g. list, dictionary) for easy parsing and filtering. This is a major advantage over LLMs generated output, whose unstructured and raw output makes it difficult to use in business or other applications.</li>
  <li> Custom examples and samples can be easily added to the prompt</li>
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
| Summarization  | [Summarization Task Examples with GPT-3](https://colab.research.google.com/drive/1PlXIAMDtrK-RyVdDhiSZy6ztcDWsNPNw?usp=sharing) | ✅    |
| Explanation    | [Explanation Task Examples with GPT-3](https://colab.research.google.com/drive/1PlXIAMDtrK-RyVdDhiSZy6ztcDWsNPNw?usp=sharing) | ✅    |
| Tabular Data | |    |
| Image Data | |     |
| More Prompts | |     |



## 💁 Contributing

We welcome any contributions to our open source project, including new features, improvements to infrastructure, and more comprehensive documentation.
