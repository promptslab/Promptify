Starter Tutorial
================
You can also check out the full documentation here. Here is a starter example for using `Promptify <installation.rst>`_. Make sure you've followed the installation steps first.

Quick Tour
----------
To immediately use a LLM model for your NLP task, we provide the Pipeline API.

.. code-block:: python
    
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


.. note:: 
    The `labels` parameter is optional. If you don't provide it, the model will automatically infer the labels from the text input. If you do provide it, the model will use the labels you provide.
    The output should be similar to the following:

.. code-block:: python

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


Features
--------
- Perform NLP tasks (such as NER and classification) in just 2 lines of code, with no training data required
- Easily add one shot, two shot, or few shot examples to the prompt
- Output always provided as a Python object (e.g. list, dictionary) for easy parsing and filtering
- Custom examples and samples can be easily added to the prompt
- Run inference on any model stored on the Huggingface Hub (see notebook guide)
- Optimized prompts to reduce OpenAI token costs (coming soon)

Community
---------
If you are interested in prompt engineering, LLMs, ChatGPT, and other latest research discussions, please consider joining `PromptsLab <https://discord.com/invite/m88xfYMbK6>`_. You can also join their `Discord community <https://discord.com/invite/m88xfYMbK6>`_.


Contributing
------------
We welcome any contributions to our open source project, including new features, improvements to infrastructure, and more comprehensive documentation. Please see the `contributing guidelines. <https://github.com/promptslab/Promptify/blob/main/contribute.md>`_.


License
-------
This project is licensed under the terms of the Apache 2.0 license. 
