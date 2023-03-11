Installation and Setup
======================

Installation from Pip
---------------------
To use Promptify, you need to have Python 3.7 or higher and openai 0.25 or higher installed on your computer.

You can install Promptify easily using the pip command in your terminal. Simply type the following command to install Promptify:

.. code-block:: bash
    
    $   pip install promptify



Installation from Source
------------------------
Alternatively,  If you want to install Promptify from source...

you can install Promptify directly from the Github repository using the following command:

.. code-block:: bash
    
    $   pip install git+https://github.com/promptslab/Promptify.git

- `pip install -e .` if you want to do an editable install (you can modify source files) of just the package itself.
- `pip install -r requirements.txt` if you want to install optional dependencies + dependencies used for development (e.g. unit testing).

Environment Setup
-----------------

By default, we use the OpenAI GPT-3 `text-davinci-003` model. In order to use this, you must have an OPENAI_API_KEY setup.
You can register an API key by logging into `OpenAI's page and creating a new API token <https://beta.openai.com/account/api-keys>`_. Once you have your API key... For example, if you want to use the `text-davinci-003` model, you can set the as follows:

.. code-block:: bash
    
    $   model = OpenAI(api_key="YOUR_API_KEY")
    $   nlp_prompter = Prompter(model)

You choose this according to the model you want to use. 
