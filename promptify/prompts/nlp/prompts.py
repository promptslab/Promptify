from prompt_utils import get_examples, get_default_config, get_template, get_shots_template


def binary(text_input, config=None):
    main_config = get_default_config("binary", config)

    examples_samples = get_examples(
        main_config["n_shots"], main_config["task"], main_config["domain"]
    )

    shots_template = get_shots_template("binary", examples_samples, main_config['description'])

    template = get_template(shots_template, main_config, text_input)
    return template


def multiclass(text_input, config=None):
    main_config = get_default_config("multiclass", config)

    examples_samples = get_examples(
        main_config["n_shots"], main_config["task"], main_config["domain"]
    )

    shots_template = get_shots_template("multiclass", examples_samples, main_config['description'])

    template = get_template(shots_template, main_config, text_input)
    return template


def multilabel(text_input, config=None):
    main_config = get_default_config("multilabel", config)

    examples_samples = get_examples(
        main_config["n_shots"], main_config["task"], main_config["domain"]
    )

    shots_template = get_shots_template("multilabel", examples_samples, main_config['description'])

    template = get_template(shots_template, main_config, text_input)
    return template


def ner(text_input, config=None):
    main_config = get_default_config("ner", config)

    examples_samples = get_examples(
        main_config["n_shots"], main_config["task"], main_config["domain"]
    )

    shots_template = get_shots_template("ner", examples_samples, main_config['description'])

    if main_config["n_ner"] != "":
        template = get_template(shots_template, main_config, text_input, isNER=True)
        return template
    else:
        template = get_template(shots_template, main_config, text_input)
        return template


def question_answer(text_input, config=None):
    main_config = get_default_config("question_answer", config)

    examples_samples = get_examples(
        main_config["n_shots"], main_config["task"], main_config["domain"]
    )

    shots_template = get_shots_template("question_answer", examples_samples, main_config['description'])

    template = get_template(shots_template, main_config, text_input)
    return template


def question_answer_gen(text_input, config=None):
    main_config = get_default_config("question_answer_gen", config)

    examples_samples = get_examples(
        main_config["n_shots"], main_config["task"], main_config["domain"]
    )

    shots_template = get_shots_template("question_answer_gen", examples_samples, main_config['description'])

    template = get_template(shots_template, main_config, text_input)
    return template


def summarization(text_input, config=None):
    main_config = get_default_config("summarization", config)

    examples_samples = get_examples(
        main_config["n_shots"], main_config["task"], main_config["domain"]
    )

    shots_template = get_shots_template("summarization", examples_samples, main_config['description'])

    template = get_template(shots_template, main_config, text_input)
    return template


def sentence_similarity(text_input, config=None):
    main_config = get_default_config("sentence_similarity", config)

    examples_samples = get_examples(
        main_config["n_shots"], main_config["task"], main_config["domain"]
    )

    shots_template = get_shots_template("sentence_similarity", examples_samples, main_config['description'])

    template = get_template(shots_template, main_config, text_input)
    return template
