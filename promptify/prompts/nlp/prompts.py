from prompt_utils import get_examples


def ner(text_input, config=None):
    main_config = {
        "task": "ner",
        "description": "",
        "domain": "",
        "n_shots": 1,
        "n_ner": "",
        "output_format": [
            {"entity_group": "", "score": "", "word": "", "start": "", "end": ""},
        ],
    }

    if config:
        main_config.update(config)

    examples_samples = get_examples(
        main_config["n_shots"], main_config["task"], main_config["domain"]
    )
    if main_config["description"] != "":
        shots_template = (
            main_config["description"]
            + "\nFollowing are the examples of Named Entity Recognition task \n\n[examples]: "
            + str(examples_samples)
        )
    else:
        shots_template = (
            "Following are the examples of Named Entity Recognition task \n\n[examples]: "
            + str(examples_samples)
        )

    if main_config["n_ner"] != "":
        template = (
            shots_template
            + "\n\nPerform "
            + str(main_config["n_ner"])
            + " Named Entity Recognition on the below paragraph, The output must be in the below form\n\n"
            + str(main_config["output_format"])
            + "\n\n[paragraph]: "
            + text_input
        )
        return template
    else:
        template = (
            shots_template
            + "\n\nPerform Named Entity Recognition on the below paragraph, The output must be in the below form\n\n"
            + str(main_config["output_format"])
            + "\n\n[paragraph]: "
            + text_input
        )
        return template
