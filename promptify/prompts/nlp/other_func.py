def binary_classification(text_input, config=None):
    default_config = {
        "description": "",
        "examples": [],
        "labels"  : [],
    }

    if config:
        default_config.update(config)

    assert (
        default_config["labels"] != []
    ), "Please provide Binary Labels as a list -> ['Positive', 'Negative']"
    labels = default_config["labels"]
    description = default_config["description"]

    if description != "":
        template = (
            description
            + f"\nPerform Binary Text Classification, the output will be either {labels[0]} or {labels[1]}\n\n"
        )
    else:
        template = f"Perform Binary Text Classification, the output will be either {labels[0]} or {labels[1]}\n\n"

    if default_config["examples"]:
        examp = format_examples(default_config["examples"])
        template = template + examp

    template = template + f"[P]:{text_input}\n[O]:"
    return template
