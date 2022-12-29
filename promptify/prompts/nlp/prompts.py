from prompt_utils import get_examples


def ner(text_input, config=None):
    main_config = {
        "task": "ner",
        "description": "",
        "domain": "medical",
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
    
    

def multilabel(text_input, config=None):
    levels = {
        1: "First",
        2: "Second",
        3: "Third",
        4: "Fourth",
        5: "Fifth",
        6: "Sixth",
        7: "Seventh",
        8: "Eighth",
        9: "Ninth",
        10: "Tenth",
        11: "Eleventh",
        12: "Twelfth",
        13: "Thirteenth",
        14: "Fourteenth",
        15: "Fifteenth",
        16: "Sixteenth",
        17: "Seventeenth",
        18: "Eighteenth",
        19: "Nineteenth",
        20: "Twentieth",
        21: "Twenty-first",
        22: "Twenty-second",
        23: "Twenty-third",
        24: "Twenty-fourth",
        25: "Twenty five",
        26: "Twenty six",
    }

    main_config = {
        "task": "multilabel",
        "description": "",
        "domain": "medical",
        "n_shots": 1,
        "n_level": 3,
        "output_format": "",
    }

    main_class = [{"main class": "", "confidence_score": ""}]
    last_info = {"branch": "", "group": ""}

    output_format = [
        {f"{levels[k+1]} level class": "", "confidence_score": ""}
        for k in range(int(main_config["n_level"]))
    ]
    main_class.extend(output_format)
    main_class.append(last_info)
    main_config["output_format"] = main_class

    if config:
        main_config.update(config)

    examples_samples = get_examples(
        main_config["n_shots"], main_config["task"], main_config["domain"]
    )
    if main_config["description"] != "":
        shots_template = (
            main_config["description"]
            + "\nFollowing are the examples of Multi-Label Text Classification \n\n[examples]: "
            + str(examples_samples)
        )
    else:
        shots_template = (
            "Following are the examples of Multi-Label Text Classification \n\n[examples]: "
            + str(examples_samples)
        )

    template = (
        shots_template
        + "\n\nPerform Multi-Label Text Classification on the below paragraph, The output must be in the below form\n\n"
        + str(main_config["output_format"])
        + "\n\n[paragraph]: "
        + text_input
    )
    return template
