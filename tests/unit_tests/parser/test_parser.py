import pytest
from promptify import Parser


@pytest.fixture
def parser():
    parser = Parser()
    return parser


def test_is_valid_json(parser):
    assert parser.is_valid_json('{"name": "Alice", "age": 30}')
    assert parser.is_valid_json("[1, 2, 3, 4]")
    assert not parser.is_valid_json('{"name": "Bob", "age": }')
    assert not parser.is_valid_json("not a JSON string")


def test_get_combinations(parser):
    combinations = parser.get_combinations(["}", "]"], 3, should_end_mark="}")
    assert combinations == ["}", "}}", "]}"]


def test_escaped_(parser):
    case_1 = """[[{'T': 'ANATOMY', 'E': 'immune system'},{'T': 'DISEASE', 'E': 'Parkinson's disease'},{'T': 'other', 'E': 'person's health'}]]"""
    case_2 = """[[{"T": "ANATOMY", "E": "immune system"},{"T": "DISEASE", "E": "Parkinson"s disease"},{"T": "other", "E": "person"s health"}]]"""
    case_3 = """[[{'T': 'ANATOMY', 'E': 'immune system'}, {'T': 'DISEASE', 'E': 'Parkinson disease'}, {'T': 'other', 'E': 'person health'}]]"""

    result_1 = parser.escaped_(case_1)
    assert eval(result_1) == [
        [
            {"T": "ANATOMY", "E": "immune system"},
            {"T": "DISEASE", "E": 'Parkinson"s disease'},
            {"T": "other", "E": 'person"s health'},
        ]
    ]

    result_2 = parser.escaped_(case_2)
    assert eval(result_2) == [
        [
            {"T": "ANATOMY", "E": "immune system"},
            {"T": "DISEASE", "E": "Parkinson's disease"},
            {"T": "other", "E": "person's health"},
        ]
    ]

    result_3 = parser.escaped_(case_3)
    assert eval(result_3) == [
        [
            {"T": "ANATOMY", "E": "immune system"},
            {"T": "DISEASE", "E": "Parkinson disease"},
            {"T": "other", "E": "person health"},
        ]
    ]


def test_complete_json_object(parser):
    assert parser.complete_json_object('{"a": 1, "b": 2', "}") == {"a": 1, "b": 2}
    assert parser.complete_json_object("[1, 2, 3", "]") == [1, 2, 3]

    with pytest.raises(ValueError):
        parser.complete_json_object('{"a": 1, "b": 2', "")


def test_get_possible_completions(parser):
    completion = parser.get_possible_completions('{"a": 1, "b": 2')["completion"]
    assert completion == {"a": 1, "b": 2}
    completion = parser.get_possible_completions("[1, 2, 3")["completion"]
    assert completion == [1, 2, 3]


def test_fit(parser):
    result = parser.fit('{"name": "Alice", "age":30}')

    case_1 = """[{'a' : 1, 'b' : 2}, {'a' : 1'"""
    assert parser.fit(case_1, 10)["data"]["completion"] == [{"a": 1, "b": 2}, {"a": 1}]

    case_2 = "[[1, 2, 3], [11, 12, 21]"
    assert parser.fit(case_2, 10)["data"]["completion"] == [[1, 2, 3], [11, 12, 21]]

    case_3 = """[{'a': [{'a': 1, 'b': 2}, {'a': 12, 'b': 23}], 'b': [{'a':"""
    assert parser.fit(case_3, 10)["data"]["completion"] == [
        {"a": [{"a": 1, "b": 2}, {"a": 12, "b": 23}], "b": [{"a"}]}
    ]

    case_4 = "[[{'a': [1, 2, 3], 'b': {'c': 4}}, {'d': 5}], {'e': {'f': {'g': 6}}}]"
    assert parser.fit(case_4, 10)["data"]["completion"] == [
        [{"a": [1, 2, 3], "b": {"c": 4}}, {"d": 5}],
        {"e": {"f": {"g": 6}}},
    ]

    case_5 = "[[{'a': [1, 2, 3], 'b': {'c': 4}}, {'d': 5}], {'e': {'f': {'g': 6]"
    assert parser.fit(case_5, 10)["data"]["completion"] == [
        [{"a": [1, 2, 3], "b": {"c": 4}}, {"d": 5}],
        {"e": {"f": {"g": 6}}},
    ]

    case_6 = "[{'a': 1}, {'b': 2"
    assert parser.fit(case_6, 10)["data"]["completion"] == [{"a": 1}, {"b": 2}]

    case_7 = """{"person": {"name": "Alice", "age": 30, "hobbies": ["reading", "running", {"favorite_movies": ["Inception", "The Matrix"]}, {"favorite_songs": ["Imagine", "Let it Be"]}"""
    assert parser.fit(case_7, 10)["data"]["completion"] == {
        "person": {
            "name": "Alice",
            "age": 30,
            "hobbies": [
                "reading",
                "running",
                {"favorite_movies": ["Inception", "The Matrix"]},
                {"favorite_songs": ["Imagine", "Let it Be"]},
            ],
        }
    }

    case_8 = """{"name": "Bob", "age": 25, "is_student": False, "scores": [85, 90, 78], "contact_info": {"email": "bob@example.com", "phone": "123-456-7890"}, "courses": [{"course_id": 101, "course_name": "Mathematics"}, {"course_id": 102, "course_name": "Physics"}"""
    assert parser.fit(case_8, 10)["data"]["completion"] == {
        "name": "Bob",
        "age": 25,
        "is_student": False,
        "scores": [85, 90, 78],
        "contact_info": {"email": "bob@example.com", "phone": "123-456-7890"},
        "courses": [
            {"course_id": 101, "course_name": "Mathematics"},
            {"course_id": 102, "course_name": "Physics"},
        ],
    }

    case_9 = """[1, 2, {"name": "Alice", "age": 30}, 4, [5, 6], {"a": 1, "b": [{"c": 2, "d": 3}], "e": 4}"""
    assert parser.fit(case_9, 10)["data"]["completion"] == [
        1,
        2,
        {"name": "Alice", "age": 30},
        4,
        [5, 6],
        {"a": 1, "b": [{"c": 2, "d": 3}], "e": 4},
    ]

    case_10 = """{"name": "Alice", "age": 30, "hobbies": ["reading", "running", {"favorite_movies": ["Inception", "The Matrix"]},"""
    assert parser.fit(case_10, 10)["data"]["completion"] == {
        "name": "Alice",
        "age": 30,
        "hobbies": [
            "reading",
            "running",
            {"favorite_movies": ["Inception", "The Matrix"]},
        ],
    }
