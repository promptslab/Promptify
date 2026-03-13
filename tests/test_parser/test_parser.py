"""Tests for safe parser — ported from v2 + new safety tests."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from promptify.core.exceptions import ParserError
from promptify.parser.parser import Parser


class TestParser:
    def setup_method(self):
        self.parser = Parser()

    # --- Safety: no eval() ---

    def test_no_eval_in_source(self):
        """Verify eval() is not used anywhere in the parser module."""
        import ast as _ast
        import pathlib
        source_path = pathlib.Path(__file__).parent.parent.parent / "promptify" / "parser" / "parser.py"
        tree = _ast.parse(source_path.read_text())
        for node in _ast.walk(tree):
            if isinstance(node, _ast.Call):
                func = node.func
                # Check for bare eval() calls
                if isinstance(func, _ast.Name) and func.id == "eval":
                    pytest.fail(f"Found bare eval() call at line {node.lineno}")

    # --- is_valid_json ---

    def test_is_valid_json_dict(self):
        assert self.parser.is_valid_json('{"name": "Alice", "age": 30}') is True

    def test_is_valid_json_list(self):
        assert self.parser.is_valid_json("[1, 2, 3, 4]") is True

    def test_is_valid_json_invalid(self):
        assert self.parser.is_valid_json('{"name": "Bob", "age": }') is False

    def test_is_valid_json_not_json(self):
        assert self.parser.is_valid_json("not a JSON string") is False

    # --- get_combinations ---

    def test_get_combinations(self):
        result = self.parser.get_combinations(["}", "]"], 3)
        assert "}" in result
        assert "]" in result
        assert "}}" in result
        assert "}]" in result

    def test_get_combinations_with_end_mark(self):
        result = self.parser.get_combinations(["}", "]"], 3, should_end_mark="}")
        assert all(c.endswith("}") for c in result)

    # --- escaped_ ---

    def test_escaped_apostrophe(self):
        result = self.parser.escaped_("it's a test")
        assert "'" not in result or '"' in result

    # --- complete_json_object ---

    def test_complete_json_object_dict(self):
        result = self.parser.complete_json_object('{"a": 1, "b": 2', "}")
        assert result == {"a": 1, "b": 2}

    def test_complete_json_object_list(self):
        result = self.parser.complete_json_object("[1, 2, 3", "]")
        assert result == [1, 2, 3]

    def test_complete_json_object_failure(self):
        with pytest.raises(ValueError, match="Couldn't fix JSON"):
            self.parser.complete_json_object("", "}")

    # --- fit (ported from v2 test_parser.py) ---

    def test_fit_complete_json(self):
        result = self.parser.fit('[{"a": 1}]')
        assert result["status"] == "completed"
        assert result["data"]["completion"] == [{"a": 1}]

    def test_fit_incomplete_list(self):
        result = self.parser.fit("[1, 2, 3")
        assert result["status"] == "completed"
        assert 1 in result["data"]["completion"]

    def test_fit_incomplete_dict(self):
        result = self.parser.fit('{"a": 1, "b": 2')
        assert result["status"] == "completed"

    def test_fit_nested(self):
        result = self.parser.fit('[{"T": "PERSON", "E": "John"}, {"T": "LOC", "E": "NYC"}]')
        assert result["status"] == "completed"
        assert len(result["data"]["completion"]) == 2

    def test_fit_deeply_nested(self):
        result = self.parser.fit('[{"outer": {"inner": [1, 2, 3]}}]')
        assert result["status"] == "completed"

    def test_fit_incomplete_nested(self):
        result = self.parser.fit('[{"T": "PERSON", "E": "John"}, {"T": "LOC", "E": "NYC"')
        assert result["status"] == "completed"

    def test_fit_garbage(self):
        result = self.parser.fit("not json at all xyz")
        assert result["status"] == "failed"

    def test_fit_empty_list(self):
        result = self.parser.fit("[]")
        assert result["status"] == "completed"
        assert result["data"]["completion"] == []

    def test_fit_empty_dict(self):
        result = self.parser.fit("{}")
        assert result["status"] == "completed"
        assert result["data"]["completion"] == {}

    def test_fit_extra_brackets(self):
        result = self.parser.fit('[{"a": 1}]}}}')
        assert result["status"] == "completed"

    # --- parse (new v3 method) ---

    def test_parse_valid_json(self):
        result = self.parser.parse('{"answer": "yes"}')
        assert result == {"answer": "yes"}

    def test_parse_with_schema(self):
        class MyModel(BaseModel):
            answer: str

        result = self.parser.parse('{"answer": "yes"}', MyModel)
        assert isinstance(result, MyModel)
        assert result.answer == "yes"

    def test_parse_incomplete_json_with_schema(self):
        class SimpleModel(BaseModel):
            a: int
            b: int

        result = self.parser.parse('{"a": 1, "b": 2', SimpleModel)
        assert isinstance(result, SimpleModel)
        assert result.a == 1

    def test_parse_failure(self):
        with pytest.raises(ParserError):
            self.parser.parse("complete garbage xyz")

    # --- extract_complete_objects ---

    def test_extract_complete_objects(self):
        text = 'Some text {"a": 1} more text [2, 3]'
        objects = self.parser.extract_complete_objects(text)
        assert len(objects) >= 1
