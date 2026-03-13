"""Safe JSON parser — no eval(), uses json.loads + ast.literal_eval fallback."""

from __future__ import annotations

import ast
import itertools
import json
import re
from operator import itemgetter
from typing import Any, Dict, List, Optional, Type, Union

from pydantic import BaseModel

from promptify.core.exceptions import ParserError


class Parser:
    """Parse and complete potentially incomplete JSON from LLM output.

    Unlike v2, this parser NEVER uses eval(). It uses json.loads() with
    ast.literal_eval() as a safe fallback.
    """

    def _safe_parse(self, text: str) -> Any:
        """Parse a string as JSON/Python literal safely — no eval()."""
        try:
            return json.loads(text)
        except (json.JSONDecodeError, ValueError):
            pass
        try:
            return ast.literal_eval(text)
        except (ValueError, SyntaxError):
            pass
        raise ValueError(f"Cannot parse: {text[:100]}")

    def is_valid_json(self, input_str: str) -> bool:
        """Check if input is valid JSON/Python literal (dict or list)."""
        try:
            output = self._safe_parse(input_str)
            return isinstance(output, (dict, list))
        except (ValueError, SyntaxError):
            return False

    def escaped_(self, data: str) -> str:
        """Handle apostrophe/quote escaping."""
        if "'" in data:
            return re.sub(r"(?<=\w)(')(?=\w)", r"\"", data)
        return re.sub(r'(?<=\w)(")(?=\w)', r"'", data)

    def get_combinations(
        self,
        candidate_marks: List[str],
        n: int,
        should_end_mark: Optional[str] = None,
    ) -> List[str]:
        """Return all combinations of closing brackets up to length n."""
        combinations = []
        for i in range(1, n):
            for comb in itertools.product(candidate_marks, repeat=i):
                if should_end_mark is not None and comb[-1] != should_end_mark:
                    continue
                combinations.append("".join(comb))
        return combinations

    def complete_json_object(self, json_str: str, completion_str: str) -> Any:
        """Complete a JSON string by appending closing brackets, trimming from end until valid."""
        text = json_str
        while text:
            try:
                return self._safe_parse(text + completion_str)
            except (ValueError, SyntaxError):
                text = text[:-1]
        raise ValueError("Couldn't fix JSON")

    def get_possible_completions(
        self, json_str: str, json_depth_limit: int = 5
    ) -> Dict[str, Any]:
        """Generate possible completions for an incomplete JSON string."""
        candidate_marks = ["}", "]"]
        if "[" not in json_str:
            candidate_marks.remove("]")
        if "{" not in json_str:
            candidate_marks.remove("}")

        stripped = json_str.strip()
        should_end_mark = "]" if stripped and stripped[0] == "[" else "}"

        completions: List[Any] = []
        for completion_str in self.get_combinations(
            candidate_marks, json_depth_limit, should_end_mark=should_end_mark
        ):
            try:
                completed_obj = self.complete_json_object(json_str, completion_str)
                completions.append(completed_obj)
            except (ValueError, SyntaxError):
                pass

        if not completions:
            raise ValueError("No valid completions found")
        return self.find_max_length(completions)

    def fit(self, json_str: str, json_depth_limit: int = 5) -> Dict[str, Any]:
        """Parse JSON string, completing it if incomplete.

        Returns dict with 'status', 'object_type', and 'data' keys.
        """
        try:
            output = self._safe_parse(json_str)
            return {
                "status": "completed",
                "object_type": type(output),
                "data": {"completion": output, "suggestions": []},
            }
        except (ValueError, SyntaxError):
            json_str = re.sub(r"[\[\]\{\}\s]+$", "", json_str)
            try:
                output = self.get_possible_completions(
                    json_str, json_depth_limit=json_depth_limit
                )
                return {
                    "status": "completed",
                    "object_type": type(output["completion"]),
                    "data": output,
                }
            except Exception as e:
                return {
                    "status": "failed",
                    "object_type": None,
                    "data": {"error_message": str(e)},
                }

    def find_max_length(self, data_list: List[Any]) -> Dict[str, Any]:
        """Return element with max length and all sorted suggestions."""
        length_dict = {i: len(str(element)) for i, element in enumerate(data_list)}
        sorted_indices = sorted(length_dict.items(), key=itemgetter(1), reverse=True)
        return {
            "completion": data_list[sorted_indices[0][0]],
            "suggestions": [data_list[i] for i, _ in sorted_indices],
        }

    def extract_complete_objects(self, string: str) -> List[Any]:
        """Extract all complete JSON objects/arrays from a string."""
        object_regex = r"(?<!\\)(\[[^][]*?(?<!\\)\]|\{[^{}]*\})"
        object_strings: List[str] = []
        opening: Dict[str, int] = {"{": 0, "[": 0}
        closing = {"}": "{", "]": "["}
        stack: List[str] = []
        start = 0

        for match in re.finditer(object_regex, string):
            if len(stack) == 0:
                start = match.start()
            stack.append(match.group(1))
            if match.group(1)[-1] in closing:
                opening_bracket = closing[match.group(1)[-1]]
                opening[opening_bracket] += 1
                nonzero = [v for v in opening.values() if v != 0]
                if opening[opening_bracket] == len(nonzero):
                    object_strings.append(string[start : match.end()])
                    stack = []
                    opening = {"{": 0, "[": 0}

        objects: List[Any] = []
        for obj_str in object_strings:
            try:
                objects.append(ast.literal_eval(obj_str))
            except (ValueError, SyntaxError):
                try:
                    objects.append(json.loads(obj_str))
                except (json.JSONDecodeError, ValueError):
                    pass
        return objects

    def parse(
        self,
        text: str,
        output_schema: Optional[Type[BaseModel]] = None,
    ) -> Union[BaseModel, Dict[str, Any], List[Any]]:
        """Parse LLM text output into structured data.

        If output_schema is provided, validates against the Pydantic model.
        """
        text = text.strip()

        # Try direct JSON parse
        try:
            data = json.loads(text)
            if output_schema:
                return output_schema.model_validate(data)
            return data
        except (json.JSONDecodeError, ValueError):
            pass

        # Try ast.literal_eval
        try:
            data = ast.literal_eval(text)
            if output_schema:
                return output_schema.model_validate(data)
            return data
        except (ValueError, SyntaxError):
            pass

        # Try JSON completion
        result = self.fit(text)
        if result["status"] == "completed" and "completion" in result.get("data", {}):
            data = result["data"]["completion"]
            if output_schema:
                return output_schema.model_validate(data)
            return data

        raise ParserError(f"Failed to parse LLM output: {text[:200]}")
