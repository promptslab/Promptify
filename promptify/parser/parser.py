import itertools
from operator import itemgetter
from typing import Any, Dict, List, Tuple, Union
import re
import ast
import json


class Parser:
    """A class to parse incomplete JSON objects and provide possible completions."""

    def __init__(self):
        pass

    def is_valid_json(self, input_str: str) -> bool:
        """
        Checks if a string is valid JSON.
        :param input_str: The string to check
        :return: True if the input is valid JSON, False otherwise
        """
        try:
            output = eval(input_str)
            if isinstance(output, (dict, list)):
                return True
            else:
                return False
        except:
            return False

    def get_combinations(self, n: int) -> List[str]:
        """
        Returns all possible combinations of } and ] characters up to length n.
        :param n: The maximum length of the combinations
        :return: A list of all possible combinations of } and ] characters up to length n
        """
        combinations = []
        for i in range(1, n):
            for comb in itertools.product("}]", repeat=i):
                combinations.append("".join(comb))
        return combinations

    def complete_json_object(self, json_str: str, completion_str: str) -> Any:
        """
        Completes a JSON object string by appending a completion string.
        :param json_str: The original JSON object string
        :param completion_str: The completion string to append
        :return: The completed JSON object as a Python object
        """
        while True:
            if not json_str:
                raise ValueError("Couldn't fix JSON")
            try:
                complete_json_str = json_str + completion_str
                python_obj = eval(complete_json_str)
            except Exception as e:
                json_str = json_str[:-1]
                continue
            return python_obj

    def get_possible_completions(
        self, json_str: str, max_completion_length: int = 5
    ) -> Union[Dict[str, Any], List[Any]]:
        """
        Returns a list of possible completions for a JSON object string.
        :param json_str: The JSON object string
        :param max_completion_length: The maximum length of the completion strings to try
        :return: A dictionary with 'completion' and 'suggestions' keys
        """
        completions = []
        for completion_str in self.get_combinations(max_completion_length):
            try:
                completed_obj = self.complete_json_object(json_str, completion_str)
                completions.append(completed_obj)
            except Exception as e:
                pass
        return self.find_max_length(completions)

    def fit(self, json_str: str, max_completion_length: int = 5) -> Dict[str, Any]:
        """
        Tries to parse the input JSON string and complete it if it is incomplete.
        :param json_str: The input JSON string
        :param max_completion_length: The maximum length of the completion strings to try
        :return: A dictionary with 'status' and 'data' keys. If the status is 'completed', the 'data'
                  key will contain the completed object and an empty list of suggestions. If the
                  status is 'failed', the 'data' key will contain an error message string. If the
                  status is 'incomplete', the 'data' key will contain a list of possible completions.
        """
        try:
            output = eval(json_str)
            return {
                "status": "completed", "object_type": type(output),
                "data": {"completion": output, "suggestions": []},
            }
        except:
            try:
                output = self.get_possible_completions(
                    json_str, max_completion_length=max_completion_length
                )
                return {"status": "completed", "object_type": type(output["completion"]), "data": output}
            except Exception as e:
                return {"status": "failed", "object_type": None, "data": {"error_message": str(e)}}

    def find_max_length(self, data_list: List[Any]) -> Dict[str, List[Any]]:
        """
        Returns a dictionary containing the element with the maximum length in the input list,
        as well as a list of all elements sorted by length in descending order.
        :param data_list: a list of elements to be compared by length
        :return: a dictionary with 'completion' and 'suggestions' keys
        """
        # Create a dictionary where the keys are the indices of the elements in the input list
        # and the values are the lengths of the corresponding elements.
        length_dict = {i: len(str(element)) for i, element in enumerate(data_list)}

        # Sort the dictionary by value (length) in descending order.
        sorted_indices = sorted(length_dict.items(), key=itemgetter(1), reverse=True)

        # Create a new dictionary with the element with the maximum length as the 'completion' value
        # and a list of all elements sorted by length as the 'suggestions' value.
        output_dict = {
            "completion": data_list[sorted_indices[0][0]],
            "suggestions": [data_list[i] for i, _ in sorted_indices],
        }
        return output_dict

    def extract_complete_objects(self, string: str) -> List[Any]:
        """
        Extracts all complete Python objects from a string.
        :param string: The string to extract objects from.
        :return: A list of all complete Python objects found in the string.
        """

        object_regex = r"(?<!\\)(\[[^][]*?(?<!\\)\]|\{[^{}]*\})"

        # The regular expression pattern matches any string starting with an opening brace or bracket,
        # followed by any number of non-brace and non-bracket characters, and ending with a closing brace
        # or bracket that is not preceded by an odd number of backslash escape characters.

        object_strings = []
        opening = {"{": 0, "[": 0}
        closing = {"}": "{", "]": "["}
        stack = []
        start = 0
        for match in re.finditer(object_regex, string):
            if len(stack) == 0:
                start = match.start()
            stack.append(match.group(1))
            if match.group(1)[-1] in closing:
                opening_bracket = closing[match.group(1)[-1]]
                opening[opening_bracket] += 1
                if opening[opening_bracket] == len(
                    [bracket for bracket in opening.values() if bracket != 0]
                ):
                    object_strings.append(string[start : match.end()])
                    stack = []
                    opening = {"{": 0, "[": 0}
                    closing = {"}": "{", "]": "["}
        if len(stack) > 0:
            print(f"Error: Incomplete object at end of string: {stack[-1]}")
        objects = []
        for object_string in object_strings:
            try:
                obj = ast.literal_eval(object_string)
                # Use ast.literal_eval() to safely evaluate the string as a Python object.
                objects.append(obj)
            except (ValueError, SyntaxError) as e:
                # If the string cannot be safely evaluated as a Python object, log an error and move on to the next object.
                print(f"Error evaluating object string '{object_string}': {str(e)}")
                pass

        return objects
