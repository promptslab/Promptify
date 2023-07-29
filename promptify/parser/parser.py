import itertools
from operator import itemgetter
from typing import Any, Dict, List, Union, Optional
import re
import ast


class Parser:
    """
    A class to parse incomplete JSON objects and provide possible completions.

    Methods
    -------
    is_valid_json() -> bool:
        Checks if a string is valid JSON.

    get_combinations() -> List[str]:
        Returns all possible combinations of } and ] characters up to length n.

    complete_json_object() -> Any:
        Completes a JSON object string by appending a completion string.

    get_possible_completions() -> Union[Dict[str, Any], List[Any]]:
        Returns a list of possible completions for a JSON object string.

    fit() -> Dict[str, Any]:
        Tries to parse the input JSON string and complete it if it is incomplete.

    find_max_length() -> Dict[str, List[Any]]:
        Returns a dictionary containing the element with the maximum length in the input list.
    """

    def __init__(self):
        pass

    def is_valid_json(self, input_str: str) -> bool:
        """
        Check if the input string is valid JSON.

        Parameters
        ----------
        input_str : str
            The string to check for validity.

        Returns
        -------
        bool
            Returns True if the input string is valid JSON, otherwise False.

        Notes
        -----
        This function uses the `json` module to check if the input string is valid JSON.
        It evaluates the input string using `eval()`, and if it successfully loads
        a JSON object (either a dictionary or a list), it returns True. Otherwise, it
        returns False.

        Examples
        --------
        >>> validator = Parser()
        >>> validator.is_valid_json('{"name": "Alice", "age": 30}')
        True
        >>> validator.is_valid_json('[1, 2, 3, 4]')
        True
        >>> validator.is_valid_json('{"name": "Bob", "age": }')
        False
        >>> validator.is_valid_json('not a JSON string')
        False
        """
        try:
            output = eval(input_str)
            if isinstance(output, (dict, list)):
                return True
            else:
                return False
        except Exception:
            return False

    def escaped_(self, data: str) -> str:
        if "'" in data:
            escaped_str = re.sub(r"(?<=\w)(')(?=\w)", r"\"", data)
        else:
            escaped_str = re.sub(r'(?<=\w)(")(?=\w)', r"\'", data)

        return escaped_str

    def get_combinations(
        self, candidate_marks: List[str], n: int, should_end_mark: Optional[str] = None
    ) -> List[str]:
        """
        Return all possible combinations of candidate marks up to length n.

        Parameters
        ----------
        candidate_marks : list of str
            Candidate marks to combine.
        n : int
            The maximum length of the combinations.
        should_end_mark : str or None, optional
            If provided, only combinations that end with this mark will be returned, by default None.

        Returns
        -------
        list of str
            A list of all possible combinations of candidate marks up to length n.
        """
        combinations = []
        for i in range(1, n):
            for comb in itertools.product(candidate_marks, repeat=i):
                if should_end_mark is not None and comb[-1] != should_end_mark:
                    # cut down search space
                    continue
                combinations.append("".join(comb))
        return combinations

    def complete_json_object(self, json_str: str, completion_str: str) -> Any:
        """
        Complete a JSON object string by appending a completion string.

        Parameters
        ----------
        json_str : str
            The original JSON object string.
        completion_str : str
            The completion string to append.

        Returns
        -------
        Any
            The completed JSON object as a Python object.

        Raises
        ------
        ValueError
            If the JSON object string cannot be fixed.

        Notes:
        ------
        - This function appends the `completion_str` to the end of `json_str` until a valid JSON object can be obtained. If `json_str` is an invalid JSON object string, the function will remove one character from the end of `json_str` and try again until it either finds a valid JSON object string or until there are no more characters left to remove.

        Examples
        --------
        >>> complete_json_object('{"a": 1, "b": 2', '}')
        {'a': 1, 'b': 2}

        >>> complete_json_object('{"a": 1, "b": 2}}}}}}}}}}}}}}}}}}}', '')
        {'a': 1, 'b': 2}

        >>> complete_json_object('{"a": 1, "b": 2', '')
        Traceback (most recent call last):
            ...
        ValueError: Couldn't fix JSON
        """
        while True:
            if not json_str:
                raise ValueError("Couldn't fix JSON")
            try:
                complete_json_str = json_str + completion_str
                python_obj = eval(complete_json_str)
            except Exception:
                json_str = json_str[:-1]
                continue
            return python_obj

    def get_possible_completions(
        self, json_str: str, json_depth_limit: int = 5
    ) -> Union[Dict[str, Any], List[Any]]:
        """
        Returns a list of possible completions for a JSON object string.

        Parameters
        ----------
        json_str : str
            The JSON object string
        json_depth_limit : int, optional
            The maximum length of the completion strings to try (default is 5)

        Returns
        -------
        Union[Dict[str, Any], List[Any]]
            If the completion strings are objects, returns a dictionary with 'completion' and 'suggestions' keys.
            If the completion strings are arrays, returns a list of suggested completions.
        """
        candidate_marks = ["}", "]"]
        if "[" not in json_str:
            candidate_marks.remove("]")
        if "{" not in json_str:
            candidate_marks.remove("}")

        # specify the mark should end with
        should_end_mark = "]" if json_str.strip()[0] == "[" else "}"
        completions = []
        for completion_str in self.get_combinations(
            candidate_marks, json_depth_limit, should_end_mark=should_end_mark
        ):
            try:
                completed_obj = self.complete_json_object(json_str, completion_str)
                completions.append(completed_obj)
            except Exception:
                pass
        return self.find_max_length(completions)

    def fit(self, json_str: str, json_depth_limit: int = 5) -> Dict[str, Any]:
        """
        Tries to parse the input JSON string and complete it if it is incomplete.

        Parameters
        ----------
        json_str : str
            The input JSON string
        json_depth_limit : int, optional
            The maximum length of the completion strings to try (default is 5)

        Returns
        -------
        Dict[str, Any]
            A dictionary with 'status' and 'data' keys. If the status is 'completed', the 'data'
            key will contain the completed object and an empty list of suggestions. If the status
            is 'failed', the 'data' key will contain an error message string. If the status is
            'incomplete', the 'data' key will contain a list of possible completions.
        """
        try:
            output = eval(json_str)
            return {
                "status": "completed",
                "object_type": type(output),
                "data": {"completion": output, "suggestions": []},
            }
        except Exception:
            # remove tail braces or brackets to speed up searching.
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

    def find_max_length(self, data_list: List[Any]) -> Dict[str, List[Any]]:
        """
        Returns a dictionary containing the element with the maximum length in the input list,
        as well as a list of all elements sorted by length in descending order.

        Parameters
        ----------
        data_list : list of any type
            A list of elements to be compared by length

        Returns
        -------
        dict
            A dictionary with keys 'completion' and 'suggestions'.

            The value of 'completion' key is the element with the maximum length in the input list.

            The value of 'suggestions' key is a list of all elements sorted by length in descending order.
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

        Parameters
        ----------
        string : str
            The string to extract objects from.

        Returns
        -------
        List[Any]
            A list of all complete Python objects found in the string.
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
