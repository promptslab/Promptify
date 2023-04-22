import json

def is_string_or_digit(obj):
    """
    Check if an object is a string or a digit (integer or float).

    Args:
        obj (any): The object to be checked.

    Returns:
        bool: True if the object is a string or a digit, False otherwise.

    Examples:
        >>> is_string_or_digit("hello")
        True
        >>> is_string_or_digit(123)
        True
        >>> is_string_or_digit(3.14)
        True
        >>> is_string_or_digit(True)
        False
    """
    return isinstance(obj, (str, int, float))
