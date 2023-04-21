import json

def read_json(json_file):
    """
    Reads JSON data from a file and returns a Python object.
    
    Args:
        json_file (str): The path to the JSON file to read.
        
    Returns:
        A Python object representing the JSON data.
    """
    with open(json_file) as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Error decoding JSON data from file {json_file}: {str(e)}")
    return data
