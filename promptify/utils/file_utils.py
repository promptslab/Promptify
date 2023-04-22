import json
import io

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


def write_json(path, data, file_name):
    """
    Writes JSON data to a file.
    
    Args:
        path (str): The path to the directory where the file should be saved.
        data (Any): The data to write to the file. This can be any JSON-serializable object.
        file_name (str): The name of the file to write, without the '.json' extension.
        
    Raises:
        IOError: If there is a problem writing the file.
    """
    full_path = os.path.join(path, f"{file_name}.json")
    try:
        with open(full_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
    except IOError as e:
        raise IOError(f"Error writing JSON file '{full_path}': {e.strerror}")
