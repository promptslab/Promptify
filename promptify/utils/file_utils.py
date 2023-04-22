import json
import uuid
import datetime
from pathlib import Path
import hashlib
import os
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
            raise ValueError(
                f"Error decoding JSON data from file {json_file}: {str(e)}"
            )
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


def calculate_hash(text: str, encoding: str = "utf-8") -> str:
    """
    Calculate the hash of a text using the specified encoding.

    Args:
        text: The text to calculate the hash for.
        encoding: The encoding to use for the text. Defaults to "utf-8".

    Returns:
        The hash of the text.
    """
    if not isinstance(text, str):
        raise TypeError("Expected a string for 'text' parameter.")

    hash_obj = hashlib.md5()
    hash_obj.update(text.encode(encoding))
    return hash_obj.hexdigest()


def setup_folder(folder_path: str, folder_name: str = None) -> str:
    """
    Creates a folder in the specified folder_path.

    Parameters
    ----------
    folder_path : str
        The path to the directory where the folder will be created.
    folder_name : str, optional
        The name of the folder. If None, a name will be generated using the
        current date and time and a UUID.

    Returns
    -------
    The path to the created folder.
    """

    if folder_name is None:
        current_date = datetime.datetime.now().strftime("%Y_%m_%d:%H:%M:%S")
        conversation_id = uuid.uuid4()
        folder_name = f"{current_date}_{conversation_id}"
        folder_name = calculate_hash(folder_name)

    folder_path = Path(folder_path)
    folder_path.mkdir(parents=True, exist_ok=True)

    folder = folder_path / folder_name
    folder.mkdir(parents=True, exist_ok=True)

    return str(folder), folder_name
