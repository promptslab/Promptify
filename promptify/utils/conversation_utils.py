import json
import uuid
import datetime
from typing import Dict, Any


def get_conversation_schema(
    conversation_id: str, llm_name: str, **llm_metadata: Any
) -> Dict[str, Any]:
    """
    Constructs a conversation schema with the specified parameters.

    Args:
    - conversation_id: A string representing the unique identifier of the conversation.
    - llm_name: A string representing the name of the language model.
    - **llm_metadata: Optional additional metadata to associate with the language model.

    Returns:
    A dictionary representing the conversation schema.
    """
    # Remove any api_key from the kwargs to avoid potential security issues
    llm_metadata.pop("api_key", None)

    # Construct the conversation schema dictionary
    conversation_schema = {
        "conversation_id": conversation_id,
        "start_time": str(datetime.datetime.now().strftime("%Y_%m_%d:%H:%M:%S")),
        "llm": {"name": llm_name, "meta_data": llm_metadata},
        "participants": [
            {"name": "User", "is_bot": False},
            {"name": "Assistant", "is_bot": True},
        ],
        "messages": [],
    }

    return conversation_schema


def create_message(
    task: str,
    prompt: str,
    response: str,
    structured_response: Any,
    prompt_file: str,
    **template_metadata: Any
) -> Dict[str, Any]:
    """
    Creates a message dictionary with the specified parameters.

    Args:
    - task: A string representing the task the message is associated with.
    - prompt: A string representing the prompt that initiated the message.
    - response: A string representing the message response.
    - structured_response: A structured representation of the message response.
    - **template_metadata: Optional metadata to associate with the message.

    Returns:
    A dictionary representing the message.
    """
    # Get the current timestamp as a formatted string
    timestamp = str(datetime.datetime.now().strftime("%Y_%m_%d:%H:%M:%S"))
    prompt_id = str(uuid.uuid4())

    # Construct the message dictionary
    
    message = {
        "request_timestamp": timestamp,
        "prompt_id": prompt_id,
        "prompt_filename": prompt_file, 
        "processing_task": task,
        "template_metadata": template_metadata,
        "input_prompt": prompt,
        "response_text": response,
        "parsed_output": structured_response,
        }

    return message


