import json
import datetime
from typing import Dict, Any

def get_conversation_schema(conversation_id: str, llm_name: str, **llm_metadata: Any) -> Dict[str, Any]:
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
    llm_metadata.pop('api_key', None)

    # Construct the conversation schema dictionary
    conversation_schema = {
        "conversation_id": conversation_id,
        "start_time": str(datetime.datetime.now().strftime("%Y_%m_%d:%H:%M:%S")),
        "llm": {
            "name": llm_name,
            "meta_data": llm_metadata
        },
        "participants": [
            {
                "name": "User",
                "is_bot": False
            },
            {
                "name": "Assistant",
                "is_bot": True
            }
        ],
        "messages": []
    }
    
    return conversation_schema
