import json
from typing import Dict, Any

def get_conversation_schema(conversation_id: str, start_time: str, llm_name: str, **llm_metadata: Any) -> Dict[str, Any]:
    """
    Constructs a conversation schema with the specified parameters.
    
    Args:
    - conversation_id: A string representing the unique identifier of the conversation.
    - start_time: A string representing the start time of the conversation.
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
        "start_time": start_time,
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
