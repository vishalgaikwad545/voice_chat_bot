from typing import Dict, Any, List, Literal, TypedDict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SupervisorOutput(TypedDict):
    """Output model for the supervisor node."""
    next_node: str

def supervisor_node(state: Dict[str, Any]) -> SupervisorOutput:
    """
    Supervisor node that determines the next step in the workflow based on the current state.
    
    Args:
        state: Current state of the workflow
        
    Returns:
        SupervisorOutput with the next node to execute
    """
    logger.info("Supervisor making decision")
    
    # Check if workflow is complete
    if state.get('is_complete', False):
        return {"next_node": "form_completion_check"}
    
    # Check if we need to handle a validation error
    if state.get('validation_success', True) is False and state.get('extraction_result', {}).get('intent') == 'provide_value':
        return {"next_node": "error_handling"}
    
    # Handle different intents
    intent = state.get('extraction_result', {}).get('intent', 'other')
    
    if intent == 'provide_value' and state.get('validation_success', False):
        # If we've validated a provided value, move to field mapping
        return {"next_node": "field_mapping"}
    elif intent in ['confirm', 'deny'] and state.get('confirmation_state', False):
        # If we're in confirmation state and got a confirm/deny response
        return {"next_node": "field_mapping"}
    elif intent in ['request_help', 'request_skip']:
        # Handle help or skip requests
        return {"next_node": "field_mapping"}
    elif state.get('transcription_success', False) is False:
        # If transcription failed
        return {"next_node": "voice_input"}
    elif state.get('extraction_success', False) is False:
        # If extraction failed
        return {"next_node": "intent_entity_extraction"}
    else:
        # Default path: process input normally
        if 'transcribed_text' in state and state.get('transcription_success', False):
            return {"next_node": "intent_entity_extraction"}
        else:
            return {"next_node": "voice_input"}