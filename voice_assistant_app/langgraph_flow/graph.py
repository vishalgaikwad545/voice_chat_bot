from typing import Dict, Any, List, Annotated, TypedDict
import logging
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from .nodes import (
    start_node,
    voice_input_node,
    intent_entity_extraction_node,
    input_validation_node,
    error_handling_node,
    field_mapping_node,
    form_completion_check_node,
    end_node
)
from .supervisor import supervisor_node

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GraphState(TypedDict):
    """Type definition for the graph state."""
    # Workflow tracking
    initialized: bool
    workflow_complete: bool
    
    # User input
    user_input: str
    transcribed_text: str
    transcription_success: bool
    transcription_error: str
    
    # Field tracking
    current_field: str
    completed_fields: List[str]
    field_values: Dict[str, Any]
    confirmation_state: bool
    extraction_attempts: int
    
    # Processing results
    extraction_result: Dict[str, Any]
    extraction_success: bool
    extraction_error: str
    current_extracted_value: Any
    
    # Validation
    validation_success: bool
    validation_error: str
    
    # Form state
    is_complete: bool
    final_output: Dict[str, Any]
    
    # Messages
    messages: List[Dict[str, str]]

def create_workflow() -> StateGraph:
    """
    Creates and returns the LangGraph workflow for the form assistant.
    
    Returns:
        StateGraph instance with the complete workflow
    """
    # Define the workflow graph
    workflow = StateGraph(GraphState)
    
    # Add all nodes
    workflow.add_node("START", start_node)
    workflow.add_node("voice_input", voice_input_node)
    workflow.add_node("intent_entity_extraction", intent_entity_extraction_node)
    workflow.add_node("input_validation", input_validation_node)
    workflow.add_node("error_handling", error_handling_node)
    workflow.add_node("field_mapping", field_mapping_node)
    workflow.add_node("form_completion_check", form_completion_check_node)
    workflow.add_node("END", end_node)
    
    # Add the supervisor node
    workflow.add_node("supervisor", supervisor_node)
    
    # Define the edges
    # From START to voice_input
    workflow.add_edge("START", "voice_input")
    
    # From voice_input to supervisor
    workflow.add_edge("voice_input", "supervisor")
    
    # From intent_entity_extraction to supervisor
    workflow.add_edge("intent_entity_extraction", "supervisor")
    
    # From input_validation to supervisor
    workflow.add_edge("input_validation", "supervisor")
    
    # From error_handling to voice_input (retry)
    workflow.add_edge("error_handling", "voice_input")
    
    # From field_mapping to supervisor
    workflow.add_edge("field_mapping", "supervisor")
    
    # From form_completion_check to END or back to supervisor
    workflow.add_conditional_edges(
        "form_completion_check",
        lambda state: "END" if state.get("is_complete", False) else "voice_input",
        {
            "END": "Form is complete, ending workflow",
            "voice_input": "Form is not complete, continuing"
        }
    )
    
    # Define supervisor conditional edges
    workflow.add_conditional_edges(
        "supervisor",
        lambda state: state["next_node"],
        {
            "voice_input": "Proceeding to voice input",
            "intent_entity_extraction": "Proceeding to intent/entity extraction",
            "input_validation": "Proceeding to input validation",
            "error_handling": "Proceeding to error handling",
            "field_mapping": "Proceeding to field mapping",
            "form_completion_check": "Proceeding to form completion check",
            "END": "Ending workflow"
        }
    )
    
    # Set the entry point
    workflow.set_entry_point("START")
    
    return workflow

# Function to process a single user input through the workflow
def process_user_input(workflow_state: Dict[str, Any], user_input: str) -> Dict[str, Any]:
    """
    Processes a single user input through the workflow.
    
    Args:
        workflow_state: Current state of the workflow
        user_input: User input text
        
    Returns:
        Updated workflow state
    """
    # Create an instance of the workflow graph
    workflow = create_workflow()
    
    # Add user input to the state
    state = {**workflow_state, "user_input": user_input}
    
    # Set the next node to be 'voice_input' in the first call
    if not state.get('initialized', False):
        state = start_node(state)
    
    # Process through the graph nodes manually since we're not using stream
    # First process voice input
    state = voice_input_node(state)
    
    # Then extract intent and entities if transcription was successful
    if state.get('transcription_success', False):
        state = intent_entity_extraction_node(state)
        
        # If extraction successful, validate
        if state.get('extraction_success', True):
            state = input_validation_node(state)
            
            # Handle validation results
            if state.get('validation_success', False):
                state = field_mapping_node(state)
            else:
                state = error_handling_node(state)
        
        # Check form completion
        if state.get('current_field', None) and state.get('field_values', {}).get(state.get('current_field')):
            state = form_completion_check_node(state)
            
            # If complete, end the workflow
            if state.get('is_complete', False):
                state = end_node(state)
    
    # Return the final state
    return state

# Function to get initial greeting
def get_initial_greeting() -> Dict[str, Any]:
    """
    Returns the initial state with greeting message.
    
    Returns:
        Initial workflow state with greeting
    """
    # Initialize the state
    state = {}
    
    # Just run the start node function directly
    output = start_node(state)
    
    return output