import json
from typing import Dict, Any, List, Tuple, Optional
import logging
from schemas import UserFormData
from utils.audio_processor import AudioProcessor
from langchain_components.llm_provider import LLMProvider
from pydantic import ValidationError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the audio processor and LLM provider
audio_processor = AudioProcessor()
llm_provider = LLMProvider()

def start_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Entry point for the LangGraph workflow.
    
    Args:
        state: Current state of the workflow
        
    Returns:
        Updated state with initialization
    """
    logger.info("Starting workflow")
    
    if 'initialized' not in state:
        state['initialized'] = True
        state['current_field'] = 'full_name'
        state['completed_fields'] = []
        state['field_values'] = {}
        state['confirmation_state'] = False
        state['extraction_attempts'] = 0
        state['is_complete'] = False
        state['messages'] = []
        
        # Add a greeting message
        greeting = "Hello! I'm your voice assistant, here to help you complete this form. Let's start with your full name. What is your full name?"
        state['messages'].append({"role": "assistant", "content": greeting})
    
    return state

def voice_input_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Captures and processes voice input from the user.
    
    Args:
        state: Current state of the workflow
        
    Returns:
        Updated state with transcribed text
    """
    logger.info("Processing voice input")
    
    # This is a placeholder for integrating with frontend
    # In the real implementation, this will be triggered by UI events
    
    # In our Streamlit app, we'll capture audio and store the transcription
    # For now, we'll simulate with the text input that was passed
    
    if 'user_input' in state and state['user_input']:
        # We already have text input provided from the frontend
        state['transcribed_text'] = state['user_input']
        state['transcription_success'] = True
        state['transcription_error'] = None
    else:
        # If no text input, mark as failure (this shouldn't happen in practice
        # since we'll always have either text or voice input from the UI)
        state['transcribed_text'] = None
        state['transcription_success'] = False
        state['transcription_error'] = "No input provided"
        
    # Add the user message to the messages list if successful
    if state['transcription_success']:
        state['messages'].append({"role": "user", "content": state['transcribed_text']})
    
    return state

def intent_entity_extraction_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extracts intent and entities from the transcribed text using Groq LLM.
    
    Args:
        state: Current state of the workflow
        
    Returns:
        Updated state with extracted intent and entities
    """
    logger.info("Extracting intent and entities")
    
    # Skip if transcription failed
    if not state.get('transcription_success', False):
        state['extraction_success'] = False
        state['extraction_error'] = "Transcription failed"
        return state
    
    try:
        # Get the current field details
        current_field = state['current_field']
        
        # Get field details from UserFormData model
        field_info = UserFormData.model_json_schema()['properties'].get(current_field, {})
        
        # Extract field description and validation rules
        field_description = field_info.get('description', f"The user's {current_field}")
        validation_rules = {
            k: v for k, v in field_info.items() 
            if k not in ['description', 'title', 'type']
        }
        
        # If we're in confirmation state, we need to check if the user confirmed or denied
        if state.get('confirmation_state', False):
            # Add a special system message for confirmation
            extraction_result = {
                "intent": "confirm" if any(word in state['transcribed_text'].lower() 
                                          for word in ["yes", "correct", "right", "sure", "yeah", "yep", "yup"]) 
                                     else "deny",
                "extracted_value": state.get('current_extracted_value'),
                "confidence": 1.0,
                "reasoning": "Direct confirmation/denial detection"
            }
        else:
            # Get chat history for context
            chat_history = state.get('messages', [])[-5:]  # Last 5 messages
            
            # Use LLM to extract intent and value
            extraction_result = llm_provider.extract_intent_and_value(
                user_input=state['transcribed_text'],
                field_name=current_field,
                field_description=field_description,
                validation_rules=validation_rules,
                chat_history=chat_history
            )
            
            # Parse the result if it's a string (depends on LLM provider implementation)
            if isinstance(extraction_result, str):
                try:
                    extraction_result = json.loads(extraction_result)
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse LLM result: {extraction_result}")
                    extraction_result = {
                        "intent": "other",
                        "extracted_value": None,
                        "confidence": 0.0,
                        "reasoning": "Failed to parse LLM result"
                    }
        
        # Store the extraction results in the state
        state['extraction_result'] = extraction_result
        state['current_extracted_value'] = extraction_result.get('extracted_value')
        state['extraction_success'] = True
        
        # Increment the extraction attempt counter
        state['extraction_attempts'] = state.get('extraction_attempts', 0) + 1
        
    except Exception as e:
        logger.error(f"Error in intent/entity extraction: {e}")
        state['extraction_success'] = False
        state['extraction_error'] = str(e)
    
    return state

def input_validation_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validates the extracted data against the Pydantic schema.
    
    Args:
        state: Current state of the workflow
        
    Returns:
        Updated state with validation results
    """
    logger.info("Validating input")
    
    # Skip if extraction failed
    if not state.get('extraction_success', False):
        state['validation_success'] = False
        state['validation_error'] = "Extraction failed"
        return state
    
    # Skip validation if the intent is not to provide a value
    if state.get('extraction_result', {}).get('intent') != 'provide_value' and not state.get('confirmation_state', False):
        state['validation_success'] = True
        state['validation_error'] = None
        return state
    
    try:
        current_field = state['current_field']
        extracted_value = state.get('current_extracted_value')
        
        # Skip validation if no value was extracted
        if extracted_value is None:
            state['validation_success'] = False
            state['validation_error'] = "No value was extracted"
            return state
        
        # Create a temporary dict with just the current field to validate
        temp_data = {current_field: extracted_value}
        
        # Use Pydantic to validate
        try:
            # Get existing values for context
            full_data = {**state.get('field_values', {}), **temp_data}
            
            # Validate just the single field
            field_schema = {
                'type': 'object',
                'properties': {
                    current_field: UserFormData.model_json_schema()['properties'][current_field]
                },
                'required': [current_field]
            }
            
            # We're using the model's validators for the specific field
            UserFormData(**full_data).model_dump()
            
            # If no exception was raised, validation passed
            state['validation_success'] = True
            state['validation_error'] = None
            
        except ValidationError as ve:
            # Extract the validation error details
            state['validation_success'] = False
            state['validation_error'] = str(ve)
            
            # Get field details from UserFormData model
            field_info = UserFormData.model_json_schema()['properties'].get(current_field, {})
            
            # Add helpful information for the error handling node
            if 'enum' in field_info:
                state['valid_options'] = field_info['enum']
            elif current_field == 'project_interests':
                state['valid_options'] = "A list of 1-5 project interests, each between 2-100 characters"
            
    except Exception as e:
        logger.error(f"Error in validation: {e}")
        state['validation_success'] = False
        state['validation_error'] = str(e)
    
    return state

def error_handling_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handles errors and provides feedback to the user.
    
    Args:
        state: Current state of the workflow
        
    Returns:
        Updated state with error handling results
    """
    logger.info("Handling errors")
    
    # Only process if validation failed
    if state.get('validation_success', True):
        return state
    
    try:
        current_field = state['current_field']
        field_info = UserFormData.model_json_schema()['properties'].get(current_field, {})
        field_description = field_info.get('description', f"The user's {current_field}")
        error_message = state.get('validation_error', "Invalid input")
        
        # Create error message based on field type and validation rules
        if current_field == 'full_name':
            error_explanation = "Your name should be between 2 and 100 characters."
            valid_examples = ["John Smith", "Maria Rodriguez", "Ahmed Khan"]
        elif current_field == 'email':
            error_explanation = "Please provide a valid email address."
            valid_examples = ["user@example.com", "name.surname@company.co.uk"]
        elif current_field == 'age':
            error_explanation = "Your age should be a number between 18 and 120."
            valid_examples = ["30", "45", "62"]
        elif current_field == 'experience_level':
            error_explanation = "Please select one of the valid experience levels."
            valid_examples = ["Beginner", "Intermediate", "Advanced", "Expert"]
        elif current_field == 'preferred_language':
            error_explanation = "Please select one of the valid programming languages."
            valid_examples = ["Python", "JavaScript", "Java", "C++", "Go", "Rust", "Other"]
        elif current_field == 'project_interests':
            error_explanation = "Please provide 1 to 5 project interests."
            valid_examples = ["Web Development", "Machine Learning, Data Analysis", "Game Development, Mobile Apps, Cloud Computing"]
        elif current_field == 'availability_per_week':
            error_explanation = "Please provide a number between 1 and 168 for weekly availability hours."
            valid_examples = ["10", "20", "40"]
        elif current_field == 'start_date':
            error_explanation = "Please provide a valid date in YYYY-MM-DD format."
            valid_examples = ["2025-06-01", "2025-07-15", "2025-08-30"]
        else:
            error_explanation = f"The provided value for {current_field} is invalid."
            valid_examples = ["Please check the requirements and try again."]
        
        # Customize error message based on field
        guidance_message = f"I'm having trouble understanding your {current_field}. {error_explanation} Could you please try again?"
        
        # Add the error handling message to the messages list
        state['messages'].append({"role": "assistant", "content": guidance_message})
        
        # Store error details in state
        state['error_details'] = {
            "error_explanation": error_explanation,
            "valid_examples": valid_examples,
            "guidance_message": guidance_message
        }
        
        # Reset confirmation state if in confirmation
        if state.get('confirmation_state', False):
            state['confirmation_state'] = False
        
    except Exception as e:
        logger.error(f"Error in error handling: {e}")
        # Fallback error message
        state['messages'].append({
            "role": "assistant", 
            "content": "I'm having trouble processing your input. Could you please try again?"
        })
    
    return state

def field_mapping_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Maps validated data to form fields and updates state.
    
    Args:
        state: Current state of the workflow
        
    Returns:
        Updated state with mapped field values
    """
    logger.info("Mapping fields")
    
    # Only process if validation succeeded or we're handling non-value intents
    if not state.get('validation_success', False) and state.get('extraction_result', {}).get('intent') == 'provide_value':
        return state
    
    try:
        current_field = state['current_field']
        intent = state.get('extraction_result', {}).get('intent')
        
        # If user confirmed a value
        if intent == 'confirm' and state.get('confirmation_state', False):
            # Store the confirmed value
            extracted_value = state.get('current_extracted_value')
            
            if extracted_value is not None:
                # Special handling for different field types
                if current_field == 'age' and not isinstance(extracted_value, int):
                    try:
                        extracted_value = int(extracted_value)
                    except (ValueError, TypeError):
                        # If conversion fails, keep as is
                        pass
                        
                elif current_field == 'project_interests' and isinstance(extracted_value, str):
                    # Convert comma-separated string to list
                    extracted_value = [item.strip() for item in extracted_value.split(',')]
                    
                elif current_field == 'availability_per_week' and not isinstance(extracted_value, int):
                    try:
                        extracted_value = int(extracted_value)
                    except (ValueError, TypeError):
                        # If conversion fails, keep as is
                        pass
                
                # Add to field values
                state['field_values'][current_field] = extracted_value
                
                # Log the mapping
                logger.info(f"Mapped field {current_field} to value: {extracted_value}")
                
                # Add to completed fields if not already there
                if current_field not in state.get('completed_fields', []):
                    state['completed_fields'] = state.get('completed_fields', []) + [current_field]
                
                # Add a confirmation message
                confirmation_message = f"Great! I've saved your {current_field}: {extracted_value}"
                state['messages'].append({"role": "assistant", "content": confirmation_message})
                
                # Reset confirmation state and move to the next field
                state['confirmation_state'] = False
                
                # Determine the next field
                field_order = list(UserFormData.model_json_schema()['properties'].keys())
                current_index = field_order.index(current_field)
                
                if current_index < len(field_order) - 1:
                    next_field = field_order[current_index + 1]
                    state['current_field'] = next_field
                    
                    # Add a prompt for the next field
                    field_info = UserFormData.model_json_schema()['properties'].get(next_field, {})
                    field_description = field_info.get('description', f"your {next_field}")
                    
                    # Customize prompt based on field type
                    if next_field == 'experience_level':
                        prompt = f"Now, please tell me your experience level. Choose from: Beginner, Intermediate, Advanced, or Expert."
                    elif next_field == 'preferred_language':
                        prompt = f"What's your preferred programming language? Options are: Python, JavaScript, Java, C++, Go, Rust, or Other."
                    elif next_field == 'project_interests':
                        prompt = f"What projects are you interested in? You can list between 1 and 5 interests."
                    elif next_field == 'start_date':
                        prompt = f"When would you like to start? Please provide a date in YYYY-MM-DD format."
                    else:
                        prompt = f"Now, please tell me {field_description}."
                    
                    state['messages'].append({"role": "assistant", "content": prompt})
                    state['extraction_attempts'] = 0
                else:
                    # All fields complete
                    state['is_complete'] = True
                    
                    # Create a summary of all collected information
                    summary = "\n".join([f"- {field}: {value}" for field, value in state['field_values'].items()])
                    
                    # Add completion message
                    completion_message = f"Excellent! We've completed all the required information. Here's a summary of what you've provided:\n\n{summary}\n\nThank you for providing all this information. The form has been submitted successfully."
                    state['messages'].append({"role": "assistant", "content": completion_message})
            
        # Rest of the function remains the same...
        # ...

        # If user denied a value
        elif intent == 'deny' and state.get('confirmation_state', False):
            # Reset confirmation state and ask again
            state['confirmation_state'] = False
            state['extraction_attempts'] = 0
            
            # Add a message asking for the correct value
            retry_message = f"I apologize for the misunderstanding. Let's try again. What is your {current_field}?"
            state['messages'].append({"role": "assistant", "content": retry_message})
        
        # If user provided a value and it passed validation
        elif intent == 'provide_value' and state.get('validation_success', False):
            # Move to confirmation state
            state['confirmation_state'] = True
            extracted_value = state.get('current_extracted_value')
            
            # Add a confirmation prompt
            confirmation_message = f"I've captured that your {current_field} is: {extracted_value}. Is that correct?"
            state['messages'].append({"role": "assistant", "content": confirmation_message})
        
        # If user requested help
        elif intent == 'request_help':
            field_info = UserFormData.model_json_schema()['properties'].get(current_field, {})
            field_description = field_info.get('description', f"your {current_field}")
            
            # Provide help based on the field
            if current_field == 'full_name':
                help_message = "I need your full name. For example, 'John Smith' or 'Maria Rodriguez'."
            elif current_field == 'email':
                help_message = "I need a valid email address where you can be contacted. For example, 'user@example.com'."
            elif current_field == 'age':
                help_message = "Please provide your age as a number between 18 and 120."
            elif current_field == 'experience_level':
                help_message = "Please select your experience level from: Beginner, Intermediate, Advanced, or Expert."
            elif current_field == 'preferred_language':
                help_message = "Please select your preferred programming language from: Python, JavaScript, Java, C++, Go, Rust, or Other."
            elif current_field == 'project_interests':
                help_message = "Please list between 1 and 5 project areas you're interested in. For example, 'Web Development, Machine Learning'."
            elif current_field == 'availability_per_week':
                help_message = "How many hours per week can you dedicate to the project? Please provide a number between 1 and 168."
            elif current_field == 'start_date':
                help_message = "When would you like to start? Please provide a date in YYYY-MM-DD format, for example, '2025-06-01'."
            else:
                help_message = f"I need information about {field_description}. Could you please provide that?"
            
            state['messages'].append({"role": "assistant", "content": help_message})
        
        # If user requested to skip
        elif intent == 'request_skip':
            # Check if the field is optional
            field_info = UserFormData.model_json_schema()
            required_fields = field_info.get('required', [])
            
            if current_field not in required_fields or current_field == 'additional_notes':
                # Field is optional, allow skipping
                state['field_values'][current_field] = None
                
                # Add to completed fields
                if current_field not in state.get('completed_fields', []):
                    state['completed_fields'] = state.get('completed_fields', []) + [current_field]
                
                # Add a skip confirmation message
                skip_message = f"No problem, we can skip the {current_field} field."
                state['messages'].append({"role": "assistant", "content": skip_message})
                
                # Move to the next field
                field_order = list(UserFormData.model_json_schema()['properties'].keys())
                current_index = field_order.index(current_field)
                
                if current_index < len(field_order) - 1:
                    next_field = field_order[current_index + 1]
                    state['current_field'] = next_field
                    
                    # Add a prompt for the next field
                    next_field_info = UserFormData.model_json_schema()['properties'].get(next_field, {})
                    next_field_description = next_field_info.get('description', f"your {next_field}")
                    prompt = f"Now, please tell me {next_field_description}."
                    
                    state['messages'].append({"role": "assistant", "content": prompt})
                    state['extraction_attempts'] = 0
                else:
                    # All fields complete
                    state['is_complete'] = True
                    
                    # Create a summary of all collected information
                    summary = "\n".join([f"- {field}: {value}" for field, value in state['field_values'].items() if value is not None])
                    
                    # Add completion message
                    completion_message = f"Excellent! We've completed all the required information. Here's a summary of what you've provided:\n\n{summary}\n\nThank you for providing all this information. The form has been submitted successfully."
                    state['messages'].append({"role": "assistant", "content": completion_message})
            else:
                # Field is required, cannot skip
                cannot_skip_message = f"I'm sorry, but {current_field} is a required field and cannot be skipped. Could you please provide this information?"
                state['messages'].append({"role": "assistant", "content": cannot_skip_message})
    
    except Exception as e:
        logger.error(f"Error in field mapping: {e}")
        # Fallback error message
        state['messages'].append({
            "role": "assistant", 
            "content": "I'm having trouble processing your information. Could you please try again?"
        })
    
    return state

def form_completion_check_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Checks if all form fields are complete and generates final output.
    
    Args:
        state: Current state of the workflow
        
    Returns:
        Updated state with form completion status
    """
    logger.info("Checking form completion")
    
    try:
        # Get all required fields from UserFormData
        schema = UserFormData.model_json_schema()
        required_fields = schema.get('required', [])
        
        # Check if all required fields have values
        completed_fields = state.get('completed_fields', [])
        is_complete = all(field in completed_fields for field in required_fields)
        
        state['is_complete'] = is_complete
        
        # If complete, prepare final JSON output
        if is_complete:
            # Get all field values
            field_values = state.get('field_values', {})
            
            # Create a clean output with only the fields that have values
            final_output = {
                field: value for field, value in field_values.items()
                if value is not None
            }
            
            state['final_output'] = final_output
            
            # If this is the first time completion is detected, add a completion message
            if not state.get('completion_message_added', False):
                # Generate a summary of all collected information
                summary_items = []
                for field, value in final_output.items():
                    if field == 'project_interests' and isinstance(value, list):
                        summary_items.append(f"- {field}: {', '.join(value)}")
                    else:
                        summary_items.append(f"- {field}: {value}")
                
                summary = "\n".join(summary_items)
                
                # Add completion message
                completion_message = f"Excellent! We've completed all the required information. Here's a summary of what you've provided:\n\n{summary}\n\nThank you for providing all this information. The form has been submitted successfully."
                state['messages'].append({"role": "assistant", "content": completion_message})
                state['completion_message_added'] = True
    
    except Exception as e:
        logger.error(f"Error in form completion check: {e}")
    
    return state

def end_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Final node in the workflow that performs cleanup and finalization.
    
    Args:
        state: Current state of the workflow
        
    Returns:
        Final state
    """
    logger.info("Ending workflow")
    
    # Add any final cleanup or processing here
    state['workflow_complete'] = True
    
    return state