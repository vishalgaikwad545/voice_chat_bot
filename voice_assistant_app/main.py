import streamlit as st
import json
import os
from datetime import datetime
from typing import Dict, Any, List, Optional
import logging
from dotenv import load_dotenv

# Import custom modules
from schemas import UserFormData
from utils.audio_processor import AudioProcessor
from utils.state_manager import StateManager
from langgraph_flow.graph import process_user_input, get_initial_greeting

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the audio processor
audio_processor = AudioProcessor()

# Page configuration
st.set_page_config(
    page_title="Voice-Controlled Form Assistant",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CSS for styling the chat interface
st.markdown("""
<style>
    /* Main containers */
    .main {
        padding: 1rem 1rem;
    }
    
    /* Form styling */
    .stForm {
        background-color: #da614e;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    /* Chat container */
    .chat-container {
        position: fixed;
        bottom: 20px;
        right: 20px;
        width: 350px;
        border-radius: 10px;
        background-color: white;
        box-shadow: 0 0 10px rgba(0, 0, 0, 0.2);
        z-index: 1000;
        max-height: 500px;
        display: flex;
        flex-direction: column;
    }
    
    /* Chat header */
    .chat-header {
        background-color: #4CAF50;
        color: white;
        padding: 10px 15px;
        border-radius: 10px 10px 0 0;
        font-weight: bold;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    /* Chat messages container */
    .chat-messages {
        padding: 15px;
        overflow-y: auto;
        flex-grow: 1;
        max-height: 350px;
    }
    
    /* User message */
    .user-message {
        background-color: #DCF8C6;
        padding: 8px 12px;
        border-radius: 10px 10px 0 10px;
        margin: 5px 0;
        max-width: 80%;
        align-self: flex-end;
        float: right;
        clear: both;
    }
    
    /* Assistant message */
    .assistant-message {
        background-color: #da614e;
        padding: 8px 12px;
        border-radius: 10px 10px 10px 0;
        margin: 5px 0;
        max-width: 80%;
        align-self: flex-start;
        float: left;
        clear: both;
    }
    
    /* Error message */
    .error-message {
        background-color: #FFEBEE;
        color: #D32F2F;
        padding: 12px 15px;
        border-radius: 10px 10px 10px 0;
        margin: 8px 0;
        max-width: 90%;
        align-self: flex-start;
        float: left;
        clear: both;
        border-left: 4px solid #D32F2F;
        font-weight: bold;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
    }
    
    /* Speech detection error specific styling */
    .speech-error {
        background-color: #FFF4E5;
        color: #E65100;
        border-left: 4px solid #FF8F00;
        font-size: 1.05em;
        display: flex;
        align-items: center;
    }
    
    /* Error icon */
    .error-message::before {
        content: "⚠️ ";
        margin-right: 8px;
        font-size: 1.2em;
    }
    
    /* Chat input area */
    .chat-input {
        padding: 10px;
        border-top: 1px solid #e0e0e0;
        display: flex;
        background-color: #da614e;
        border-radius: 0 0 10px 10px;
    }
    
    /* Voice button */
    .voice-button {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        background-color: #4CAF50;
        color: white;
        border: none;
        border-radius: 50%;
        width: 40px;
        height: 40px;
        cursor: pointer;
        margin-left: 10px;
    }
    
    .voice-button:hover {
        background-color: #45a049;
    }
    
    /* Form field with validation error */
    .field-error .stTextInput, .field-error .stNumberInput, .field-error .stDateInput, .field-error .stSelectbox {
        border-color: red !important;
    }
    
    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Loading spinner for voice input */
    .listening-indicator {
        display: inline-block;
        width: 20px;
        height: 20px;
        border: 3px solid rgba(255,255,255,.3);
        border-radius: 50%;
        border-top-color: white;
        animation: spin 1s ease-in-out infinite;
    }
    
    @keyframes spin {
        to { transform: rotate(360deg); }
    }
</style>
""", unsafe_allow_html=True)

def render_chat_message(message: Dict[str, str]):
    """
    Renders a single chat message with appropriate styling.
    
    Args:
        message: Dictionary with 'role' and 'content' keys
    """
    role = message.get('role', 'assistant')
    content = message.get('content', '')
    
    if role == 'user':
        st.markdown(f'<div class="user-message">{content}</div>', unsafe_allow_html=True)
    elif 'No speech detected' in content:
        # Special styling for speech detection errors
        st.markdown(f'<div class="error-message speech-error">{content}</div>', unsafe_allow_html=True)
    elif role == 'error' or 'Voice input error' in content:
        st.markdown(f'<div class="error-message">{content}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="assistant-message">{content}</div>', unsafe_allow_html=True)

def render_chat_interface(on_submit=None):
    """
    Renders the chat interface in the bottom-right corner.
    
    Args:
        on_submit: Optional callback function for text input submission
    """
    # Chat container
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
    # Chat header
    listening_indicator = ""
    if st.session_state.is_listening:
        listening_indicator = '<div class="listening-indicator"></div>'
    
    st.markdown(
        f'<div class="chat-header">Form Assistant {listening_indicator}</div>',
        unsafe_allow_html=True
    )
    
    # Chat messages container
    st.markdown('<div class="chat-messages">', unsafe_allow_html=True)
    
    # Render chat messages
    chat_placeholder = st.empty()
    with chat_placeholder.container():
        for message in st.session_state.chat_messages:
            render_chat_message(message)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Chat input area
    st.markdown('<div class="chat-input">', unsafe_allow_html=True)
    
    # Text input and voice button
    col1, col2 = st.columns([5, 1])
    
    with col1:
        if on_submit:
            # Use the on_change parameter to trigger the callback
            user_input = st.text_input(
                "", 
                key="user_text_input", 
                label_visibility="collapsed",
                on_change=on_submit
            )
        else:
            # Default behavior without callback
            user_input = st.text_input("", key="user_text_input", label_visibility="collapsed")
    
    with col2:
        if st.button("🎤", key="voice_button"):
            # Set listening state to true to show indicator
            StateManager.set_listening_state(True)
            st.rerun()  # Rerun to show the listening indicator
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Close chat container
    st.markdown('</div>', unsafe_allow_html=True)
    
    return user_input

def generate_form():
    """
    Generates a dynamic form based on the Pydantic model and current state.
    """
    st.title("Voice-Controlled Form")
    st.write("Fill out this form by talking to the assistant in the bottom-right corner.")
    
    # Get the form data and state
    form_data = StateManager.get_form_data()
    form_state = StateManager.get_form_state()
    
    # Get form fields from Pydantic model
    form_schema = UserFormData.model_json_schema()
    properties = form_schema.get('properties', {})
    required_fields = form_schema.get('required', [])
    
    # Create a form
    with st.form(key="user_form"):
        # For each field in the Pydantic model
        for field_name, field_info in properties.items():
            field_type = field_info.get('type')
            field_description = field_info.get('description', '')
            field_value = form_data.get(field_name)
            
            # Check if field is required
            is_required = field_name in required_fields
            required_label = " *" if is_required else ""
            
            # Style based on completion status
            is_completed = field_name in form_state.get('completed_fields', [])
            field_style = "" if is_completed else "field-incomplete"
            
            # Check if current field is being processed
            is_current = field_name == form_state.get('current_field')
            if is_current:
                field_style += " field-current"
            
            # Create different input types based on field type
            st.write(f"### {field_name.replace('_', ' ').title()}{required_label}")
            st.write(f"<small>{field_description}</small>", unsafe_allow_html=True)
            
            if field_name == 'full_name':
                st.text_input(
                    "Full Name", 
                    value=field_value if field_value else "",
                    key=f"form_{field_name}",
                    disabled=True
                )
            elif field_name == 'email':
                st.text_input(
                    "Email", 
                    value=field_value if field_value else "",
                    key=f"form_{field_name}",
                    disabled=True
                )
            elif field_name == 'age':
                st.number_input(
                    "Age", 
                    min_value=18, 
                    max_value=120,
                    value=field_value if field_value else 18,
                    key=f"form_{field_name}",
                    disabled=True
                )
            elif field_name == 'occupation':
                st.text_input(
                    "Occupation", 
                    value=field_value if field_value else "",
                    key=f"form_{field_name}",
                    disabled=True
                )
            elif field_name == 'experience_level':
                options = field_info.get('enum', ["Beginner", "Intermediate", "Advanced", "Expert"])
                st.selectbox(
                    "Experience Level", 
                    options=options,
                    index=options.index(field_value) if field_value in options else 0,
                    key=f"form_{field_name}",
                    disabled=True
                )
            elif field_name == 'preferred_language':
                options = field_info.get('enum', ["Python", "JavaScript", "Java", "C++", "Go", "Rust", "Other"])
                st.selectbox(
                    "Preferred Language", 
                    options=options,
                    index=options.index(field_value) if field_value in options else 0,
                    key=f"form_{field_name}",
                    disabled=True
                )
            elif field_name == 'project_interests':
                if field_value and isinstance(field_value, list):
                    interests_text = ", ".join(field_value)
                else:
                    interests_text = ""
                st.text_area(
                    "Project Interests", 
                    value=interests_text,
                    key=f"form_{field_name}",
                    disabled=True,
                    help="Enter interests separated by commas"
                )
            elif field_name == 'availability_per_week':
                st.number_input(
                    "Availability (hours per week)", 
                    min_value=1, 
                    max_value=168,
                    value=field_value if field_value else 1,
                    key=f"form_{field_name}",
                    disabled=True
                )
            elif field_name == 'start_date':
                if field_value:
                    if isinstance(field_value, str):
                        try:
                            date_value = datetime.strptime(field_value, "%Y-%m-%d").date()
                        except ValueError:
                            date_value = datetime.now().date()
                    else:
                        date_value = field_value
                else:
                    date_value = datetime.now().date()
                
                st.date_input(
                    "Start Date", 
                    value=date_value,
                    key=f"form_{field_name}",
                    disabled=True
                )
            elif field_name == 'additional_notes':
                st.text_area(
                    "Additional Notes", 
                    value=field_value if field_value else "",
                    key=f"form_{field_name}",
                    disabled=True
                )
            
            # Add some spacing between fields
            st.write("")
        
        # Submit button (disabled since we're using voice control)
        st.form_submit_button("Submit Form", disabled=True)
    
    # If form is complete, show the output
    if form_state.get('is_complete', False):
        st.success("Form completed successfully!")
        
        # Display the JSON output
        st.write("### Form Data (JSON Output)")
        st.json(form_data)

def handle_voice_input():
    """Handles voice input if the listening state is active."""
    if StateManager.is_listening():
        try:
            # Get the selected device index from session state
            device_index = st.session_state.get('selected_audio_device', None)
            if device_index == "default":
                device_index = None
                
            # Capture and transcribe audio
            success, text, error = audio_processor.capture_and_transcribe(
                device_index=device_index
            )
            
            # Update listening state
            StateManager.set_listening_state(False)
            
            if success and text:
                # Process the transcribed text
                process_text_input(text)
            elif error:
                # Create a more visible error message based on error type
                if "NO SPEECH DETECTED" in error:
                    error_msg = f"⚠️ {error}"
                else:
                    error_msg = f"⚠️ Voice input error: {error}"
                
                # Add error message to chat with error role
                StateManager.add_message("error", error_msg)
                
        except Exception as e:
            # Handle any unexpected errors during voice processing
            error_msg = f"⚠️ Error with voice input: {str(e)}"
            StateManager.add_message("error", error_msg)
            StateManager.set_listening_state(False)
            
        # Force app to refresh
        st.rerun()

def process_text_input(text: str):
    """
    Processes text input through the LangGraph workflow.
    
    Args:
        text: User input text
    """
    if not text.strip():
        return
    
    # Add user message to chat
    StateManager.add_message("user", text)
    
    # Get current LangGraph state
    langgraph_state = st.session_state.get('langgraph_state', {})
    
    # Process the input through the workflow
    updated_state = process_user_input(langgraph_state, text)
    
    # Store the updated state
    st.session_state.langgraph_state = updated_state
    
    # Update the StateManager with field values and state
    if 'field_values' in updated_state:
        for field, value in updated_state['field_values'].items():
            StateManager.set_field_value(field, value)
    
    if 'current_field' in updated_state:
        StateManager.set_current_field(updated_state['current_field'])
    
    # Update form state
    form_state_updates = {
        'current_field': updated_state.get('current_field', ''),
        'confirmation_state': updated_state.get('confirmation_state', False),
        'extraction_attempts': updated_state.get('extraction_attempts', 0),
        'is_complete': updated_state.get('is_complete', False)
    }
    StateManager.update_form_state(form_state_updates)
    
    # Add assistant messages to chat
    new_messages = updated_state.get('messages', [])
    if new_messages:
        # Just add the last assistant message that wasn't added yet
        current_messages = StateManager.get_chat_history()
        current_message_contents = [m['content'] for m in current_messages]
        
        for message in reversed(new_messages):
            if message['role'] == 'assistant' and message['content'] not in current_message_contents:
                StateManager.add_message("assistant", message['content'])
                break

def main():
    """Main application function."""
    # Initialize the session state
    StateManager.initialize_state()
    
    # Add an audio device selector in the sidebar
    st.sidebar.title("Audio Settings")
    
    try:
        # Get available audio devices
        audio_devices = audio_processor.get_available_devices()
        
        # Create a list of options for the selectbox
        device_options = [{"label": "Default Microphone", "value": "default"}]
        device_options.extend([
            {"label": f"{device['name']}", "value": device['index']}
            for device in audio_devices
        ])
        
        # Create a dictionary mapping labels to values for easier lookup
        device_map = {item["label"]: item["value"] for item in device_options}
        
        # Initialize selected device in session state if not present
        if 'selected_audio_device' not in st.session_state:
            st.session_state.selected_audio_device = "default"
        
        # Display the dropdown for device selection
        selected_device_label = st.sidebar.selectbox(
            "Select Microphone Device",
            options=[item["label"] for item in device_options],
            index=0,
            key="audio_device_selector"
        )
        
        # Update the selected device in session state when changed
        st.session_state.selected_audio_device = device_map[selected_device_label]
        
        # Add a help text
        st.sidebar.info("If voice input isn't working, try selecting a different microphone device.")
    
    except Exception as e:
        st.sidebar.error(f"Error loading audio devices: {str(e)}")
        st.sidebar.info("Using default microphone")
        st.session_state.selected_audio_device = None
    
    # Handle voice input if in listening state
    handle_voice_input()
    
    # Generate the dynamic form
    generate_form()
    
    # Define a callback to handle text input submission
    def on_text_input_submit():
        user_input = st.session_state.user_text_input
        if user_input:
            process_text_input(user_input)
            # We'll rely on Streamlit's automatic form reset behavior
            st.rerun()
    
    # Render the chat interface with the callback
    render_chat_interface(on_submit=on_text_input_submit)
    
    # Initialize with greeting if no messages yet
    if not st.session_state.chat_messages:
        initial_state = get_initial_greeting()
        
        # Store the initial state
        st.session_state.langgraph_state = initial_state
        
        # Add the greeting message
        if 'messages' in initial_state and initial_state['messages']:
            for message in initial_state['messages']:
                if message['role'] == 'assistant':
                    StateManager.add_message('assistant', message['content'])
        
        # Force refresh
        st.rerun()
        
if __name__ == "__main__":
    main()