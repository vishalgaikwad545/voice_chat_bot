import streamlit as st
from typing import Dict, Any, List, Optional
from schemas import UserFormData, FormState

class StateManager:
    """
    Manages the application state within the Streamlit session state.
    """
    
    @staticmethod
    def initialize_state():
        """Initialize all required session state variables if they don't exist."""
        if 'chat_messages' not in st.session_state:
            st.session_state.chat_messages = []
            
        if 'form_state' not in st.session_state:
            st.session_state.form_state = FormState().model_dump()
            
        if 'form_data' not in st.session_state:
            st.session_state.form_data = {}
            
        if 'is_listening' not in st.session_state:
            st.session_state.is_listening = False
            
        if 'langgraph_state' not in st.session_state:
            st.session_state.langgraph_state = {}
            
    @staticmethod
    def add_message(role: str, content: str):
        """
        Add a message to the chat history.
        
        Args:
            role: Either 'user', 'assistant', or 'error'
            content: The message content
        """
        st.session_state.chat_messages.append({"role": role, "content": content})
    
    @staticmethod
    def get_chat_history() -> List[Dict[str, str]]:
        """Get the current chat history."""
        return st.session_state.chat_messages
    
    @staticmethod
    def update_form_state(updates: Dict[str, Any]):
        """
        Update the form state with new values.
        
        Args:
            updates: Dictionary of values to update
        """
        st.session_state.form_state.update(updates)
    
    @staticmethod
    def get_form_state() -> Dict[str, Any]:
        """Get the current form state."""
        return st.session_state.form_state
    
    @staticmethod
    def set_field_value(field_name: str, value: Any):
        """
        Set a value for a specific form field.
        
        Args:
            field_name: The name of the field to update
            value: The value to set
        """
        # Print debug info
        print(f"Setting field value: {field_name} = {value}")
        
        # Update the form data
        if value is not None:
            st.session_state.form_data[field_name] = value
        
        # Update the completed fields in form state if not already there and value is not None
        if field_name not in st.session_state.form_state['completed_fields'] and value is not None:
            st.session_state.form_state['completed_fields'].append(field_name)
        
        # Update field values in form state
        if 'field_values' not in st.session_state.form_state:
            st.session_state.form_state['field_values'] = {}
            
        st.session_state.form_state['field_values'][field_name] = value
    
    @staticmethod
    def get_form_data() -> Dict[str, Any]:
        """Get the current form data dictionary."""
        return st.session_state.form_data
    
    @staticmethod
    def get_current_field() -> str:
        """Get the name of the current field being processed."""
        return st.session_state.form_state['current_field']
    
    @staticmethod
    def set_current_field(field_name: str):
        """
        Set the current field being processed.
        
        Args:
            field_name: The name of the field
        """
        st.session_state.form_state['current_field'] = field_name
        st.session_state.form_state['confirmation_state'] = False
        st.session_state.form_state['extraction_attempts'] = 0
    
    @staticmethod
    def check_form_completion() -> bool:
        """
        Check if all required fields in the form have been completed.
        
        Returns:
            Boolean indicating if the form is complete
        """
        # Get all fields from UserFormData excluding optional ones
        required_fields = [
            field for field, model_field in UserFormData.__annotations__.items()
            if not str(model_field).startswith('Optional')
        ]
        
        completed_fields = st.session_state.form_state['completed_fields']
        is_complete = all(field in completed_fields for field in required_fields)
        
        st.session_state.form_state['is_complete'] = is_complete
        return is_complete
    
    @staticmethod
    def set_listening_state(is_listening: bool):
        """
        Set whether the application is currently listening for audio input.
        
        Args:
            is_listening: Boolean indicating listening state
        """
        st.session_state.is_listening = is_listening
    
    @staticmethod
    def is_listening() -> bool:
        """Check if the application is currently listening for audio."""
        return st.session_state.is_listening
    
    @staticmethod
    def reset_form():
        """Reset the form and chat state."""
        st.session_state.form_state = FormState().model_dump()
        st.session_state.form_data = {}
        st.session_state.chat_messages = []
        st.session_state.langgraph_state = {}