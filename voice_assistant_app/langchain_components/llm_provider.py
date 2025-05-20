import os
from langchain_groq import ChatGroq
from langchain.schema import SystemMessage, HumanMessage
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from typing import List, Dict, Any, Optional
import logging
from dotenv import load_dotenv


_ = load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMProvider:
    """
    Configures and provides access to the Groq LLaMA 70B model via LangChain.
    """
    
    def __init__(self):
        """
        Initialize the LLM provider with Groq's LLaMA 70B model.
        Requires GROQ_API_KEY environment variable.
        """
        self.model_name = "llama3-70b-8192"
        
        if not os.getenv("GROQ_API_KEY"):
            logger.warning("GROQ_API_KEY environment variable not set")
            
        self.llm = ChatGroq(
            model_name=self.model_name,
            temperature=0.7,
            groq_api_key=os.getenv("GROQ_API_KEY")
        )
    
    def create_chain_with_system_prompt(self, system_prompt: str):
        """
        Creates a LangChain chain with a specified system prompt.
        
        Args:
            system_prompt: The system prompt to use
            
        Returns:
            LangChain chain object
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])
        
        return prompt | self.llm
    
    def extract_intent_and_value(self, 
                               user_input: str, 
                               field_name: str, 
                               field_description: str,
                               validation_rules: Dict[str, Any],
                               chat_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """
        Extracts user intent and field values from text input.
        
        Args:
            user_input: The transcribed user input
            field_name: The current field being processed
            field_description: Description of the field
            validation_rules: Dictionary of validation rules for the field
            chat_history: Optional chat history for context
            
        Returns:
            Dictionary with extracted intent and values
        """
        if chat_history is None:
            chat_history = []
            
        system_prompt = f"""
        You are an AI assistant helping extract structured information from user input.
        Your task is to identify the user's intent and extract the value for the field: '{field_name}'.
        
        Field description: {field_description}
        Validation rules: {validation_rules}
        
        You should return a JSON with the following structure:
        {{
            "intent": "provide_value" | "confirm" | "deny" | "request_help" | "request_skip" | "other",
            "extracted_value": The extracted value (if any) that matches the field type requirements,
            "confidence": A number between 0 and 1 indicating your confidence in the extraction,
            "reasoning": Brief explanation of your extraction logic
        }}
        
        If the user is confirming something, set intent to "confirm".
        If the user is denying or correcting something, set intent to "deny".
        If the user is asking for help or clarification, set intent to "request_help".
        If the user wants to skip this field, set intent to "request_skip".
        If the user is providing a value for the field, set intent to "provide_value" and extract the value.
        Otherwise, set intent to "other".
        
        Only extract values that directly relate to the current field ({field_name}).
        """
        
        # Convert chat history to LangChain message format
        messages = []
        for message in chat_history[-5:]:  # Only use the last 5 messages for context
            if message["role"] == "user":
                messages.append(HumanMessage(content=message["content"]))
            else:
                messages.append(SystemMessage(content=message["content"]))
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            *[(msg.type, msg.content) for msg in messages],
            ("human", user_input)
        ])
        
        chain = prompt | self.llm
        
        try:
            result = chain.invoke({})
            logger.info(f"LLM extraction result: {result.content}")
            # The result should be a JSON string that we can parse
            # But for simplicity in this example, we'll assume it's already structured properly
            return result.content
        except Exception as e:
            logger.error(f"Error in LLM extraction: {e}")
            return {
                "intent": "other",
                "extracted_value": None,
                "confidence": 0.0,
                "reasoning": f"Error occurred during extraction: {str(e)}"
            }