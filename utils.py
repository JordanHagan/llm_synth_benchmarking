import json
import logging
from typing import Any, Dict, Union

from agent_config import MODEL_CONFIG
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.base import RunnableSerializable
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('debug.log'),
        logging.StreamHandler()  # This will still print to console
    ]
)
logger = logging.getLogger(__name__)

def create_agent(
    system_prompt: str,
    model_name: str,
    agent_type: str,
    **llm_kwargs
) -> RunnableSerializable[Dict, str]:
    """
    Create a Langchain agent based on a system prompt and agent type.
    
    Args:
        system_prompt: The system message to use in the prompt.
        model_name: The name of the language model to use.
        agent_type: The type of agent to create (e.g., 'json_generator', 'conversation_generator', 'validator', 'model_A', 'model_B').
        **llm_kwargs: Additional keyword arguments to pass to the language model.
        
    Returns:
        A serializable runnable that takes a dictionary of inputs and returns a string.
    """
    if agent_type == 'json_generator':
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{id}\n{response_schema}\n{schema}\n{sample_size}")
        ])
    elif agent_type == 'conversation_generator':
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{id}\n{scenarios}\n{sample_size}")
        ])
    elif agent_type == 'validator':
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{test_cases}")
        ])
    elif agent_type == 'report_generator':
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "Metrics Data:\n{metrics_data}\n\nFeature Flags:\n{feature_flags}")
        ])
    else:  # model_A and model_B
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{prompt}")
        ])

    llm = ChatGroq(model=model_name, **llm_kwargs)
    
    return prompt | llm | StrOutputParser()

def parse_json_response(response: Any) -> list:
    """Parse JSON response with improved error handling for array format."""
    
    def is_valid_array_structure(data):
        """Check if data follows our expected array structure."""
        if not isinstance(data, list):
            return False
        for case in data:
            if not isinstance(case, list):
                return False
            # Each case should be a list of key-value pair arrays
            for pair in case:
                if not isinstance(pair, list) or len(pair) != 2:
                    return False
        return True

    if isinstance(response, list):
        if is_valid_array_structure(response):
            return response
        logger.warning(f"Response is a list but doesn't match expected structure: {response[:100]}...")
            
    if isinstance(response, str):
        try:
            # First try to parse as is
            parsed = json.loads(response)
            if is_valid_array_structure(parsed):
                return parsed
                
            # Try to find array patterns in the string
            import re
            array_pattern = r'\[\s*\[\s*\[.*?\]\s*\]\s*\]'
            matches = re.findall(array_pattern, response, re.DOTALL)
            
            for match in matches:
                try:
                    parsed_match = json.loads(match)
                    if is_valid_array_structure(parsed_match):
                        return parsed_match
                except json.JSONDecodeError:
                    continue
                    
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.debug(f"Problematic response: {response[:500]}...")
            
    logger.warning(f"Unexpected response type or structure: {type(response)}")
    return []