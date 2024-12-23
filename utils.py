import json
import logging
from typing import Any, Dict, Union

from agent_config import MODEL_CONFIG
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.base import RunnableSerializable
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser

logging.basicConfig(
    level=logging.DEBUG,
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
    else:  # model_A and model_B
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{prompt}")
        ])

    llm = ChatGroq(model=model_name, **llm_kwargs)
    
    return prompt | llm | StrOutputParser()

def parse_json_response(response: Any) -> list:
    """Parse JSON response with improved error handling.
    
    Args:
        response: Response to parse. Can be:
            - A list of test cases
            - A single test case object
            - A string containing JSON (either array or object)
            - A string with multiple JSON objects
        
    Returns:
        List of test cases
    """
    if isinstance(response, list):
        return response
        
    if isinstance(response, dict):
        return [response]

    if isinstance(response, str):
        try:
            # First try to parse as is
            parsed = json.loads(response)
            if isinstance(parsed, list):
                return parsed
            if isinstance(parsed, dict):
                return [parsed]

            # If that fails, try to find JSON objects in the string
            test_cases = []
            # Look for patterns like {...} or [{...}]
            import re
            json_objects = re.findall(r'\{[^{}]*\}', response)
            
            for json_str in json_objects:
                try:
                    test_case = json.loads(json_str)
                    if all(key in test_case for key in ['id', 'prompt', 'golden_response', 'test_case']):
                        test_cases.append(test_case)
                except json.JSONDecodeError:
                    continue
            
            if test_cases:
                return test_cases
                
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.debug(f"Problematic response: {response}")
            
    logger.warning(f"Unexpected response type: {type(response)}")
    return []