import json
import logging
import random
import asyncio
from typing import Any, Dict, Union, Callable
from functools import wraps

from agent_config import MODEL_CONFIG
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.base import RunnableSerializable
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser

# Enhanced logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('debug.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def groq_rate_limit(max_retries: int = 3, base_delay: float = 1.0, max_delay: float = 60.0):
    """
    A decorator for handling Groq API rate limits with exponential backoff.
    """
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(max_retries):
                try:
                    # If the function is async, await it
                    if asyncio.iscoroutinefunction(func):
                        return await func(*args, **kwargs)
                    # If the function is sync, just call it
                    return func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    error_str = str(e).lower()
                    
                    if any(code in error_str for code in ["429", "too many requests"]):
                        delay = min(base_delay * (2 ** attempt) + random.uniform(0, 1), max_delay)
                        logger.info(f"Rate limit hit. Waiting {delay:.2f}s before retry {attempt + 1}/{max_retries}")
                    elif "503" in error_str:
                        delay = min(base_delay * (2 ** attempt), max_delay)
                        logger.warning(f"Service unavailable. Waiting {delay:.2f}s before retry {attempt + 1}/{max_retries}")
                    else:
                        raise last_error
                        
                    await asyncio.sleep(delay)
            
            raise last_error
            
        return wrapper
    return decorator

@groq_rate_limit()
async def create_agent(
    system_prompt: str,
    model_name: str,
    agent_type: str,
    **llm_kwargs
) -> Any:
    """
    Create an agent that directly interfaces with Groq's API for tool use.
    
    Args:
        system_prompt: The system message to use in the prompt
        model_name: The name of the language model to use
        agent_type: The type of agent to create
        **llm_kwargs: Additional keyword arguments including tools
        
    Returns:
        An agent object with async invoke capabilities
    """
    from groq import AsyncGroq
    
    client = AsyncGroq()
    
    async def ainvoke(self, inputs: Dict) -> str:
        try:
            messages = [{"role": "system", "content": system_prompt}]
            content = inputs.get('input', json.dumps(inputs))
            messages.append({"role": "user", "content": content})
            
            response = await client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=llm_kwargs.get('temperature', 0.0),
                tools=llm_kwargs.get('tools', []),
                tool_choice="auto"
            )
            
            if response.choices[0].message.tool_calls:
                tool_call = response.choices[0].message.tool_calls[0]
                return tool_call.function.arguments
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Agent invocation failed: {str(e)}")
            raise
    
    return type('Agent', (), {'ainvoke': ainvoke})()

def parse_json_response(response: Any) -> Union[list, dict]:
    """
    Parse JSON response with enhanced error handling and validation.
    
    Args:
        response: The response to parse, can be string or structured data
        
    Returns:
        Parsed JSON data as list or dict, empty list if parsing fails
    """
    def is_valid_array_structure(data: Any) -> bool:
        """Validate array structure matches expected format."""
        if not isinstance(data, list):
            return False
            
        return all(
            isinstance(case, list) and 
            all(isinstance(pair, list) and len(pair) == 2 for pair in case)
            for case in data
        )

    # Handle direct list input
    if isinstance(response, list):
        return response if is_valid_array_structure(response) else []

    # Handle string input
    if isinstance(response, str):
        try:
            parsed = json.loads(response)
            
            # If parsing succeeds, validate structure
            if isinstance(parsed, (list, dict)):
                if isinstance(parsed, list) and not is_valid_array_structure(parsed):
                    logger.warning("Parsed JSON does not match expected array structure")
                return parsed
                
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing failed: {str(e)}")
            logger.debug(f"Problematic response: {response[:500]}...")
            
    logger.warning(f"Unexpected response type: {type(response)}")
    return []

def setup_logging(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Configure logging settings for a module with consistent formatting.
    
    Args:
        name: Name of the logger (typically __name__)
        level: Logging level to use (default: INFO)
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Create handlers if they don't exist
    if not logger.handlers:
        # File handler
        file_handler = logging.FileHandler('debug.log')
        file_handler.setLevel(level)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Add formatter to handlers
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers to logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    
    return logger