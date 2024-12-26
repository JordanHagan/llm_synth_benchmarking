import uuid
import json
import logging
import asyncio
import pandas as pd

from typing import Dict, List
from agent_config import MODEL_CONFIG
from utils import create_agent, parse_json_response
from benchmark_config import BenchmarkConfig
from test_config import TestConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('debug.log'),
        logging.StreamHandler()  # This will still print to console
    ]
)
logger = logging.getLogger(__name__)

class TestGenerator:
    """Generates test cases for both JSON and conversation-based testing."""
    
    def __init__(self, config: BenchmarkConfig, test_config: TestConfig):
        """Initialize with configuration objects."""
        self.config = config
        self.test_config = test_config
        self._initialize_generators()
        
    def _initialize_generators(self):
        """Initialize test case generators."""
        try:
            self.json_generator = create_agent(
                system_prompt=MODEL_CONFIG['json_generator']['prompt'],
                model_name=MODEL_CONFIG['json_generator']['model_name'],
                temperature=MODEL_CONFIG['json_generator']['temperature'],
                agent_type='json_generator'
                
            )
            self.conversation_generator = create_agent(
                system_prompt=MODEL_CONFIG['conversation_generator']['prompt'],
                model_name=MODEL_CONFIG['conversation_generator']['model_name'],
                temperature=MODEL_CONFIG['conversation_generator']['temperature'],
                agent_type='conversation_generator'
            )
        except Exception as e:
            logger.error(f"Generator initialization failed: {e}")
            raise

    def _array_to_dict(self, array_data):
        """Convert array format to dictionary format recursively."""
        if isinstance(array_data, list):
            # If it's a key-value pair
            if len(array_data) == 2 and isinstance(array_data[0], str):
                key, value = array_data
                # Recursively convert nested arrays
                if isinstance(value, list):
                    # Special case for next_steps which should remain an array
                    if key == "next_steps":
                        return key, value
                    return key, self._array_to_dict(value)
                return key, value
            # If it's a list of key-value pairs
            result = {}
            for item in array_data:
                if isinstance(item, list) and len(item) == 2:
                    key, value = self._array_to_dict(item)
                    result[key] = value
            return result
        return array_data

    def _format_json_cases(self, raw_cases: List) -> pd.DataFrame:
        """Format JSON test cases into required DataFrame structure."""
        formatted_cases = []
        
        try:
            for case_array in raw_cases:
                # Convert the array format to dictionary
                case_dict = self._array_to_dict(case_array)
                
                formatted_case = {
                    'id': case_dict.get('id', str(uuid.uuid4())),
                    'prompt': case_dict.get('prompt', ''),
                    'golden_response': case_dict.get('golden_response', {}),
                    'test_case': case_dict.get('test_case', 'json')
                }
                formatted_cases.append(formatted_case)
                
                logger.debug(f"Formatted case: {formatted_case}")
                
        except Exception as e:
            logger.error(f"Error formatting cases: {e}")
            logger.error(f"Raw cases: {raw_cases}")
            raise
            
        return pd.DataFrame(formatted_cases)

    def _format_conversation_cases(self, raw_cases: List) -> pd.DataFrame:
        formatted_cases = []
        try:
            logger.debug(f"Raw conversation cases type: {type(raw_cases)}")
            logger.debug(f"First case type: {type(raw_cases[0])}")
            logger.debug(f"Raw cases content: {raw_cases}")

            for case_array in raw_cases:
                case_dict = self._array_to_dict(case_array)
                formatted_case = {
                    'id': case_dict.get('id', str(uuid.uuid4())),
                    'prompt': case_dict.get('prompt', ''),
                    'golden_response': case_dict.get('golden_response', ''),
                    'test_case': case_dict.get('test_case', 'conversation')
                }
                formatted_cases.append(formatted_case)
                
        except Exception as e:
            logger.error(f"Error in _format_conversation_cases: {e}")
            logger.error(f"Raw cases: {raw_cases}")
            raise
            
        return pd.DataFrame(formatted_cases)
    
    async def generate_test_cases(self, json_count=None, conv_count=None) -> pd.DataFrame:
        MAX_RETRIES = 3
        
        for attempt in range(MAX_RETRIES):
            try:
                cases = []
                
                # Only generate JSON cases if enabled
                if self.test_config.enable_json_tests:
                    json_params = {
                        "id": str(uuid.uuid4()),
                        "schema": json.dumps(self.test_config.response_schema, indent=2),
                        "response_schema": json.dumps(self.test_config.response_schema, indent=2),
                        "sample_size": json_count or self.test_config.test_categories['structured_json_output']['sample_size']
                    }
                    json_cases = await self.json_generator.ainvoke(json_params)
                    json_cases = parse_json_response(json_cases)
                    if json_cases:
                        json_df = self._format_json_cases(json_cases)
                        cases.append(json_df)

                # Generate conversation cases
                conv_params = {
                    "id": str(uuid.uuid4()),
                    "scenarios": ', '.join(self.test_config.test_categories['customer_support']['scenarios']),
                    "sample_size": conv_count or self.test_config.test_categories['customer_support']['sample_size']
                }
                conv_cases = await self.conversation_generator.ainvoke(conv_params)
                conv_cases = parse_json_response(conv_cases)
                if conv_cases:
                    conv_df = self._format_conversation_cases(conv_cases)
                    cases.append(conv_df)

                if cases:
                    all_cases = pd.concat(cases, ignore_index=True)
                    logger.info(f"Generated {len(all_cases)} total test cases")
                    if not all_cases.empty:
                        return all_cases
                        
            except Exception as e:
                if "Rate limit reached" in str(e):
                    retry_time = float(str(e).split("Please try again in ")[1].split(".")[0])
                    logger.warning(f"Rate limit exceeded. Retrying in {retry_time} seconds...")
                    await asyncio.sleep(retry_time)
                else:
                    logger.error(f"Test generation attempt {attempt + 1} failed: {e}")
                    if attempt == MAX_RETRIES - 1:
                        raise RuntimeError(f"All test generation attempts failed: {str(e)}")
            
        raise ValueError("No valid test cases generated")