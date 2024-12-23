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
    level=logging.DEBUG,
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

    def _format_json_cases(self, raw_cases: List[Dict]) -> pd.DataFrame:
        """Format JSON test cases into required DataFrame structure."""
        formatted_cases = []
        for case in raw_cases:
            formatted_case = {
                'id': case.get('id', str(uuid.uuid4())),
                'prompt': case.get('prompt', ''),
                'golden_response': case.get('golden_response', {}),
                'test_case': 'json'
            }
            formatted_cases.append(formatted_case)
        return pd.DataFrame(formatted_cases)

    def _format_conversation_cases(self, raw_cases: List[Dict]) -> pd.DataFrame:
        """Format conversation test cases into required DataFrame structure."""
        formatted_cases = []
        for case in raw_cases:
            formatted_case = {
                'id': case.get('id', str(uuid.uuid4())),
                'prompt': case.get('prompt', ''),
                'golden_response': case.get('golden_response', ''),
                'test_case': 'conversation'
            }
            formatted_cases.append(formatted_case)
        return pd.DataFrame(formatted_cases)
    
    async def generate_test_cases(self, json_count=None, conv_count=None) -> pd.DataFrame:
        MAX_RETRIES = 3
        
        for attempt in range(MAX_RETRIES):
            try:
                # Generate JSON test cases
                json_params = {
                    "id": str(uuid.uuid4()),
                    "schema": json.dumps(self.test_config.response_schema, indent=2),
                    "response_schema": json.dumps(self.test_config.response_schema, indent=2),
                    "sample_size": json_count or self.test_config.test_categories['structured_json_output']['sample_size']
                }
                logger.debug(f"Generating JSON test cases with params: {json_params}")
                json_cases = await self.json_generator.ainvoke(json_params)
                
                # Generate conversation test cases
                conv_params = {
                    "id": str(uuid.uuid4()),
                    "scenarios": ', '.join(self.test_config.test_categories['customer_support']['scenarios']),
                    "sample_size": conv_count or self.test_config.test_categories['customer_support']['sample_size']
                }
                logger.debug(f"Generating conversation test cases with params: {conv_params}")
                conv_cases = await self.conversation_generator.ainvoke(conv_params)
                
                # Parse responses
                json_cases = parse_json_response(json_cases)
                conv_cases = parse_json_response(conv_cases)
                
                if not json_cases:
                    logger.error(f"Failed to parse JSON responses on attempt {attempt + 1}")
                    continue
                    
                if not conv_cases:
                    logger.error(f"Failed to parse conversation responses on attempt {attempt + 1}")
                    continue

                # Format cases
                json_df = self._format_json_cases(json_cases)
                conv_df = self._format_conversation_cases(conv_cases)
                
                # Log the number of cases generated
                logger.info(f"Generated {len(json_df)} JSON test cases and {len(conv_df)} conversation test cases")
                
                # Combine cases
                all_cases = pd.concat([json_df, conv_df], ignore_index=True)
                
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