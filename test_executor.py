import asyncio
import logging
import random
import json
from typing import Dict, List
import pandas as pd
from dataclasses import dataclass
from datetime import datetime

from utils import create_agent
from agent_config import MODEL_CONFIG

logging.basicConfig(
    # level=logging.DEBUG,
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('debug.log'),
        logging.StreamHandler()  # This will still print to console
    ]
)

logger = logging.getLogger(__name__)

@dataclass
class TestAttempt:
    """Data class for tracking failed test attempts."""
    id: str
    prompt: str
    timestamp: str
    attempt_number: int
    error: str = None
    response: str = None
    
class TestExecutor:
    def __init__(self, config):
        self.config = config
        self.models = {}
        self.failed_attempts: List[TestAttempt] = []
        self._initialize_models()
            
    def _initialize_models(self):
        """Initialize all language models specified in the configuration."""
        for name, model_path in self.config.TEST_MODELS.items():
            try:
                model_config = MODEL_CONFIG['executors'][name]
                if name == 'model_A':
                    self.models[name] = create_agent(
                        system_prompt=model_config['prompt'],
                        model_name=model_config['model_name'],
                        agent_type='model_A',
                        temperature=model_config['temperature']
                    )
                elif name == 'model_B':
                    self.models[name] = create_agent(
                        system_prompt=model_config['prompt'],
                        model_name=model_config['model_name'],
                        agent_type='model_B',
                        temperature=model_config['temperature']
                    )
                else:
                    self.models[name] = create_agent(
                        system_prompt=model_config['prompt'],
                        model_name=model_config['model_name'],
                        temperature=model_config['temperature']
                    )
            except Exception as e:
                logger.error(f"Failed to initialize model {name}: {e}")
                raise RuntimeError(f"Model initialization failed: {str(e)}")

    def _extract_customer_query(self, row) -> str:
        """Extract the actual customer query from a test case."""
        try:
            if row['test_case'] == 'json':
                response_obj = row['golden_response']
                if isinstance(response_obj, str):
                    try:
                        response_obj = json.loads(response_obj)
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse JSON for test case {row['id']}")
                        return self._clean_query_text(row['prompt'])
                
                try:
                    interaction = response_obj.get('customer_interaction', {}).get('interaction', {})
                    query = interaction.get('summary', '')
                    return self._clean_query_text(query) if query else self._clean_query_text(row['prompt'])
                except (KeyError, AttributeError):
                    return self._clean_query_text(row['prompt'])
                    
            else:  # conversation test case
                return self._clean_query_text(row['prompt'])
                
        except Exception as e:
            logger.error(f"Error extracting query: {e}")
            return "Could not extract query"

    def _clean_query_text(self, text: str) -> str:
        """Clean and normalize query text."""
        if not text:
            return "No query provided"
            
        text = str(text).strip()
        text = text.replace('\n', ' ').replace('\r', ' ')
        text = ' '.join(text.split())  # Normalize whitespace
        
        # Remove any surrounding quotes
        if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
            text = text[1:-1]
            
        return text
            
    async def run_tests(self, test_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Run all test cases against all models for specified number of rounds."""
        raw_results = {}
        
        # Run tests for each model
        for model_name, model in self.models.items():
            model_results = []
            
            # Run multiple rounds
            for round_num in range(1, self.config.TEST_ROUNDS + 1):
                try:
                    round_results = await self._run_single_round(model_name, model, test_df, round_num)
                    model_results.append(round_results)
                except Exception as e:
                    logger.error(f"Failed to run round {round_num} for model {model_name}: {e}")
                    continue
            
            # Combine all rounds for this model
            if model_results:
                raw_results[model_name] = pd.concat(model_results, ignore_index=True)
            else:
                logger.error(f"No valid results for model {model_name}")
                raw_results[model_name] = pd.DataFrame()  # Empty DataFrame with correct structure
        
        # Store failed attempts
        self._save_failed_attempts()
        
        return raw_results

    async def _run_single_round(self, model_name: str, model, test_df: pd.DataFrame, round_num: int) -> pd.DataFrame:
        """Run a single round of testing for one model against all test cases."""
        responses = []
        for _, row in test_df.iterrows():
            for attempt in range(self.config.MAX_RETRIES):
                try:
                    await self._exponential_backoff(attempt)
                    
                    customer_query = self._extract_customer_query(row)
                    logger.debug(f"Extracted query: {customer_query} from test case {row['id']}")
                    
                    model_input = {
                        'id': row['id'],
                        'prompt': customer_query
                    }
                    
                    response = await asyncio.wait_for(
                        model.ainvoke(model_input),
                        timeout=self.config.TIMEOUT
                    )
                    
                    # Clean the response string
                    if isinstance(response, str):
                        response = response.replace("\n", " ").replace("\r", " ").strip()
                    
                    response_dict = self._create_response_dict(row, response, round_num)
                    response_dict['actual_query'] = customer_query
                    responses.append(response_dict)
                    break
                except Exception as e:
                    self._handle_test_failure(row, attempt, e, round_num)
                    if attempt == self.config.MAX_RETRIES - 1:
                        responses.append(self._create_error_response(row, str(e), round_num))
        
        return pd.DataFrame(responses)
        

    async def _exponential_backoff(self, attempt: int):
        """Implement exponential backoff between retry attempts."""
        if attempt > 0:
            delay = min(300, (2 ** attempt) + (random.uniform(0, 1)))
            await asyncio.sleep(delay)

    def _create_response_dict(self, row, response, round_num):
        """Create a dictionary containing test response data."""
        try:
            return {
                'id': str(row['id']),  # Ensure ID is string
                'prompt': str(row['prompt']),
                'model_response': str(response),  # Ensure response is string
                'test_case': str(row['test_case']),
                'round': int(round_num),
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error creating response dictionary: {e}")
            return {
                'id': str(row['id']),
                'prompt': str(row['prompt']),
                'model_response': f"ERROR: Failed to format response - {str(e)}",
                'test_case': str(row['test_case']),
                'round': int(round_num),
                'timestamp': datetime.now().isoformat()
            }

    def _create_error_response(self, row, error, round_num):
        """Create a dictionary containing error response data."""
        return {
            'id': row['id'],
            'prompt': row['prompt'],
            'model_response': f"ERROR: {error}",
            'test_case': row['test_case'],
            'round': round_num,
            'timestamp': datetime.now().isoformat()
        }

    def _handle_test_failure(self, row, attempt, error, round_num):
        """Record a failed test attempt."""
        attempt_record = TestAttempt(
            id=row['id'],
            prompt=row['prompt'],
            timestamp=datetime.now().isoformat(),
            attempt_number=attempt + 1,
            error=str(error)
        )
        self.failed_attempts.append(attempt_record)
        logger.error(f"Test failed - ID: {row['id']}, Attempt: {attempt + 1}, Error: {error}")

    def _save_failed_attempts(self):
        """Save all failed test attempts to a CSV file."""
        if self.failed_attempts:
            df = pd.DataFrame([vars(attempt) for attempt in self.failed_attempts])
            df.to_csv(f"failed_attempts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", index=False)