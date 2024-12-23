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
        """Extract the actual customer query from a test case.
        
        Args:
            row: DataFrame row containing test case data
            
        Returns:
            str: Extracted customer query
        """
        try:
            if row['test_case'] == 'json':
                # Skip empty JSON responses
                if not row['golden_response'] or row['golden_response'] == '{}':
                    logger.warning(f"Empty JSON response for test case {row['id']}")
                    return row['prompt'] if row['prompt'] else "Could not extract query"
                    
                # Parse the JSON response if it's a string
                response_obj = row['golden_response']
                if isinstance(response_obj, str):
                    try:
                        print(response_obj)
                        response_obj = json.loads(response_obj)
                    except json.JSONDecodeError:
                        logger.error(f"Failed to parse JSON for test case {row['id']}")
                        return row['prompt']
                
                # Extract query based on our expected JSON structure
                try:
                    # Navigate the nested structure to find the query
                    interaction = response_obj.get('customer_interaction', {}).get('interaction', {})
                    query = interaction.get('summary', '')
                    if query:
                        return query
                    else:
                        logger.warning(f"No query found in JSON structure for test case {row['id']}")
                        return row['prompt']
                except (KeyError, AttributeError) as e:
                    logger.error(f"Error parsing JSON structure: {e}")
                    return row['prompt']
                    
            else:  # conversation test case
                # Extract query from different conversation formats
                query_text = row['prompt']
                
                # Look for text between single quotes (the actual customer query)
                if "'" in query_text:
                    parts = query_text.split("'")
                    if len(parts) >= 3:  # Ensure we have opening and closing quotes
                        return parts[1]
                
                # Fallback: try to find query after a colon
                if ':' in query_text:
                    return query_text.split(':', 1)[1].strip()
                
                return query_text
                
        except Exception as e:
            logger.error(f"Error extracting query: {e}")
            return row['prompt'] if row['prompt'] else "Could not extract query"
        
    async def run_tests(self, test_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Run all test cases against all models for specified number of rounds."""
        results = {}
        for model_name, model in self.models.items():
            model_results = []
            for round in range(1, self.config.TEST_ROUNDS + 1):
                round_results = await self._run_single_round(model_name, model, test_df, round)
                model_results.append(round_results)
            results[model_name] = pd.concat(model_results, ignore_index=True)
        
        self._save_failed_attempts()
        return results

    async def execute_tests(self, test_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
            """Run all test cases against all models for specified number of rounds."""
            results = {}
            for model_name, model in self.models.items():
                model_results = []
                for round in range(1, self.config.TEST_ROUNDS + 1):
                    round_results = await self._run_single_round(model_name, model, test_df, round)
                    model_results.append(round_results)
                results[model_name] = pd.concat(model_results, ignore_index=True)
            
            self._save_failed_attempts()
            return results

    async def _run_single_round(self, model_name: str, model, test_df: pd.DataFrame, round_num: int) -> pd.DataFrame:
        """Run a single round of testing for one model against all test cases."""
        responses = []
        for _, row in test_df.iterrows():
            for attempt in range(self.config.MAX_RETRIES):
                try:
                    await self._exponential_backoff(attempt)
                    
                    # Extract the actual customer query
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
                    
                    response_dict = self._create_response_dict(row, response, round_num)
                    response_dict['actual_query'] = customer_query  # Add the actual query for reference
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
        return {
            'id': row['id'],
            'prompt': row['prompt'],
            'model_response': response,
            'test_case': row['test_case'],
            'round': round_num,
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