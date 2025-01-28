import asyncio
import logging
import json
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
from datetime import datetime
import pandas as pd
from groq import Groq

from agent_config import MODEL_CONFIG
from utils import groq_rate_limit

logger = logging.getLogger(__name__)

@dataclass
class TestAttempt:
    """Records details of test execution attempts, including any failures."""
    id: str
    prompt: str
    timestamp: str
    attempt_number: int
    error: Optional[str] = None
    response: Optional[str] = None

class TestExecutor:
    """
    Executes test cases against multiple models with proper error handling and rate limiting.
    
    This class manages the execution of test cases across different models, handles retries,
    and records test results and failures.
    """
    
    def __init__(self, config):
        """Initialize the test executor with configuration and models."""
        self.config = config
        self.client = Groq()
        self.models = {}
        self.failed_attempts: List[TestAttempt] = []
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize model configurations from the provided config."""
        for name, model_path in self.config.TEST_MODELS.items():
            model_config = MODEL_CONFIG['executors'][name]
            self.models[name] = {
                'model': model_config['model_name'],
                'prompt': model_config['prompt'],
                'temperature': model_config['temperature']
            }
    
    def _extract_customer_query(self, row: Dict) -> str:
        """
        Extract the customer query from a test case row.
        
        Args:
            row: Dictionary containing test case data
            
        Returns:
            Extracted and cleaned query text
        """
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
            
            return self._clean_query_text(row['prompt'])
            
        except Exception as e:
            logger.error(f"Error extracting query: {e}")
            return "Could not extract query"
    
    def _clean_query_text(self, text: str) -> str:
        """
        Clean and normalize query text for consistent processing.
        
        Args:
            text: Raw query text
            
        Returns:
            Cleaned and normalized text
        """
        if not text:
            return "No query provided"
        
        text = str(text).strip()
        text = text.replace('\n', ' ').replace('\r', ' ')
        text = ' '.join(text.split())
        
        if (text.startswith('"') and text.endswith('"')) or \
           (text.startswith("'") and text.endswith("'")):
            text = text[1:-1]
        
        return text
    
    @groq_rate_limit(max_retries=3, base_delay=1.0)
    async def run_tests(self, test_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Run all test cases against all models for specified number of rounds.
        
        Args:
            test_df: DataFrame containing test cases
            
        Returns:
            Dictionary mapping model names to their results DataFrame
        """
        raw_results = {}
        
        for model_name, model in self.models.items():
            model_results = []
            logger.info(f"Starting test runs for model: {model_name}")
            
            for round_num in range(1, self.config.TEST_ROUNDS + 1):
                try:
                    logger.info(f"Starting round {round_num} for model {model_name}")
                    round_results = await self._run_single_round(model_name, model, test_df, round_num)
                    model_results.append(round_results)
                except Exception as e:
                    logger.error(f"Failed to run round {round_num} for model {model_name}: {e}")
                    continue
            
            if model_results:
                raw_results[model_name] = pd.concat(model_results, ignore_index=True)
                logger.info(f"Completed all rounds for model {model_name}")
            else:
                logger.error(f"No valid results for model {model_name}")
                raw_results[model_name] = pd.DataFrame()
        
        self._save_failed_attempts()
        return raw_results
    
    @groq_rate_limit(max_retries=3, base_delay=0.01)
    async def _run_single_round(
        self,
        model_name: str,
        model: Dict,
        test_df: pd.DataFrame,
        round_num: int
    ) -> pd.DataFrame:
        """Execute a single round of tests for a specific model."""
        responses = []
        request_count = 0
        
        for _, row in test_df.iterrows():
            try:
                customer_query = self._extract_customer_query(row)
                response = await self._execute_model_query(model, customer_query)
                
                response_dict = self._create_response_dict(
                    row,
                    response.choices[0].message.content,
                    round_num,
                    customer_query
                )
                responses.append(response_dict)
                request_count += 1
                
            except Exception as e:
                self._handle_test_failure(row, request_count, e, round_num)
                responses.append(self._create_error_response(row, str(e), round_num))
        
        return pd.DataFrame(responses)
    
    @groq_rate_limit(max_retries=3, base_delay=1.0)
    async def _execute_model_query(self, model: Dict, query: str):
        """Execute a query using Groq's tool use capabilities."""
        try:
            from groq import Groq
            client = Groq()
            
            response = client.chat.completions.create(
                model=model['model'],
                messages=[
                    {
                        "role": "system",
                        "content": model['prompt']
                    },
                    {
                        "role": "user",
                        "content": query
                    }
                ],
                tools=model.get('tools', []),
                tool_choice="auto",
                temperature=model['temperature']
            )

            return response.content if hasattr(response, 'content') else response
            
        except Exception as e:
            logger.error(f"Model query execution failed: {e}")
            raise
        
    def _create_response_dict(
        self,
        row: Dict,
        response: str,
        round_num: int,
        customer_query: Optional[str] = None
    ) -> Dict:
        """Create a standardized response dictionary."""
        try:
            response_dict = {
                'id': str(row['id']),
                'prompt': str(row['prompt']),
                'model_response': str(response),
                'test_case': str(row['test_case']),
                'round': int(round_num),
                'timestamp': datetime.now().isoformat()
            }
            
            if customer_query:
                response_dict['actual_query'] = customer_query
                
            return response_dict
            
        except Exception as e:
            logger.error(f"Error creating response dictionary: {e}")
            return self._create_error_response(row, str(e), round_num)
    
    def _create_error_response(self, row: Dict, error: str, round_num: int) -> Dict:
        """Create a standardized error response dictionary."""
        return {
            'id': str(row['id']),
            'prompt': str(row['prompt']),
            'model_response': f"ERROR: {error}",
            'test_case': str(row['test_case']),
            'round': int(round_num),
            'timestamp': datetime.now().isoformat()
        }
    
    def _handle_test_failure(
        self,
        row: Dict,
        attempt: int,
        error: Exception,
        round_num: int
    ) -> None:
        """Record and log test failures."""
        attempt_record = TestAttempt(
            id=row['id'],
            prompt=row['prompt'],
            timestamp=datetime.now().isoformat(),
            attempt_number=attempt + 1,
            error=str(error)
        )
        self.failed_attempts.append(attempt_record)
        logger.error(f"Test failed - ID: {row['id']}, Attempt: {attempt + 1}, Error: {error}")
    
    def _save_failed_attempts(self) -> None:
        """Save failed test attempts to a CSV file."""
        if self.failed_attempts:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            df = pd.DataFrame([vars(attempt) for attempt in self.failed_attempts])
            filename = f"failed_attempts_{timestamp}.csv"
            df.to_csv(filename, index=False)
            logger.info(f"Saved {len(self.failed_attempts)} failed attempts to {filename}")