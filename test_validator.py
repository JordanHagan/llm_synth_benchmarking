import os
import json
import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import pandas as pd

from test_generator import TestGenerator
from test_config import TestConfig, TestCaseType
from agent_config import MODEL_CONFIG, VALIDATOR_TOOL
from utils import create_agent, groq_rate_limit

logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    """Container for comprehensive validation results including quality scores and metadata."""
    id: str
    prompt_quality_score: int
    response_quality_score: int
    validation_message: Optional[str] = None
    prompt: Optional[str] = None
    response: Optional[str] = None
    timestamp: Optional[str] = None
    test_type: Optional[str] = None
    validation_details: Optional[Dict] = None

@dataclass
class ValidationData:
    """Container for processed validation results and statistics."""
    rows: List[Dict]
    issues: List[Dict]
    regeneration_counts: Dict[str, int]
    all_passed: bool
    requires_regeneration: bool
    low_scoring_pairs: List[Dict]
    validation_stats: Dict[str, float]

class TestValidator:
    """
    Enhanced validator for test cases with comprehensive result tracking and analysis.
    Handles both JSON and conversation test cases with detailed validation scoring.
    """
    
    def __init__(self, config, test_config: TestConfig):
        self.config = config
        self.test_config = test_config
        self.max_regeneration_depth = 3
        self.validator = None
        self.results_dir = "validation_results"
        self._setup_storage()
        
    def _setup_storage(self):
        """Initialize storage directories for validation results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.validation_dir = os.path.join(self.results_dir, f"validation_run_{timestamp}")
        self.low_scoring_dir = os.path.join(self.validation_dir, "low_scoring")
        self.stats_dir = os.path.join(self.validation_dir, "statistics")
        
        for directory in [self.validation_dir, self.low_scoring_dir, self.stats_dir]:
            os.makedirs(directory, exist_ok=True)

    async def initialize(self):
        """Initialize the validator agent with proper error handling."""
        try:
            config = MODEL_CONFIG['validator']
            self.validator = await create_agent(
                system_prompt=config['prompt'],
                model_name=config['model_name'],
                temperature=config['temperature'],
                agent_type='validator',
                tools=[VALIDATOR_TOOL]
            )
            logger.info("Validator agent initialized successfully")
            return self
            
        except Exception as e:
            logger.error(f"Validator initialization failed: {e}")
            raise RuntimeError(f"Failed to initialize validator: {str(e)}")

    def _passes_validation(self, result: ValidationResult) -> bool:
            """
            Determine if a validation result meets quality thresholds.
            
            Args:
                result: The ValidationResult to evaluate
                
            Returns:
                True if both prompt and response scores meet quality thresholds
            """
            if not result or not isinstance(result, ValidationResult):
                return False
                
            threshold = 4  # Minimum score required to pass validation
            return (result.prompt_quality_score >= threshold and 
                    result.response_quality_score >= threshold)
                    
    async def _validate_single_case(self, test_case: Dict) -> ValidationResult:
        timestamp = datetime.now().isoformat()
        
        try:
            # Restructure the validation input to ensure proper tool use
            validation_input = {
                "messages": [
                    {
                        "role": "user",
                        "content": json.dumps({
                            "id": test_case['id'],
                            "test_case": test_case['prompt'],
                            "golden_response": test_case['golden_response']
                        })
                    }
                ],
                "tools": [VALIDATOR_TOOL],
                "tool_choice": {"type": "function", "function": {"name": "validate_test_case"}}
            }
            
            logger.info(f"Validating test case {test_case['id']}")
            
            # Get validation response
            response = await self.validator.ainvoke(validation_input)
            
            # Add debug logging to see the raw response
            logger.debug(f"Raw validation response: {response}")
            
            # Handle the tool response appropriately
            if not response:
                logger.warning(f"Empty validation response for test case {test_case['id']}")
                return ValidationResult(
                    id=test_case['id'],
                    prompt_quality_score=0,
                    response_quality_score=0,
                    validation_message="Empty validation response",
                    timestamp=timestamp,
                    test_type=test_case['test_case']
                )
            
            validation_data = self._parse_validation_response(response, test_case)
            
            return ValidationResult(
                id=validation_data['id'],
                prompt_quality_score=validation_data['prompt_quality_score'],
                response_quality_score=validation_data['response_quality_score'],
                validation_message=validation_data.get('message'),
                prompt=test_case['prompt'],
                response=test_case['golden_response'],
                timestamp=timestamp,
                test_type=test_case['test_case']
            )
            
        except Exception as e:
            logger.error(
                f"Validation failed for test case {test_case['id']}: {str(e)}", 
                exc_info=True
            )
            
            return ValidationResult(
                id=test_case['id'],
                prompt_quality_score=0,
                response_quality_score=0,
                validation_message=f"Validation error: {str(e)}",
                prompt=test_case.get('prompt'),
                response=test_case.get('golden_response'),
                timestamp=timestamp,
                test_type=test_case.get('test_case')
            )

    def _store_validation_result(self, result: ValidationResult):
        """Store validation results with proper organization."""
        result_data = {
            'id': result.id,
            'timestamp': result.timestamp,
            'test_type': result.test_type,
            'prompt': result.prompt,
            'response': result.response,
            'prompt_score': result.prompt_quality_score,
            'response_score': result.response_quality_score,
            'message': result.validation_message,
            'details': result.validation_details
        }
        
        # Store full result
        filename = f"{self.validation_dir}/{result.id}_{result.timestamp}.json"
        with open(filename, 'w') as f:
            json.dump(result_data, f, indent=2)
            
        # Store low scoring pairs separately
        if not self._passes_validation(result):
            low_score_file = f"{self.low_scoring_dir}/low_score_{result.id}.json"
            with open(low_score_file, 'w') as f:
                json.dump(result_data, f, indent=2)

    async def validate_tests(
        self,
        df: pd.DataFrame,
        depth: int = 0
    ) -> Tuple[bool, List[Dict], pd.DataFrame]:
        """Validate test cases with comprehensive result tracking."""
        if self.validator is None:
            await self.initialize()
            
        if not self.test_config.enable_validation:
            return True, [], df

        if depth >= self.max_regeneration_depth:
            logger.warning("Max regeneration depth reached")
            return False, [], df

        test_cases = df.to_dict('records')
        validation_results = []
        
        batch_size = 5
        for i in range(0, len(test_cases), batch_size):
            batch = test_cases[i:i + batch_size]
            batch_results = await asyncio.gather(
                *[self._validate_single_case(case) for case in batch]
            )
            validation_results.extend(batch_results)
            
        validated_data = self._process_validation_results(validation_results, df)
        self._save_validation_statistics(validated_data)
        
        if validated_data.requires_regeneration and depth < self.max_regeneration_depth:
            additional_cases = await self._handle_regeneration(
                validated_data.regeneration_counts,
                depth
            )
            if not additional_cases.empty:
                validated_data.rows.extend(additional_cases.to_dict('records'))

        validated_df = pd.DataFrame(validated_data.rows)
        return validated_data.all_passed, validated_data.issues, validated_df
    
    def _process_validation_results(
        self,
        results: List[ValidationResult],
        df: pd.DataFrame
    ) -> ValidationData:
        validated_rows = []
        regeneration_needed = {test_type.value: 0 for test_type in TestCaseType}
        validation_issues = []
        low_scoring_pairs = []
        
        for result in results:
            try:
                row = df[df['id'] == result.id].iloc[0].to_dict()
                
                # Add validation scores to the row data
                enhanced_row = {
                    **row,
                    'prompt_quality_score': result.prompt_quality_score,
                    'response_quality_score': result.response_quality_score,
                    'validation_message': result.validation_message,
                    'validation_timestamp': result.timestamp
                }
                
                if self._passes_validation(result):
                    validated_rows.append(enhanced_row)
                else:
                    test_type = row['test_case']
                    regeneration_needed[test_type] = regeneration_needed.get(test_type, 0) + 1
                    validation_issues.append(self._create_validation_issue(result, row))
                    low_scoring_pairs.append(self._create_low_scoring_pair(result, row))
                    
            except Exception as e:
                logger.error(f"Error processing validation result: {e}")

        return ValidationData(
            rows=validated_rows,
            issues=validation_issues,
            regeneration_counts=regeneration_needed,
            all_passed=len(validation_issues) == 0,
            requires_regeneration=bool(sum(regeneration_needed.values())),
            low_scoring_pairs=low_scoring_pairs,
            validation_stats={}  # Empty dict instead of calculated stats
        )

    def _create_validation_issue(self, result: ValidationResult, row: Dict) -> Dict:
        """
        Create a structured record of validation issues for review.
        
        Args:
            result: The validation result containing scores and messages
            row: The original test case data
            
        Returns:
            Dictionary containing validation issue details
        """
        return {
            'id': result.id,
            'test_type': row['test_case'],
            'prompt': row['prompt'],
            'response': row['golden_response'],
            'prompt_score': result.prompt_quality_score,
            'response_score': result.response_quality_score,
            'message': result.validation_message,
            'timestamp': datetime.now().isoformat()
        }

    async def _handle_regeneration(
        self,
        regeneration_counts: Dict[str, int],
        current_depth: int
    ) -> pd.DataFrame:
        """
        Handle regeneration of failed test cases.
        
        Args:
            regeneration_counts: Dictionary tracking how many cases need regeneration
            current_depth: Current recursion depth for regeneration attempts
            
        Returns:
            DataFrame containing newly generated and validated test cases
        """
        if not any(regeneration_counts.values()):
            return pd.DataFrame()

        try:
            test_generator = TestGenerator(self.config, self.test_config)
            
            # Generate new test cases
            additional_cases = await test_generator.generate_test_cases(
                json_count=regeneration_counts.get('json', 0),
                conv_count=regeneration_counts.get('conversation', 0)
            )
            
            # Validate new cases
            _, _, validated_cases = await self.validate_tests(
                additional_cases,
                depth=current_depth + 1
            )
            
            return validated_cases
            
        except Exception as e:
            logger.error(f"Failed to generate additional cases: {str(e)}")
            return pd.DataFrame()

    def _create_low_scoring_pair(self, result: ValidationResult, row: Dict) -> Dict:
        """
        Create a record of low-scoring prompt/response pairs for analysis.
        
        Args:
            result: The validation result containing scores
            row: The original test case data
            
        Returns:
            Dictionary containing details of the low-scoring pair
        """
        return {
            'id': result.id,
            'test_type': row['test_case'],
            'prompt': row['prompt'],
            'response': row['golden_response'],
            'prompt_score': result.prompt_quality_score,
            'response_score': result.response_quality_score,
            'validation_message': result.validation_message,
            'timestamp': datetime.now().isoformat()
        }

    def _save_validation_statistics(self, validated_data: ValidationData):
        """Save validation statistics for analysis."""
        stats_file = f"{self.stats_dir}/validation_stats.json"
        with open(stats_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'statistics': validated_data.validation_stats,
                'regeneration_counts': validated_data.regeneration_counts,
                'total_issues': len(validated_data.issues)
            }, f, indent=2)
            
    def _parse_validation_response(self, response: str, test_case: Dict) -> Dict:
        """
        Parse and validate the response from the validator agent.
        
        Args:
            response: Raw response string from the validator
            test_case: Original test case for reference
            
        Returns:
            Dictionary containing parsed validation data
        """
        try:
            if isinstance(response, str):
                parsed_response = json.loads(response)
                if isinstance(parsed_response, dict):
                    return {
                        'id': parsed_response.get('id', test_case['id']),
                        'prompt_quality_score': parsed_response.get('prompt_quality_score', 0),
                        'response_quality_score': parsed_response.get('response_quality_score', 0),
                        'message': parsed_response.get('message', 'No validation message provided'),
                        'details': parsed_response.get('details', {})
                    }
                    
            logger.warning(f"Unexpected validation response structure: {response}")
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse validation response: {e}")
            
        return {
            'id': test_case['id'],
            'prompt_quality_score': 0,
            'response_quality_score': 0,
            'message': f"Failed to parse validation response",
            'details': {}
        }