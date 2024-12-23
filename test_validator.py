import logging
import uuid
import pandas as pd
import json
from test_generator import TestGenerator
from agent_config import MODEL_CONFIG
from utils import create_agent

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('debug.log'),
        logging.StreamHandler()  # This will still print to console
    ]
)
logger = logging.getLogger(__name__)

class TestValidator:
    def __init__(self, config, test_config):
        self.config = config
        self.test_config = test_config
        self.validator = create_agent(
            system_prompt=MODEL_CONFIG['validator']['prompt'],
            model_name=MODEL_CONFIG['validator']['model_name'],
            temperature=MODEL_CONFIG['validator']['temperature'],
            agent_type='validator',
        )
    
    def _parse_validation_results(self, validation_results: str):
        """Parse validation results with improved error handling and JSON cleanup."""
        try:
            # Extract just the JSON array
            start_idx = validation_results.find('[')
            end_idx = validation_results.rfind(']') + 1
            
            if start_idx == -1 or end_idx == 0:
                logger.error("Could not find JSON array markers")
                raise ValueError("No JSON array found in validation results")
                
            json_str = validation_results[start_idx:end_idx]
            
            # Clean up common JSON formatting issues
            import re
            
            # Remove any whitespace between lines
            json_str = re.sub(r'\s+', ' ', json_str)
            
            # Remove any trailing commas before closing brackets
            json_str = re.sub(r',(\s*[\]}])', r'\1', json_str)
            
            # Ensure arrays are properly terminated
            json_str = re.sub(r'}\s*]$', '}]', json_str)
            
            logger.debug(f"Cleaned JSON string starts with: {json_str[:100]}")
            
            try:
                # Parse the JSON
                parsed_results = json.loads(json_str)
                logger.info("Successfully parsed validation results JSON")
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error: {str(e)}")
                logger.error(f"Position {e.pos}, line {e.lineno}, col {e.colno}")
                # Print the problematic section
                start = max(0, e.pos - 50)
                end = min(len(json_str), e.pos + 50)
                logger.error(f"Context: ...{json_str[start:end]}...")
                raise
                
            # Convert array format to dictionaries
            formatted_results = []
            for result_array in parsed_results:
                result_dict = {}
                for key_value_pair in result_array:
                    if len(key_value_pair) == 2:
                        key, value = key_value_pair
                        result_dict[key] = value
                formatted_results.append(result_dict)
                
            return formatted_results
                
        except Exception as e:
            logger.error(f"Failed to parse validation results: {e}")
            logger.error(f"Full raw validation results: {validation_results}")
            raise ValueError("Failed to parse validation results") from e

    async def validate_tests(self, df):
        test_cases = df.to_dict('records')
        logger.info(f"Validating {len(test_cases)} test cases")
        
        validation_results = await self.validator.ainvoke({
            'test_cases': test_cases
        })
        
        # Log raw response for debugging
        logger.info("Received validation results")
        logger.debug(f"Raw validation results: {validation_results}")
        
        # Parse the validation results using our new method
        try:
            parsed_results = self._parse_validation_results(validation_results)
            logger.info("Successfully parsed validation results")
        except ValueError as e:
            logger.error(f"Validation results parsing failed: {e}")
            raise
            
        # Process validation results
        validated_rows = []
        invalid_rows = []
        validation_issues = []
        
        for result in parsed_results:
            try:
                # Convert array format to dict if necessary
                if isinstance(result, list):
                    result_dict = {}
                    for item in result:
                        if len(item) == 2:  # Expecting ["key", "value"] format
                            key, value = item
                            result_dict[key] = value
                    result = result_dict
                
                test_id = result.get('id')
                prompt_score = result.get('prompt_quality_score', 0)
                response_score = result.get('response_quality_score', 0)
                
                if prompt_score > 3 and response_score > 3:
                    try:
                        row = df[df['id'] == test_id].iloc[0]
                        validated_rows.append(row)
                        logger.debug(f"Test case passed validation - ID: {test_id}")
                    except IndexError:
                        logger.error(f"Could not find original test case with ID: {test_id}")
                else:
                    invalid_rows.append(test_id)
                    validation_issues.append({
                        'id': test_id,
                        'prompt_score': prompt_score,
                        'response_score': response_score,
                        'reason': 'Scores below threshold'
                    })
                    logger.info(f"Test case failed validation - ID: {test_id}, "
                            f"Prompt Score: {prompt_score}, Response Score: {response_score}")
            except Exception as e:
                logger.error(f"Error processing validation result: {e}")
                logger.error(f"Problematic result: {result}")
                continue
        
        validated_df = pd.DataFrame(validated_rows)
        validation_passed = len(validation_issues) == 0
        
        logger.info(f"Validation complete: {len(validated_rows)} passed, {len(validation_issues)} failed")
        
        return validation_passed, validation_issues, validated_df

    async def _generate_additional_cases(self, json_count, conv_count):
        test_generator = TestGenerator(self.config, self.test_config)
        additional_cases = await test_generator.generate_test_cases(json_count, conv_count)
        return additional_cases
    
    