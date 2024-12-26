import re
import json
import logging
import pandas as pd
from test_generator import TestGenerator
from test_config import TestConfig
from agent_config import MODEL_CONFIG
from utils import create_agent

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
            # Extract just the array content
            start_idx = validation_results.find('[')
            end_idx = validation_results.rfind(']') + 1
            
            if start_idx == -1 or end_idx == 0:
                logger.error("Could not find JSON array markers")
                raise ValueError("No JSON array found in validation results")
                
            json_str = validation_results[start_idx:end_idx]
            
            # Clean up the JSON string
            # Remove any line breaks and normalize whitespace
            json_str = ' '.join(json_str.split())
            
            # Fix common JSON formatting issues
            json_str = json_str.replace('}, ]', '}]')
            json_str = json_str.replace('},]', '}]')
            json_str = json_str.replace(',,', ',')
            
            # Remove any trailing commas before closing brackets
            json_str = re.sub(r',(\s*[\]}])', r'\1', json_str)
            
            # Try to parse the cleaned JSON
            try:
                parsed_results = json.loads(json_str)
            except json.JSONDecodeError as e:
                # Log the problematic section
                error_pos = e.pos
                context_start = max(0, error_pos - 50)
                context_end = min(len(json_str), error_pos + 50)
                logger.error(f"JSON parse error near position {error_pos}:")
                logger.error(f"Context: ...{json_str[context_start:context_end]}...")
                raise
                
            # Convert to expected format
            formatted_results = []
            for result in parsed_results:
                if isinstance(result, list):
                    # Handle array format
                    result_dict = {}
                    for item in result:
                        if isinstance(item, list) and len(item) >= 2:
                            key, value = item[0], item[1]
                            result_dict[key] = value
                    formatted_results.append(result_dict)
                else:
                    # Already in dictionary format
                    formatted_results.append(result)
                    
            return formatted_results
                        
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {str(e)}")
            logger.error(f"Problematic JSON string: {validation_results}")
            # Handle the error gracefully, e.g., return an empty list or default values
            return []
        
        except Exception as e:
            logger.error(f"Unexpected error during validation results parsing: {str(e)}")
            logger.error(f"Validation results: {validation_results}")
            raise ValueError("Failed to parse validation results") from e
                
    def _debug_json_structure(self, json_str: str):
        """Debug helper to analyze JSON structure."""
        try:
            open_square = json_str.count('[')
            close_square = json_str.count(']')
            open_curly = json_str.count('{')
            close_curly = json_str.count('}')
            
            if open_square != close_square or open_curly != close_curly:
                logger.error(f"Unbalanced brackets/braces: [{open_square}:{close_square}] {{:{open_curly}:{close_curly}}}")
                
        except Exception as e:
            logger.error(f"Error in debug analysis: {e}")

    async def validate_tests(self, df):
        """Validate test cases and regenerate low-scoring cases."""
        if not self.test_config.enable_validation:
            logger.info("Validation disabled, returning original dataset")
            return True, [], df
            
        test_cases = df.to_dict('records')
        if not self.test_config.enable_json_tests:
            test_cases = [case for case in test_cases if case['test_case'] != 'json']
            
        logger.info(f"Validating {len(test_cases)} test cases")
        validation_results = await self.validator.ainvoke({'test_cases': test_cases})
        self._debug_json_structure(validation_results)
        
        try:
            parsed_results = self._parse_validation_results(validation_results)
        except ValueError as e:
            logger.error(f"Validation results parsing failed: {e}")
            raise
        
        validated_rows = []
        regeneration_needed = {'json': 0, 'conversation': 0}
        validation_issues = []
        
        for result in parsed_results:
            try:
                test_id = result.get('id')
                prompt_score = result.get('prompt_quality_score', 0)
                response_score = result.get('response_quality_score', 0)
                
                row = df[df['id'] == test_id].iloc[0]
                test_type = row['test_case']
                
                if prompt_score >= 4 and response_score >= 4:
                    print(validated_rows)
                    validated_rows.append(row)
                    logger.debug(f"Test case passed validation - ID: {test_id}")
                else:
                    regeneration_needed[test_type] = regeneration_needed.get(test_type, 0) + 1
                    print(validation_issues)
                    validation_issues.append({
                        'id': test_id,
                        'test_type': test_type,
                        'prompt_score': prompt_score,
                        'response_score': response_score,
                    })
            except Exception as e:
                logger.error(f"Error processing result {test_id}: {e}")
                continue
        
        # Regenerate cases that didn't meet quality threshold
        if validation_issues:
            logger.info(f"Regenerating {sum(regeneration_needed.values())} low-scoring test cases")
            additional_cases = await self._generate_additional_cases(
                json_count=regeneration_needed.get('json', 0),
                conv_count=regeneration_needed.get('conversation', 0)
            )
            if not additional_cases.empty:
                validated_rows.extend(additional_cases.to_dict('records'))
        
        validated_df = pd.DataFrame(validated_rows)
        validation_passed = len(validation_issues) == 0
        
        logger.info(f"Validation complete: {len(validated_rows)} passed, {len(validation_issues)} regenerated")
        
        print(validated_df.head())
        
        return validation_passed, validation_issues, validated_df

    async def _generate_additional_cases(self, json_count, conv_count):
        """Generate additional test cases to replace low-scoring ones."""
        if json_count == 0 and conv_count == 0:
            return pd.DataFrame()
            
        test_generator = TestGenerator(self.config, self.test_config)
        additional_cases = await test_generator.generate_test_cases(json_count, conv_count)
        
        # Validate new cases immediately
        _, _, validated_cases = await self.validate_tests(additional_cases)
        return validated_cases