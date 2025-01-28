import uuid
import json
import logging
import asyncio 
from typing import Dict, List, Optional
import pandas as pd
from groq import Groq

from agent_config import MODEL_CONFIG, TestCase 
from benchmark_config import BenchmarkConfig
from test_config import TestConfig, TestCaseType
from utils import groq_rate_limit

logger = logging.getLogger(__name__)

class TestGenerator:
    """
    Generates test cases for model evaluation using Groq's API.
    Handles JSON and conversational test cases with error handling and rate limiting.
    """
    
    def __init__(self, config: BenchmarkConfig, test_config: TestConfig):
        self.config = config
        self.test_config = test_config
        self.client = Groq()
        self._initialize_generators()
    
    def _initialize_generators(self):
        try:
            self.json_config = MODEL_CONFIG['json_generator']
            self.conv_config = MODEL_CONFIG['conversation_generator'] 
            logger.info("Generator configurations initialized")
        except KeyError as e:
            logger.error(f"Missing required configuration: {e}")
            raise
        except Exception as e:
            logger.error(f"Generator initialization failed: {e}") 
            raise
    
    @groq_rate_limit(max_retries=3, base_delay=1.0)
    async def _generate_single_case(
        self,
        model_config: Dict,
        prompt: str,
        case_type: TestCaseType
    ) -> Dict:
        try:
            messages = [
                {"role": "system", "content": model_config['prompt']},
                {"role": "user", "content": prompt}
            ]
            
            tools = model_config.get('tools', [])
            tool_name = tools[0]['function']['name'] if tools else None
            
            response = self.client.chat.completions.create(
                model=model_config['model_name'],
                messages=messages,
                tools=tools,
                tool_choice={
                    "type": "function",
                    "function": {"name": tool_name}
                } if tool_name else "auto",
                temperature=model_config.get('temperature', 0.0)
            )
            
            if not response.choices[0].message.tool_calls:
                raise ValueError(f"No tool calls for {case_type.value}")
                
            tool_outputs = []
            for tool_call in response.choices[0].message.tool_calls:
                tool_output = json.loads(tool_call.function.arguments)
                tool_outputs.append(tool_output)
                
            combined_output = self._process_tool_outputs(
                tool_outputs[0] if tool_outputs else {},
                case_type
            )
            
            if not self._validate_generated_case(combined_output, case_type):
                raise ValueError(f"Validation failed for {case_type.value}")
                
            return combined_output
            
        except Exception as e:
            logger.error(f"Error generating {case_type.value} case: {e}")
            raise
            
    def _process_tool_outputs(
        self,
        raw_output: Dict,
        case_type: TestCaseType
    ) -> Dict:
        """Process raw tool outputs into final test case format."""
        processed_output = {
            "id": raw_output["id"],
            "prompt": raw_output["prompt"], 
            "test_case": case_type.value
        }
        
        if case_type == TestCaseType.CONVERSATION:
            processed_output["golden_response"] = raw_output["golden_response"]
            
        elif case_type == TestCaseType.JSON:
            processed_output["golden_response"] = {
                "customer_interaction": {
                    "interaction_id": str(uuid.uuid4()),
                    "timestamp": raw_output["golden_response"]["interaction_details"]["timestamp"],
                    "customer": {
                        "id": str(uuid.uuid4()),  
                        "segment": raw_output["golden_response"]["support_context"]["customer_tier"],
                        "priority_level": raw_output["golden_response"]["support_context"]["issue_priority"]
                    },
                    "interaction": {
                        "type": "customer_service",
                        "summary": raw_output["prompt"],
                        "category": raw_output["golden_response"]["interaction_details"]["category"], 
                        "resolution_status": raw_output["golden_response"]["resolution"]["status"],
                        "next_steps": raw_output["golden_response"]["resolution"]["next_steps"]
                    },
                    "metrics": {
                        "response_time": raw_output["golden_response"]["resolution"]["response_time_seconds"],
                        "satisfaction_score": raw_output["golden_response"]["interaction_details"].get("satisfaction_score", -1),
                        "resolution_time": raw_output["golden_response"]["resolution"]["response_time_seconds"]
                    }
                }
            }
            
        return processed_output
    
    def _validate_generated_case(self, case: Dict, case_type: TestCaseType) -> bool:
        required_fields = ['id', 'prompt', 'golden_response', 'test_case']
        
        try:
            if not all(field in case for field in required_fields):
                return False
            
            uuid.UUID(case['id']) # Check valid UUID
            
            if case['test_case'] not in [e.value for e in TestCaseType]:
                return False
                
            return True
            
        except ValueError:
            return False
    
    async def generate_test_cases(
        self,
        json_count: Optional[int] = None,
        conv_count: Optional[int] = None 
    ) -> pd.DataFrame:
        cases = []
        
        try:
            if self.test_config.enable_json_tests:
                json_cases = await self._generate_json_cases(json_count)
                if json_cases:
                    cases.append(pd.DataFrame(json_cases))
            
            conv_cases = await self._generate_conversation_cases(conv_count)
            if conv_cases:
                cases.append(pd.DataFrame(conv_cases))
            
            if not cases:
                raise ValueError("No test cases generated")
            
            all_cases = pd.concat(cases, ignore_index=True)
            logger.info(f"Generated {len(all_cases)} test cases")
            return all_cases
            
        except Exception as e:
            logger.error(f"Failed to generate test cases: {e}")
            raise
    
    async def _generate_json_cases(self, count: Optional[int] = None) -> List[Dict]:
        json_cases = []
        sample_size = count or self.test_config.test_categories['structured_json_output']['sample_size']
        
        for _ in range(sample_size): 
            case = await self._generate_single_case(
                self.json_config,
                f"Generate JSON test case with ID: {str(uuid.uuid4())}",
                TestCaseType.JSON
            )
            json_cases.append(case)
            
        return json_cases
    
    async def _generate_conversation_cases(self, count: Optional[int] = None) -> List[Dict]:
        conv_cases = []
        scenarios = self.test_config.test_categories['customer_support']['scenarios'] 
        conv_size = count or self.test_config.test_categories['customer_support']['sample_size']
        
        for i in range(conv_size):
            case = await self._generate_single_case(
                self.conv_config,
                f"Generate conversation test case for {scenarios[i % len(scenarios)]} with ID: {str(uuid.uuid4())}",   
                TestCaseType.CONVERSATION
            )
            conv_cases.append(case)
            
        return conv_cases