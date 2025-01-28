from typing import Dict
from enum import Enum

class TestCaseType(str, Enum):
    """Enumeration of valid test case types."""
    JSON = "json" 
    CONVERSATION = "conversation"

class TestConfig:
    """
    Configuration for test generation, validation, and metrics.
    Manages feature flags, schemas, categories, and specs.
    """
    
    def __init__(self):
        """Initialize with default settings."""
        # Feature flags 
        self.enable_json_tests = False
        self.enable_validation = True
        self.sample_size = 5
        
        self._initialize_schemas()
        self._initialize_test_categories() 
        self._initialize_metrics()
        
    def _initialize_schemas(self):
        """Define JSON schemas for test cases and responses."""
        self.test_case_schema = {
            "type": "object",
            "required": ["id", "prompt", "golden_response", "test_case"],
            "properties": {
                "id": {"type": "string", "format": "uuid"},
                "prompt": {"type": "string", "minLength": 1}, 
                "golden_response": {"type": "object"},
                "test_case": {
                    "type": "string",
                    "enum": [e.value for e in TestCaseType]
                }
            },
            "additionalProperties": False
        }
        
        self.response_schema = {
            "type": "object", 
            "properties": {
                "customer_interaction": {
                    "type": "object",
                    "required": [
                        "id", 
                        "timestamp",
                        "customer",
                        "interaction",
                        "metrics"
                    ],
                    "properties": {
                        "interaction_id": {"type": "string"},
                        "timestamp": {
                            "type": "string", 
                            "format": "date-time"
                        },
                        "customer": {
                            "type": "object",
                            "required": [
                                "id",
                                "segment",
                                "priority_level"  
                            ],
                            "properties": {
                                "id": {"type": "string"},
                                "segment": {"type": "string"},
                                "priority_level": {
                                    "type": "integer",
                                    "minimum": 0
                                }
                            }
                        },
                        "interaction": {
                            "type": "object",
                            "required": [
                                "type",
                                "summary", 
                                "category",
                                "resolution_status",
                                "next_steps"
                            ],
                            "properties": {
                                "type": {"type": "string"},
                                "summary": {"type": "string"},
                                "category": {"type": "string"},
                                "resolution_status": {"type": "string"},
                                "next_steps": {
                                    "type": "array",
                                    "items": {"type": "string"}
                                }
                            }
                        },
                        "metrics": {
                            "type": "object",
                            "required": [
                                "response_time",
                                "satisfaction_score",
                                "resolution_time" 
                            ],
                            "properties": {
                                "response_time": {
                                    "type": "integer",
                                    "minimum": 0
                                },
                                "satisfaction_score": {
                                    "type": "integer", 
                                    "minimum": 0,
                                    "maximum": 10
                                },
                                "resolution_time": {
                                    "type": "integer",
                                    "minimum": 0
                                }
                            }
                        }
                    }
                }
            }
        }
    
    def _initialize_test_categories(self):
        """Define test categories and configurations."""
        self.test_categories = {
            'customer_support': {
                'scenarios': [
                    'technical_support',
                    'billing_inquiry', 
                    'product_question',
                    'account_management',
                    'service_complaint'  
                ],
                'sample_size': self.sample_size
            }
        }
        
        if self.enable_json_tests:
            self.test_categories['structured_json_output'] = {
                'schema': self.response_schema,
                'sample_size': self.sample_size
            }
    
    def _initialize_metrics(self):
        """Define metric configurations."""
        self.metrics = {
            'conversation_metrics': [
                'response_relevance',
                'clarity',
                'task_completion',
                'bleu_score',
                'wer_score'
            ]
        }
        
        if self.enable_json_tests:
            self.metrics['json_tests'] = [
                'schema_compliance_rate',
                'field_accuracy',
                'structural_consistency' 
            ]
    
    def update_feature_flags(
        self, 
        enable_json: bool = None,
        enable_validation: bool = None
    ):
        """Update feature flags and reinitialize configs."""
        if enable_json is not None:
            self.enable_json_tests = enable_json
        if enable_validation is not None: 
            self.enable_validation = enable_validation
            
        self._initialize_test_categories()
        self._initialize_metrics()
    
    def get_schema_for_type(self, test_type: TestCaseType) -> Dict:
        """Get schema for given test type."""
        if test_type == TestCaseType.JSON:
            return self.response_schema
        return self.test_case_schema