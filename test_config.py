class TestConfig:
    def __init__(self):
        self.test_case_schema = {
            "type": "object",
            "required": ["id", "prompt", "golden_response", "test_case"],
            "properties": {
                "id": {"type": "string", "format": "uuid"},
                "prompt": {"type": "string", "minLength": 1},
                "golden_response": {"type": "object"},
                "test_case": {"type": "string", "enum": ["json", "conversation"]}
            },
            "additionalProperties": False
        }

        self.response_schema = {
            "type": "object",
            "properties": {
                "customer_interaction": {
                    "type": "object",
                    "required": ["id", "timestamp", "customer", "interaction", "metrics"],
                    "properties": {
                        "interaction_id": {"type": "string"},
                        "timestamp": {"type": "string", "format": "date-time"},
                        "customer": {
                            "type": "object",
                            "required": ["id", "segment", "priority_level"],
                            "properties": {
                                "id": {"type": "string"},
                                "segment": {"type": "string"},
                                "priority_level": {"type": "integer", "minimum": 0}
                            }
                        },
                        "interaction": {
                            "type": "object",
                            "required": ["type", "summary", "category", "resolution_status", "next_steps"],
                            "properties": {
                                "type": {"type": "string"},
                                "summary": {"type": "string"},
                                "category": {"type": "string"},
                                "resolution_status": {"type": "string"},
                                "next_steps": {"type": "array", "items": {"type": "string"}}
                            }
                        },
                        "metrics": {
                            "type": "object",
                            "required": ["response_time", "satisfaction_score", "resolution_time"],
                            "properties": {
                                "response_time": {"type": "integer", "minimum": 0},
                                "satisfaction_score": {"type": "integer", "minimum": 0, "maximum": 10},
                                "resolution_time": {"type": "integer", "minimum": 0}
                            }
                        }
                    }
                }
            }
        }

        self.test_categories = {
            'structured_json_output': {
                'schema': self.response_schema,
                'sample_size': 5
            },
            'customer_support': {
                'scenarios': ['technical_support', 'billing_inquiry', 'product_question', 
                            'account_management', 'service_complaint'],
                'sample_size': 5
            }
        }

        self.metrics = {
            'json_tests': ['schema_compliance_rate', 'field_accuracy', 'structural_consistency'],
            'conversation_metrics': ['response_relevance', 'clarity', 'task_completion', 
                                   'bleu_score', 'wer_score']
        }