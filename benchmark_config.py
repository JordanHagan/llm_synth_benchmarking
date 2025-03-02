import os
from agent_config import MODEL_CONFIG
from test_config import TestConfig

class BenchmarkConfig:
    """Configuration for the benchmark system."""
    
    def __init__(self):
        self.test_config = TestConfig()
        self.GROQ_API_KEY = os.getenv("GROQ_API_KEY")
        if not self.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY not set")
        
        # Model configurations
        self.GEN_MODEL = MODEL_CONFIG['json_generator']['model_name']
        self.VAL_MODEL = MODEL_CONFIG['validator']['model_name']
        
        # Use direct references to the models rather than string paths
        self.TEST_MODELS = {
            'model_A': 'model_A',
            'model_B': 'model_B'
        }
        
        # Test execution parameters
        self.TEST_ROUNDS = 3
        self.MAX_RETRIES = 3
        self.TIMEOUT = 30.0
        
        # Rate limiting parameters
        self.RETRY_DELAY = 2  # Base delay in seconds
        self.MAX_WAIT_TIME = 60  # Maximum wait time in seconds (1 minute)