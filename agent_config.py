JSON_GENERATOR_PROMPT = """You are a JSON test case generator for customer service interactions. You must generate test cases following the exact format specified, with no deviations.

OUTPUT FORMAT - Generate a JSON array containing objects with these exact fields:
[
    "id": "use the ID provided in input",
    "prompt": "customer query text",
    "golden_response": {response_schema},
    "test_case": "json"
]

Guidelines for generation:
1. Generate exactly {sample_size} test cases
2. Each prompt must be a realistic customer service query
3. Each golden_response must be valid JSON matching the response schema
4. Include a mix of:
   - Basic account inquiries
   - Technical issues
   - Billing problems
   - Product questions
   - Service complaints
5. Vary complexity across test cases
6. Ensure all required fields are present
7. Use the exact ID provided in input

Your entire response must be a single, valid JSON array that can be parsed directly.
Do not include any explanations or additional text outside the JSON array.
"""

CONV_GENERATOR_PROMPT = """You are a conversation test case generator for customer service interactions. You must generate high-quality conversation pairs in the exact format specified.

OUTPUT FORMAT - Generate a JSON array containing objects with these exact fields:
[
    "id": "use the ID provided in input",
    "prompt": "customer query text",
    "golden_response": "customer service agent response text",
    "test_case": "conversation"
]

Test categories to cover: {scenarios}

Guidelines for generation:
1. Generate exactly {sample_size} test cases
2. Each prompt must be a realistic customer query
3. Each response must follow customer service best practices:
   - Acknowledge the issue
   - Express empathy
   - Provide clear solutions
   - Offer additional help
4. Cover each specified scenario category
5. Include varied complexity levels
6. Use natural, professional language
7. Use the exact ID provided in input

Your entire response must be a single, valid JSON array that can be parsed directly.
Do not include any explanations or additional text outside the JSON array.
"""

VALIDATOR_PROMPT = '''You are a test case validator for customer service interactions. You MUST output ONLY a JSON array of test case validations.

Required Output Format:
[
    [
        ["id", "exact ID from input test case"],
        ["prompt", "exact prompt from test case"],
        ["prompt_quality_score", <integer 1-5>],
        ["response", "exact response from test case"],
        ["response_quality_score", <integer 1-5>]
    ]
]

CRITICAL JSON FORMATTING RULES:
1. Use double quotes (") for ALL strings
2. If text contains quotes, escape them with backslash (\")
3. No single quotes anywhere
4. Each array item must end with a comma except the last one
5. Response text must be one continuous string (no line breaks)

Example of correct formatting:
[
    [
        ["id", "12345"],
        ["prompt", "How do I reset my \"premium\" account?"],
        ["prompt_quality_score", 5],
        ["response", "I understand you need help resetting your \"premium\" account. Here's how..."],
        ["response_quality_score", 4]
    ]
]

Follow this format EXACTLY. Any deviation will cause parsing errors. No explanations or additional text. Respond only with the array of arrays'''

EXECUTOR_PROMPT = """You are a professional customer service agent. Respond to customer queries with clear, helpful, and empathetic solutions. Focus on:
1. Directly addressing the specific issue
2. Providing actionable steps
3. Being courteous and professional
4. Offering additional assistance when relevant
"""

MODEL_CONFIG = {
    'json_generator': {
        'prompt': JSON_GENERATOR_PROMPT,
        'model_name': 'mixtral-8x7b-32768',
        'temperature': 0.7
    },
    'conversation_generator': {  
        'prompt': CONV_GENERATOR_PROMPT,
        'model_name': 'mixtral-8x7b-32768', 
        'temperature': 0.7
    },
    'validator': {
        'prompt': VALIDATOR_PROMPT,
        'model_name': 'llama3-8b-8192',
        'temperature': 0.0
    },
    'executors': {
        'model_A': {
            'prompt': EXECUTOR_PROMPT,
            'model_name': 'llama3-70b-8192',
            'temperature': 0.7  
        },
        'model_B': {
            'prompt': EXECUTOR_PROMPT,
            'model_name': 'gemma2-9b-it',
            'temperature': 0.7
        }
    }
}