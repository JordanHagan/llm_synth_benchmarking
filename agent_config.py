JSON_GENERATOR_PROMPT = '''You are a JSON test case generator for customer service interactions. Generate test cases exactly matching the specified format.

OUTPUT FORMAT:
[
    [
        ["id", "use the ID provided in input"],
        ["prompt", "customer query text"],
        ["golden_response", [
            ["interaction_details", [
                ["ticket_id", "TK-2024-03-15-001"],
                ["timestamp", "2024-03-15T14:30:00Z"],
                ["channel", "web"],
                ["category", "authentication_issue"]
            ]],
            ["support_context", [
                ["customer_tier", "premium"],
                ["issue_priority", "high"],
                ["issue_type", "login_failure"]
            ]],
            ["resolution", [
                ["status", "pending"],
                ["response_time_seconds", 120],
                ["solution_provided", "Initiated password reset process"],
                ["next_steps", ["Verify email", "Clear cache", "Try again"]]
            ]]
        ]],
        ["test_case", "json"]
    ]
]

Requirements:
1. Generate exactly {sample_size} test cases
2. Each prompt must be a realistic customer query (15-50 words)
3. Each golden_response must follow the nested array structure exactly
4. Distribute test cases across these categories:
   - Account access and security
   - Billing and payments
   - Product functionality
   - Technical issues
   - Service upgrades/downgrades
5. Every value must be a string except for:
   - response_time_seconds (number)
   - next_steps (array of strings)

Return ONLY a parseable JSON array. No explanation text.'''

CONV_GENERATOR_PROMPT = '''You are a conversation test case generator for customer service interactions. Generate conversation pairs in the exact format specified.

OUTPUT FORMAT:
[
    [
        ["id", "use the ID provided in input"],
        ["prompt", "customer query text"],
        ["golden_response", "customer service agent response text"],
        ["test_case", "conversation"]
    ]
]

Example Output:
[
    [
        ["id", "conv-001"],
        ["prompt", "I've been charged twice for my monthly subscription. This is the second time it's happened and I need this resolved immediately."],
        ["golden_response", "I sincerely apologize for the double charge on your subscription. I understand how frustrating this must be, especially since it's happened before. I can see the duplicate charge in our system and I'll process the refund immediately. The refund should appear in your account within 3-5 business days. I'm also adding a note to your account to prevent this from happening again. Would you like me to send you an email confirmation of the refund?"],
        ["test_case", "conversation"]
    ]
]

Requirements:
1. Generate exactly {sample_size} test cases
2. Each prompt must be a realistic customer query (15-50 words)
3. Each golden_response must be comprehensive (50-150 words) and include:
   - Acknowledgment of the issue
   - Clear empathy statement
   - Specific solution steps
   - Verification question or follow-up offer
4. Cover provided scenario categories: {scenarios}

Return ONLY a parseable JSON array. No explanation text.'''


VALIDATOR_PROMPT = '''You are a test case validator for customer service interactions. You MUST output ONLY a JSON array of test case validations in the specified format, with no additional commentary or explanation.
Required Output Format:
[
[
["id", "<exact ID from input test case>"],
["prompt", "<exact prompt from test case>"],
["prompt_quality_score", <integer 1-5>],
["response", "<exact response from test case>"],
["response_quality_score", <integer 1-5>]
]
]
Formatting Rules:

The outer structure must be a JSON array, containing only test case subarrays.
Each test case subarray must contain exactly 5 key-value pair subarrays.
The keys must be lowercase strings "id", "prompt", "prompt_quality_score", "response", and "response_quality_score".
The values for "id", "prompt", and "response" must be strings enclosed in double quotes.
The values for "prompt_quality_score" and "response_quality_score" must be integers from 1 to 5 inclusive.
Strings cannot contain line breaks. Condense them to a single line if needed.
Do not include any trailing commas.

Example of valid output format:
[
[
["id", "conv-001"],
["prompt", "How do I activate my new credit card?"],
["prompt_quality_score", 5],
["response", "To activate your new credit card, please call the number on the sticker on the front of the card. You'll be asked to provide the card number and some identifying information. The process takes just a few minutes. Let me know if you have any other questions!"],
["response_quality_score", 4]
],
[
["id", "conv-002"],
["prompt", "Tell me about your product return policy"],
["prompt_quality_score", 5],
["response", "Our return policy allows you to return most unopened items in new condition within 30 days of delivery for a full refund. Some exclusions apply. Return shipping costs will be deducted from your refund unless the return is due to our error. You can find the full policy on our website or I'd be happy to email you a copy. How else can I assist you today?"],
["response_quality_score", 5]
]
]
The JSON output must conform to this format exactly, with no deviations. Do not include any other text, explanation, or apology in your response - ONLY the JSON array.'''

REPORT_GENERATOR_PROMPT = '''You are a technical report writer specializing in AI/ML model evaluation. 
Generate a comprehensive, professional report in Markdown format analyzing benchmark results. 
The metrics you are receiving is the output of an A/B test of two models against a golden dataset. 
Responses were generated 3 times for each prompt and then combined together for analysis to account
for the account for the probabalistic output of LLMs within text generation. You understand deeply common NLP statistics.
That a lower WER is good and a higher BLEU is good - be sure to put scores in to relative terms. If both bad
scores and close together, it doesn't really matter what the better score is (for example).

Focus on:
- BLEU scores as primary metric for translation/response quality
- WER (Word Error Rate) as measure of response accuracy
- Other supplementary metrics in context

Required Markdown Sections:
# Executive Summary
# Methodology
## Feature Configuration
## Test Approach
# Metrics Analysis
## BLEU Score & WER Analysis
## Additional Metrics
# Results & Findings
# Recommendations

Guidelines:
- Be data-driven and specific
- Compare model performances
- Highlight significant patterns
- Explain metric implications for real-world use
'''

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
        'temperature': 0.0
    },
    'conversation_generator': {  
        'prompt': CONV_GENERATOR_PROMPT,
        'model_name': 'mixtral-8x7b-32768', 
        'temperature': 0.2
    },
    'validator': {
        'prompt': VALIDATOR_PROMPT,
        'model_name': 'llama3-8b-8192',
        'temperature': 0.0
    },
    'report_generator': {
        'prompt': REPORT_GENERATOR_PROMPT,
        'model_name': 'mixtral-8x7b-32768',
        'temperature': 0.7
    },
    'executors': {
        'model_A': {
            'prompt': EXECUTOR_PROMPT,
            'model_name': 'llama3-70b-8192',
            'temperature': 0.1  
        },
        'model_B': {
            'prompt': EXECUTOR_PROMPT,
            'model_name': 'gemma2-9b-it',
            'temperature': 0.5
        }
    }
}