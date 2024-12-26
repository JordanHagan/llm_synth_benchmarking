import json
import pandas as pd
import numpy as np
import jsonschema
import nltk
from nltk.translate.bleu_score import sentence_bleu
from jiwer import wer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.metrics.distance import edit_distance


nltk.download('punkt_tab','punkt','stopwords')

class MetricsCalculator:
    def __init__(self, test_config):
        self.test_config = test_config
        self._load_metrics()
        self.stop_words = set(stopwords.words('english'))

    def _load_metrics(self):
        if self.test_config.enable_json_tests: 
            self.json_metrics = self.test_config.metrics['json_tests']
            self.conv_metrics = self.test_config.metrics['conversation_metrics']
        else:
            self.conv_metrics = self.test_config.metrics['conversation_metrics']

    def calculate_all_metrics(self, results: dict, conversation_tests: pd.DataFrame, json_tests: pd.DataFrame) -> dict:
        metrics = {}
        print("DEBUG: Starting metrics calculation")
        print(f"DEBUG: Conversation tests available: {len(conversation_tests)}")
        
        for model, model_results in results.items():
            print(f"DEBUG: Processing model: {model}")
            print(f"DEBUG: Model results shape: {model_results.shape}")
            print(f"DEBUG: Test types found: {model_results['test_case'].unique()}")
            
            model_metrics = {}

            # Updated to handle 'customer, agent' test case
            conv_responses = model_results[model_results['test_case'] == 'customer, agent']
            print(f"DEBUG: Conversation responses found: {len(conv_responses)}")
            
            if not conv_responses.empty:  
                print("DEBUG: Calculating conversation metrics...")
                model_metrics['conversation_metrics'] = self._calculate_conversation_metrics(conv_responses, conversation_tests)
                
            print(model_metrics)

            metrics[model] = model_metrics
        
        return metrics

    def _calculate_json_metrics(self, responses: pd.DataFrame, golden_tests: pd.DataFrame) -> dict:
        total_tests = len(responses)
        if total_tests == 0:
            return {'schema_compliance_rate': 0, 'field_accuracy': 0, 'structural_consistency': 0}

        valid_responses = 0
        field_matches = 0
        structure_matches = 0

        for _, response in responses.iterrows():
            try:
                parsed_response = json.loads(response['model_response'])
                golden = golden_tests.loc[golden_tests['id'] == response['id']].iloc[0]
                if self._validate_schema(parsed_response, self.test_config.response_schema):
                    valid_responses += 1
                    field_matches += self._check_field_accuracy(parsed_response, golden['golden_response'])
                    structure_matches += self._check_structure(parsed_response, self.test_config.response_schema)
            except (json.JSONDecodeError, IndexError):
                continue

        return {
            'schema_compliance_rate': valid_responses / total_tests,
            'field_accuracy': field_matches / total_tests,
            'structural_consistency': structure_matches / total_tests
        }

    def _calculate_conversation_metrics(self, responses: pd.DataFrame, golden_tests: pd.DataFrame) -> dict:
        print("❤️ working on some metrics....")
        total_tests = len(responses)
        if total_tests == 0:
            return {metric: 0 for metric in self.conv_metrics}

        bleu_scores = []
        wer_scores = []
        relevance_scores = []
        clarity_scores = []
        completion_scores = []

        for idx, response in responses.iterrows():
            golden = golden_tests.loc[golden_tests['id'] == response['id']].iloc[0]
            
            bleu_scores.append(sentence_bleu([golden['golden_response'].split()], response['model_response'].split()))
            wer_scores.append(wer(golden['golden_response'], response['model_response']))
            
            # Add custom scoring for relevance, clarity, and task completion
            relevance_scores.append(self._score_relevance(response['model_response'], golden['prompt']))
            clarity_scores.append(self._score_clarity(response['model_response']))
            completion_scores.append(self._score_task_completion(response['model_response'], golden['prompt']))

        return {
            'bleu_score': np.mean(bleu_scores),
            'wer_score': np.mean(wer_scores),
            'response_relevance': np.mean(relevance_scores),
            'clarity': np.mean(clarity_scores),
            'task_completion': np.mean(completion_scores)
        }

    def _validate_schema(self, response, schema):
        try:
            validator = jsonschema.Draft7Validator(schema)
            errors = list(validator.iter_errors(response))
            return len(errors) == 0
        except Exception:
            return False

    def _check_field_accuracy(self, response, golden):
        """Check field accuracy for array-formatted responses."""
        total_fields = 0
        matched_fields = 0
        
        def compare_arrays(resp_arr, gold_arr):
            nonlocal total_fields, matched_fields
            if not isinstance(resp_arr, list) or not isinstance(gold_arr, list):
                return
                
            # Handle flat key-value arrays
            if len(resp_arr) == 2 and isinstance(resp_arr[0], str):
                key = resp_arr[0]
                value = resp_arr[1]
                if len(gold_arr) == 2 and gold_arr[0] == key:
                    total_fields += 1
                    if gold_arr[1] == value:
                        matched_fields += 1
                return
                
            # Handle nested arrays
            for resp_item, gold_item in zip(resp_arr, gold_arr):
                if isinstance(resp_item, list):
                    compare_arrays(resp_item, gold_item)
        
        compare_arrays(response, golden)
        return matched_fields / total_fields if total_fields > 0 else 0

    def _check_structure(self, response, schema):
        """Check structure compliance for array-formatted responses."""
        required_fields = set()
        
        def collect_required_fields(schema_arr):
            if isinstance(schema_arr, list):
                if len(schema_arr) == 2 and isinstance(schema_arr[0], str):
                    required_fields.add(schema_arr[0])
                else:
                    for item in schema_arr:
                        collect_required_fields(item)
                        
        def check_response_fields(resp_arr):
            found_fields = set()
            if isinstance(resp_arr, list):
                if len(resp_arr) == 2 and isinstance(resp_arr[0], str):
                    found_fields.add(resp_arr[0])
                else:
                    for item in resp_arr:
                        found_fields.update(check_response_fields(item))
            return found_fields
        
        collect_required_fields(schema)
        found_fields = check_response_fields(response)
        
        return len(required_fields.intersection(found_fields)) / len(required_fields) if required_fields else 0

    def _score_relevance(self, response, prompt):
        # Tokenize and remove stop words
        prompt_tokens = set(word_tokenize(prompt.lower())) - self.stop_words
        response_tokens = set(word_tokenize(response.lower())) - self.stop_words
        
        # Calculate Jaccard similarity
        intersection = len(prompt_tokens.intersection(response_tokens))
        union = len(prompt_tokens.union(response_tokens))
        
        return intersection / union if union > 0 else 0

    def _score_clarity(self, response):
        sentences = sent_tokenize(response)
        if not sentences:
            return 0
            
        scores = []
        for sentence in sentences:
            words = word_tokenize(sentence)
            if len(words) > 50:  # Penalize very long sentences
                scores.append(0.5)
            elif len(words) < 3:  # Penalize very short sentences
                scores.append(0.3)
            else:
                scores.append(1.0)
                
            # Check for common clarity markers
            if any(marker in sentence.lower() for marker in ['however', 'but', 'although']):
                scores[-1] *= 0.9
                
        return np.mean(scores)

    def _score_task_completion(self, response, prompt):
        # Extract key task elements from prompt
        task_elements = set(word_tokenize(prompt.lower())) - self.stop_words
        response_elements = set(word_tokenize(response.lower())) - self.stop_words
        
        # Calculate completion ratio
        addressed_elements = len(task_elements.intersection(response_elements))
        total_elements = len(task_elements)
        
        # Check for resolution indicators
        resolution_markers = ['resolved', 'completed', 'fixed', 'solution', 'answer']
        has_resolution = any(marker in response.lower() for marker in resolution_markers)
        
        completion_score = addressed_elements / total_elements if total_elements > 0 else 0
        return completion_score * 1.2 if has_resolution else completion_score