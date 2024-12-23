import os
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
        self.json_metrics = self.test_config.metrics['json_tests']
        self.conv_metrics = self.test_config.metrics['conversation_metrics']

    def calculate_all_metrics(self, results: dict, conversation_tests: pd.DataFrame, json_tests: pd.DataFrame) -> dict:
        metrics = {}
        for model, model_results in results.items():
            conv_responses = model_results[model_results['test_case'] == 'conversation']
            json_responses = model_results[model_results['test_case'] == 'json']
            
            metrics[model] = {
                'json_metrics': self._calculate_json_metrics(json_responses, json_tests),
                'conversation_metrics': self._calculate_conversation_metrics(conv_responses, conversation_tests)
            }
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
        total_fields = 0
        matched_fields = 0
        
        def compare_fields(resp_obj, gold_obj):
            nonlocal total_fields, matched_fields
            for key in gold_obj:
                if isinstance(gold_obj[key], dict):
                    if key in resp_obj and isinstance(resp_obj[key], dict):
                        compare_fields(resp_obj[key], gold_obj[key])
                else:
                    total_fields += 1
                    if key in resp_obj and resp_obj[key] == gold_obj[key]:
                        matched_fields += 1
        
        compare_fields(response, golden)
        return matched_fields / total_fields if total_fields > 0 else 0

    def _check_structure(self, response, schema):
        required_keys = set()
        
        def collect_required_keys(schema_obj):
            if isinstance(schema_obj, dict):
                if 'required' in schema_obj:
                    required_keys.update(schema_obj['required'])
                for value in schema_obj.values():
                    collect_required_keys(value)
                    
        collect_required_keys(schema)
        
        found_keys = set()
        def check_keys(obj):
            if isinstance(obj, dict):
                found_keys.update(obj.keys())
                for value in obj.values():
                    check_keys(value)
                    
        check_keys(response)
        
        return len(required_keys.intersection(found_keys)) / len(required_keys) if required_keys else 0

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