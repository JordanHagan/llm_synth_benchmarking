import json
import pandas as pd
import numpy as np
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from jiwer import wer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')

class MetricsCalculator:
    """
    Calculates metrics for evaluating model responses, handling both direct text and tool-based outputs.
    """
    
    def __init__(self, test_config):
        self.test_config = test_config
        self.stop_words = set(stopwords.words('english'))
        self._load_metrics()

    def _load_metrics(self):
        """Load appropriate metrics based on test configuration."""
        self.conv_metrics = self.test_config.metrics['conversation_metrics']
        if self.test_config.enable_json_tests:
            self.json_metrics = self.test_config.metrics['json_tests']

    def calculate_all_metrics(self, results: dict, conversation_tests: pd.DataFrame, json_tests: pd.DataFrame) -> dict:
        """Calculate metrics for all model results."""
        metrics = {}
        print("Starting metrics calculation")
        
        for model, model_results in results.items():
            print(f"Processing model: {model}")
            model_metrics = {}

            # Handle conversation metrics
            conv_responses = model_results[model_results['test_case'].isin(['conversation', 'customer, agent'])]
            if not conv_responses.empty:
                print(f"Calculating conversation metrics for {len(conv_responses)} responses")
                model_metrics['conversation_metrics'] = self._calculate_conversation_metrics(conv_responses, conversation_tests)

            metrics[model] = model_metrics

        return metrics

    def _extract_response_text(self, response: str) -> str:
        """
        Extract clean response text from tool-based output format.
        Handles both JSON tool outputs and direct text responses.
        """
        try:
            # Check for JSON tool output format
            if '```json' in response:
                json_content = response.split('```json')[1].split('```')[0].strip()
                parsed_json = json.loads(json_content)
                return parsed_json.get('response_text', '')
            
            # Check for direct JSON format
            if response.strip().startswith('{'):
                parsed_json = json.loads(response)
                return parsed_json.get('response_text', '')
            
            # Return cleaned text for direct responses
            return response.replace('```', '').strip()
            
        except (json.JSONDecodeError, IndexError):
            return response.strip()

    def _calculate_conversation_metrics(self, responses: pd.DataFrame, golden_tests: pd.DataFrame) -> dict:
        """Calculate metrics for conversation-based responses."""
        print("Processing conversation metrics...")
        total_tests = len(responses)
        if total_tests == 0:
            return {metric: 0 for metric in self.conv_metrics}

        bleu_scores = []
        wer_scores = []
        relevance_scores = []
        clarity_scores = []
        completion_scores = []

        for _, response in responses.iterrows():
            try:
                golden = golden_tests.loc[golden_tests['id'] == response['id']].iloc[0]
                response_text = self._extract_response_text(response['model_response'])
                golden_text = self._extract_response_text(golden['golden_response'])
                
                # Calculate metrics
                bleu_scores.append(self._calculate_bleu(golden_text, response_text))
                wer_scores.append(self._calculate_wer(golden_text, response_text))
                relevance_scores.append(self._score_relevance(response_text, response['prompt']))
                clarity_scores.append(self._score_clarity(response_text))
                completion_scores.append(self._score_task_completion(response_text, response['prompt']))
                
            except Exception as e:
                print(f"Error processing response {response['id']}: {str(e)}")
                continue

        return {
            'bleu_score': np.mean(bleu_scores) if bleu_scores else 0,
            'wer_score': np.mean(wer_scores) if wer_scores else 0,
            'response_relevance': np.mean(relevance_scores) if relevance_scores else 0,
            'clarity': np.mean(clarity_scores) if clarity_scores else 0,
            'task_completion': np.mean(completion_scores) if completion_scores else 0
        }

    def _calculate_bleu(self, reference: str, candidate: str) -> float:
        """Calculate BLEU score between reference and candidate texts with smoothing."""
        try:
            reference_tokens = reference.split()
            candidate_tokens = candidate.split()
            smoothing = SmoothingFunction().method1
            return sentence_bleu([reference_tokens], candidate_tokens, smoothing_function=smoothing)
        except Exception:
            return 0

    def _calculate_wer(self, reference: str, candidate: str) -> float:
        """Calculate Word Error Rate between reference and candidate texts."""
        try:
            return wer(reference, candidate)
        except Exception:
            return 1.0

    def _score_relevance(self, response: str, prompt: str) -> float:
        """Score response relevance based on keyword overlap."""
        prompt_tokens = set(word_tokenize(prompt.lower())) - self.stop_words
        response_tokens = set(word_tokenize(response.lower())) - self.stop_words
        
        intersection = len(prompt_tokens.intersection(response_tokens))
        union = len(prompt_tokens.union(response_tokens))
        
        return intersection / union if union > 0 else 0

    def _score_clarity(self, response: str) -> float:
        """Score response clarity based on sentence structure and length."""
        sentences = sent_tokenize(response)
        if not sentences:
            return 0
            
        scores = []
        for sentence in sentences:
            words = word_tokenize(sentence)
            if len(words) > 50:
                scores.append(0.5)
            elif len(words) < 3:
                scores.append(0.3)
            else:
                scores.append(1.0)
                
        return np.mean(scores)

    def _score_task_completion(self, response: str, prompt: str) -> float:
        """Score task completion based on response comprehensiveness."""
        task_tokens = set(word_tokenize(prompt.lower())) - self.stop_words
        response_tokens = set(word_tokenize(response.lower())) - self.stop_words
        
        addressed_elements = len(task_tokens.intersection(response_tokens))
        total_elements = len(task_tokens)
        
        completion_score = addressed_elements / total_elements if total_elements > 0 else 0
        
        # Check for solution indicators in the response
        resolution_indicators = ['here', 'follow', 'steps', 'solution', 'resolve']
        has_resolution = any(indicator in response.lower() for indicator in resolution_indicators)
        
        return completion_score * 1.2 if has_resolution else completion_score