import re
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

            # Handle JSON metrics
            json_responses = model_results[model_results['test_case'] == 'json']
            if not json_responses.empty and not json_tests.empty and self.test_config.enable_json_tests:
                print(f"Calculating JSON metrics for {len(json_responses)} responses")
                model_metrics['json_metrics'] = self._calculate_json_metrics(json_responses, json_tests)

            metrics[model] = model_metrics

        return metrics

    def _extract_response_text(self, response: str) -> str:
        """
        Extract clean response text from tool-based output format.
        Handles both JSON tool outputs, direct text responses, and removes <think> tags.
        """
        try:
            # Clean out <think> tags first
            if isinstance(response, str) and '<think>' in response and '</think>' in response:
                response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
            
            # Check for JSON tool output format
            if isinstance(response, str) and '```json' in response:
                json_content = response.split('```json')[1].split('```')[0].strip()
                parsed_json = json.loads(json_content)
                return parsed_json.get('response_text', '')
            
            # Check for direct JSON format with process_customer_response
            if isinstance(response, str) and '"name": "process_customer_response"' in response:
                parsed_json = json.loads(response)
                return parsed_json.get('arguments', {}).get('response_text', '')
                
            # Check for direct JSON format
            if isinstance(response, str) and response.strip().startswith('{'):
                parsed_json = json.loads(response)
                return parsed_json.get('response_text', '')
            
            # Return cleaned text for direct responses
            return response.replace('```', '').strip()
            
        except (json.JSONDecodeError, IndexError) as e:
            print(f"Error extracting response text: {str(e)}")
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
    
    def _extract_json_content(self, response_text: str) -> dict:
        """Extract the JSON content from various response formats."""
        try:
            # Handle different response formats
            if isinstance(response_text, str):
                # Clean out <think> tags first
                if '<think>' in response_text and '</think>' in response_text:
                    response_text = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL).strip()
                    
                # Check for JSON code blocks
                if '```json' in response_text:
                    json_content = response_text.split('```json')[1].split('```')[0].strip()
                    return json.loads(json_content)
                    
                # Check for process_customer_response format
                if '"name": "process_customer_response"' in response_text:
                    parsed = json.loads(response_text)
                    if 'arguments' in parsed:
                        return parsed['arguments']
                    return parsed
                    
                # Direct JSON string
                if response_text.strip().startswith('{'):
                    return json.loads(response_text)
                    
            # If it's already a dict, just return it
            elif isinstance(response_text, dict):
                return response_text
                
        except (json.JSONDecodeError, IndexError) as e:
            print(f"Error extracting JSON content: {str(e)}")
        
        return {}
    
    def _calculate_json_metrics(self, responses: pd.DataFrame, golden_tests: pd.DataFrame) -> dict:
        """Calculate metrics for JSON-based responses."""
        print("Processing JSON metrics...")
        total_tests = len(responses)
        if total_tests == 0:
            return {metric: 0 for metric in self.json_metrics}

        schema_compliance_scores = []
        field_accuracy_scores = []
        structural_consistency_scores = []

        for _, response in responses.iterrows():
            try:
                # Find the corresponding golden test
                golden_df = golden_tests[golden_tests['id'] == response['id']]
                if golden_df.empty:
                    print(f"No golden test found for response {response['id']}")
                    continue
                    
                golden = golden_df.iloc[0]
                
                # Extract JSON from response and golden test
                response_json = self._extract_json_content(response['model_response'])
                
                # Extract golden response
                golden_response = golden['golden_response']
                if isinstance(golden_response, str):
                    try:
                        golden_json = json.loads(golden_response)
                    except json.JSONDecodeError:
                        golden_json = {}
                else:
                    golden_json = golden_response
                    
                # For customer service responses, we'll match expected response format
                # rather than the full customer_interaction schema
                expected_response_fields = ["response_text", "response_type", "next_steps"]
                if all(field in response_json for field in expected_response_fields):
                    # Calculate compliance against response format
                    schema_compliance = 1.0
                else:
                    # Traditional schema compliance against customer_interaction
                    schema_compliance = self._calculate_schema_compliance(response_json)
                    
                # Calculate field accuracy - match what we can between the structures
                field_accuracy = self._calculate_flexible_field_accuracy(response_json, golden_json)
                
                # Calculate structural consistency
                structural_consistency = self._calculate_flexible_structural_consistency(response_json, golden_json)
                
                schema_compliance_scores.append(schema_compliance)
                field_accuracy_scores.append(field_accuracy)
                structural_consistency_scores.append(structural_consistency)
                
            except Exception as e:
                print(f"Error processing JSON response {response['id']}: {str(e)}")
                continue

        return {
            'schema_compliance_rate': np.mean(schema_compliance_scores) if schema_compliance_scores else 0,
            'field_accuracy': np.mean(field_accuracy_scores) if field_accuracy_scores else 0,
            'structural_consistency': np.mean(structural_consistency_scores) if structural_consistency_scores else 0
        }

    def _calculate_flexible_field_accuracy(self, response_json: dict, golden_json: dict) -> float:
        """
        Calculate field accuracy in a more flexible way, comparing what we can
        even if structures don't exactly match.
        """
        if not response_json:
            return 0.0
            
        # If we have a process_customer_response format, adapt
        if 'customer_interaction' in golden_json and 'response_text' in response_json:
            # Extract what we can compare
            expected_fields = [
                ('response_text', True),  # Field name, required
                ('response_type', True),
                ('next_steps', False)
            ]
            
            score = 0.0
            total = 0
            
            for field, required in expected_fields:
                if field in response_json:
                    score += 1.0
                    total += 1
                elif required:
                    total += 1
                    
            return score / total if total > 0 else 0.0
        
        # Fall back to existing field comparison
        return self._compare_json_fields(response_json, golden_json)

    def _calculate_flexible_structural_consistency(self, response_json: dict, golden_json: dict) -> float:
        """
        More flexible structural consistency check that accounts for different
        valid response formats.
        """
        if not response_json:
            return 0.0
            
        # Check if it's a valid customer service response
        if 'response_text' in response_json and 'response_type' in response_json:
            valid_types = ['answer', 'clarification', 'solution']
            if response_json.get('response_type') in valid_types:
                # It's a valid customer service response
                return 1.0
                
        # Fall back to traditional structural comparison
        return self._calculate_structural_consistency(response_json, golden_json)

    def _get_structure_signature(self, json_obj, path="") -> dict:
        """
        Generate a structure signature for a JSON object.
        
        Returns a dictionary mapping paths to data types.
        """
        signature = {}
        
        if isinstance(json_obj, dict):
            for key, value in json_obj.items():
                new_path = f"{path}.{key}" if path else key
                
                if isinstance(value, (dict, list)):
                    signature.update(self._get_structure_signature(value, new_path))
                else:
                    signature[new_path] = type(value).__name__
        elif isinstance(json_obj, list) and json_obj:
            # For lists, check the first item's type
            sample_item = json_obj[0]
            new_path = f"{path}[]"
            
            if isinstance(sample_item, (dict, list)):
                signature.update(self._get_structure_signature(sample_item, new_path))
            else:
                signature[new_path] = type(sample_item).__name__
        
        return signature
    
    def _calculate_schema_compliance(self, response_json: dict) -> float:
        """
        Calculate how well a response complies with the expected schema structure.
        
        Returns a score between 0 and 1.
        """
        if not response_json:
            return 0.0
        
        # Check for expected top-level keys
        expected_keys = ['customer_interaction']
        if 'customer_interaction' not in response_json:
            return 0.0
        
        customer_interaction = response_json.get('customer_interaction', {})
        if not isinstance(customer_interaction, dict):
            return 0.1
        
        # Check for expected fields in customer_interaction
        expected_ci_fields = ['interaction_id', 'timestamp', 'customer', 'interaction', 'metrics']
        existing_ci_fields = [field for field in expected_ci_fields if field in customer_interaction]
        ci_compliance = len(existing_ci_fields) / len(expected_ci_fields)
        
        # Check substructures if they exist
        subscore = 0.0
        subscore_count = 0
        
        # Check customer field
        if 'customer' in customer_interaction and isinstance(customer_interaction['customer'], dict):
            customer = customer_interaction['customer']
            expected_customer_fields = ['id', 'segment', 'priority_level']
            existing_customer_fields = [field for field in expected_customer_fields if field in customer]
            customer_compliance = len(existing_customer_fields) / len(expected_customer_fields)
            subscore += customer_compliance
            subscore_count += 1
        
        # Check interaction field
        if 'interaction' in customer_interaction and isinstance(customer_interaction['interaction'], dict):
            interaction = customer_interaction['interaction']
            expected_interaction_fields = ['type', 'summary', 'category', 'resolution_status', 'next_steps']
            existing_interaction_fields = [field for field in expected_interaction_fields if field in interaction]
            interaction_compliance = len(existing_interaction_fields) / len(expected_interaction_fields)
            subscore += interaction_compliance
            subscore_count += 1
        
        # Check metrics field
        if 'metrics' in customer_interaction and isinstance(customer_interaction['metrics'], dict):
            metrics = customer_interaction['metrics']
            expected_metrics_fields = ['response_time', 'satisfaction_score', 'resolution_time']
            existing_metrics_fields = [field for field in expected_metrics_fields if field in metrics]
            metrics_compliance = len(existing_metrics_fields) / len(expected_metrics_fields)
            subscore += metrics_compliance
            subscore_count += 1
        
        # Calculate final compliance score
        if subscore_count > 0:
            # Weight the top-level compliance and the substructure compliance
            return 0.4 * ci_compliance + 0.6 * (subscore / subscore_count)
        else:
            return 0.4 * ci_compliance

    def _compare_json_fields(self, response_obj: dict, golden_obj: dict) -> float:
        """Recursively compare fields between two JSON objects."""
        if not isinstance(response_obj, dict) or not isinstance(golden_obj, dict):
            # If they're both the same type and value, return 1, otherwise 0
            if type(response_obj) == type(golden_obj) and response_obj == golden_obj:
                return 1.0
            return 0.0
        
        # Find common keys
        response_keys = set(response_obj.keys())
        golden_keys = set(golden_obj.keys())
        common_keys = response_keys.intersection(golden_keys)
        
        if not common_keys:
            return 0.0
        
        field_scores = []
        
        for key in common_keys:
            resp_value = response_obj[key]
            gold_value = golden_obj[key]
            
            if isinstance(resp_value, dict) and isinstance(gold_value, dict):
                # Recursively compare nested objects
                field_scores.append(self._compare_json_fields(resp_value, gold_value))
            elif isinstance(resp_value, list) and isinstance(gold_value, list):
                # For lists, compare length similarity and sample a few items
                len_similarity = min(len(resp_value), len(gold_value)) / max(len(resp_value), len(gold_value)) if max(len(resp_value), len(gold_value)) > 0 else 0
                
                # Sample comparison for list items
                item_scores = []
                for i in range(min(len(resp_value), len(gold_value), 3)):  # Compare up to 3 items
                    if isinstance(resp_value[i], dict) and isinstance(gold_value[i], dict):
                        item_scores.append(self._compare_json_fields(resp_value[i], gold_value[i]))
                    elif resp_value[i] == gold_value[i]:
                        item_scores.append(1.0)
                    else:
                        item_scores.append(0.0)
                
                avg_item_score = np.mean(item_scores) if item_scores else 0.0
                field_scores.append(0.5 * len_similarity + 0.5 * avg_item_score)
            else:
                # For simple values, direct comparison
                field_scores.append(1.0 if resp_value == gold_value else 0.0)
        
        return np.mean(field_scores) if field_scores else 0.0

    def _calculate_structural_consistency(self, response_json: dict, golden_json: dict) -> float:
        """
        Calculate structural consistency between response and golden JSONs.
        
        Measures how well the response structure matches the expected structure.
        Returns a score between 0 and 1.
        """
        if not response_json or not golden_json:
            return 0.0
        
        # Get nested structure signatures
        response_structure = self._get_structure_signature(response_json)
        golden_structure = self._get_structure_signature(golden_json)
        
        # Calculate similarity between structures
        if response_structure == golden_structure:
            return 1.0
        
        # Calculate structure similarity score
        common_keys = set(response_structure.keys()).intersection(set(golden_structure.keys()))
        
        if not common_keys:
            return 0.0
        
        similarity_scores = []
        
        for key in common_keys:
            resp_value = response_structure[key]
            gold_value = golden_structure[key]
            
            if resp_value == gold_value:
                similarity_scores.append(1.0)
            else:
                # For different types or nested structures, calculate partial match
                similarity_scores.append(0.5)
        
        return np.mean(similarity_scores) if similarity_scores else 0.0