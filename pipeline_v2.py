import os
import json
import nltk
import asyncio
import logging
import pandas as pd
from datetime import datetime

from benchmark_config import BenchmarkConfig
from metrics_calculator import MetricsCalculator
from test_config import TestConfig
from test_executor import TestExecutor
from test_generator import TestGenerator
from test_validator import TestValidator

nltk.download('punkt_tab')

logging.basicConfig(
    # level=logging.DEBUG,
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('debug.log'),
        logging.StreamHandler()  # This will still print to console
    ]
)
logger = logging.getLogger(__name__)

class BenchmarkPipeline:
    def __init__(self):
        self.config = BenchmarkConfig()
        self.test_config = TestConfig()
        self.generator = TestGenerator(self.config, self.test_config)
        self.validator = TestValidator(self.config, self.test_config) 
        self.executor = TestExecutor(self.config)
        self.calculator = MetricsCalculator(self.test_config)

    async def run(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = f"benchmark_results/run_{timestamp}"
        os.makedirs(results_dir, exist_ok=True)

        test_df = await self._generate_and_validate_tests(results_dir)
        test_results = await self.executor.run_tests(test_df)
        
        # Save individual model results
        for model_name, model_df in test_results.items():
            output_path = f"{results_dir}/{model_name}_all_responses.csv"
            model_df.to_csv(output_path, index=False)
            logger.info(f"Saved {model_name} results to {output_path}")
        
        # Create and save combined results
        combined_results = []
        for test_id in test_df['id'].unique():
            for round_num in range(1, self.config.TEST_ROUNDS + 1):
                row_data = {
                    'test_id': test_id,
                    'round': round_num,
                    'prompt': test_df[test_df['id'] == test_id]['prompt'].iloc[0],
                    'test_case': test_df[test_df['id'] == test_id]['test_case'].iloc[0]
                }
                
                # Add each model's response
                for model_name, model_df in test_results.items():
                    model_response = model_df[
                        (model_df['id'] == test_id) & 
                        (model_df['round'] == round_num)
                    ]['model_response'].iloc[0]
                    row_data[f'{model_name}_response'] = model_response
                
                combined_results.append(row_data)
        
        combined_df = pd.DataFrame(combined_results)
        combined_output_path = f"{results_dir}/combined_model_responses.csv"
        combined_df.to_csv(combined_output_path, index=False)
        logger.info(f"Saved combined results to {combined_output_path}")

        # Calculate and save metrics
        metrics = self._calculate_and_save_metrics(test_results, test_df, results_dir)
        self._generate_report(metrics, test_results, results_dir)
        
        return metrics

    async def _generate_and_validate_tests(self, results_dir):
        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            test_df = await self.generator.generate_test_cases()
            test_df.to_csv(f"{results_dir}/initial_tests_attempt_{attempt}.csv", index=False)

            validation_passed, issues, valid_df = await self.validator.validate_tests(test_df)
            
            if validation_passed:
                valid_df.to_csv(f"{results_dir}/validated_tests.csv", index=False)
                return valid_df
            else:
                logger.warning(f"Validation attempt {attempt} failed. Issues: {issues}")
                if attempt < max_attempts:
                    logger.info(f"Retrying validation (attempt {attempt + 1})...")
                else:
                    raise RuntimeError("All validation attempts failed")

    def _calculate_and_save_metrics(self, test_results, test_df, results_dir):
        conversation_tests = test_df[test_df['test_case'] == 'conversation']
        json_tests = test_df[test_df['test_case'] == 'json']
        
        metrics = self.calculator.calculate_all_metrics(test_results, conversation_tests, json_tests)
        
        metrics_file = f"{results_dir}/metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        return metrics

    def _generate_report(self, metrics, test_results, results_dir):
        """Generate and save final report"""
        logger.info("Generating final report")
        
        report = {
            'run_timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
            'configuration': {
                'models_tested': list(test_results.keys()),
                'total_test_cases': len(next(iter(test_results.values()))),
                'rounds_per_model': self.config.TEST_ROUNDS
            },
            'results_summary': {
                model_name: {
                    'avg_json_compliance': metrics[model_name]['json_metrics']['schema_compliance_rate'],
                    'avg_bleu_score': metrics[model_name]['conversation_metrics']['bleu_score'],
                    'avg_task_completion': metrics[model_name]['conversation_metrics']['task_completion']
                }
                for model_name in metrics
            },
            'detailed_metrics': metrics
        }
        
        # Save report
        report_file = f"{results_dir}/metrics_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)

async def main():
    pipeline = BenchmarkPipeline()
    metrics = await pipeline.run()
    logger.info("Benchmark completed successfully!")
    
if __name__ == "__main__":
    asyncio.run(main())