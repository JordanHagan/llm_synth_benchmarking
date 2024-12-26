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
from utils import create_agent
from agent_config import MODEL_CONFIG

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


       # Calculate metrics
        metrics = self._calculate_and_save_metrics(test_results, test_df, results_dir)
        metrics_report_path = f"{results_dir}/metrics_report.json"
        self._generate_report(metrics, test_results, results_dir)
        
        # Generate analysis report
        await self._generate_analysis_report(metrics_report_path, results_dir)
            
        return metrics

    async def _generate_and_validate_tests(self, results_dir):
        try:
            test_df = await self.generator.generate_test_cases()
            test_df.to_csv(f"{results_dir}/initial_tests.csv", index=False)
            
            if not self.test_config.enable_validation:
                logger.info("Validation disabled, proceeding with unvalidated test cases")
                return test_df
                
            validation_passed, issues, valid_df = await self.validator.validate_tests(test_df)
            valid_df.to_csv(f"{results_dir}/validated_tests.csv", index=False)
            
            if not validation_passed:
                logger.warning("Validation issues detected, proceeding with partially validated dataset")
                
            return valid_df
            
        except Exception as e:
            logger.error(f"Test generation/validation failed: {e}")
            raise

    def _calculate_and_save_metrics(self, test_results, test_df, results_dir):
        # Update test case filtering
        conversation_tests = test_df[test_df['test_case'] == 'customer, agent']
        json_tests = test_df[test_df['test_case'] == 'json'] if self.test_config.enable_json_tests else pd.DataFrame()

        print(f"DEBUG: Found {len(conversation_tests)} conversation tests")
        print(f"DEBUG: Found {len(json_tests)} JSON tests")

        metrics = self.calculator.calculate_all_metrics(test_results, conversation_tests, json_tests)

        metrics_file = f"{results_dir}/metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)

        return metrics

    def _generate_report(self, metrics, test_results, results_dir):
        logger.info("Generating final report")

        report = {
            'run_timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
            'configuration': {
                'models_tested': list(test_results.keys()),
                'total_test_cases': len(next(iter(test_results.values()))),
                'rounds_per_model': self.config.TEST_ROUNDS,
                'json_tests_enabled': self.test_config.enable_json_tests,
                'validation_enabled': self.test_config.enable_validation
            },
            'results_summary': {
                model_name: {
                    'avg_bleu_score': metrics[model_name]['conversation_metrics']['bleu_score'], 
                    'avg_task_completion': metrics[model_name]['conversation_metrics']['task_completion']
                } 
                for model_name in metrics
            },
            'detailed_metrics': metrics
        }

        if self.test_config.enable_json_tests:
            for model_name in metrics:
                report['results_summary'][model_name]['avg_json_compliance'] = metrics[model_name]['json_metrics']['schema_compliance_rate']

        report_file = f"{results_dir}/metrics_report.json"   
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
    
    async def _generate_analysis_report(self, metrics_report_path: str, results_dir: str):
        """Generate final analysis report with rate limit handling."""
        MAX_RETRIES = 3
        RETRY_DELAY = 60  # seconds
        
        for attempt in range(MAX_RETRIES):
            try:
                # Read metrics report
                with open(metrics_report_path, 'r') as f:
                    metrics_data = json.load(f)
                
                # Create report generator agent using MODEL_CONFIG instead of test_config
                report_generator = create_agent(
                    system_prompt=MODEL_CONFIG['report_generator']['prompt'],
                    model_name=MODEL_CONFIG['report_generator']['model_name'],
                    temperature=MODEL_CONFIG['report_generator']['temperature'],
                    agent_type='report_generator'
                )
                
                # Prepare input for report generator
                report_input = {
                        'metrics_data': metrics_data,
                        'feature_flags': {
                            'json_tests_enabled': self.test_config.enable_json_tests,
                            'validation_enabled': self.test_config.enable_validation
                    }
                }
                
                # Generate report
                report = await report_generator.ainvoke(report_input)
                
                # Save report
                report_path = f"{results_dir}/analysis_report.md"
                with open(report_path, 'w') as f:
                    f.write(report)
                
                logger.info(f"Analysis report generated and saved to {report_path}")
                return report
                
            except Exception as e:
                if "Rate limit" in str(e):
                    if attempt < MAX_RETRIES - 1:
                        wait_time = RETRY_DELAY * (2 ** attempt)
                        logger.warning(f"Rate limit reached. Waiting {wait_time} seconds before retry...")
                        await asyncio.sleep(wait_time)
                    else:
                        logger.error("Max retries reached for rate limit. Failed to generate report.")
                        raise
                else:
                    logger.error(f"Error generating report: {str(e)}")
                    raise
    

async def main():
    pipeline = BenchmarkPipeline()
    metrics = await pipeline.run()
    logger.info("Benchmark completed successfully!")
    
if __name__ == "__main__":
    asyncio.run(main())