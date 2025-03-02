import os
import json
import nltk
import asyncio
import logging
import pandas as pd
from datetime import datetime
from typing import Dict, Optional, Tuple

from benchmark_config import BenchmarkConfig
from metrics_calculator import MetricsCalculator
from test_config import TestConfig
from test_executor import TestExecutor
from test_generator import TestGenerator
from test_validator import TestValidator
from utils import create_agent, groq_rate_limit, setup_logging
from agent_config import MODEL_CONFIG

# Initialize NLTK and logging
nltk.download('punkt_tab', quiet=True)
logger = setup_logging(__name__)

class BenchmarkPipeline:
    """
    Main pipeline for running model benchmarking experiments.
    
    Handles test generation, validation, execution, and analysis in a coordinated
    workflow with proper error handling and resource management.
    """
    
    def __init__(self):
        """Initialize pipeline components and configurations."""
        self.config = BenchmarkConfig()
        self.test_config = TestConfig()
        self.generator = TestGenerator(self.config, self.test_config)
        self.validator = None
        self.executor = TestExecutor(self.config)
        self.calculator = MetricsCalculator(self.test_config)
        
    async def _initialize_validator(self):
        """
        Asynchronously initialize the validator.
        
        Returns:
            Initialized TestValidator instance
        """
        validator = TestValidator(self.config, self.test_config)
        self.validator = await validator.initialize()
        return self.validator
        
    async def run(self) -> Dict:
        """
        Execute the complete benchmarking pipeline.
        
        Returns:
            Dict containing the computed metrics and analysis results
        """
        await self._initialize_validator()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = f"benchmark_results/run_{timestamp}"
        os.makedirs(results_dir, exist_ok=True)
        
        try:
            # Generate and validate test cases
            test_df = await self._generate_and_validate_tests(results_dir)
            if test_df.empty:
                raise ValueError("No valid test cases generated")
                
            # Execute tests across models
            test_results = await self._execute_and_save_tests(test_df, results_dir)
            
            # Process results and generate reports
            metrics = await self._process_and_analyze_results(
                test_results, 
                test_df, 
                results_dir
            )
            
            logger.info("Benchmark pipeline completed successfully")
            return metrics
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {str(e)}")
            raise
            
    async def _generate_and_validate_tests(self, results_dir: str) -> pd.DataFrame:
        """Generate test cases and optionally validate them."""
        try:
            test_df = await self.generator.generate_test_cases()
            test_df.to_csv(f"{results_dir}/initial_tests.csv", index=False)
            logger.info(f"Generated {len(test_df)} initial test cases")
            
            if not self.test_config.enable_validation:
                logger.info("Validation disabled, proceeding with unvalidated test cases")
                return test_df
                
            validation_passed, issues, valid_df = await self.validator.validate_tests(test_df)
            valid_df.to_csv(f"{results_dir}/validated_tests.csv", index=False)
            
            if not validation_passed:
                logger.warning(
                    f"Validation issues detected in {len(issues)} cases. "
                    "Proceeding with partially validated dataset"
                )
                
            return valid_df
            
        except Exception as e:
            logger.error(f"Test generation/validation failed: {str(e)}")
            raise
            
    async def _execute_and_save_tests(
        self, 
        test_df: pd.DataFrame, 
        results_dir: str
    ) -> Dict[str, pd.DataFrame]:
        """Execute tests and save results for all models."""
        test_results = await self.executor.run_tests(test_df)
        
        # Save individual model results
        for model_name, model_df in test_results.items():
            output_path = f"{results_dir}/{model_name}_all_responses.csv"
            model_df.to_csv(output_path, index=False)
            logger.info(f"Saved {model_name} results to {output_path}")
            
        # Create and save combined results
        combined_results = self._combine_test_results(test_df, test_results)
        combined_df = pd.DataFrame(combined_results)
        combined_output_path = f"{results_dir}/combined_model_responses.csv"
        combined_df.to_csv(combined_output_path, index=False)
        logger.info(f"Saved combined results to {combined_output_path}")
        
        return test_results
        
    def _combine_test_results(
        self, 
        test_df: pd.DataFrame, 
        test_results: Dict[str, pd.DataFrame]
    ) -> list:
        """Combine results from all models into a single format."""
        combined_results = []
        
        for test_id in test_df['id'].unique():
            for round_num in range(1, self.config.TEST_ROUNDS + 1):
                row_data = {
                    'test_id': test_id,
                    'round': round_num,
                    'prompt': test_df[test_df['id'] == test_id]['prompt'].iloc[0],
                    'test_case': test_df[test_df['id'] == test_id]['test_case'].iloc[0]
                }
                
                for model_name, model_df in test_results.items():
                    model_response = model_df[
                        (model_df['id'] == test_id) & 
                        (model_df['round'] == round_num)
                    ]['model_response'].iloc[0]
                    row_data[f'{model_name}_response'] = model_response
                    
                combined_results.append(row_data)
                
        return combined_results
        
    async def _process_and_analyze_results(
        self, 
        test_results: Dict[str, pd.DataFrame],
        test_df: pd.DataFrame,
        results_dir: str
    ) -> Dict:
        """Process test results and generate analysis reports."""
        # Calculate metrics
        metrics = self._calculate_and_save_metrics(test_results, test_df, results_dir)
        
        # Generate reports
        metrics_report_path = f"{results_dir}/metrics_report.json"
        self._generate_metrics_report(metrics, test_results, results_dir)
        await self._generate_analysis_report(metrics_report_path, results_dir)
        
        return metrics
        
    def _calculate_and_save_metrics(
        self, 
        test_results: Dict[str, pd.DataFrame],
        test_df: pd.DataFrame,
        results_dir: str
    ) -> Dict:
        """Calculate and save metrics with error handling."""
        try:
            conversation_tests = test_df[test_df['test_case'].isin(['customer, agent', 'conversation'])]
            logger.info(f"Processing {len(conversation_tests)} conversation tests")
            
            json_tests = pd.DataFrame()
            if self.test_config.enable_json_tests:
                json_tests = test_df[test_df['test_case'] == 'json']
                logger.info(f"Processing {len(json_tests)} JSON tests")
                
            metrics = self.calculator.calculate_all_metrics(
                test_results,
                conversation_tests,
                json_tests
            )
            
            metrics_file = f"{results_dir}/metrics.json"
            with open(metrics_file, 'w') as f:
                json.dump(metrics, f, indent=2)
                
            logger.info(f"Metrics calculated and saved to {metrics_file}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}")
            raise
            
    def _generate_metrics_report(self, metrics, test_results, results_dir):
        """Generate and save the metrics report."""
        logger.info("Generating metrics report")
        
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
        
        # Add JSON metrics only if they exist
        if self.test_config.enable_json_tests:
            for model_name in metrics:
                if 'json_metrics' in metrics[model_name]:
                    report['results_summary'][model_name]['avg_json_compliance'] = \
                        metrics[model_name]['json_metrics']['schema_compliance_rate']
                else:
                    # Include a placeholder or default value if json_metrics is missing
                    report['results_summary'][model_name]['avg_json_compliance'] = 0.0
                    logger.warning(f"json_metrics not found for model {model_name}, using default value")
        
        report_file = f"{results_dir}/metrics_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
            
    @groq_rate_limit(max_retries=3, base_delay=60.0)
    async def _generate_analysis_report(
        self,
        metrics_report_path: str,
        results_dir: str
    ) -> str:
        """Generate final analysis report using the report generator agent."""
        try:
            with open(metrics_report_path, 'r') as f:
                metrics_data = json.load(f)

            report_generator = await create_agent(
                system_prompt=MODEL_CONFIG['report_generator']['prompt'],
                model_name=MODEL_CONFIG['report_generator']['model_name'],
                temperature=MODEL_CONFIG['report_generator']['temperature'],
                agent_type='report_generator'
            )

            report_input = {
                'metrics_data': metrics_data,
                'feature_flags': {
                    'json_tests_enabled': self.test_config.enable_json_tests,
                    'validation_enabled': self.test_config.enable_validation
                }
            }

            report = await report_generator.ainvoke(report_input)
            report_path = f"{results_dir}/analysis_report.md"

            try:
                with open(report_path, 'w') as f:
                    f.write(report)
                logger.info(f"Analysis report generated and saved to {report_path}")
            except IOError as e:
                logger.error(f"Error writing analysis report: {str(e)}")
                raise

            return report

        except FileNotFoundError:
            logger.error(f"Metrics report file not found at {metrics_report_path}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in metrics report: {str(e)}")
            raise    
        except Exception as e:
            logger.exception(f"Unexpected error generating analysis report: {str(e)}")
            raise
        
async def main():
    """Execute the benchmark pipeline."""
    try:
        pipeline = BenchmarkPipeline()
        metrics = await pipeline.run()
        logger.info("Benchmark completed successfully!")
        return metrics
    except Exception as e:
        logger.error(f"Benchmark failed: {str(e)}")
        raise
        
if __name__ == "__main__":
    asyncio.run(main())