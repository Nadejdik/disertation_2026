"""
Main runner script for LLM evaluation experiments
Can be used as an alternative to Jupyter notebook
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import argparse
from datetime import datetime
from pathlib import Path

from config import Config
from llm_interface import ModelFactory
from evaluation_framework import LLMEvaluator
from metrics_calculator import MetricsCalculator


def main():
    parser = argparse.ArgumentParser(description='Run LLM evaluation experiments')
    parser.add_argument('--models', nargs='+', choices=['gpt-4', 'llama-3', 'phi-3'],
                        default=['gpt-4'], help='Models to evaluate')
    parser.add_argument('--tasks', nargs='+', 
                        choices=['fault_classification', 'anomaly_detection', 'trace_analysis'],
                        default=['fault_classification'], help='Tasks to evaluate')
    parser.add_argument('--n-samples', type=int, default=50, help='Number of samples per dataset')
    parser.add_argument('--n-runs', type=int, default=3, help='Number of runs per sample')
    parser.add_argument('--skip-generation', action='store_true', help='Skip dataset generation')
    
    args = parser.parse_args()
    
    # Create directories
    Config.create_directories()
    
    print("=" * 60)
    print("LLM EVALUATION RUNNER")
    print("=" * 60)
    
    # Generate datasets if needed
    if not args.skip_generation:
        print("\nGenerating datasets...")
        os.system('python setup_datasets.py')
    
    # Initialize models
    print(f"\nInitializing models: {args.models}")
    models = []
    
    for model_type in args.models:
        try:
            config = Config.get_model_config(model_type)
            model = ModelFactory.create_model(model_type, **config)
            models.append(model)
            print(f"  ✓ {model_type} initialized")
        except Exception as e:
            print(f"  ✗ {model_type} failed: {e}")
    
    if not models:
        print("\n✗ No models initialized. Exiting.")
        return
    
    # Initialize evaluator
    evaluator = LLMEvaluator(models, output_dir=str(Config.RESULTS_DIR))
    
    # Run evaluations
    print(f"\nRunning evaluations for tasks: {args.tasks}")
    print(f"Samples: {args.n_samples}, Runs: {args.n_runs}\n")
    
    for task in args.tasks:
        print(f"\n{'='*60}")
        print(f"Task: {task.upper().replace('_', ' ')}")
        print(f"{'='*60}")
        
        if task == 'fault_classification':
            evaluator.evaluate_fault_classification(
                str(Config.DATASETS_DIR / 'synthetic_telecom_faults.csv'),
                n_samples=args.n_samples,
                n_runs=args.n_runs
            )
        elif task == 'anomaly_detection':
            evaluator.evaluate_anomaly_detection(
                str(Config.DATASETS_DIR / 'kpi_anomaly_detection.csv'),
                n_samples=args.n_samples,
                n_runs=args.n_runs
            )
        elif task == 'trace_analysis':
            evaluator.evaluate_trace_analysis(
                str(Config.DATASETS_DIR / 'microservices_traces.csv'),
                n_samples=args.n_samples,
                n_runs=args.n_runs
            )
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = evaluator.save_results(f'evaluation_results_{timestamp}.csv')
    print(f"\n✓ Results saved to: {results_file}")
    
    # Calculate metrics
    print("\nCalculating metrics...")
    calculator = MetricsCalculator()
    
    import pandas as pd
    results_df = pd.DataFrame(evaluator.results)
    
    metrics = {}
    for task in args.tasks:
        task_data = results_df[results_df['task'] == task]
        if not task_data.empty:
            if task == 'fault_classification':
                metrics[task] = calculator.calculate_fault_classification_metrics(task_data)
            elif task == 'anomaly_detection':
                metrics[task] = calculator.calculate_anomaly_detection_metrics(task_data)
            elif task == 'trace_analysis':
                metrics[task] = calculator.calculate_trace_analysis_metrics(task_data)
    
    # Save metrics
    metrics_file = Config.RESULTS_DIR / f'metrics_report_{timestamp}.json'
    calculator.save_metrics_report(metrics, str(metrics_file))
    print(f"✓ Metrics saved to: {metrics_file}")
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    
    for task_name, task_metrics in metrics.items():
        print(f"\n{task_name.upper().replace('_', ' ')}:")
        if 'error' not in task_metrics:
            for model_name, model_metrics in task_metrics.items():
                print(f"\n  {model_name}:")
                for metric, value in model_metrics.items():
                    print(f"    {metric}: {value}")
    
    print("\n" + "="*60)
    print("COMPLETE!")
    print("="*60)


if __name__ == '__main__':
    main()
