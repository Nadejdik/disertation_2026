"""
LLM Evaluation Framework
Evaluates different LLM models on telecom fault detection tasks
"""
import pandas as pd
import json
import time
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging
from datetime import datetime

from .llm_interface import ModelFactory, BaseLLMModel


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PromptTemplate:
    """Prompt templates for different tasks"""
    
    FAULT_CLASSIFICATION = """You are an expert in telecom systems and fault analysis.

Analyze the following fault log and provide:
1. Fault type classification
2. Severity level
3. Root cause analysis
4. Recommended action

Fault Log:
{log_message}

Service: {service}
Response Time: {response_time_ms}ms
Retry Count: {retry_count}
Error Rate: {error_rate}

Provide your analysis in a structured format."""

    ANOMALY_DETECTION = """You are an expert in time-series analysis and anomaly detection for telecom KPIs.

Analyze the following KPI time-series data and determine if there is an anomaly:

KPI Type: {kpi_type}
Current Value: {value}
Rolling Mean (12 periods): {rolling_mean_12}
Rolling Std (12 periods): {rolling_std_12}
Rolling Max (24 periods): {rolling_max_24}

Is this an anomaly? Explain your reasoning and provide confidence level (0-100%)."""

    TRACE_ANALYSIS = """You are an expert in distributed systems and microservices debugging.

Analyze the following distributed trace and identify issues:

Service: {service_name}
Operation: {operation_name}
Duration: {duration_ms}ms
Status: {status}
Fault Scenario: {fault_scenario}

Analyze the trace and provide:
1. Is there a performance issue or error?
2. What is the root cause?
3. Which service is responsible?
4. Recommended remediation"""

    ROOT_CAUSE_ANALYSIS = """You are an expert in telecom system reliability and root cause analysis.

Given the following system state, identify the root cause of the fault:

{context}

Provide:
1. Primary root cause
2. Contributing factors
3. Evidence from the data
4. Confidence level (0-100%)"""


class LLMEvaluator:
    """Evaluate LLM models on fault detection tasks"""
    
    def __init__(self, models: List[BaseLLMModel], output_dir: str = 'results'):
        self.models = models
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results = []
        
    def evaluate_fault_classification(self, dataset_path: str, n_samples: int = 100, n_runs: int = 3):
        """Evaluate models on fault classification task"""
        logger.info(f"Evaluating fault classification on {dataset_path}")
        
        # Load dataset
        df = pd.read_csv(dataset_path)
        samples = df.sample(n=min(n_samples, len(df)), random_state=42)
        
        task_results = []
        
        for model in self.models:
            logger.info(f"Evaluating model: {model.model_name}")
            
            for run_idx in range(n_runs):
                logger.info(f"  Run {run_idx + 1}/{n_runs}")
                
                for idx, row in samples.iterrows():
                    # Create prompt
                    prompt = PromptTemplate.FAULT_CLASSIFICATION.format(
                        log_message=row['log_message'],
                        service=row['service'],
                        response_time_ms=row['response_time_ms'],
                        retry_count=row['retry_count'],
                        error_rate=row['error_rate']
                    )
                    
                    # Generate response
                    result = model.generate(prompt)
                    
                    # Store result
                    task_results.append({
                        'task': 'fault_classification',
                        'model': model.model_name,
                        'run': run_idx,
                        'sample_id': row['id'],
                        'true_fault_type': row['fault_type'],
                        'true_outcome': row['outcome'],
                        'true_severity': row['severity'],
                        'prompt': prompt,
                        'response': result['response'],
                        'latency_seconds': result['latency_seconds'],
                        'tokens_used': result['tokens_used'],
                        'success': result['success'],
                        'error': result['error'],
                        'timestamp': datetime.now().isoformat()
                    })
                    
                    time.sleep(0.5)  # Rate limiting
        
        self.results.extend(task_results)
        return task_results
    
    def evaluate_anomaly_detection(self, dataset_path: str, n_samples: int = 100, n_runs: int = 3):
        """Evaluate models on anomaly detection task"""
        logger.info(f"Evaluating anomaly detection on {dataset_path}")
        
        # Load dataset
        df = pd.read_csv(dataset_path)
        
        # Sample balanced: 50% anomalies, 50% normal
        anomalies = df[df['is_anomaly'] == 1].sample(n=min(n_samples // 2, (df['is_anomaly'] == 1).sum()), random_state=42)
        normal = df[df['is_anomaly'] == 0].sample(n=min(n_samples // 2, (df['is_anomaly'] == 0).sum()), random_state=42)
        samples = pd.concat([anomalies, normal]).sample(frac=1, random_state=42)
        
        task_results = []
        
        for model in self.models:
            logger.info(f"Evaluating model: {model.model_name}")
            
            for run_idx in range(n_runs):
                logger.info(f"  Run {run_idx + 1}/{n_runs}")
                
                for idx, row in samples.iterrows():
                    # Create prompt
                    prompt = PromptTemplate.ANOMALY_DETECTION.format(
                        kpi_type=row['kpi_type'],
                        value=row['value'],
                        rolling_mean_12=row.get('rolling_mean_12', 'N/A'),
                        rolling_std_12=row.get('rolling_std_12', 'N/A'),
                        rolling_max_24=row.get('rolling_max_24', 'N/A')
                    )
                    
                    # Generate response
                    result = model.generate(prompt)
                    
                    # Store result
                    task_results.append({
                        'task': 'anomaly_detection',
                        'model': model.model_name,
                        'run': run_idx,
                        'sample_id': idx,
                        'kpi_type': row['kpi_type'],
                        'true_anomaly': row['is_anomaly'],
                        'value': row['value'],
                        'prompt': prompt,
                        'response': result['response'],
                        'latency_seconds': result['latency_seconds'],
                        'tokens_used': result['tokens_used'],
                        'success': result['success'],
                        'error': result['error'],
                        'timestamp': datetime.now().isoformat()
                    })
                    
                    time.sleep(0.5)  # Rate limiting
        
        self.results.extend(task_results)
        return task_results
    
    def evaluate_trace_analysis(self, dataset_path: str, n_samples: int = 100, n_runs: int = 3):
        """Evaluate models on distributed trace analysis task"""
        logger.info(f"Evaluating trace analysis on {dataset_path}")
        
        # Load dataset
        df = pd.read_csv(dataset_path)
        
        # Sample traces with errors
        samples = df.sample(n=min(n_samples, len(df)), random_state=42)
        
        task_results = []
        
        for model in self.models:
            logger.info(f"Evaluating model: {model.model_name}")
            
            for run_idx in range(n_runs):
                logger.info(f"  Run {run_idx + 1}/{n_runs}")
                
                for idx, row in samples.iterrows():
                    # Create prompt
                    prompt = PromptTemplate.TRACE_ANALYSIS.format(
                        service_name=row['service_name'],
                        operation_name=row['operation_name'],
                        duration_ms=row['duration_ms'],
                        status=row['status'],
                        fault_scenario=row['fault_scenario']
                    )
                    
                    # Generate response
                    result = model.generate(prompt)
                    
                    # Store result
                    task_results.append({
                        'task': 'trace_analysis',
                        'model': model.model_name,
                        'run': run_idx,
                        'sample_id': row['trace_id'],
                        'span_id': row['span_id'],
                        'service': row['service_name'],
                        'true_status': row['status'],
                        'true_fault_scenario': row['fault_scenario'],
                        'prompt': prompt,
                        'response': result['response'],
                        'latency_seconds': result['latency_seconds'],
                        'tokens_used': result['tokens_used'],
                        'success': result['success'],
                        'error': result['error'],
                        'timestamp': datetime.now().isoformat()
                    })
                    
                    time.sleep(0.5)  # Rate limiting
        
        self.results.extend(task_results)
        return task_results
    
    def save_results(self, filename: Optional[str] = None):
        """Save evaluation results"""
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'evaluation_results_{timestamp}.csv'
        
        output_path = self.output_dir / filename
        df = pd.DataFrame(self.results)
        df.to_csv(output_path, index=False)
        logger.info(f"Results saved to {output_path}")
        
        # Also save as JSON for richer structure
        json_path = output_path.with_suffix('.json')
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        logger.info(f"Results saved to {json_path}")
        
        return output_path
    
    def get_summary_statistics(self) -> pd.DataFrame:
        """Get summary statistics for evaluation"""
        if not self.results:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.results)
        
        summary = df.groupby(['task', 'model']).agg({
            'latency_seconds': ['mean', 'std', 'min', 'max'],
            'tokens_used': ['mean', 'sum'],
            'success': ['sum', 'count']
        }).round(3)
        
        return summary


# Example usage
if __name__ == '__main__':
    print("LLM Evaluation Framework")
    print("=" * 50)
    
    # Note: Uncomment and configure when ready to run evaluations
    # models = [
    #     ModelFactory.create_model('gpt-4', api_key='your-key'),
    #     ModelFactory.create_model('llama-3', model_path='path-to-model'),
    #     ModelFactory.create_model('phi-3', model_path='path-to-model')
    # ]
    
    # evaluator = LLMEvaluator(models)
    # evaluator.evaluate_fault_classification('datasets/synthetic_telecom_faults.csv', n_samples=10, n_runs=1)
    # evaluator.save_results()
    # print(evaluator.get_summary_statistics())
    
    print("\nFramework ready. Configure models and run evaluations.")
