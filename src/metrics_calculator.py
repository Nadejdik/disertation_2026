"""
Metrics Collection and Analysis Module
Calculates evaluation metrics for LLM performance
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import json
import re
from collections import defaultdict
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MetricsCalculator:
    """Calculate various metrics for LLM evaluation"""
    
    def __init__(self):
        self.metrics = defaultdict(list)
        
    def extract_classification_from_response(self, response: str, possible_classes: List[str]) -> Optional[str]:
        """Extract classification from LLM response using pattern matching"""
        if not response:
            return None
        
        response_lower = response.lower()
        
        # Try to find exact matches first
        for cls in possible_classes:
            if cls.lower() in response_lower:
                return cls
        
        return None
    
    def extract_binary_decision(self, response: str) -> Optional[int]:
        """Extract binary decision (yes/no, anomaly/normal) from response"""
        if not response:
            return None
        
        response_lower = response.lower()
        
        # Positive indicators
        positive_patterns = ['yes', 'anomaly', 'abnormal', 'issue', 'problem', 'fault', 'error']
        negative_patterns = ['no', 'normal', 'expected', 'typical', 'healthy']
        
        positive_score = sum(1 for pattern in positive_patterns if pattern in response_lower)
        negative_score = sum(1 for pattern in negative_patterns if pattern in response_lower)
        
        if positive_score > negative_score:
            return 1
        elif negative_score > positive_score:
            return 0
        else:
            return None  # Ambiguous
    
    def extract_confidence_score(self, response: str) -> Optional[float]:
        """Extract confidence score from response"""
        if not response:
            return None
        
        # Look for percentage patterns like "95%", "0.95", "95 percent"
        patterns = [
            r'(\d+(?:\.\d+)?)\s*%',  # 95%
            r'confidence[:\s]+(\d+(?:\.\d+)?)',  # confidence: 95
            r'(\d+(?:\.\d+)?)\s*percent',  # 95 percent
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response.lower())
            if match:
                value = float(match.group(1))
                if value > 1:  # Assume percentage
                    return value / 100
                return value
        
        return None
    
    def calculate_fault_classification_metrics(self, results_df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate metrics for fault classification task"""
        logger.info("Calculating fault classification metrics...")
        
        metrics = {}
        
        # Extract predicted fault types from responses
        fault_types = results_df['true_fault_type'].unique().tolist()
        
        results_df['predicted_fault_type'] = results_df['response'].apply(
            lambda x: self.extract_classification_from_response(x, fault_types)
        )
        
        # Remove rows where prediction couldn't be extracted
        valid_predictions = results_df.dropna(subset=['predicted_fault_type'])
        
        if len(valid_predictions) == 0:
            logger.warning("No valid predictions extracted for fault classification")
            return {'error': 'No valid predictions'}
        
        # Calculate accuracy by model
        for model in valid_predictions['model'].unique():
            model_data = valid_predictions[valid_predictions['model'] == model]
            
            y_true = model_data['true_fault_type']
            y_pred = model_data['predicted_fault_type']
            
            accuracy = accuracy_score(y_true, y_pred)
            precision, recall, f1, support = precision_recall_fscore_support(
                y_true, y_pred, average='weighted', zero_division=0
            )
            
            metrics[model] = {
                'accuracy': round(accuracy, 3),
                'precision': round(precision, 3),
                'recall': round(recall, 3),
                'f1_score': round(f1, 3),
                'samples_evaluated': len(model_data),
                'avg_latency': round(model_data['latency_seconds'].mean(), 3),
                'avg_tokens': round(model_data['tokens_used'].mean(), 1),
                'success_rate': round(model_data['success'].mean(), 3)
            }
        
        return metrics
    
    def calculate_anomaly_detection_metrics(self, results_df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate metrics for anomaly detection task"""
        logger.info("Calculating anomaly detection metrics...")
        
        metrics = {}
        
        # Extract binary predictions from responses
        results_df['predicted_anomaly'] = results_df['response'].apply(
            self.extract_binary_decision
        )
        
        results_df['confidence'] = results_df['response'].apply(
            self.extract_confidence_score
        )
        
        # Remove rows where prediction couldn't be extracted
        valid_predictions = results_df.dropna(subset=['predicted_anomaly'])
        
        if len(valid_predictions) == 0:
            logger.warning("No valid predictions extracted for anomaly detection")
            return {'error': 'No valid predictions'}
        
        # Calculate metrics by model
        for model in valid_predictions['model'].unique():
            model_data = valid_predictions[valid_predictions['model'] == model]
            
            y_true = model_data['true_anomaly']
            y_pred = model_data['predicted_anomaly']
            
            accuracy = accuracy_score(y_true, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average='binary', zero_division=0
            )
            
            # Confusion matrix
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            
            # Calculate additional metrics
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
            
            metrics[model] = {
                'accuracy': round(accuracy, 3),
                'precision': round(precision, 3),
                'recall': round(recall, 3),
                'f1_score': round(f1, 3),
                'specificity': round(specificity, 3),
                'false_positive_rate': round(false_positive_rate, 3),
                'true_positives': int(tp),
                'true_negatives': int(tn),
                'false_positives': int(fp),
                'false_negatives': int(fn),
                'samples_evaluated': len(model_data),
                'avg_latency': round(model_data['latency_seconds'].mean(), 3),
                'avg_tokens': round(model_data['tokens_used'].mean(), 1),
                'avg_confidence': round(model_data['confidence'].mean(), 3) if model_data['confidence'].notna().any() else None,
                'success_rate': round(model_data['success'].mean(), 3)
            }
        
        return metrics
    
    def calculate_trace_analysis_metrics(self, results_df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate metrics for trace analysis task"""
        logger.info("Calculating trace analysis metrics...")
        
        metrics = {}
        
        # Extract status predictions
        statuses = ['ok', 'error']
        results_df['predicted_status'] = results_df['response'].apply(
            lambda x: self.extract_classification_from_response(x, statuses)
        )
        
        valid_predictions = results_df.dropna(subset=['predicted_status'])
        
        if len(valid_predictions) == 0:
            logger.warning("No valid predictions extracted for trace analysis")
            return {'error': 'No valid predictions'}
        
        # Calculate metrics by model
        for model in valid_predictions['model'].unique():
            model_data = valid_predictions[valid_predictions['model'] == model]
            
            y_true = model_data['true_status']
            y_pred = model_data['predicted_status']
            
            accuracy = accuracy_score(y_true, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average='binary', pos_label='error', zero_division=0
            )
            
            metrics[model] = {
                'accuracy': round(accuracy, 3),
                'precision': round(precision, 3),
                'recall': round(recall, 3),
                'f1_score': round(f1, 3),
                'samples_evaluated': len(model_data),
                'avg_latency': round(model_data['latency_seconds'].mean(), 3),
                'avg_tokens': round(model_data['tokens_used'].mean(), 1),
                'success_rate': round(model_data['success'].mean(), 3)
            }
        
        return metrics
    
    def calculate_cross_task_comparison(self, results_df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate cross-task comparison metrics"""
        logger.info("Calculating cross-task comparison...")
        
        comparison = {}
        
        for model in results_df['model'].unique():
            model_data = results_df[results_df['model'] == model]
            
            comparison[model] = {
                'total_requests': len(model_data),
                'successful_requests': model_data['success'].sum(),
                'success_rate': round(model_data['success'].mean(), 3),
                'avg_latency_all_tasks': round(model_data['latency_seconds'].mean(), 3),
                'total_tokens_used': model_data['tokens_used'].sum(),
                'avg_tokens_per_request': round(model_data['tokens_used'].mean(), 1),
                'tasks_evaluated': model_data['task'].unique().tolist()
            }
            
            # Per-task breakdown
            for task in model_data['task'].unique():
                task_data = model_data[model_data['task'] == task]
                comparison[model][f'{task}_latency'] = round(task_data['latency_seconds'].mean(), 3)
                comparison[model][f'{task}_tokens'] = round(task_data['tokens_used'].mean(), 1)
                comparison[model][f'{task}_success_rate'] = round(task_data['success'].mean(), 3)
        
        return comparison
    
    def generate_comparison_matrix(self, metrics_by_task: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """Generate comparison matrix (models vs tasks)"""
        
        # Flatten metrics into matrix format
        rows = []
        
        for task_name, task_metrics in metrics_by_task.items():
            if 'error' in task_metrics:
                continue
                
            for model_name, model_metrics in task_metrics.items():
                row = {
                    'task': task_name,
                    'model': model_name,
                    **model_metrics
                }
                rows.append(row)
        
        df = pd.DataFrame(rows)
        return df
    
    def save_metrics_report(self, metrics: Dict[str, Any], output_path: str):
        """Save metrics report to file"""
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"Metrics report saved to {output_path}")


# Example usage
if __name__ == '__main__':
    print("Metrics Calculation Module")
    print("=" * 50)
    
    # Example: Load results and calculate metrics
    # results_df = pd.read_csv('results/evaluation_results.csv')
    # calculator = MetricsCalculator()
    # metrics = calculator.calculate_anomaly_detection_metrics(results_df)
    # print(json.dumps(metrics, indent=2))
    
    print("\nMetrics calculator ready.")
