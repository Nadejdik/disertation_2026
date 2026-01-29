
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import random
import uuid

np.random.seed(42)
random.seed(42)


class MicroservicesFaultGenerator:
    """Generate synthetic microservices fault data"""
    
    def __init__(self, n_traces=1000):
        self.n_traces = n_traces
        self.services = {
            'gateway': {'downstream': ['auth', 'catalog', 'order']},
            'auth': {'downstream': ['user-db']},
            'catalog': {'downstream': ['catalog-db', 'cache']},
            'order': {'downstream': ['payment', 'inventory', 'shipping']},
            'payment': {'downstream': ['payment-gateway', 'fraud-detection']},
            'inventory': {'downstream': ['inventory-db']},
            'shipping': {'downstream': ['logistics-api']},
            'user-db': {'downstream': []},
            'catalog-db': {'downstream': []},
            'cache': {'downstream': []},
            'payment-gateway': {'downstream': []},
            'fraud-detection': {'downstream': []},
            'inventory-db': {'downstream': []},
            'logistics-api': {'downstream': []}
        }
        
        self.fault_scenarios = [
            'normal', 'timeout', 'cascading_failure', 'resource_exhaustion',
            'network_latency', 'database_deadlock', 'circuit_breaker_open'
        ]
        
    def generate_dataset(self):
        """Generate complete microservices traces and logs"""
        traces = []
        logs = []
        metrics = []
        
        for i in range(self.n_traces):
            trace_data = self._generate_trace(i)
            traces.extend(trace_data['spans'])
            logs.extend(trace_data['logs'])
            metrics.extend(trace_data['metrics'])
        
        traces_df = pd.DataFrame(traces)
        logs_df = pd.DataFrame(logs)
        metrics_df = pd.DataFrame(metrics)
        
        return {
            'traces': traces_df,
            'logs': logs_df,
            'metrics': metrics_df
        }
    
    def _generate_trace(self, trace_idx):
        """Generate a single distributed trace with spans"""
        trace_id = str(uuid.uuid4())
        timestamp = datetime.now() - timedelta(days=random.randint(0, 30))
        
        # Select fault scenario
        fault_scenario = np.random.choice(
            self.fault_scenarios,
            p=[0.7, 0.08, 0.05, 0.05, 0.05, 0.04, 0.03]  # Normal is most common
        )
        
        # Start with gateway
        spans = []
        logs = []
        metrics = []
        
        root_span = self._create_span(
            trace_id=trace_id,
            span_id=str(uuid.uuid4()),
            parent_id=None,
            service='gateway',
            operation='http.request',
            timestamp=timestamp,
            fault_scenario=fault_scenario
        )
        
        spans.append(root_span)
        
        # Generate downstream calls
        self._generate_downstream_spans(
            trace_id, root_span, timestamp, fault_scenario, spans, logs, metrics
        )
        
        # Calculate total duration
        if spans:
            max_end_time = max(s['end_time'] for s in spans)
            min_start_time = min(s['start_time'] for s in spans)
            total_duration = (max_end_time - min_start_time).total_seconds() * 1000
        else:
            total_duration = 0
        
        return {
            'spans': spans,
            'logs': logs,
            'metrics': metrics,
            'trace_id': trace_id,
            'total_duration': total_duration,
            'fault_scenario': fault_scenario
        }
    
    def _create_span(self, trace_id, span_id, parent_id, service, operation, timestamp, fault_scenario):
        """Create a single span with fault injection"""
        
        # Base duration based on service
        base_duration = {
            'gateway': 10, 'auth': 50, 'catalog': 30, 'order': 100,
            'payment': 200, 'inventory': 40, 'shipping': 150,
            'user-db': 15, 'catalog-db': 20, 'cache': 5,
            'payment-gateway': 300, 'fraud-detection': 100,
            'inventory-db': 25, 'logistics-api': 180
        }.get(service, 50)
        
        # Apply fault scenario
        if fault_scenario == 'timeout':
            duration = base_duration * random.uniform(20, 50) if random.random() < 0.3 else base_duration
            status = 'error' if duration > base_duration * 10 else 'ok'
            
        elif fault_scenario == 'cascading_failure':
            # Services fail in cascade
            failure_prob = 0.5 if parent_id else 0.2
            duration = base_duration * random.uniform(10, 30) if random.random() < failure_prob else base_duration
            status = 'error' if duration > base_duration * 5 else 'ok'
            
        elif fault_scenario == 'resource_exhaustion':
            duration = base_duration * random.uniform(5, 15)
            status = 'error' if random.random() < 0.2 else 'ok'
            
        elif fault_scenario == 'network_latency':
            duration = base_duration * random.uniform(3, 8)
            status = 'ok'
            
        elif fault_scenario == 'database_deadlock':
            if 'db' in service:
                duration = base_duration * random.uniform(10, 40)
                status = 'error' if random.random() < 0.4 else 'ok'
            else:
                duration = base_duration
                status = 'ok'
                
        elif fault_scenario == 'circuit_breaker_open':
            duration = 1  # Fast fail
            status = 'error' if random.random() < 0.6 else 'ok'
            
        else:  # normal
            duration = base_duration * random.uniform(0.8, 1.5)
            status = 'error' if random.random() < 0.01 else 'ok'
        
        # Add jitter
        duration += np.random.normal(0, duration * 0.1)
        duration = max(1, duration)
        
        start_time = timestamp
        end_time = start_time + timedelta(milliseconds=duration)
        
        return {
            'trace_id': trace_id,
            'span_id': span_id,
            'parent_span_id': parent_id,
            'service_name': service,
            'operation_name': operation,
            'start_time': start_time,
            'end_time': end_time,
            'duration_ms': round(duration, 2),
            'status': status,
            'fault_scenario': fault_scenario,
            'http_status_code': self._get_http_status(status),
            'error_message': self._get_error_message(service, status, fault_scenario)
        }
    
    def _generate_downstream_spans(self, trace_id, parent_span, timestamp, fault_scenario, spans, logs, metrics):
        """Recursively generate downstream service calls"""
        service = parent_span['service_name']
        downstream = self.services.get(service, {}).get('downstream', [])
        
        for down_service in downstream:
            # Some calls might be skipped if parent failed
            if parent_span['status'] == 'error' and random.random() < 0.3:
                continue
            
            span_id = str(uuid.uuid4())
            operation = f'{service}.call_{down_service}'
            
            # Add slight delay from parent start
            call_timestamp = parent_span['start_time'] + timedelta(milliseconds=random.uniform(5, 20))
            
            span = self._create_span(
                trace_id=trace_id,
                span_id=span_id,
                parent_id=parent_span['span_id'],
                service=down_service,
                operation=operation,
                timestamp=call_timestamp,
                fault_scenario=fault_scenario
            )
            
            spans.append(span)
            
            # Generate log for this span
            log_entry = {
                'trace_id': trace_id,
                'span_id': span_id,
                'service_name': down_service,
                'timestamp': span['start_time'],
                'level': 'ERROR' if span['status'] == 'error' else 'INFO',
                'message': self._generate_log_message(down_service, span['status'], fault_scenario)
            }
            logs.append(log_entry)
            
            # Generate metrics
            metric_entry = {
                'trace_id': trace_id,
                'service_name': down_service,
                'timestamp': span['start_time'],
                'metric_name': 'request_duration_ms',
                'value': span['duration_ms'],
                'status': span['status']
            }
            metrics.append(metric_entry)
            
            # Recurse to downstream
            self._generate_downstream_spans(
                trace_id, span, call_timestamp, fault_scenario, spans, logs, metrics
            )
    
    def _get_http_status(self, status):
        """Get HTTP status code"""
        if status == 'error':
            return random.choice([500, 502, 503, 504, 408])
        return 200
    
    def _get_error_message(self, service, status, fault_scenario):
        """Generate error message"""
        if status != 'error':
            return None
        
        messages = {
            'timeout': f'{service}: Connection timeout after 30s',
            'cascading_failure': f'{service}: Upstream service unavailable',
            'resource_exhaustion': f'{service}: Resource limit exceeded, throttling requests',
            'network_latency': f'{service}: High network latency detected',
            'database_deadlock': f'{service}: Database deadlock detected, transaction aborted',
            'circuit_breaker_open': f'{service}: Circuit breaker open, fast-failing requests'
        }
        
        return messages.get(fault_scenario, f'{service}: Internal server error')
    
    def _generate_log_message(self, service, status, fault_scenario):
        """Generate log message"""
        if status == 'error':
            return self._get_error_message(service, status, fault_scenario)
        return f'{service}: Request processed successfully'


def main():
    """Generate and save microservices fault dataset"""
    print("Generating Microservices Fault Dataset...")
    
    generator = MicroservicesFaultGenerator(n_traces=1000)
    datasets = generator.generate_dataset()
    
    # Save traces
    traces_path = 'datasets/microservices_traces.csv'
    datasets['traces'].to_csv(traces_path, index=False)
    print(f"✓ Saved traces: {traces_path}")
    
    # Save logs
    logs_path = 'datasets/microservices_logs.csv'
    datasets['logs'].to_csv(logs_path, index=False)
    print(f"✓ Saved logs: {logs_path}")
    
    # Save metrics
    metrics_path = 'datasets/microservices_metrics.csv'
    datasets['metrics'].to_csv(metrics_path, index=False)
    print(f"✓ Saved metrics: {metrics_path}")
    
    # Statistics
    print("\n=== Dataset Statistics ===")
    print(f"Total traces: {datasets['traces']['trace_id'].nunique()}")
    print(f"Total spans: {len(datasets['traces'])}")
    print(f"Total logs: {len(datasets['logs'])}")
    print(f"Total metrics: {len(datasets['metrics'])}")
    
    print(f"\nFault scenario distribution:")
    print(datasets['traces'].groupby('fault_scenario')['trace_id'].nunique())
    
    print(f"\nStatus distribution:")
    print(datasets['traces']['status'].value_counts())
    
    # Save metadata
    metadata = {
        'name': 'Microservices Fault Dataset',
        'version': '1.0',
        'traces': datasets['traces']['trace_id'].nunique(),
        'spans': len(datasets['traces']),
        'logs': len(datasets['logs']),
        'services': list(datasets['traces']['service_name'].unique()),
        'fault_scenarios': list(datasets['traces']['fault_scenario'].unique()),
        'description': 'Distributed traces, logs, and metrics for telecom-like microservices architecture'
    }
    
    with open('datasets/microservices_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("\n✓ Microservices dataset generation complete!")


if __name__ == '__main__':
    main()
