
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

np.random.seed(42)


class KPIAnomalyGenerator:
    """Generate synthetic KPI time-series with anomalies"""
    
    def __init__(self, n_timestamps=5000, anomaly_ratio=0.05):
        self.n_timestamps = n_timestamps
        self.anomaly_ratio = anomaly_ratio
        self.kpi_types = ['latency', 'throughput', 'error_rate', 'cpu_usage', 'memory_usage']
        
    def generate_dataset(self):
        """Generate complete KPI anomaly dataset"""
        all_data = []
        
        for kpi_type in self.kpi_types:
            df = self._generate_kpi_series(kpi_type)
            all_data.append(df)
        
        # Combine all KPIs
        combined_df = pd.concat(all_data, ignore_index=True)
        return combined_df
    
    def _generate_kpi_series(self, kpi_type):
        """Generate time-series for a specific KPI"""
        start_time = datetime.now() - timedelta(days=30)
        timestamps = [start_time + timedelta(minutes=i*5) for i in range(self.n_timestamps)]
        
        # Generate base signal with trend and seasonality
        t = np.arange(self.n_timestamps)
        
        if kpi_type == 'latency':
            # Base latency: 50-200ms with daily pattern
            base = 100 + 50 * np.sin(2 * np.pi * t / (24 * 12))  # 24h cycle (5min intervals)
            noise = np.random.normal(0, 10, self.n_timestamps)
            anomaly_magnitude = np.random.uniform(300, 800)
            
        elif kpi_type == 'throughput':
            # Base throughput: 1000-5000 req/s with daily pattern
            base = 3000 + 2000 * np.sin(2 * np.pi * t / (24 * 12))
            noise = np.random.normal(0, 200, self.n_timestamps)
            anomaly_magnitude = np.random.uniform(-2000, -500)  # Drop in throughput
            
        elif kpi_type == 'error_rate':
            # Base error rate: 0.1-1% with occasional spikes
            base = 0.5 + 0.3 * np.sin(2 * np.pi * t / (24 * 12))
            noise = np.random.normal(0, 0.1, self.n_timestamps)
            anomaly_magnitude = np.random.uniform(5, 20)  # Spike in errors
            
        elif kpi_type == 'cpu_usage':
            # Base CPU: 40-70% with load pattern
            base = 55 + 15 * np.sin(2 * np.pi * t / (24 * 12))
            noise = np.random.normal(0, 5, self.n_timestamps)
            anomaly_magnitude = np.random.uniform(25, 45)  # CPU spike
            
        else:  # memory_usage
            # Base memory: 50-75% with slow growth
            base = 60 + 10 * np.sin(2 * np.pi * t / (24 * 12)) + t * 0.001
            noise = np.random.normal(0, 3, self.n_timestamps)
            anomaly_magnitude = np.random.uniform(20, 35)  # Memory leak
        
        # Generate clean signal
        values = base + noise
        values = np.clip(values, 0, None)  # No negative values
        
        # Inject anomalies
        n_anomalies = int(self.n_timestamps * self.anomaly_ratio)
        anomaly_indices = np.random.choice(self.n_timestamps, n_anomalies, replace=False)
        labels = np.zeros(self.n_timestamps, dtype=int)
        
        for idx in anomaly_indices:
            # Create different anomaly patterns
            anomaly_type = np.random.choice(['spike', 'drop', 'gradual', 'level_shift'])
            
            if anomaly_type == 'spike':
                values[idx] += anomaly_magnitude
                labels[idx] = 1
                
            elif anomaly_type == 'drop':
                values[idx] = max(0, values[idx] - abs(anomaly_magnitude))
                labels[idx] = 1
                
            elif anomaly_type == 'gradual':
                # Gradual increase over 10 points
                for j in range(min(10, self.n_timestamps - idx)):
                    values[idx + j] += anomaly_magnitude * (j / 10)
                    labels[idx + j] = 1
                    
            else:  # level_shift
                # Sudden level shift lasting 20 points
                for j in range(min(20, self.n_timestamps - idx)):
                    values[idx + j] += anomaly_magnitude * 0.5
                    labels[idx + j] = 1
        
        # Create DataFrame
        df = pd.DataFrame({
            'timestamp': timestamps,
            'kpi_type': kpi_type,
            'value': values,
            'is_anomaly': labels,
            'service_id': f'SVC_{kpi_type.upper()[:3]}'
        })
        
        return df
    
    def add_context_features(self, df):
        """Add rolling statistics and context features"""
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        for kpi in df['kpi_type'].unique():
            mask = df['kpi_type'] == kpi
            values = df.loc[mask, 'value'].values
            
            # Rolling statistics
            df.loc[mask, 'rolling_mean_12'] = pd.Series(values).rolling(12, min_periods=1).mean().values
            df.loc[mask, 'rolling_std_12'] = pd.Series(values).rolling(12, min_periods=1).std().fillna(0).values
            df.loc[mask, 'rolling_max_24'] = pd.Series(values).rolling(24, min_periods=1).max().values
            df.loc[mask, 'rolling_min_24'] = pd.Series(values).rolling(24, min_periods=1).min().values
        
        return df


def main():
    """Generate and save KPI anomaly dataset"""
    print("Generating KPI Anomaly Detection Dataset...")
    
    generator = KPIAnomalyGenerator(n_timestamps=5000, anomaly_ratio=0.05)
    df = generator.generate_dataset()
    df = generator.add_context_features(df)
    
    # Save dataset
    csv_path = 'datasets/kpi_anomaly_detection.csv'
    df.to_csv(csv_path, index=False)
    print(f"✓ Saved CSV: {csv_path}")
    
    # Save JSON format
    json_path = 'datasets/kpi_anomaly_detection.json'
    df.to_json(json_path, orient='records', indent=2, date_format='iso')
    print(f"✓ Saved JSON: {json_path}")
    
    # Statistics
    print("\n=== Dataset Statistics ===")
    print(f"Total samples: {len(df)}")
    print(f"\nKPI type distribution:")
    print(df['kpi_type'].value_counts())
    print(f"\nAnomaly ratio:")
    print(df['is_anomaly'].value_counts(normalize=True))
    
    # Save metadata
    metadata = {
        'name': 'KPI Anomaly Detection Dataset',
        'version': '1.0',
        'samples': len(df),
        'features': list(df.columns),
        'kpi_types': list(df['kpi_type'].unique()),
        'anomaly_ratio': float(df['is_anomaly'].mean()),
        'time_range': {
            'start': df['timestamp'].min().isoformat(),
            'end': df['timestamp'].max().isoformat()
        },
        'description': 'Time-series KPIs with injected anomalies for telecom operations analysis'
    }
    
    with open('datasets/kpi_anomaly_detection_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    
    print("\n✓ KPI dataset generation complete!")


if __name__ == '__main__':
    main()
