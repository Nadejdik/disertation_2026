"""
Process real telecom/microservices datasets and prepare them for LLM evaluation
"""
import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime

# Dataset paths - Updated to use Downloads folder
KPI_DATASET = Path(r"C:\Users\Leore\Downloads\KPI-Anomaly-Detection-master\KPI-Anomaly-Detection-master\Finals_dataset\phase2_train.csv")
SOCKSHOP_DIR = Path(r"C:\Users\Leore\Downloads\sockshop_stress_test\generated_csvs_4")

# Output paths
OUTPUT_DIR = Path("datasets/processed")
OUTPUT_DIR.mkdir(exist_ok=True)

def process_kpi_dataset(n_samples=1000):
    """
    Process KPI Anomaly Detection dataset
    Real dataset from AIOps challenge
    """
    print(f"Processing KPI dataset from {KPI_DATASET}...")
    
    # Read dataset (168MB, so read in chunks)
    df = pd.read_csv(KPI_DATASET, nrows=n_samples)
    
    print(f"Loaded {len(df)} KPI samples")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nSample data:")
    print(df.head())
    
    # Save processed version
    output_file = OUTPUT_DIR / "kpi_anomaly_real.csv"
    df.to_csv(output_file, index=False)
    
    # Create metadata
    metadata = {
        "dataset": "KPI Anomaly Detection (Real)",
        "source": "AIOps Challenge Dataset",
        "samples": len(df),
        "features": df.columns.tolist(),
        "description": "Real-world KPI time-series data from production systems",
        "processed_date": datetime.now().isoformat()
    }
    
    with open(OUTPUT_DIR / "kpi_anomaly_real_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ Saved to {output_file}")
    return df

def process_sockshop_dataset():
    """
    Process Sock Shop microservices stress test data
    Real distributed system metrics
    """
    print(f"\nProcessing Sock Shop dataset from {SOCKSHOP_DIR}...")
    
    all_data = []
    
    # Process each microservice
    for service_dir in SOCKSHOP_DIR.iterdir():
        if not service_dir.is_dir():
            continue
            
        service_name = service_dir.name
        print(f"  Processing {service_name}...")
        
        # Get first CSV file from each service
        csv_files = list(service_dir.glob("*.csv"))
        if csv_files:
            try:
                df = pd.read_csv(csv_files[0], nrows=100)  # Sample 100 rows per service
                df['service'] = service_name
                df['source_file'] = csv_files[0].name
                all_data.append(df)
                print(f"    ✓ Loaded {len(df)} metrics from {csv_files[0].name}")
            except Exception as e:
                print(f"    ✗ Error reading {csv_files[0]}: {e}")
    
    if all_data:
        # Combine all service data
        combined_df = pd.concat(all_data, ignore_index=True)
        
        print(f"\nCombined {len(combined_df)} total metrics from {len(all_data)} services")
        print(f"Columns: {combined_df.columns.tolist()[:10]}...")  # Show first 10 columns
        
        # Save processed version
        output_file = OUTPUT_DIR / "sockshop_microservices_real.csv"
        combined_df.to_csv(output_file, index=False)
        
        # Create metadata
        metadata = {
            "dataset": "Sock Shop Microservices (Real)",
            "source": "Sock Shop Stress Test",
            "services": [service_dir.name for service_dir in SOCKSHOP_DIR.iterdir() if service_dir.is_dir()],
            "total_metrics": len(combined_df),
            "features": combined_df.columns.tolist(),
            "description": "Real microservices metrics from sock shop application under stress",
            "processed_date": datetime.now().isoformat()
        }
        
        with open(OUTPUT_DIR / "sockshop_microservices_real_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✓ Saved to {output_file}")
        return combined_df
    else:
        print("✗ No data processed")
        return None

def create_llm_evaluation_samples(kpi_df, sockshop_df, n_samples=50):
    """
    Create evaluation samples for LLM testing
    Format: scenario description + data snippet
    """
    print(f"\nCreating LLM evaluation samples...")
    
    samples = []
    
    # KPI Anomaly Detection samples
    if kpi_df is not None:
        for i in range(min(n_samples // 2, len(kpi_df))):
            row = kpi_df.iloc[i]
            sample = {
                "id": f"kpi_{i}",
                "task": "kpi_anomaly_detection",
                "dataset": "real_kpi",
                "data": row.to_dict(),
                "question": "Analyze this KPI time-series data and determine if there is an anomaly. Explain your reasoning."
            }
            samples.append(sample)
    
    # Microservices Fault Detection samples
    if sockshop_df is not None:
        for i in range(min(n_samples // 2, len(sockshop_df))):
            row = sockshop_df.iloc[i]
            sample = {
                "id": f"microservice_{i}",
                "task": "microservice_fault_detection",
                "dataset": "real_sockshop",
                "service": row.get('service', 'unknown'),
                "data": {k: v for k, v in row.to_dict().items() if k not in ['service', 'source_file']},
                "question": "Analyze these microservice metrics and identify any potential faults or anomalies. What could be the root cause?"
            }
            samples.append(sample)
    
    # Save samples
    output_file = OUTPUT_DIR / "llm_evaluation_samples.json"
    with open(output_file, "w") as f:
        json.dump(samples, f, indent=2)
    
    print(f"✓ Created {len(samples)} evaluation samples")
    print(f"  - {sum(1 for s in samples if s['task'] == 'kpi_anomaly_detection')} KPI samples")
    print(f"  - {sum(1 for s in samples if s['task'] == 'microservice_fault_detection')} microservice samples")
    print(f"✓ Saved to {output_file}")
    
    return samples

def main():
    print("=" * 70)
    print("REAL DATASET PROCESSING FOR DISSERTATION")
    print("=" * 70)
    
    # Process KPI dataset
    kpi_df = process_kpi_dataset(n_samples=1000)
    
    # Process Sock Shop dataset
    sockshop_df = process_sockshop_dataset()
    
    # Create LLM evaluation samples
    samples = create_llm_evaluation_samples(kpi_df, sockshop_df, n_samples=100)
    
    print("\n" + "=" * 70)
    print("PROCESSING COMPLETE")
    print("=" * 70)
    print(f"\nProcessed datasets saved to: {OUTPUT_DIR}")
    print("\nNext steps:")
    print("1. Review the processed datasets in datasets/processed/")
    print("2. Run: python run_experiments.py --use-real-data")
    print("3. Or use the Jupyter notebook to explore the real data")

if __name__ == "__main__":
    main()
