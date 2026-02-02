"""
Exploratory Data Analysis (EDA) for Dissertation Datasets
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

# Set plot style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Dataset path
KPI_DATASET = Path(r"C:\Users\Leore\Downloads\KPI-Anomaly-Detection-master\KPI-Anomaly-Detection-master\Preliminary_dataset\train.csv")

def print_section(title):
    """Print formatted section header"""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)

def analyze_kpi_dataset():
    """Perform comprehensive EDA on KPI Anomaly Detection dataset"""
    print_section("EXPLORATORY DATA ANALYSIS: KPI ANOMALY DETECTION DATASET")
    
    # Load dataset
    print("\n📂 Loading dataset...")
    print(f"Path: {KPI_DATASET}")
    
    if not KPI_DATASET.exists():
        print(f"❌ ERROR: Dataset not found at {KPI_DATASET}")
        return None
    
    df = pd.read_csv(KPI_DATASET)
    print(f"✓ Dataset loaded successfully!")
    
    # Basic Information
    print_section("1. DATASET OVERVIEW")
    print(f"Dataset Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print(f"\nColumn Names: {df.columns.tolist()}")
    print(f"\nData Types:\n{df.dtypes}")
    
    # First rows
    print_section("2. SAMPLE DATA (First 10 Rows)")
    print(df.head(10).to_string())
    
    # Statistical Summary
    print_section("3. STATISTICAL SUMMARY")
    print(df.describe().to_string())
    
    # Missing Values
    print_section("4. DATA QUALITY CHECK")
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print("Missing Values:")
        print(missing[missing > 0].to_string())
    else:
        print("✓ No missing values detected!")
    
    print(f"\nDuplicate Rows: {df.duplicated().sum():,}")
    
    # Anomaly Distribution
    print_section("5. TARGET VARIABLE ANALYSIS")
    if 'label' in df.columns:
        label_col = 'label'
    elif 'anomaly' in df.columns:
        label_col = 'anomaly'
    else:
        label_col = df.columns[-1]  # Assume last column is label
    
    print(f"Target Column: '{label_col}'")
    print(f"\nLabel Distribution:")
    value_counts = df[label_col].value_counts()
    print(value_counts.to_string())
    
    if len(value_counts) == 2:
        anomaly_rate = value_counts.iloc[1] / len(df) * 100 if value_counts.iloc[1] < value_counts.iloc[0] else value_counts.iloc[0] / len(df) * 100
        print(f"\nAnomaly Rate: {anomaly_rate:.2f}%")
        print(f"Normal Rate: {100-anomaly_rate:.2f}%")
        print(f"Class Imbalance Ratio: 1:{value_counts.iloc[0]/value_counts.iloc[1]:.2f}")
    
    # Feature Analysis
    print_section("6. FEATURE STATISTICS")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    print(f"Numeric Features: {len(numeric_cols)}")
    
    for col in numeric_cols:
        if col != label_col:
            print(f"\n{col}:")
            print(f"  Range: [{df[col].min():.4f}, {df[col].max():.4f}]")
            print(f"  Mean: {df[col].mean():.4f}")
            print(f"  Std Dev: {df[col].std():.4f}")
            print(f"  Median: {df[col].median():.4f}")
            print(f"  Skewness: {df[col].skew():.4f}")
            print(f"  Kurtosis: {df[col].kurtosis():.4f}")
    
    # Correlation Analysis
    print_section("7. CORRELATION ANALYSIS")
    if len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols].corr()
        print("Correlation Matrix:")
        print(corr_matrix.to_string())
        
        # Find highly correlated features
        print("\n🔍 Highly Correlated Features (|r| > 0.7):")
        high_corr = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > 0.7:
                    high_corr.append((
                        corr_matrix.columns[i],
                        corr_matrix.columns[j],
                        corr_matrix.iloc[i, j]
                    ))
        
        if high_corr:
            for feat1, feat2, corr_val in high_corr:
                print(f"  {feat1} ↔ {feat2}: {corr_val:.3f}")
        else:
            print("  No highly correlated features found.")
    
    # Unique Values
    print_section("8. UNIQUE VALUES PER FEATURE")
    for col in df.columns:
        n_unique = df[col].nunique()
        print(f"{col}: {n_unique:,} unique values", end="")
        if n_unique <= 10:
            print(f" → {df[col].unique()[:10].tolist()}")
        else:
            print()
    
    # Time-series Analysis (if timestamp exists)
    print_section("9. TEMPORAL ANALYSIS")
    timestamp_cols = [col for col in df.columns if 'time' in col.lower() or 'date' in col.lower()]
    if timestamp_cols:
        print(f"Timestamp columns found: {timestamp_cols}")
        for ts_col in timestamp_cols:
            try:
                df[ts_col] = pd.to_datetime(df[ts_col])
                print(f"\n{ts_col}:")
                print(f"  Date Range: {df[ts_col].min()} to {df[ts_col].max()}")
                print(f"  Duration: {(df[ts_col].max() - df[ts_col].min()).days} days")
            except:
                print(f"  Could not parse as datetime")
    else:
        print("No timestamp columns detected")
    
    # Export Summary Report
    print_section("10. GENERATING SUMMARY REPORT")
    
    summary_report = {
        "dataset": "KPI Anomaly Detection - Preliminary Dataset",
        "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
        "columns": df.columns.tolist(),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing_values": {col: int(count) for col, count in missing[missing > 0].items()},
        "label_distribution": value_counts.to_dict() if label_col in df.columns else {},
        "numeric_features": len(numeric_cols),
        "memory_mb": float(df.memory_usage(deep=True).sum() / 1024**2)
    }
    
    output_file = Path("datasets/processed/eda_report.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(summary_report, f, indent=2)
    
    print(f"✓ Summary report saved to: {output_file}")
    
    print_section("EDA COMPLETE")
    print("✅ Exploratory Data Analysis finished successfully!")
    print(f"\nDataset is ready for LLM evaluation experiments.")
    print(f"Total samples: {len(df):,}")
    print(f"Features: {len(df.columns)}")
    
    return df

def main():
    """Run EDA analysis"""
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║  DISSERTATION PROJECT - EXPLORATORY DATA ANALYSIS             ║
    ║  LLM Evaluation for Fault Detection in Telecom Systems        ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)
    
    df = analyze_kpi_dataset()
    
    if df is not None:
        print("\n" + "="*80)
        print("📊 Next Steps:")
        print("  1. Review the EDA findings above")
        print("  2. Check datasets/processed/eda_report.json for summary")
        print("  3. Run: python run_real_experiments.py (to evaluate LLMs)")
        print("  4. Open: notebooks/dissertation_experiments.ipynb (for visualizations)")
        print("="*80)

if __name__ == "__main__":
    main()
