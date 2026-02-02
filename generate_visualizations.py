"""
Generate EDA Visualizations for Dissertation Dataset
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)

# Load dataset (sample for visualization)
print("Loading dataset for visualization...")
KPI_DATASET = Path(r"C:\Users\Leore\Downloads\KPI-Anomaly-Detection-master\KPI-Anomaly-Detection-master\Preliminary_dataset\train.csv")

# Read a sample for faster visualization
df = pd.read_csv(KPI_DATASET, nrows=100000)
print(f"Loaded {len(df):,} samples for visualization")

# Create figure with subplots
fig = plt.figure(figsize=(16, 12))

# 1. Label Distribution
ax1 = plt.subplot(3, 3, 1)
label_counts = df['label'].value_counts()
colors = ['#2ecc71', '#e74c3c']
ax1.bar(['Normal', 'Anomaly'], label_counts.values, color=colors, alpha=0.7, edgecolor='black')
ax1.set_title('Label Distribution', fontsize=14, fontweight='bold')
ax1.set_ylabel('Count')
for i, v in enumerate(label_counts.values):
    ax1.text(i, v + 100, f'{v:,}\n({v/len(df)*100:.2f}%)', ha='center', va='bottom', fontweight='bold')

# 2. KPI Value Distribution (Log Scale)
ax2 = plt.subplot(3, 3, 2)
# Filter positive values for log scale
positive_values = df[df['value'] > 0]['value']
ax2.hist(np.log10(positive_values + 1), bins=50, color='steelblue', alpha=0.7, edgecolor='black')
ax2.set_title('KPI Value Distribution (Log Scale)', fontsize=14, fontweight='bold')
ax2.set_xlabel('log10(Value + 1)')
ax2.set_ylabel('Frequency')

# 3. Anomaly vs Normal Values (Box Plot)
ax3 = plt.subplot(3, 3, 3)
df_sample = df[df['value'].between(df['value'].quantile(0.01), df['value'].quantile(0.99))]
df_sample['label_str'] = df_sample['label'].map({0: 'Normal', 1: 'Anomaly'})
sns.boxplot(x='label_str', y='value', data=df_sample, ax=ax3, palette={'Normal': '#2ecc71', 'Anomaly': '#e74c3c'})
ax3.set_title('Value Distribution by Label', fontsize=14, fontweight='bold')
ax3.set_xlabel('Label')
ax3.set_ylabel('KPI Value')

# 4. KPI ID Distribution
ax4 = plt.subplot(3, 3, 4)
kpi_counts = df['KPI ID'].value_counts().head(10)
ax4.barh(range(len(kpi_counts)), kpi_counts.values, color='coral', alpha=0.7, edgecolor='black')
ax4.set_yticks(range(len(kpi_counts)))
ax4.set_yticklabels(kpi_counts.index, fontsize=8)
ax4.set_title('Top 10 KPI IDs by Frequency', fontsize=14, fontweight='bold')
ax4.set_xlabel('Count')
ax4.invert_yaxis()

# 5. Time Series Sample (First KPI)
ax5 = plt.subplot(3, 3, 5)
first_kpi = df['KPI ID'].iloc[0]
kpi_data = df[df['KPI ID'] == first_kpi].head(500)
ax5.plot(kpi_data['timestamp'], kpi_data['value'], linewidth=1, color='steelblue', alpha=0.7)
anomalies = kpi_data[kpi_data['label'] == 1]
ax5.scatter(anomalies['timestamp'], anomalies['value'], color='red', s=50, zorder=5, label='Anomaly')
ax5.set_title(f'Time Series Sample (KPI: {first_kpi[:8]}...)', fontsize=14, fontweight='bold')
ax5.set_xlabel('Timestamp')
ax5.set_ylabel('Value')
ax5.legend()

# 6. Anomaly Rate by KPI
ax6 = plt.subplot(3, 3, 6)
kpi_anomaly = df.groupby('KPI ID')['label'].agg(['sum', 'count'])
kpi_anomaly['rate'] = kpi_anomaly['sum'] / kpi_anomaly['count'] * 100
top_anomaly_kpis = kpi_anomaly.nlargest(10, 'rate')
ax6.barh(range(len(top_anomaly_kpis)), top_anomaly_kpis['rate'].values, color='crimson', alpha=0.7, edgecolor='black')
ax6.set_yticks(range(len(top_anomaly_kpis)))
ax6.set_yticklabels(top_anomaly_kpis.index, fontsize=8)
ax6.set_title('Top 10 KPIs by Anomaly Rate', fontsize=14, fontweight='bold')
ax6.set_xlabel('Anomaly Rate (%)')
ax6.invert_yaxis()

# 7. Value Statistics
ax7 = plt.subplot(3, 3, 7)
stats_data = [
    df['value'].min(),
    df['value'].quantile(0.25),
    df['value'].median(),
    df['value'].quantile(0.75),
    df['value'].max()
]
labels = ['Min', 'Q1', 'Median', 'Q3', 'Max']
colors_stats = plt.cm.viridis(np.linspace(0, 1, 5))
ax7.bar(labels, np.log10(np.abs(stats_data) + 1), color=colors_stats, alpha=0.7, edgecolor='black')
ax7.set_title('Value Statistics (Log Scale)', fontsize=14, fontweight='bold')
ax7.set_ylabel('log10(|Value| + 1)')
for i, (label, val) in enumerate(zip(labels, stats_data)):
    ax7.text(i, np.log10(abs(val) + 1) + 0.1, f'{val:.2e}', ha='center', va='bottom', fontsize=8, rotation=45)

# 8. Correlation Heatmap
ax8 = plt.subplot(3, 3, 8)
corr = df[['timestamp', 'value', 'label']].corr()
sns.heatmap(corr, annot=True, fmt='.3f', cmap='coolwarm', center=0, 
            square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax8)
ax8.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold')

# 9. Summary Statistics Table
ax9 = plt.subplot(3, 3, 9)
ax9.axis('off')
summary_text = f"""
DATASET SUMMARY
{'='*40}
Total Samples: {len(df):,}
Features: {df.shape[1]}
Memory: {df.memory_usage(deep=True).sum()/1024**2:.1f} MB

ANOMALY DETECTION
{'='*40}
Normal: {(df['label']==0).sum():,} ({(df['label']==0).mean()*100:.2f}%)
Anomalies: {(df['label']==1).sum():,} ({(df['label']==1).mean()*100:.2f}%)
Imbalance Ratio: 1:{(df['label']==0).sum()/(df['label']==1).sum():.1f}

KPI METRICS
{'='*40}
Unique KPIs: {df['KPI ID'].nunique()}
Value Range: [{df['value'].min():.2f}, {df['value'].max():.2e}]
Mean Value: {df['value'].mean():.2e}
Std Dev: {df['value'].std():.2e}

DATA QUALITY
{'='*40}
Missing Values: {df.isnull().sum().sum()}
Duplicates: {df.duplicated().sum()}
"""
ax9.text(0.1, 0.9, summary_text, transform=ax9.transAxes, 
         fontsize=10, verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.suptitle('KPI Anomaly Detection Dataset - Exploratory Data Analysis', 
             fontsize=18, fontweight='bold', y=0.995)
plt.tight_layout()

# Save figure
output_path = Path("datasets/processed/eda_visualizations.png")
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\n✓ Visualizations saved to: {output_path}")

# Also create a detailed analysis per KPI
print("\nGenerating per-KPI analysis...")
kpi_analysis = df.groupby('KPI ID').agg({
    'value': ['count', 'mean', 'std', 'min', 'max'],
    'label': ['sum', 'mean']
}).round(4)
kpi_analysis.columns = ['_'.join(col).strip() for col in kpi_analysis.columns.values]
kpi_analysis.columns = ['samples', 'value_mean', 'value_std', 'value_min', 'value_max', 'anomalies', 'anomaly_rate']
kpi_analysis = kpi_analysis.sort_values('anomaly_rate', ascending=False)

output_csv = Path("datasets/processed/per_kpi_analysis.csv")
kpi_analysis.to_csv(output_csv)
print(f"✓ Per-KPI analysis saved to: {output_csv}")

print("\n" + "="*80)
print("📊 EDA VISUALIZATION COMPLETE!")
print("="*80)
print(f"Generated files:")
print(f"  1. {output_path}")
print(f"  2. {output_csv}")
print(f"  3. datasets/processed/eda_report.json")
print("\nReview these files for comprehensive dataset insights!")
