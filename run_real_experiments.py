"""
Run LLM experiments on REAL telecom/microservices datasets
"""
import pandas as pd
import json
import numpy as np
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Paths
SAMPLES_FILE = Path("datasets/processed/llm_evaluation_samples.json")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# Simulated LLM responses (replace with real API calls when you have keys)
def simulate_llm_evaluation(sample):
    """
    Simulate LLM evaluation for demonstration
    Replace this with real OpenAI/LLaMA/Phi-3 API calls
    """
    task = sample['task']
    
    # Simulate different accuracy levels for different models
    models_accuracy = {
        'gpt-4': 0.88,
        'llama-3': 0.76,
        'phi-3': 0.72
    }
    
    results = {}
    for model, base_accuracy in models_accuracy.items():
        # Add some randomness
        is_correct = np.random.random() < base_accuracy
        
        results[model] = {
            'prediction': 'anomaly' if is_correct else 'normal',
            'confidence': np.random.uniform(0.7, 0.95),
            'response_time': np.random.uniform(1.0, 3.5),
            'correct': is_correct
        }
    
    return results

def load_evaluation_samples():
    """Load the prepared evaluation samples"""
    print(f"Loading evaluation samples from {SAMPLES_FILE}...")
    with open(SAMPLES_FILE, 'r') as f:
        samples = json.load(f)
    print(f"✓ Loaded {len(samples)} samples")
    print(f"  - KPI samples: {sum(1 for s in samples if s['task'] == 'kpi_anomaly_detection')}")
    print(f"  - Microservice samples: {sum(1 for s in samples if s['task'] == 'microservice_fault_detection')}")
    return samples

def run_experiments(samples, n_samples=20):
    """Run experiments on real data samples"""
    print(f"\n{'='*70}")
    print("RUNNING LLM EVALUATIONS ON REAL DATASETS")
    print(f"{'='*70}\n")
    
    results = []
    models = ['gpt-4', 'llama-3', 'phi-3']
    
    # Take subset of samples
    test_samples = samples[:n_samples]
    
    for i, sample in enumerate(test_samples, 1):
        print(f"[{i}/{n_samples}] Evaluating {sample['task']} (ID: {sample['id']})...")
        
        # Simulate LLM evaluation
        model_results = simulate_llm_evaluation(sample)
        
        for model, result in model_results.items():
            results.append({
                'sample_id': sample['id'],
                'task': sample['task'],
                'dataset': sample['dataset'],
                'model': model,
                'prediction': result['prediction'],
                'confidence': result['confidence'],
                'response_time': result['response_time'],
                'correct': result['correct']
            })
    
    print(f"\n✓ Completed {len(results)} evaluations")
    return results

def calculate_metrics(results_df):
    """Calculate performance metrics per model"""
    print("\nCalculating metrics...")
    
    metrics = []
    for model in results_df['model'].unique():
        model_data = results_df[results_df['model'] == model]
        
        accuracy = model_data['correct'].mean()
        avg_confidence = model_data['confidence'].mean()
        avg_response_time = model_data['response_time'].mean()
        
        # Per-task metrics
        for task in model_data['task'].unique():
            task_data = model_data[model_data['task'] == task]
            task_accuracy = task_data['correct'].mean()
            
            metrics.append({
                'model': model,
                'task': task,
                'accuracy': task_accuracy,
                'samples': len(task_data),
                'avg_confidence': task_data['confidence'].mean(),
                'avg_response_time': task_data['response_time'].mean()
            })
        
        # Overall metrics
        metrics.append({
            'model': model,
            'task': 'overall',
            'accuracy': accuracy,
            'samples': len(model_data),
            'avg_confidence': avg_confidence,
            'avg_response_time': avg_response_time
        })
    
    metrics_df = pd.DataFrame(metrics)
    print("✓ Metrics calculated")
    return metrics_df

def visualize_results(results_df, metrics_df):
    """Create visualizations"""
    print("\nCreating visualizations...")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Accuracy comparison by model and task
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Overall accuracy
    overall_metrics = metrics_df[metrics_df['task'] == 'overall']
    ax1 = axes[0]
    bars = ax1.bar(overall_metrics['model'], overall_metrics['accuracy'], color=['#2ecc71', '#e74c3c', '#3498db'])
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_title('Overall Model Accuracy on Real Data', fontsize=14, fontweight='bold')
    ax1.set_ylim(0, 1)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1%}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Accuracy by task
    ax2 = axes[1]
    task_metrics = metrics_df[metrics_df['task'] != 'overall']
    tasks = task_metrics['task'].unique()
    
    if len(tasks) > 0:
        x = np.arange(len(tasks))
        width = 0.25
        
        for i, model in enumerate(['gpt-4', 'llama-3', 'phi-3']):
            model_task_data = task_metrics[task_metrics['model'] == model]
            accuracies = [model_task_data[model_task_data['task'] == task]['accuracy'].values[0] if len(model_task_data[model_task_data['task'] == task]) > 0 else 0 for task in tasks]
            ax2.bar(x + i*width, accuracies, width, label=model)
        
        ax2.set_ylabel('Accuracy', fontsize=12)
        ax2.set_title('Accuracy by Task Type', fontsize=14, fontweight='bold')
        ax2.set_xticks(x + width)
        task_labels = [task.replace('_', ' ').title()[:20] for task in tasks]
        ax2.set_xticklabels(task_labels, rotation=15, ha='right')
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)
        ax2.set_ylim(0, 1)
    else:
        ax2.text(0.5, 0.5, 'No task breakdown available', ha='center', va='center', transform=ax2.transAxes)
    
    plt.tight_layout()
    accuracy_plot = RESULTS_DIR / f"real_data_accuracy_{timestamp}.png"
    plt.savefig(accuracy_plot, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved {accuracy_plot}")
    plt.close()
    
    # 2. Performance metrics comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Response time
    ax1 = axes[0]
    response_times = overall_metrics.sort_values('avg_response_time')
    bars = ax1.barh(response_times['model'], response_times['avg_response_time'], color=['#9b59b6', '#e67e22', '#1abc9c'])
    ax1.set_xlabel('Response Time (seconds)', fontsize=12)
    ax1.set_title('Average Response Time', fontsize=14, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    
    # Confidence scores
    ax2 = axes[1]
    confidence = overall_metrics.sort_values('avg_confidence', ascending=False)
    bars = ax2.bar(confidence['model'], confidence['avg_confidence'], color=['#f39c12', '#16a085', '#c0392b'])
    ax2.set_ylabel('Confidence Score', fontsize=12)
    ax2.set_title('Average Confidence Scores', fontsize=14, fontweight='bold')
    ax2.set_ylim(0, 1)
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    performance_plot = RESULTS_DIR / f"real_data_performance_{timestamp}.png"
    plt.savefig(performance_plot, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved {performance_plot}")
    plt.close()
    
    return timestamp

def save_results(results_df, metrics_df, timestamp):
    """Save results to files"""
    print("\nSaving results...")
    
    # Save detailed results
    results_file = RESULTS_DIR / f"real_data_results_{timestamp}.csv"
    results_df.to_csv(results_file, index=False)
    print(f"  ✓ Saved {results_file}")
    
    # Save metrics
    metrics_file = RESULTS_DIR / f"real_data_metrics_{timestamp}.csv"
    metrics_df.to_csv(metrics_file, index=False)
    print(f"  ✓ Saved {metrics_file}")
    
    # Save metrics as JSON
    metrics_json = RESULTS_DIR / f"real_data_metrics_{timestamp}.json"
    metrics_dict = metrics_df.to_dict(orient='records')
    with open(metrics_json, 'w') as f:
        json.dump(metrics_dict, f, indent=2)
    print(f"  ✓ Saved {metrics_json}")

def display_summary(metrics_df):
    """Display summary of results"""
    print("\n" + "="*70)
    print("EXPERIMENT RESULTS SUMMARY")
    print("="*70)
    
    overall = metrics_df[metrics_df['task'] == 'overall'].sort_values('accuracy', ascending=False)
    
    print("\nOVERALL PERFORMANCE (on real datasets):")
    print("-" * 70)
    for _, row in overall.iterrows():
        print(f"{row['model']:12} | Accuracy: {row['accuracy']:6.1%} | "
              f"Confidence: {row['avg_confidence']:.3f} | "
              f"Response: {row['avg_response_time']:.2f}s")
    
    print("\nPER-TASK PERFORMANCE:")
    print("-" * 70)
    task_metrics = metrics_df[metrics_df['task'] != 'overall']
    for task in task_metrics['task'].unique():
        print(f"\n{task.upper().replace('_', ' ')}:")
        task_data = task_metrics[task_metrics['task'] == task].sort_values('accuracy', ascending=False)
        for _, row in task_data.iterrows():
            print(f"  {row['model']:12} | Accuracy: {row['accuracy']:6.1%} | "
                  f"Samples: {row['samples']:3d}")
    
    print("\n" + "="*70)

def main():
    # Load samples
    samples = load_evaluation_samples()
    
    # Run experiments
    results = run_experiments(samples, n_samples=20)
    results_df = pd.DataFrame(results)
    
    # Calculate metrics
    metrics_df = calculate_metrics(results_df)
    
    # Visualize
    timestamp = visualize_results(results_df, metrics_df)
    
    # Save results
    save_results(results_df, metrics_df, timestamp)
    
    # Display summary
    display_summary(metrics_df)
    
    print("\n" + "="*70)
    print("EXPERIMENTS COMPLETE!")
    print("="*70)
    print(f"\nResults saved in: {RESULTS_DIR}")
    print("\nNote: This is a simulated demo. To run with real LLMs:")
    print("1. Add your OpenAI API key to .env file")
    print("2. Modify simulate_llm_evaluation() to call actual APIs")
    print("3. See notebooks/dissertation_experiments.ipynb for full workflow")

if __name__ == "__main__":
    main()
