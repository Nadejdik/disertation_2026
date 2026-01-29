# Getting Started Guide

## Overview
This guide will help you set up and run the LLM evaluation experiments for your dissertation.

## Prerequisites
- Python 3.10 or higher
- At least 8GB RAM (16GB recommended for local models)
- OpenAI API key (for GPT-4 evaluation)
- Optional: GPU with CUDA support for local models

## Installation Steps

### 1. Set Up Python Environment

```bash
# Navigate to project directory
cd mypythondata

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure API Keys

```bash
# Copy the example environment file
copy .env.example .env  # Windows
cp .env.example .env    # macOS/Linux

# Edit .env file and add your API keys
# Use any text editor:
notepad .env  # Windows
nano .env     # macOS/Linux
```

Required configuration in `.env`:
```
OPENAI_API_KEY=your-actual-api-key-here
```

### 3. Generate Datasets

```bash
python setup_datasets.py
```

This will create three datasets:
- `datasets/synthetic_telecom_faults.csv` (1,000 samples)
- `datasets/kpi_anomaly_detection.csv` (5,000 samples)
- `datasets/microservices_traces.csv` (1,000+ spans)

## Running Experiments

You have two options to run experiments:

### Option 1: Jupyter Notebook (Recommended for exploration)

```bash
# Start Jupyter
jupyter notebook

# Open: notebooks/dissertation_experiments.ipynb
# Run cells sequentially
```

The notebook provides:
- Interactive exploration
- Inline visualizations
- Step-by-step guidance
- Immediate results

### Option 2: Command Line (Recommended for batch processing)

```bash
# Run all experiments with GPT-4 only
python run_experiments.py --models gpt-4 --n-samples 50 --n-runs 3

# Run specific tasks
python run_experiments.py --models gpt-4 --tasks fault_classification anomaly_detection

# Run multiple models (when configured)
python run_experiments.py --models gpt-4 llama-3 phi-3 --n-samples 100
```

## Expected Outputs

### Datasets
- Located in `datasets/` folder
- CSV and JSON formats
- Metadata files included

### Results
Located in `results/` folder:
- `evaluation_results_[timestamp].csv` - Raw evaluation data
- `metrics_report_[timestamp].json` - Calculated performance metrics
- `final_comparison_matrix_[timestamp].csv` - Model comparison table
- `*.png` - Visualization charts
- `README_[timestamp].md` - Results documentation

## Understanding the Results

### Key Metrics

1. **Accuracy** - Overall correctness of predictions
2. **Precision** - When model predicts positive, how often is it correct?
3. **Recall** - Of all actual positives, how many did model find?
4. **F1 Score** - Harmonic mean of precision and recall
5. **Latency** - Time taken per request
6. **Token Usage** - Computational/cost efficiency

### Comparison Matrix

The final comparison matrix shows:
- Rows: Different LLM models
- Columns: Tasks and metrics
- Values: Performance scores

This is the main table for your dissertation results section.

## Troubleshooting

### Issue: "OpenAI API key not found"
**Solution:** Ensure `.env` file exists and contains valid `OPENAI_API_KEY`

### Issue: "Module not found"
**Solution:** Make sure virtual environment is activated and dependencies are installed:
```bash
pip install -r requirements.txt
```

### Issue: "Out of memory" when running local models
**Solution:** 
- Reduce `N_SAMPLES` in configuration
- Use smaller models (Phi-3 instead of LLaMA-3)
- Use API endpoints instead of local inference

### Issue: Rate limiting errors
**Solution:** Increase `DELAY_BETWEEN_REQUESTS` in `.env`:
```
DELAY_BETWEEN_REQUESTS=1.0
```

### Issue: Dataset not found
**Solution:** Run dataset generation first:
```bash
python setup_datasets.py
```

## Configuration Options

Edit `.env` to customize:

```bash
# Experiment size
N_SAMPLES=100          # Number of samples per dataset
N_RUNS=3               # Runs per sample for consistency

# Model parameters
TEMPERATURE=0.7        # 0.0 = deterministic, 1.0 = creative
MAX_TOKENS=2048        # Maximum response length

# Rate limiting
DELAY_BETWEEN_REQUESTS=0.5  # Seconds between API calls
```

## Next Steps

1. **Generate datasets** - Run `setup_datasets.py`
2. **Configure API keys** - Edit `.env` file
3. **Run small test** - Start with 10 samples to verify setup
4. **Run full evaluation** - Use 100-500 samples for dissertation
5. **Analyze results** - Use Jupyter notebook for visualization
6. **Document findings** - Update notebook with conclusions



