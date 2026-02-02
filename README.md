# A Data-Driven Evaluation of Large Language Models
# Dissertation Project

This repository contains the experimental framework and results for evaluating Large Language Models (LLMs) on fault detection and quality assessment tasks in telecom software systems.

## Project Overview

**Thesis Title:** A Data-Driven Evaluation of Large Language Models for Fault Detection and Quality Assessment in Telecom Software Systems

### Research Objectives
1. Evaluate closed-source and open-source LLMs on fault detection tasks
2. Compare performance across different model sizes (large vs small)
3. Analyze model errors and failure modes
4. Assess potential as early-warning tools for telecom operations

### Models Evaluated
- **GPT-4 / GPT-4.1** - Closed-source, state-of-the-art baseline
- **LLaMA-3 8B** - Open-source, mid-size model
- **Phi-3 Mini** - Small, resource-efficient model

## Project Structure

```
mypythondata/
├── datasets/                      # Dataset generators and data
│   ├── generate_synthetic_telecom_faults.py
│   ├── generate_kpi_anomalies.py
│   ├── generate_microservices_faults.py
│   ├── synthetic_telecom_faults.csv
│   ├── kpi_anomaly_detection.csv
│   ├── microservices_traces.csv
│   ├── microservices_logs.csv
│   └── microservices_metrics.csv
├── src/                           # Source code
│   ├── llm_interface.py          # Unified LLM API interface
│   ├── evaluation_framework.py    # Evaluation orchestration
│   └── metrics_calculator.py      # Performance metrics
├── notebooks/                     # Jupyter notebooks
│   └── dissertation_experiments.ipynb  # Main experimental notebook
├── results/                       # Evaluation results and visualizations
├── requirements.txt               # Python dependencies
├── .env.example                   # Configuration template
└── README.md                      # This file
```

## Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd mypythondata
```

2. **Create a virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Configure API keys:**
```bash
cp .env.example .env
# Edit .env and add your API keys
```

## Quick Start

### 1. Generate Synthetic Datasets

```bash
python datasets/generate_synthetic_telecom_faults.py
python datasets/generate_kpi_anomalies.py
python datasets/generate_microservices_faults.py
```

### 2. Run Experiments

Open the Jupyter notebook:
```bash
jupyter notebook notebooks/dissertation_experiments.ipynb
```

Follow the notebook cells to:
- Generate datasets
- Configure LLM models
- Run evaluations
- Calculate metrics
- Generate visualizations

## Datasets

### 1. Synthetic Telecom Fault Dataset
- **Size:** 1,000 samples
- **Features:** Fault scenarios with timeouts, retries, QoS violations
- **Labels:** Failure, degradation, SLA breach outcomes
- **Purpose:** Controlled fault analysis

### 2. KPI Anomaly Detection Dataset
- **Size:** 5,000 time-series samples
- **KPIs:** Latency, throughput, error rate, CPU, memory
- **Labels:** Binary anomaly detection (5% anomaly ratio)
- **Purpose:** Time-series pattern recognition

### 3. Microservices Fault Dataset
- **Size:** 1,000 distributed traces
- **Data:** Logs, traces, metrics from 14 microservices
- **Scenarios:** Timeouts, cascading failures, resource exhaustion
- **Purpose:** Distributed system debugging

## Evaluation Tasks

### Task 1: Fault Classification
- Classify fault types from log messages
- Determine severity levels
- Identify root causes

### Task 2: Anomaly Detection
- Detect anomalies in KPI time-series
- Explain reasoning and provide confidence

### Task 3: Trace Analysis
- Analyze distributed traces
- Identify performance issues
- Determine responsible services

## Metrics

Performance metrics calculated for each model:
- **Accuracy, Precision, Recall, F1 Score** - Classification performance
- **Latency** - Response time per request
- **Token Usage** - Computational cost
- **Success Rate** - Request completion rate
- **Confidence Scores** - Model certainty

## Results

Results are saved in the `results/` directory:
- `evaluation_results_[timestamp].csv` - Raw evaluation data
- `metrics_report_[timestamp].json` - Calculated metrics
- `final_comparison_matrix_[timestamp].csv` - Model comparison table
- Visualization PNG files

## Usage Example

```python
from src.llm_interface import ModelFactory
from src.evaluation_framework import LLMEvaluator

# Initialize models
gpt4 = ModelFactory.create_model('gpt-4', api_key='your-key')
llama3 = ModelFactory.create_model('llama-3', model_path='path/to/model')
phi3 = ModelFactory.create_model('phi-3')

# Run evaluation
evaluator = LLMEvaluator([gpt4, llama3, phi3])
evaluator.evaluate_fault_classification('datasets/synthetic_telecom_faults.csv')
evaluator.save_results()
```

## Key Findings

[To be updated after experiments]

## Contributing

This is a dissertation project. For questions or collaboration inquiries, please contact the author.

## License

[To be determined]

## Citation

If you use this work, please cite:

```
[Citation to be added after publication]
```

## Acknowledgments

- Datasets inspired by AIOps benchmarks and telecom operations research
- Framework built on OpenAI, Hugging Face, and scikit-learn libraries

## Contact

[Your contact information]

---

**Status:** Active Development  
**Last Updated:** January 2026
