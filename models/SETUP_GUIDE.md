# LLaMA-3 & Phi-3 Local Model Setup Guide

## Overview
This guide will help you download and run LLaMA-3 8B and Phi-3 Mini models locally on your machine.

## Prerequisites

### System Requirements
- **RAM**: Minimum 8GB (16GB recommended for better performance)
- **Storage**: ~10GB free space for models
- **Python**: 3.8 or higher
- **Optional**: NVIDIA GPU with CUDA for faster inference

## Installation Steps

### 1. Install Python Dependencies

First, install the base requirements:
```powershell
pip install -r requirements.txt
```

### 2. Install llama-cpp-python

**For CPU-only (simpler, works on all systems):**
```powershell
pip install llama-cpp-python
```

**For GPU acceleration with CUDA (if you have NVIDIA GPU):**
```powershell
$env:CMAKE_ARGS="-DLLAMA_CUBLAS=on"
pip install llama-cpp-python --force-reinstall --no-cache-dir
```

### 3. Download Models

Run the download script to fetch LLaMA-3 and Phi-3 models:
```powershell
cd models
python download_models.py
```

This will:
- Download LLaMA-3 8B GGUF model (~5GB)
- Download Phi-3 Mini GGUF model (~2.5GB)
- Create a `model_config.json` file with model paths

**Note**: Download may take 10-30 minutes depending on your internet speed.

## Running the Models

### Option 1: Interactive Model Runner

Run the interactive script to test both models:
```powershell
cd models
python run_models.py
```

This provides a menu to:
1. Run LLaMA-3 only
2. Run Phi-3 only
3. Run both and compare results
4. Enter custom prompts

### Option 2: Use in Your Code

```python
from src.llm_interface import ModelFactory

# Create LLaMA-3 model
llama = ModelFactory.create_model('llama-3')

# Create Phi-3 model
phi = ModelFactory.create_model('phi-3')

# Generate response
system_prompt = "You are an expert in telecom fault detection."
prompt = "Analyze these metrics: CPU 95%, Memory 89%, Latency 250ms"

llama_result = llama.generate(prompt, system_prompt)
phi_result = phi.generate(prompt, system_prompt)

print("LLaMA-3:", llama_result['response'])
print("Phi-3:", phi_result['response'])
```

### Option 3: Run Experiments

Use the models in the evaluation framework:
```powershell
python run_experiments.py
```

## Model Information

### LLaMA-3 8B
- **Size**: ~5GB (quantized)
- **Context Window**: 8,192 tokens
- **Best For**: Complex reasoning, detailed analysis
- **Speed**: Moderate (slower but more accurate)

### Phi-3 Mini
- **Size**: ~2.5GB (quantized)
- **Context Window**: 4,096 tokens  
- **Best For**: Fast inference, resource-constrained environments
- **Speed**: Fast (2-3x faster than LLaMA-3)

## Troubleshooting

### Issue: "llama-cpp-python not installed"
**Solution**: Run `pip install llama-cpp-python`

### Issue: "Model file not found"
**Solution**: Run `python models/download_models.py` to download models

### Issue: Out of memory (OOM) errors
**Solution**: 
- Close other applications
- Use Phi-3 instead (smaller model)
- Reduce context window in code

### Issue: Very slow inference
**Solution**:
- Install GPU-accelerated version (if you have NVIDIA GPU)
- Use smaller max_tokens (e.g., 256 instead of 512)
- Reduce n_threads parameter

### Issue: GPU not being used
**Solution**: Reinstall with CUDA support:
```powershell
$env:CMAKE_ARGS="-DLLAMA_CUBLAS=on"
pip uninstall llama-cpp-python -y
pip install llama-cpp-python --force-reinstall --no-cache-dir
```

## Performance Optimization

### For Laptops/Lower-end Systems
- Use Phi-3 Mini (smaller, faster)
- Set `n_threads=2` in model initialization
- Reduce `max_tokens` to 256-512

### For Desktop/High-end Systems
- Use LLaMA-3 8B for better quality
- Install GPU version for 5-10x speedup
- Increase `n_threads=8` for better CPU utilization

## File Structure

```
models/
├── download_models.py      # Downloads models from HuggingFace
├── run_models.py           # Interactive test runner
├── model_config.json       # Auto-generated model paths
├── LLaMA-3-8B/            # LLaMA-3 model files
│   └── *.gguf
└── Phi-3-Mini/            # Phi-3 model files
    └── *.gguf
```

## Next Steps

1. Test the models with `python models/run_models.py`
2. Integrate into your experiments with `run_experiments.py`
3. Compare GPT-4 vs LLaMA-3 vs Phi-3 performance
4. Analyze results in `results/` directory

## Additional Resources

- [LLaMA-3 Documentation](https://ai.meta.com/llama/)
- [Phi-3 Documentation](https://azure.microsoft.com/en-us/products/phi-3)
- [llama-cpp-python GitHub](https://github.com/abetlen/llama-cpp-python)

## Support

If you encounter issues:
1. Check the Troubleshooting section above
2. Ensure all dependencies are installed: `pip install -r requirements.txt`
3. Verify models are downloaded: check `models/LLaMA-3-8B/` and `models/Phi-3-Mini/`
4. Check available RAM: models need 8-16GB to run
