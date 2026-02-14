# Models Directory

This directory contains scripts for downloading and running LLaMA-3 and Phi-3 models locally.

## Quick Start

### 1. Setup and Test Everything
```powershell
python models/setup_and_test.py
```

This will:
- Check if dependencies are installed
- Offer to install missing packages
- Check if models are downloaded
- Offer to download models
- Run a quick test

### 2. Manual Setup

If you prefer manual setup:

```powershell
# Install dependencies
pip install llama-cpp-python huggingface-hub

# Download models
python models/download_models.py

# Test models
python models/run_models.py
```

## Files in This Directory

| File | Description |
|------|-------------|
| `setup_and_test.py` | **START HERE** - Automated setup and testing |
| `download_models.py` | Downloads LLaMA-3 and Phi-3 from HuggingFace |
| `run_models.py` | Interactive script to test models |
| `SETUP_GUIDE.md` | Detailed setup instructions and troubleshooting |
| `model_config.json` | Auto-generated model paths (created by download script) |
| `LLaMA-3-8B/` | LLaMA-3 model files (created after download) |
| `Phi-3-Mini/` | Phi-3 model files (created after download) |

## Model Information

### LLaMA-3 8B
- **Size**: ~5GB
- **Type**: GGUF quantized
- **Context**: 8,192 tokens
- **Use**: High-quality responses, complex reasoning

### Phi-3 Mini
- **Size**: ~2.5GB
- **Type**: GGUF quantized
- **Context**: 4,096 tokens
- **Use**: Fast inference, resource-efficient

## Usage Examples

### Simple Test
```python
from src.llm_interface import ModelFactory

# Create model
model = ModelFactory.create_model('llama-3')

# Generate response
result = model.generate("Explain CPU usage metrics in telecom systems")
print(result['response'])
```

### Compare Both Models
```python
from models.run_models import run_comparison

results = run_comparison(
    prompt="Analyze these metrics: CPU 95%, Memory 89%",
    system_prompt="You are a telecom fault detection expert"
)

print("LLaMA-3:", results['llama3'])
print("Phi-3:", results['phi3'])
```

## System Requirements

- **RAM**: 8GB minimum (16GB recommended)
- **Storage**: 10GB free space
- **Python**: 3.8+
- **Optional**: NVIDIA GPU with CUDA

## Troubleshooting

See [SETUP_GUIDE.md](SETUP_GUIDE.md) for detailed troubleshooting.

### Common Issues

**"llama-cpp-python not installed"**
```powershell
pip install llama-cpp-python
```

**"Model file not found"**
```powershell
python models/download_models.py
```

**Out of memory**
- Use Phi-3 (smaller)
- Close other applications
- Reduce max_tokens in code

## GPU Acceleration (Optional)

For NVIDIA GPUs with CUDA:
```powershell
$env:CMAKE_ARGS="-DLLAMA_CUBLAS=on"
pip install llama-cpp-python --force-reinstall --no-cache-dir
```

This can provide 5-10x speedup in inference.

## Next Steps

After setup:
1. Test models: `python models/run_models.py`
2. Run experiments: `python run_experiments.py`
3. View results: Check `results/` directory
4. Compare with GPT-4: Update experiments to include all models

## Support

For detailed help, see [SETUP_GUIDE.md](SETUP_GUIDE.md)
