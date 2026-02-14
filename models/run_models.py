"""
Model Runner - Load and test LLaMA-3 and Phi-3 models
This script demonstrates how to run the downloaded models locally
"""
import os
import sys
import json
from pathlib import Path

# Add parent directory to path to import from src
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False
    print("⚠️  llama-cpp-python not installed. Install with: pip install llama-cpp-python")

def load_model_config():
    """Load model configuration"""
    config_path = Path(__file__).parent / "model_config.json"
    if config_path.exists():
        with open(config_path) as f:
            return json.load(f)
    return None

def run_llama3(prompt: str, system_prompt: str = None):
    """Run LLaMA-3 model"""
    if not LLAMA_CPP_AVAILABLE:
        return "Error: llama-cpp-python not installed"
    
    config = load_model_config()
    if not config or 'llama3' not in config:
        return "Error: Model configuration not found. Run download_models.py first"
    
    llama_config = config['llama3']
    model_path = Path(llama_config['model_dir']) / llama_config['model_file']
    
    if not model_path.exists():
        return f"Error: Model file not found at {model_path}"
    
    print(f"\n🤖 Loading LLaMA-3 from {model_path}...")
    
    try:
        llm = Llama(
            model_path=str(model_path),
            n_ctx=llama_config['context_length'],
            n_threads=4,
            n_gpu_layers=-1  # Use GPU if available
        )
        
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        
        print("🔄 Generating response...")
        output = llm(
            full_prompt,
            max_tokens=512,
            temperature=0.7,
            stop=["</s>"],
            echo=False
        )
        
        response = output['choices'][0]['text']
        print(f"\n✅ LLaMA-3 Response:\n{response}\n")
        return response
        
    except Exception as e:
        return f"Error running LLaMA-3: {e}"

def run_phi3(prompt: str, system_prompt: str = None):
    """Run Phi-3 model"""
    if not LLAMA_CPP_AVAILABLE:
        return "Error: llama-cpp-python not installed"
    
    config = load_model_config()
    if not config or 'phi3' not in config:
        return "Error: Model configuration not found. Run download_models.py first"
    
    phi_config = config['phi3']
    model_path = Path(phi_config['model_dir']) / phi_config['model_file']
    
    if not model_path.exists():
        return f"Error: Model file not found at {model_path}"
    
    print(f"\n🤖 Loading Phi-3 from {model_path}...")
    
    try:
        llm = Llama(
            model_path=str(model_path),
            n_ctx=phi_config['context_length'],
            n_threads=4,
            n_gpu_layers=-1  # Use GPU if available
        )
        
        # Format prompt for Phi-3
        if system_prompt:
            formatted_prompt = f"<|system|>\n{system_prompt}<|end|>\n<|user|>\n{prompt}<|end|>\n<|assistant|>"
        else:
            formatted_prompt = f"<|user|>\n{prompt}<|end|>\n<|assistant|>"
        
        print("🔄 Generating response...")
        output = llm(
            formatted_prompt,
            max_tokens=512,
            temperature=0.7,
            stop=["<|end|>"],
            echo=False
        )
        
        response = output['choices'][0]['text']
        print(f"\n✅ Phi-3 Response:\n{response}\n")
        return response
        
    except Exception as e:
        return f"Error running Phi-3: {e}"

def run_comparison(prompt: str, system_prompt: str = None):
    """Run both models and compare results"""
    print("="*70)
    print("🔬 Running Model Comparison")
    print("="*70)
    print(f"\nPrompt: {prompt}")
    if system_prompt:
        print(f"System: {system_prompt}")
    print("\n" + "-"*70)
    
    print("\n[1/2] Running LLaMA-3...")
    llama_response = run_llama3(prompt, system_prompt)
    
    print("\n" + "-"*70)
    print("\n[2/2] Running Phi-3...")
    phi_response = run_phi3(prompt, system_prompt)
    
    print("\n" + "="*70)
    print("Comparison Complete!")
    print("="*70)
    
    return {
        'llama3': llama_response,
        'phi3': phi_response
    }

def main():
    """Main function with example usage"""
    print("\n" + "#"*70)
    print("#" + " "*68 + "#")
    print("#  LLaMA-3 & Phi-3 Model Runner".ljust(69) + "#")
    print("#" + " "*68 + "#")
    print("#"*70)
    
    if not LLAMA_CPP_AVAILABLE:
        print("\n❌ Required dependency missing!")
        print("\nInstall llama-cpp-python:")
        print("  pip install llama-cpp-python")
        print("\nFor GPU support (CUDA):")
        print("  CMAKE_ARGS=\"-DLLAMA_CUBLAS=on\" pip install llama-cpp-python")
        return
    
    # Example telecom fault detection prompt
    system_prompt = """You are an expert in telecom systems fault detection. 
Analyze the given metrics and identify potential issues."""
    
    prompt = """Analyze these metrics:
- CPU usage: 95% (normal: 30-50%)
- Memory usage: 89% (normal: 40-60%)
- Network latency: 250ms (normal: 10-50ms)
- Error rate: 5% (normal: <1%)

Identify the fault and suggest root cause."""
    
    print("\n📋 Choose an option:")
    print("1. Run LLaMA-3 only")
    print("2. Run Phi-3 only")
    print("3. Run both and compare")
    print("4. Custom prompt")
    print("0. Exit")
    
    try:
        choice = input("\nEnter choice (0-4): ").strip()
        
        if choice == "0":
            print("Exiting...")
            return
        elif choice == "1":
            run_llama3(prompt, system_prompt)
        elif choice == "2":
            run_phi3(prompt, system_prompt)
        elif choice == "3":
            run_comparison(prompt, system_prompt)
        elif choice == "4":
            custom_prompt = input("\nEnter your prompt: ").strip()
            run_comparison(custom_prompt, system_prompt)
        else:
            print("Invalid choice")
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main()
