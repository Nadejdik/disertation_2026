"""
Quick Setup and Test Script for LLaMA-3 & Phi-3
Checks installation and runs a simple test
"""
import sys
import subprocess
from pathlib import Path

def print_header(text):
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70)

def check_dependency(package_name, import_name=None):
    """Check if a package is installed"""
    if import_name is None:
        import_name = package_name
    
    try:
        __import__(import_name)
        print(f"✅ {package_name} installed")
        return True
    except ImportError:
        print(f"❌ {package_name} NOT installed")
        return False

def check_models():
    """Check if models are downloaded"""
    models_dir = Path(__file__).parent
    
    llama_dir = models_dir / "LLaMA-3-8B"
    phi_dir = models_dir / "Phi-3-Mini"
    
    llama_exists = llama_dir.exists() and list(llama_dir.glob("*.gguf"))
    phi_exists = phi_dir.exists() and list(phi_dir.glob("*.gguf"))
    
    if llama_exists:
        print(f"✅ LLaMA-3 model found in {llama_dir}")
    else:
        print(f"❌ LLaMA-3 model NOT found in {llama_dir}")
    
    if phi_exists:
        print(f"✅ Phi-3 model found in {phi_dir}")
    else:
        print(f"❌ Phi-3 model NOT found in {phi_dir}")
    
    return llama_exists and phi_exists

def install_missing_deps():
    """Install missing dependencies"""
    print("\nAttempting to install missing dependencies...")
    
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "llama-cpp-python"], 
                      check=True, capture_output=True)
        print("✅ llama-cpp-python installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install: {e}")
        return False

def download_models():
    """Download models"""
    print("\nDownloading models (this may take 10-30 minutes)...")
    script_path = Path(__file__).parent / "download_models.py"
    
    try:
        subprocess.run([sys.executable, str(script_path)], check=True)
        print("✅ Models downloaded successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to download models: {e}")
        return False

def run_test():
    """Run a quick test"""
    print("\nRunning quick test...")
    
    try:
        from llama_cpp import Llama
        
        # Find a model file
        models_dir = Path(__file__).parent
        llama_files = list((models_dir / "LLaMA-3-8B").glob("*.gguf"))
        
        if not llama_files:
            print("❌ No model files found for testing")
            return False
        
        print(f"Loading model: {llama_files[0].name}...")
        llm = Llama(
            model_path=str(llama_files[0]),
            n_ctx=2048,
            n_threads=2,
            n_gpu_layers=-1,
            verbose=False
        )
        
        print("Generating test response...")
        output = llm(
            "What is 2+2?",
            max_tokens=50,
            temperature=0.7,
            stop=["</s>"],
            echo=False
        )
        
        response = output['choices'][0]['text']
        print(f"\n✅ Test successful! Model response:\n{response}\n")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def main():
    print_header("LLaMA-3 & Phi-3 Setup Checker")
    
    # Step 1: Check dependencies
    print_header("Step 1: Checking Dependencies")
    deps_ok = True
    
    deps_ok &= check_dependency("huggingface_hub", "huggingface_hub")
    llama_cpp_ok = check_dependency("llama-cpp-python", "llama_cpp")
    deps_ok &= llama_cpp_ok
    
    if not deps_ok:
        print("\n⚠️  Missing dependencies detected")
        response = input("\nInstall missing dependencies? (y/n): ").strip().lower()
        if response == 'y':
            if not llama_cpp_ok:
                install_missing_deps()
        else:
            print("\n❌ Cannot proceed without dependencies")
            print("Install manually with: pip install llama-cpp-python")
            return
    
    # Step 2: Check models
    print_header("Step 2: Checking Models")
    models_ok = check_models()
    
    if not models_ok:
        print("\n⚠️  Models not found")
        response = input("\nDownload models now? (y/n): ").strip().lower()
        if response == 'y':
            if not download_models():
                print("\n❌ Setup incomplete - models not downloaded")
                return
        else:
            print("\n⚠️  Cannot run models without downloading them")
            print("Download manually with: python models/download_models.py")
            return
    
    # Step 3: Run test
    print_header("Step 3: Running Test")
    test_ok = run_test()
    
    # Summary
    print_header("Setup Summary")
    if deps_ok and models_ok and test_ok:
        print("✅ All checks passed!")
        print("\nYou can now:")
        print("  1. Run interactive test: python models/run_models.py")
        print("  2. Use in code: from src.llm_interface import ModelFactory")
        print("  3. Run experiments: python run_experiments.py")
    else:
        print("⚠️  Setup incomplete. Please review the errors above.")
        print("\nFor help, see: models/SETUP_GUIDE.md")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nSetup interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
