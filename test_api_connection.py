"""
Quick test script to verify ChatGPT API connection
Run this after setting up your .env file
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.config import Config
from src.llm_interface import GPT4Model


def test_api_connection():
    """Test OpenAI API connection"""
    print("=" * 80)
    print("TESTING CHATGPT API CONNECTION")
    print("=" * 80)
    
    # Check if API key is configured
    if not Config.OPENAI_API_KEY or Config.OPENAI_API_KEY == "YOUR_NEW_API_KEY_HERE":
        print("\nERROR: OpenAI API key not configured!")
        print("\nPlease:")
        print("1. Open the .env file")
        print("2. Replace YOUR_NEW_API_KEY_HERE with your actual API key")
        print("3. Save the file and run this script again")
        return False
    print(f"\nAPI Key loaded: {Config.OPENAI_API_KEY[:20]}...{Config.OPENAI_API_KEY[-4:]}")
    print(f"Model: GPT-3.5-Turbo (trying multiple models)")
    print(f"Temperature: {Config.TEMPERATURE}")
    print(f"Max Tokens: {Config.MAX_TOKENS}")
    
    # Try different models in order of preference
    models_to_try = ["gpt-4o-mini", "gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"]
    
    for model_name in models_to_try:
        try:
            print("\n" + "-" * 80)
            print(f"Trying model: {model_name}...")
            print("-" * 80)
            
            # Initialize model
            model = GPT4Model(
                api_key=Config.OPENAI_API_KEY,
                model_name=model_name,
                temperature=0.7,
                max_tokens=150
            )
            
            # Test prompt
            test_prompt = "Say 'API connection successful!' and briefly explain what you can help with."
            system_prompt = "You are a helpful AI assistant for a telecommunications fault detection research project."
            
            # Generate response
            result = model.generate(test_prompt, system_prompt)
            
            if result['success']:
                print(f"\nSUCCESS! API Connection Working with {model_name}")
                print("=" * 80)
                print(f"\n{model_name} Response:")
                print(result['response'])
                print(f"\nTokens Used: {result['tokens_used']}")
                print(f"Latency: {result['latency_seconds']:.2f} seconds")
                print("\n" + "=" * 80)
                print("\nYour dissertation project is ready to run!")
                print(f"\nNote: Using model '{model_name}' for experiments")
                print("\nNext steps:")
                print("1. Run: python run_real_experiments.py")
                print("2. Or open Jupyter: jupyter notebook notebooks/dissertation_experiments.ipynb")
                return True
            else:
                print(f"Failed with {model_name}: {result.get('error', 'Unknown error')}")
                continue
                
        except Exception as e:
            print(f"Error with {model_name}: {str(e)}")
            continue
    
    # If we reach here, all models failed
    print("\nERROR: All models failed")
    print("\nPossible issues:")
    print("- Invalid API key (generate a new one)")
    print("- Insufficient credits (add funds to your OpenAI account)")
    print("- Rate limit exceeded (wait a moment and try again)")
    return False


if __name__ == "__main__":
    print("\n")
    success = test_api_connection()
    print("\n")
    
    if success:
        sys.exit(0)
    else:
        sys.exit(1)
