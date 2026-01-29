"""
Configuration loader for the LLM evaluation framework
"""
import os
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """Configuration management for LLM evaluation"""
    
    # Paths
    PROJECT_ROOT = Path(__file__).parent.parent
    DATASETS_DIR = PROJECT_ROOT / 'datasets'
    RESULTS_DIR = PROJECT_ROOT / 'results'
    MODELS_CACHE_DIR = PROJECT_ROOT / 'models_cache'
    LOGS_DIR = PROJECT_ROOT / 'logs'
    
    # API Keys
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
    LLAMA_API_ENDPOINT = os.getenv('LLAMA_API_ENDPOINT', 'http://localhost:8000/generate')
    PHI3_API_ENDPOINT = os.getenv('PHI3_API_ENDPOINT', 'http://localhost:8001/generate')
    
    # Model Paths
    LLAMA3_MODEL_PATH = os.getenv('LLAMA3_MODEL_PATH', 'meta-llama/Meta-Llama-3-8B')
    PHI3_MODEL_PATH = os.getenv('PHI3_MODEL_PATH', 'microsoft/Phi-3-mini-4k-instruct')
    
    # Evaluation Configuration
    N_SAMPLES = int(os.getenv('N_SAMPLES', 100))
    N_RUNS = int(os.getenv('N_RUNS', 3))
    TEMPERATURE = float(os.getenv('TEMPERATURE', 0.7))
    MAX_TOKENS = int(os.getenv('MAX_TOKENS', 2048))
    
    # Rate Limiting
    REQUESTS_PER_MINUTE = int(os.getenv('REQUESTS_PER_MINUTE', 60))
    DELAY_BETWEEN_REQUESTS = float(os.getenv('DELAY_BETWEEN_REQUESTS', 0.5))
    
    # Dataset Configuration
    TELECOM_FAULTS_SAMPLES = int(os.getenv('TELECOM_FAULTS_SAMPLES', 1000))
    KPI_ANOMALY_SAMPLES = int(os.getenv('KPI_ANOMALY_SAMPLES', 5000))
    MICROSERVICES_TRACES = int(os.getenv('MICROSERVICES_TRACES', 1000))
    ANOMALY_RATIO = float(os.getenv('ANOMALY_RATIO', 0.05))
    
    # Logging
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FILE = os.getenv('LOG_FILE', 'logs/evaluation.log')
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories if they don't exist"""
        for directory in [cls.DATASETS_DIR, cls.RESULTS_DIR, cls.MODELS_CACHE_DIR, cls.LOGS_DIR]:
            directory.mkdir(exist_ok=True)
    
    @classmethod
    def get_model_config(cls, model_type: str) -> Dict[str, Any]:
        """Get configuration for a specific model type"""
        configs = {
            'gpt-4': {
                'api_key': cls.OPENAI_API_KEY,
                'temperature': cls.TEMPERATURE,
                'max_tokens': cls.MAX_TOKENS
            },
            'llama-3': {
                'model_path': cls.LLAMA3_MODEL_PATH,
                'api_endpoint': cls.LLAMA_API_ENDPOINT,
                'temperature': cls.TEMPERATURE,
                'max_tokens': cls.MAX_TOKENS
            },
            'phi-3': {
                'model_path': cls.PHI3_MODEL_PATH,
                'temperature': cls.TEMPERATURE,
                'max_tokens': cls.MAX_TOKENS
            }
        }
        return configs.get(model_type, {})
    
    @classmethod
    def validate(cls) -> bool:
        """Validate configuration"""
        issues = []
        
        if not cls.OPENAI_API_KEY:
            issues.append("OPENAI_API_KEY not set")
        
        if not cls.DATASETS_DIR.exists():
            issues.append(f"Datasets directory not found: {cls.DATASETS_DIR}")
        
        if issues:
            print("Configuration issues:")
            for issue in issues:
                print(f"  - {issue}")
            return False
        
        return True
    
    @classmethod
    def print_config(cls):
        """Print current configuration"""
        print("="*60)
        print("LLM EVALUATION CONFIGURATION")
        print("="*60)
        print(f"Project Root: {cls.PROJECT_ROOT}")
        print(f"Datasets: {cls.DATASETS_DIR}")
        print(f"Results: {cls.RESULTS_DIR}")
        print(f"\nEvaluation:")
        print(f"  Samples per dataset: {cls.N_SAMPLES}")
        print(f"  Runs per sample: {cls.N_RUNS}")
        print(f"  Temperature: {cls.TEMPERATURE}")
        print(f"  Max tokens: {cls.MAX_TOKENS}")
        print(f"\nAPI Keys:")
        print(f"  OpenAI: {'✓ Set' if cls.OPENAI_API_KEY else '✗ Not set'}")
        print("="*60)


if __name__ == '__main__':
    Config.create_directories()
    Config.print_config()
    
    if Config.validate():
        print("\n✓ Configuration is valid")
    else:
        print("\n✗ Configuration has issues")
