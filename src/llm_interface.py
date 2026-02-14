"""
"""
LLM Model Interface - Unified API for different LLM models
Supports: GPT-4 (OpenAI), LLaMA-3 (local/API), Phi-3 (local)
"""
import os
import json
import time
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from pathlib import Path
from openai import OpenAI

try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False
    print("[WARNING] llama-cpp-python not available. Local models will not work.")


class BaseLLMModel(ABC):
    """Base class for all LLM models"""
    
    def __init__(self, model_name: str, temperature: float = 0.7, max_tokens: int = 2048):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.total_tokens_used = 0
        self.total_requests = 0
        
    @abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """Generate response from the model"""
        pass
    
    def get_stats(self) -> Dict[str, Any]:
        """Get usage statistics"""
        return {
            'model_name': self.model_name,
            'total_requests': self.total_requests,
            'total_tokens_used': self.total_tokens_used
        }


class GPT4Model(BaseLLMModel):
    """OpenAI GPT-4 model interface"""
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gpt-4", **kwargs):
        super().__init__(model_name, **kwargs)
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
        
        # Initialize OpenAI client (v1.x API)
        self.client = OpenAI(api_key=self.api_key)
        
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """Generate response using GPT-4"""
        try:
            messages = []
            
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            
            messages.append({"role": "user", "content": prompt})
            
            start_time = time.time()
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            end_time = time.time()
            
            self.total_requests += 1
            self.total_tokens_used += response.usage.total_tokens
            
            return {
                'response': response.choices[0].message.content,
                'model': self.model_name,
                'tokens_used': response.usage.total_tokens,
                'latency_seconds': end_time - start_time,
                'finish_reason': response.choices[0].finish_reason,
                'success': True,
                'error': None
            }
            
        except Exception as e:
            return {
                'response': None,
                'model': self.model_name,
                'tokens_used': 0,
                'latency_seconds': 0,
                'finish_reason': None,
                'success': False,
                'error': str(e)
            }


class LLaMA3Model(BaseLLMModel):
    """LLaMA-3 model interface (supports both local and API)"""
    
    def __init__(self, model_path: Optional[str] = None, use_api: bool = False, 
                 api_endpoint: Optional[str] = None, **kwargs):
        super().__init__("llama-3-8b", **kwargs)
        self.model_path = model_path
        self.use_api = use_api
        self.api_endpoint = api_endpoint
        
        if use_api:
            if not api_endpoint:
                raise ValueError("API endpoint required when use_api=True")
        else:
            # For local inference, you'd load the model here
            # This is a placeholder for the actual implementation
            self._load_local_model()
    
    def _load_local_model(self):
        """Load LLaMA-3 model locally using llama-cpp-python"""
        if not LLAMA_CPP_AVAILABLE:
            print("[ERROR] llama-cpp-python not installed. Install with: pip install llama-cpp-python")
            self.model_loaded = False
            return
        
        try:
            # Find model file
            model_path = self._find_model_file()
            if not model_path:
                print(f"[ERROR] Model file not found in: {self.model_path}")
                self.model_loaded = False
                return
            
            print(f"[INFO] Loading LLaMA-3 from: {model_path}")
            
            self.model = Llama(
                model_path=str(model_path),
                n_ctx=8192,  # Context window
                n_threads=4,
                n_gpu_layers=-1,  # Use GPU if available
                verbose=False
            )
            
            self.model_loaded = True
            print(f"[SUCCESS] LLaMA-3 model loaded successfully")
            
        except Exception as e:
            print(f"[ERROR] Failed to load LLaMA-3 model: {e}")
            self.model_loaded = False
    
    def _find_model_file(self):
        """Find the GGUF model file"""
        if not self.model_path:
            # Try to load from models directory
            models_dir = Path(__file__).parent.parent / "models" / "LLaMA-3-8B"
            if not models_dir.exists():
                return None
            self.model_path = str(models_dir)
        
        search_path = Path(self.model_path)
        
        # Look for GGUF files
        gguf_files = list(search_path.glob("*.gguf"))
        if gguf_files:
            return str(gguf_files[0])
        
        return None
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """Generate response using LLaMA-3"""
        
        if self.use_api:
            return self._generate_via_api(prompt, system_prompt)
        else:
            return self._generate_local(prompt, system_prompt)
    
    def _generate_via_api(self, prompt: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """Generate using API endpoint"""
        try:
            import requests
            
            full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
            
            start_time = time.time()
            
            response = requests.post(
                self.api_endpoint,
                json={
                    'prompt': full_prompt,
                    'temperature': self.temperature,
                    'max_tokens': self.max_tokens
                },
                timeout=60
            )
            
            end_time = time.time()
            
            if response.status_code == 200:
                data = response.json()
                self.total_requests += 1
                
                return {
                    'response': data.get('text', ''),
                    'model': self.model_name,
                    'tokens_used': data.get('tokens_used', 0),
                    'latency_seconds': end_time - start_time,
                    'finish_reason': 'stop',
                    'success': True,
                    'error': None
                }
            else:
                return {
                    'response': None,
                    'model': self.model_name,
                    'tokens_used': 0,
                    'latency_seconds': end_time - start_time,
                    'finish_reason': None,
                    'success': False,
                    'error': f"API error: {response.status_code}"
                }
                
        except Exception as e:
            return {
                'response': None,
                'model': self.model_name,
                'tokens_used': 0,
                'latency_seconds': 0,
                'finish_reason': None,
                'success': False,
                'error': str(e)
            }
    
    def _generate_local(self, prompt: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """Generate using local model with llama-cpp-python"""
        if not hasattr(self, 'model_loaded') or not self.model_loaded:
            return {
                'response': None,
                'model': self.model_name,
                'tokens_used': 0,
                'latency_seconds': 0,
                'finish_reason': None,
                'success': False,
                'error': 'Local model not loaded. Run download_models.py first.'
            }
        
        try:
            # Format prompt
            full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
            
            start_time = time.time()
            
            # Generate response
            output = self.model(
                full_prompt,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                stop=["</s>", "<|end|>"],
                echo=False
            )
            
            end_time = time.time()
            
            response_text = output['choices'][0]['text']
            tokens_used = output['usage']['total_tokens']
            
            self.total_requests += 1
            self.total_tokens_used += tokens_used
            
            return {
                'response': response_text,
                'model': self.model_name,
                'tokens_used': tokens_used,
                'latency_seconds': end_time - start_time,
                'finish_reason': output['choices'][0].get('finish_reason', 'stop'),
                'success': True,
                'error': None
            }
            
        except Exception as e:
            return {
                'response': None,
                'model': self.model_name,
                'tokens_used': 0,
                'latency_seconds': 0,
                'finish_reason': None,
                'success': False,
                'error': str(e)
            }


class Phi3Model(BaseLLMModel):
    """Microsoft Phi-3 Mini model interface"""
    
    def __init__(self, model_path: Optional[str] = None, **kwargs):
        super().__init__("phi-3-mini", **kwargs)
        self.model_path = model_path or "microsoft/Phi-3-mini-4k-instruct"
        self._load_model()
    
    def _load_model(self):
        """Load Phi-3 model using llama-cpp-python"""
        if not LLAMA_CPP_AVAILABLE:
            print("[ERROR] llama-cpp-python not installed. Install with: pip install llama-cpp-python")
            self.model_loaded = False
            return
        
        try:
            # Find model file
            model_path = self._find_model_file()
            if not model_path:
                print(f"[ERROR] Model file not found in: {self.model_path}")
                self.model_loaded = False
                return
            
            print(f"[INFO] Loading Phi-3 from: {model_path}")
            
            self.model = Llama(
                model_path=str(model_path),
                n_ctx=4096,  # Phi-3 has 4k context window
                n_threads=4,
                n_gpu_layers=-1,  # Use GPU if available
                verbose=False
            )
            
            self.model_loaded = True
            print(f"[SUCCESS] Phi-3 model loaded successfully")
            
        except Exception as e:
            print(f"[ERROR] Failed to load Phi-3 model: {e}")
            self.model_loaded = False
    
    def _find_model_file(self):
        """Find the GGUF model file"""
        if self.model_path == "microsoft/Phi-3-mini-4k-instruct":
            # Use local downloaded model
            models_dir = Path(__file__).parent.parent / "models" / "Phi-3-Mini"
            if not models_dir.exists():
                return None
            self.model_path = str(models_dir)
        
        search_path = Path(self.model_path)
        
        # Look for GGUF files
        gguf_files = list(search_path.glob("*.gguf"))
        if gguf_files:
            return str(gguf_files[0])
        
        return None
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """Generate response using Phi-3 with llama-cpp-python"""
        
        if not hasattr(self, 'model_loaded') or not self.model_loaded:
            return {
                'response': None,
                'model': self.model_name,
                'tokens_used': 0,
                'latency_seconds': 0,
                'finish_reason': None,
                'success': False,
                'error': 'Phi-3 model not loaded. Run download_models.py first.'
            }
        
        try:
            # Format prompt for Phi-3 chat template
            formatted_prompt = f"<|system|>\n{system_prompt}<|end|>\n<|user|>\n{prompt}<|end|>\n<|assistant|>" if system_prompt else f"<|user|>\n{prompt}<|end|>\n<|assistant|>"
            
            start_time = time.time()
            
            # Generate response
            output = self.model(
                formatted_prompt,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                stop=["<|end|>", "</s>"],
                echo=False
            )
            
            end_time = time.time()
            
            response_text = output['choices'][0]['text']
            tokens_used = output['usage']['total_tokens']
            
            self.total_requests += 1
            self.total_tokens_used += tokens_used
            
            return {
                'response': response_text,
                'model': self.model_name,
                'tokens_used': tokens_used,
                'latency_seconds': end_time - start_time,
                'finish_reason': output['choices'][0].get('finish_reason', 'stop'),
                'success': True,
                'error': None
            }
            
        except Exception as e:
            return {
                'response': None,
                'model': self.model_name,
                'tokens_used': 0,
                'latency_seconds': 0,
                'finish_reason': None,
                'success': False,
                'error': str(e)
            }


class ModelFactory:
    """Factory for creating LLM model instances"""
    
    @staticmethod
    def create_model(model_type: str, **kwargs) -> BaseLLMModel:
        """Create a model instance based on type"""
        
        model_map = {
            'gpt-4': GPT4Model,
            'gpt-4-turbo': lambda **k: GPT4Model(model_name='gpt-4-turbo', **k),
            'llama-3': LLaMA3Model,
            'phi-3': Phi3Model
        }
        
        if model_type not in model_map:
            raise ValueError(f"Unknown model type: {model_type}. Available: {list(model_map.keys())}")
        
        return model_map[model_type](**kwargs)


# Example usage
if __name__ == '__main__':
    print("LLM Model Interface")
    print("=" * 50)
    
    # Example: Create GPT-4 model (requires API key)
    # model = ModelFactory.create_model('gpt-4')
    # result = model.generate("Explain what is a timeout in telecom systems.")
    # print(result)
    
    print("\nAvailable models:")
    print("- gpt-4: OpenAI GPT-4 (requires API key)")
    print("- gpt-4-turbo: OpenAI GPT-4 Turbo (requires API key)")
    print("- llama-3: LLaMA-3 8B (local or API)")
    print("- phi-3: Microsoft Phi-3 Mini (local)")
