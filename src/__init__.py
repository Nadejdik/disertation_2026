"""
LLM Evaluation Framework for Telecom Fault Detection
Dissertation Project - Source Code Package
"""

__version__ = '1.0.0'
__author__ = 'Your Name'

from .llm_interface import BaseLLMModel, GPT4Model, LLaMA3Model, Phi3Model, ModelFactory
from .evaluation_framework import LLMEvaluator, PromptTemplate
from .metrics_calculator import MetricsCalculator
from .config import Config

__all__ = [
    'BaseLLMModel',
    'GPT4Model',
    'LLaMA3Model',
    'Phi3Model',
    'ModelFactory',
    'LLMEvaluator',
    'PromptTemplate',
    'MetricsCalculator',
    'Config'
]
