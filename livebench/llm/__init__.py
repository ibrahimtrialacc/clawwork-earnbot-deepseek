"""
DeepSeek LLM Module for ClawWork
"""

from .deepseek_client import (
    DeepSeekClient,
    get_client,
    call_deepseek,
    evaluate_deepseek,
)

__all__ = [
    "DeepSeekClient",
    "get_client",
    "call_deepseek",
    "evaluate_deepseek",
]

__version__ = "1.0.0"
__author__ = "ClawWork Team"
__description__ = "DeepSeek API integration for ClawWork"