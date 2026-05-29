"""
DeepSeek API Client for ClawWork
Unified interface for all DeepSeek API calls
"""

import os
import json
from typing import List, Dict, Any, Optional
from litellm import completion
import logging

logger = logging.getLogger(__name__)


class DeepSeekClient:
    """Client for DeepSeek API with cost tracking"""
    
    def __init__(self):
        self.api_key = os.getenv("DEEPSEEK_API_KEY")
        self.api_base = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com/v1")
        self.model = os.getenv("AGENT_MODEL", "deepseek-chat")
        self.eval_model = os.getenv("EVALUATION_MODEL", "deepseek-chat")
        
        if not self.api_key:
            raise ValueError("DEEPSEEK_API_KEY environment variable not set")
        
        logger.info(f"DeepSeek client initialized with model: {self.model}")
    
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2000,
        model: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Call DeepSeek chat completion API"""
        
        model_to_use = model or self.model
        
        try:
            response = completion(
                model=f"deepseek/{model_to_use}",
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                api_key=self.api_key,
                api_base=self.api_base,
                **kwargs
            )
            
            content = response.choices[0].message.content
            usage = {
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
            
            cost = (usage["input_tokens"] * 0.14 + usage["output_tokens"] * 0.28) / 1_000_000
            
            logger.debug(f"API call successful. Tokens: {usage['total_tokens']}, Cost: ${cost:.6f}")
            
            return {
                "content": content,
                "usage": usage,
                "cost": cost
            }
        
        except Exception as e:
            logger.error(f"DeepSeek API error: {str(e)}")
            raise
    
    def evaluate_work(
        self,
        task_description: str,
        submission: str,
        rubric: str = ""
    ) -> Dict[str, Any]:
        """Evaluate work submission using DeepSeek"""
        
        prompt = f"""
You are an expert evaluator. Evaluate the following work submission.

TASK:
{task_description}

SUBMISSION:
{submission}

{f"RUBRIC:{rubric}" if rubric else ""}

Provide:
1. A quality score from 0.0 to 1.0
2. Brief feedback (2-3 sentences)

Format your response as JSON:
{{
    "score": 0.85,
    "feedback": "Your feedback here"
}}
"""
        
        messages = [{"role": "user", "content": prompt}]
        
        result = self.chat_completion(
            messages,
            temperature=0.3,
            max_tokens=500,
            model=self.eval_model
        )
        
        try:
            response_text = result["content"]
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1
            json_str = response_text[json_start:json_end]
            evaluation = json.loads(json_str)
            
            return {
                "score": float(evaluation.get("score", 0.0)),
                "feedback": evaluation.get("feedback", ""),
                "cost": result["cost"]
            }
        except Exception as e:
            logger.error(f"Failed to parse evaluation response: {str(e)}")
            return {
                "score": 0.0,
                "feedback": "Evaluation failed",
                "cost": result["cost"]
            }
    
    def get_model_info(self) -> Dict[str, str]:
        """Get information about configured models"""
        return {
            "agent_model": self.model,
            "eval_model": self.eval_model,
            "api_base": self.api_base,
            "provider": "deepseek"
        }


_client: Optional[DeepSeekClient] = None


def get_client() -> DeepSeekClient:
    """Get or create DeepSeek client"""
    global _client
    if _client is None:
        _client = DeepSeekClient()
    return _client


def call_deepseek(
    messages: List[Dict[str, str]],
    **kwargs
) -> Dict[str, Any]:
    """Convenience function to call DeepSeek"""
    client = get_client()
    return client.chat_completion(messages, **kwargs)


def evaluate_deepseek(
    task_description: str,
    submission: str,
    rubric: str = ""
) -> Dict[str, Any]:
    """Convenience function to evaluate with DeepSeek"""
    client = get_client()
    return client.evaluate_work(task_description, submission, rubric)