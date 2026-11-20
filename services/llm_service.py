import time
from typing import Dict, Any, List, Optional
import openai
from openai import OpenAI

from config import Config
from prompts.company_system_prompt import build_system_prompt, build_rag_user_prompt


class LLMService:
    """
    Manages interactions with OpenAI Chat Completion models (gpt-4o-mini)
    with cost tracking, strict company prompt enforcement, and latency monitoring.
    """

    def __init__(self):
        self.model = Config.OPENAI_CHAT_MODEL
        self.temperature = Config.LLM_TEMPERATURE
        self.max_tokens = Config.MAX_TOKENS

        # Telemetry metrics
        self._total_api_calls = 0
        self._total_prompt_tokens = 0
        self._total_completion_tokens = 0
        self._total_cost_usd = 0.0

    def _get_api_key(self) -> str:
        key = Config.OPENAI_API_KEY
        if not key:
            raise openai.OpenAIError(
                "OPENAI_API_KEY is not set. Please add OPENAI_API_KEY to your .env file."
            )
        return key

    def generate_answer(
        self,
        question: str,
        context: str,
        allowed_departments: List[str],
        conversation_history: Optional[List[Dict[str, str]]] = None,
        company_name: Optional[str] = None,
        company_role: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Executes grounded company Q&A generation via OpenAI.
        Returns: {
            "answer": str,
            "tokens_used": int,
            "cost_usd": float,
            "latency_ms": float,
            "model": str,
        }
        """
        api_key = self._get_api_key()
        client = OpenAI(api_key=api_key)

        comp_name = company_name or Config.COMPANY_NAME
        comp_role = company_role or Config.COMPANY_ROLE

        # Build enterprise prompts
        system_prompt = build_system_prompt(
            company_name=comp_name,
            company_role=comp_role,
            allowed_departments=allowed_departments,
            response_style=Config.RESPONSE_STYLE,
        )

        user_prompt = build_rag_user_prompt(
            question=question,
            context=context,
            conversation_history=conversation_history,
        )

        start_time = time.time()

        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        latency_ms = (time.time() - start_time) * 1000.0

        answer = response.choices[0].message.content or ""
        usage = response.usage
        prompt_tokens = getattr(usage, "prompt_tokens", len(user_prompt) // 4)
        completion_tokens = getattr(usage, "completion_tokens", len(answer) // 4)
        total_tokens = prompt_tokens + completion_tokens

        # Compute cost based on gpt-4o-mini pricing
        pricing = Config.PRICING.get(self.model, {"input": 0.15, "output": 0.60})
        cost = (
            (prompt_tokens / 1_000_000.0) * pricing["input"]
            + (completion_tokens / 1_000_000.0) * pricing["output"]
        )

        # Update telemetry metrics
        self._total_api_calls += 1
        self._total_prompt_tokens += prompt_tokens
        self._total_completion_tokens += completion_tokens
        self._total_cost_usd += cost

        return {
            "answer": answer,
            "tokens_used": total_tokens,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cost_usd": round(cost, 6),
            "latency_ms": round(latency_ms, 2),
            "model": self.model,
        }

    def get_metrics(self) -> Dict[str, Any]:
        """Returns cumulative LLM usage statistics."""
        return {
            "api_calls_count": self._total_api_calls,
            "total_tokens_used": self._total_prompt_tokens + self._total_completion_tokens,
            "prompt_tokens": self._total_prompt_tokens,
            "completion_tokens": self._total_completion_tokens,
            "total_cost_usd": round(self._total_cost_usd, 5),
            "model": self.model,
        }
