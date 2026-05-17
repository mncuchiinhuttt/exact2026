"""OpenAI-compatible vLLM client."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

import openai


class VLLMClient:
    """Small wrapper around a vLLM OpenAI-compatible chat endpoint."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
    ) -> None:
        """Create a client using explicit args or environment defaults."""
        self.client = openai.OpenAI(
            base_url=base_url or os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1"),
            api_key=api_key or os.environ.get("VLLM_API_KEY", "EMPTY"),
        )
        self.model = model or os.environ.get("VLLM_MODEL", "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B")

    def generate_one(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.6,
        top_p: float = 0.95,
        max_tokens: int = 4096,
        enable_thinking: bool = True,
    ) -> dict[str, str]:
        """Generate one chat completion."""
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            n=1,
            extra_body={"chat_template_kwargs": {"enable_thinking": enable_thinking}},
        )
        choice = resp.choices[0]
        return {
            "content": choice.message.content or "",
            "reasoning": getattr(choice.message, "reasoning_content", None) or "",
        }

    def generate_k(self, messages: list[dict[str, str]], k: int = 5, **kwargs) -> list[dict[str, str]]:
        """Run k independent generations in parallel."""
        with ThreadPoolExecutor(max_workers=max(1, k)) as executor:
            futures = [executor.submit(self.generate_one, messages, **kwargs) for _ in range(k)]
            return [future.result() for future in futures]

