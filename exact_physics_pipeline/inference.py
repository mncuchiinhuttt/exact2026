"""vLLM OpenAI-compatible inference calls."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from exact_physics_pipeline import config
from exact_physics_pipeline.parsing import parse_generation_answer


def create_client() -> Any:
    """Create an OpenAI-compatible client pointed at the local vLLM server."""
    from openai import OpenAI

    return OpenAI(base_url=config.BASE_URL, api_key=config.API_KEY)


def call_vllm_once(
    messages: list[dict[str, str]],
    max_tokens: int | None = None,
    enable_thinking: bool | None = None,
) -> dict[str, Any]:
    """Run one independent vLLM chat completion and parse its final answer."""
    client = create_client()
    extra_body = dict(config.EXTRA_BODY)
    if enable_thinking is not None:
        extra_body = {"chat_template_kwargs": {"enable_thinking": enable_thinking}}
    response = client.chat.completions.create(
        model=config.MODEL_NAME,
        messages=messages,
        temperature=config.TEMPERATURE,
        top_p=config.TOP_P,
        max_tokens=max_tokens or config.MAX_TOKENS,
        n=1,
        extra_body=extra_body,
    )
    message = response.choices[0].message
    think_text = getattr(message, "reasoning_content", None)
    final_text = message.content
    parsed_answer = parse_generation_answer(final_text, think_text)
    if parsed_answer.get("error") == "parse_failed":
        parsed_answer["final_excerpt"] = (final_text or "")[-600:]
        parsed_answer["reasoning_excerpt"] = (think_text or "")[-600:]
    return {
        "think_text": think_text or "",
        "final_text": final_text or "",
        "parsed_answer": parsed_answer,
    }


def run_self_consistency(
    messages: list[dict[str, str]],
    k: int = config.DEFAULT_K,
    max_tokens: int | None = None,
    enable_thinking: bool | None = None,
) -> list[dict[str, Any]]:
    """Run k independent vLLM calls in parallel and return generation records."""
    results: list[dict[str, Any] | None] = [None] * k
    with ThreadPoolExecutor(max_workers=min(config.MAX_WORKERS, k)) as executor:
        future_to_index = {
            executor.submit(call_vllm_once, messages, max_tokens, enable_thinking): index
            for index in range(k)
        }
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            try:
                results[index] = future.result()
            except Exception as exc:
                results[index] = {
                    "think_text": "",
                    "final_text": "",
                    "parsed_answer": {
                        "answer": None,
                        "unit": None,
                        "error": f"inference_failed: {exc}",
                    },
                }
    return [result for result in results if result is not None]
