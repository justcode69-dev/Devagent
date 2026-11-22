"""LLM factory: Gemini free tier first, Groq free tier as automatic fallback.

GEMINI_MODEL and GROQ_MODEL accept comma-separated lists. Gemini's free tier has a small
*per-model* daily quota (~20 requests/day), so chaining several models multiplies it; every
model in the chain is tried in order and a 429 / error falls through to the next one.
"""
from __future__ import annotations

from langchain_core.language_models import BaseChatModel

from app.config import get_settings


def _names(value: str) -> list[str]:
    return [n.strip() for n in value.split(",") if n.strip()]


def default_models() -> list[BaseChatModel]:
    s = get_settings()
    models: list[BaseChatModel] = []
    if s.google_api_key:
        from langchain_google_genai import ChatGoogleGenerativeAI
        for name in _names(s.gemini_model):
            # No retries: a 429 on the free tier is usually the daily quota, so fall through fast.
            models.append(ChatGoogleGenerativeAI(
                model=name, google_api_key=s.google_api_key.get_secret_value(),
                temperature=0, max_retries=0,
                # A hung or overloaded (503) request must not stall the run forever.
                timeout=s.llm_timeout_s))
    if s.groq_api_key:
        from langchain_groq import ChatGroq
        for name in _names(s.groq_model):
            models.append(ChatGroq(model=name, api_key=s.groq_api_key.get_secret_value(),
                                   temperature=0, max_retries=2,
                                   # Groq counts max_tokens against the per-minute token
                                   # limit, so a huge default reservation alone can 413.
                                   max_tokens=s.groq_max_tokens,
                                   request_timeout=s.llm_timeout_s))
    if not models:
        raise RuntimeError("Set GOOGLE_API_KEY and/or GROQ_API_KEY in .env")
    return models


def _structured(model: BaseChatModel, schema):
    # Groq's reasoning models (gpt-oss, qwen) break on forced tool-choice structured output
    # ("Tool choice is required, but model did not call a tool"); JSON schema mode is reliable.
    if type(model).__name__ == "ChatGroq":
        return model.with_structured_output(schema, method="json_schema")
    return model.with_structured_output(schema)


class LLM:
    """Wraps an ordered list of chat models and applies fallbacks per capability."""

    def __init__(self, models: list[BaseChatModel]):
        self.models = models

    def _chain(self, runnables):
        head, *rest = runnables
        return head.with_fallbacks(rest) if rest else head

    def plain(self):
        return self._chain(list(self.models))

    def with_tools(self, tools):
        return self._chain([m.bind_tools(tools) for m in self.models])

    def structured(self, schema):
        return self._chain([_structured(m, schema) for m in self.models])
