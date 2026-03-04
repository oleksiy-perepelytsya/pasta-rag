import asyncio
import logging
from typing import Optional

import google.generativeai as genai

from config import settings

logger = logging.getLogger(__name__)


def _build_gemini_prompt(messages: list[dict], system_prompt: Optional[str]) -> str:
    parts = []
    if system_prompt:
        parts.append(system_prompt)
        parts.append("")
    for msg in messages:
        role = "User" if msg["role"] == "user" else "Assistant"
        parts.append(f"{role}: {msg['content']}")
    return "\n".join(parts)


class LLMClient:
    def __init__(self):
        self._gemini_models: dict[str, genai.GenerativeModel] = {}

        if settings.gemini_api_key:
            genai.configure(api_key=settings.gemini_api_key)

        if settings.anthropic_api_key:
            from anthropic import Anthropic
            self._anthropic = Anthropic(api_key=settings.anthropic_api_key)
        else:
            self._anthropic = None

        if settings.openai_api_key:
            from openai import AsyncOpenAI
            self._openai = AsyncOpenAI(api_key=settings.openai_api_key)
        else:
            self._openai = None

        logger.info("LLMClient initialized")

    def _get_gemini_model(self, model: str) -> genai.GenerativeModel:
        if model not in self._gemini_models:
            self._gemini_models[model] = genai.GenerativeModel(model)
        return self._gemini_models[model]

    async def call_gemini(
        self,
        model: str,
        messages: list[dict],
        max_output_tokens: 800,
        system_prompt: Optional[str] = None,
    ) -> str:
        full_prompt = _build_gemini_prompt(messages, system_prompt)
        loop = asyncio.get_running_loop()

        def _sync() -> str:
            m = self._get_gemini_model(model)
            resp = m.generate_content(full_prompt)
            return resp.text

        return await loop.run_in_executor(None, _sync)

    async def call_claude(
        self,
        model: str,
        messages: list[dict],
        system_prompt: Optional[str] = None,
    ) -> str:
        if not self._anthropic:
            raise ValueError("Anthropic API key not configured")
        loop = asyncio.get_running_loop()

        def _sync() -> str:
            params: dict = {
                "model": model,
                "max_tokens": 4096,
                "messages": messages,
            }
            if system_prompt:
                params["system"] = system_prompt
            resp = self._anthropic.messages.create(**params)
            return resp.content[0].text

        return await loop.run_in_executor(None, _sync)

    async def call_openai(
        self,
        model: str,
        messages: list[dict],
        system_prompt: Optional[str] = None,
    ) -> str:
        if not self._openai:
            raise ValueError("OpenAI API key not configured")
        all_messages = []
        if system_prompt:
            all_messages.append({"role": "system", "content": system_prompt})
        all_messages.extend(messages)
        resp = await self._openai.chat.completions.create(
            model=model,
            messages=all_messages,
            max_tokens=4096,
        )
        return resp.choices[0].message.content

    async def call_model(
        self,
        model: str,
        messages: list[dict],
        system_prompt: Optional[str] = None,
    ) -> str:
        logger.info(f"LLM call: model={model}, messages={len(messages)}")
        if model.startswith("gemini-") or model.startswith("models/gemini"):
            return await self.call_gemini(model, messages, system_prompt)
        elif model.startswith("claude-"):
            return await self.call_claude(model, messages, system_prompt)
        elif model.startswith(("gpt-", "o1", "o3", "o4")):
            return await self.call_openai(model, messages, system_prompt)
        else:
            logger.warning(f"Unknown model prefix '{model}', defaulting to Gemini")
            return await self.call_gemini(model, messages, system_prompt)
