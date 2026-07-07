from __future__ import annotations
import threading
import time


import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from backend.config.settings import load_agent_config


@dataclass
class ModelStatus:
    mode: str
    provider: str
    model: str
    configured: bool
    reason: str


class ModelFactory:
    """OpenAI-compatible LLM facade with deterministic rule fallback."""

    def __init__(self) -> None:
        self.config = load_agent_config()
        """QPS限制控制变量"""
        self._last_call_time = 0.0
        self._qps_lock = threading.Lock()
        self._min_interval = 1.0 / 5

    def status(self) -> ModelStatus:
        configured = bool(self.config.get("llm_api_key"))
        provider = self.config.get("llm_provider") or "rule"
        model = self.config.get("llm_model") or "rule-agent"
        return ModelStatus(
            mode="llm" if configured else "rule",
            provider=provider,
            model=model,
            configured=configured,
            reason="LLM key configured" if configured else "LLM_API_KEY is empty; using deterministic rule agent",
        )

    def chat(self, messages: list[dict[str, str]], temperature: float = 0.2, timeout: int = 45) -> str:
        #新增QPS限流闸门
        with self._qps_lock:
            now = time.time()
            wait_time = self._last_call_time + self._min_interval - now
            if wait_time > 0:
                time.sleep(wait_time)
            self._last_call_time = time.time()
        status = self.status()
        if not status.configured:
            raise RuntimeError("LLM_API_KEY is empty")
        base_url = (self.config.get("llm_base_url") or "").rstrip("/")
        if not base_url:
            raise RuntimeError("LLM_BASE_URL is empty")
        url = f"{base_url}/chat/completions"
        payload: dict[str, Any] = {
            "model": status.model,
            "messages": messages,
            "temperature": temperature,
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.config['llm_api_key']}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")[:500]
            raise RuntimeError(f"LLM HTTP {exc.code}: {detail}") from exc
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError("LLM response has no choices")
        message = choices[0].get("message") or {}
        content = message.get("content") or choices[0].get("text")
        if not content:
            raise RuntimeError("LLM response has no content")
        return str(content).strip()
