"""Ollama 호출. 모든 호출은 JSON 스키마로 출력 형식을 강제한다."""

import json
from typing import Any, Protocol

import httpx

DEFAULT_MODEL = "qwen3:14b"
DEFAULT_HOST = "http://localhost:11434"


class LLMError(Exception):
    """출력 형식 오류 등. 역할 함수에서 기본값으로 대체 처리한다."""


class LLMUnavailable(Exception):
    """Ollama 서버에 연결할 수 없음. 게임을 진행할 수 없으므로 UI까지 올려보낸다."""


class LLM(Protocol):
    def chat_json(
        self, system: str, user: str, schema: dict, temperature: float = 0.3
    ) -> dict[str, Any]: ...


class OllamaClient:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        host: str = DEFAULT_HOST,
        timeout: float = 120.0,
        retries: int = 2,
        think: bool = False,
    ):
        """think: qwen3 thinking 모드. 정확도는 오르지만 느려진다."""
        self.model = model
        self.host = host.rstrip("/")
        self.retries = retries
        self.think = think
        self._http = httpx.Client(timeout=timeout)

    def chat_json(
        self, system: str, user: str, schema: dict, temperature: float = 0.3
    ) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "stream": False,
            "think": self.think,
            "format": schema,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "options": {"temperature": temperature},
        }
        last_error: Exception | None = None
        for _ in range(self.retries + 1):
            try:
                try:
                    resp = self._http.post(f"{self.host}/api/chat", json=payload)
                except httpx.ConnectError as e:
                    raise LLMUnavailable(
                        f"Ollama 서버({self.host})에 연결할 수 없습니다. Ollama가 실행 중인지 확인하세요."
                    ) from e
                resp.raise_for_status()
                content = resp.json()["message"]["content"]
                data = json.loads(content)
                missing = [k for k in schema.get("required", []) if k not in data]
                if missing:
                    raise LLMError(f"필수 필드 누락: {missing}")
                return data
            except (httpx.HTTPError, json.JSONDecodeError, KeyError, LLMError) as e:
                last_error = e
        raise LLMError(f"LLM 호출 실패: {last_error}")
