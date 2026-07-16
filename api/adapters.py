# -*- coding: utf-8 -*-
"""
OpenAI-compatible adapter — works with Grok (xAI), DeepSeek, and any
/chat/completions endpoint. Zero external dependencies (stdlib only).
"""
from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass


class AdapterError(Exception):
    pass


@dataclass
class OpenAICompatAdapter:
    base_url: str
    model: str
    api_key: str | None = None
    temperature: float = 0.3
    max_tokens: int = 400
    timeout: int = 30
    name: str = ""

    def complete(self, system: str, user: str) -> str:
        body = json.dumps({
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }).encode()
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(
            self.base_url.rstrip("/") + "/chat/completions",
            data=body, headers=headers,
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                data = json.loads(r.read())
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")[:200]
            raise AdapterError(f"{self.name or self.model}: HTTP {e.code} — {body}") from e
        except Exception as e:
            raise AdapterError(f"{self.name or self.model}: {e}") from e
        try:
            return data["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError) as e:
            raise AdapterError(f"{self.name or self.model}: invalid response") from e
