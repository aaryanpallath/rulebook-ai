"""A thin client for a local Ollama server: chat (with optional tools) and embeddings. Everything runs on the
laptop, so evaluation costs no API money; token counts are reported so cost on a hosted model can be estimated."""

from __future__ import annotations

import http.client
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
CHAT_MODEL = os.environ.get("RULEBOOK_MODEL", "gemma4")
EMBED_MODEL = os.environ.get("RULEBOOK_EMBED_MODEL", "nomic-embed-text")


def _post(path: str, body: dict, timeout: float = 600, attempts: int = 4) -> dict:
    """POST to Ollama, retrying dropped connections and server errors (a local server can restart mid-run)."""
    for i in range(attempts):
        req = urllib.request.Request(
            f"{HOST}{path}", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())
        except (urllib.error.URLError, ConnectionError, http.client.HTTPException):
            if i == attempts - 1:
                raise
            time.sleep(5 * 2**i)  # 5, 10, 20 s: long enough for the server to come back
    raise RuntimeError("unreachable")


def embed(texts: list[str], batch: int = 32) -> list[list[float]]:
    out: list[list[float]] = []
    for i in range(0, len(texts), batch):
        out += _post("/api/embed", {"model": EMBED_MODEL, "input": texts[i : i + batch]})["embeddings"]
    return out


@dataclass
class Reply:
    content: str
    tool_calls: list[dict] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    seconds: float = 0.0
    message: dict = field(default_factory=dict)


def chat(messages: list[dict], tools: list[dict] | None = None, model: str = CHAT_MODEL) -> Reply:
    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
        "think": False,
        # 8k fits the longest agent run; a 16k context pushed this 16 GB laptop past its memory and crashed Ollama
        "options": {"temperature": 0, "seed": 7, "num_ctx": 8192},
    }
    if tools:
        body["tools"] = tools
    t0 = time.monotonic()
    r = _post("/api/chat", body)
    msg = r["message"]
    return Reply(
        msg.get("content", ""),
        msg.get("tool_calls") or [],
        r.get("prompt_eval_count", 0),
        r.get("eval_count", 0),
        time.monotonic() - t0,
        msg,
    )
