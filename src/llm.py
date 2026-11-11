"""Model access through the OpenAI SDK pointed at OpenRouter, with a response cache and retry with backoff.

- Calls are made at temperature 0 and cached on (model, prompt text), so a rerun of the same tickets gives the same
  replies, costs nothing and is quick. The cache is a local SQLite file (git-ignored).
- Transient failures (timeouts, rate limits, 5xx) are retried with exponential backoff and jitter. Other failures are
  not retried. complete() never raises: it returns an LLMResult with `error` set so the caller can escalate (A11).
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import sqlite3
import threading
import time
from dataclasses import dataclass

from .config import Settings, get_settings

log = logging.getLogger(__name__)


@dataclass
class LLMResult:
    text: str = ""
    model: str = ""
    model_version: str = ""  # the model id the provider reports having served
    cached: bool = False
    attempts: int = 0
    error: str | None = None


class LLMClient:
    def __init__(self, settings: Settings | None = None, client=None, sleep=time.sleep):
        self.settings = settings or get_settings()
        self._client = client
        self._sleep = sleep
        self._lock = threading.Lock()
        self.calls = 0  # real network calls made (cache misses)
        self.cache_hits = 0
        path = self.settings.cache_path
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._db = sqlite3.connect(path, check_same_thread=False)
            self._db.execute("create table if not exists cache (k text primary key, v text)")
            self._db.commit()
        except Exception as exc:  # the cache is an optimisation, never a dependency
            log.warning("LLM cache unavailable: %s", exc)
            self._db = None

    # ------------------------------------------------------------------ cache
    @staticmethod
    def _key(model: str, system: str, user: str, max_tokens: int) -> str:
        return hashlib.sha256(json.dumps([model, system, user, max_tokens]).encode()).hexdigest()

    def _get(self, k: str):
        if self._db is None:
            return None
        try:
            with self._lock:
                row = self._db.execute("select v from cache where k=?", (k,)).fetchone()
            return json.loads(row[0]) if row else None
        except Exception:
            return None

    def _put(self, k: str, v: dict) -> None:
        if self._db is None:
            return
        try:
            with self._lock:
                self._db.execute("insert or replace into cache values (?,?)", (k, json.dumps(v)))
                self._db.commit()
        except Exception as exc:
            log.warning("LLM cache write failed: %s", exc)

    # ---------------------------------------------------------------- network
    def _sdk(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(
                api_key=self.settings.openrouter_api_key,
                base_url=self.settings.openrouter_base_url,
                timeout=self.settings.llm_timeout_seconds,
                max_retries=0,  # retries are ours, so they are counted and visible
            )
        return self._client

    @staticmethod
    def _retryable(exc: Exception) -> bool:
        status = getattr(exc, "status_code", None)
        if status is not None:
            return status == 429 or status >= 500
        return type(exc).__name__ in ("APITimeoutError", "APIConnectionError", "TimeoutError", "ConnectionError")

    def complete(self, system: str, user: str, max_tokens: int = 700) -> LLMResult:
        model = self.settings.model_name
        key = self._key(model, system, user, max_tokens)
        hit = self._get(key)
        if hit:
            self.cache_hits += 1
            return LLMResult(hit["text"], model, hit.get("model_version", ""), cached=True)
        if not self.settings.openrouter_api_key:
            return LLMResult(model=model, error="no_api_key")
        last = ""
        for attempt in range(1, self.settings.llm_max_retries + 2):
            try:
                self.calls += 1
                r = self._sdk().chat.completions.create(
                    model=model,
                    messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                    temperature=0,
                    max_tokens=max_tokens,
                )
                text = (r.choices[0].message.content or "").strip()
                if not text:
                    raise ValueError("empty completion")
                version = getattr(r, "model", "") or model
                self._put(key, {"text": text, "model_version": version})
                return LLMResult(text, model, version, attempts=attempt)
            except Exception as exc:
                last = f"{type(exc).__name__}: {str(exc)[:160]}"
                retry = attempt <= self.settings.llm_max_retries and (
                    self._retryable(exc) or isinstance(exc, ValueError)
                )
                log.warning("LLM call failed (attempt %s): %s%s", attempt, last, "; retrying" if retry else "")
                if not retry:
                    return LLMResult(model=model, attempts=attempt, error=last)
                self._sleep(min(30.0, 2 ** (attempt - 1)) * (0.5 + random.random()))
        return LLMResult(model=model, attempts=attempt, error=last)  # pragma: no cover
