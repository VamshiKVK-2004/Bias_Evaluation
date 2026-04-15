"""Gemini provider client with a unified generate interface."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any


class GeminiClient:
    """Minimal Google Gemini generateContent wrapper."""

    def __init__(self, api_key: str | None = None, timeout_s: int = 60) -> None:
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.timeout_s = timeout_s

    def generate(self, prompt: str, model: str, temperature: float, seed: int | None = None) -> dict[str, Any]:
        """Generate text for a prompt.

        Returns a standard response schema:
        {
            "response_text": str,
            "latency_ms": int,
            "error": str | None,
            "raw": dict | None,
        }
        """
        start = time.perf_counter()
        if not self.api_key:
            return {
                "response_text": "",
                "latency_ms": int((time.perf_counter() - start) * 1000),
                "error": "missing GEMINI_API_KEY/GOOGLE_API_KEY",
                "raw": None,
            }

        endpoint = self._build_endpoint(model)
        payload: dict[str, Any] = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": temperature},
        }
        if seed is not None:
            payload["generationConfig"]["seed"] = seed

        request = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                raw = json.loads(response.read().decode("utf-8"))
            candidates = raw.get("candidates", [])
            parts = candidates[0].get("content", {}).get("parts", []) if candidates else []
            text = "".join(part.get("text", "") for part in parts)

            if not str(text).strip():
                prompt_feedback = raw.get("promptFeedback") or {}
                block_reason = prompt_feedback.get("blockReason")
                finish_reason = candidates[0].get("finishReason") if candidates else None
                reason_bits = [
                    value for value in [block_reason, finish_reason] if value
                ]
                reason = ", ".join(reason_bits) if reason_bits else "empty response from Gemini"
                return {
                    "response_text": "",
                    "latency_ms": int((time.perf_counter() - start) * 1000),
                    "error": f"gemini empty output ({reason})",
                    "raw": raw,
                }

            return {
                "response_text": text,
                "latency_ms": int((time.perf_counter() - start) * 1000),
                "error": None,
                "raw": raw,
            }
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            fallback_result = self._try_model_fallback(
                exc=exc,
                body=body,
                prompt=prompt,
                model=model,
                temperature=temperature,
                seed=seed,
                start=start,
            )
            if fallback_result is not None:
                return fallback_result
            return {
                "response_text": "",
                "latency_ms": int((time.perf_counter() - start) * 1000),
                "error": f"http_{exc.code}: {body}",
                "raw": None,
            }
        except Exception as exc:  # pragma: no cover - network/runtime dependent
            return {
                "response_text": "",
                "latency_ms": int((time.perf_counter() - start) * 1000),
                "error": str(exc),
                "raw": None,
            }

    def _build_endpoint(self, model: str) -> str:
        return (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{urllib.parse.quote(model)}:generateContent?key={urllib.parse.quote(self.api_key or '')}"
        )

    def _try_model_fallback(
        self,
        *,
        exc: urllib.error.HTTPError,
        body: str,
        prompt: str,
        model: str,
        temperature: float,
        seed: int | None,
        start: float,
    ) -> dict[str, Any] | None:
        if exc.code != 404 or "-latest" in model:
            return None
        if "is not found" not in body:
            return None

        fallback_model = f"{model}-latest"
        request = urllib.request.Request(
            self._build_endpoint(fallback_model),
            data=json.dumps(
                {
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": (
                        {"temperature": temperature, **({"seed": seed} if seed is not None else {})}
                    ),
                }
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                raw = json.loads(response.read().decode("utf-8"))
            candidates = raw.get("candidates", [])
            parts = candidates[0].get("content", {}).get("parts", []) if candidates else []
            text = "".join(part.get("text", "") for part in parts)
            if not str(text).strip():
                return {
                    "response_text": "",
                    "latency_ms": int((time.perf_counter() - start) * 1000),
                    "error": f"gemini empty output after fallback model={fallback_model}",
                    "raw": raw,
                }
            return {
                "response_text": text,
                "latency_ms": int((time.perf_counter() - start) * 1000),
                "error": None,
                "raw": raw,
            }
        except urllib.error.HTTPError:
            return None
