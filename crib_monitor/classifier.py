"""Ask a vision model how the baby is lying."""

from __future__ import annotations

import asyncio
import base64
import json
import re
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .config import ClassifierConfig, ModelConfig
from .labels import Position

PROMPT = """You are checking an image from a camera above a baby's crib to see how the baby is lying.
The image may be grayscale night vision (infrared). There are no blankets in the crib.

Classify the baby's position:
- "back": lying face up, chest and face toward the ceiling.
- "stomach": lying face down, chest against the mattress, back of the head or back toward the ceiling.
- "side": lying on either side.
- "unclear": a baby is in the crib but you cannot tell the position.
- "not_visible": no baby is visible in the crib.

If you are unsure between "stomach" and another position, answer "stomach".

Respond with JSON only: {"position": "<one of the values above>"}"""

RESPONSE_FORMAT: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "crib_position",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {"position": {"type": "string", "enum": [p.value for p in Position]}},
            "required": ["position"],
            "additionalProperties": False,
        },
    },
}

_FENCE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.S)

Completion = Callable[..., Awaitable[Any]]


@dataclass(frozen=True)
class ClassifyResult:
    model_name: str
    position: Position | None
    latency_s: float
    error: str | None = None


def parse_position(content: str | None) -> Position:
    if not content:
        raise ValueError("empty response")
    text = content.strip()
    fenced = _FENCE.match(text)
    if fenced:
        text = fenced.group(1)
    data = json.loads(text)
    if not isinstance(data, dict) or "position" not in data:
        raise ValueError(f"no position in {text[:80]!r}")
    return Position(data["position"])


async def _litellm_completion(**kwargs: Any) -> Any:
    import litellm  # imported lazily: slow to import, and tests inject a fake

    return await litellm.acompletion(**kwargs)


class Classifier:
    def __init__(self, cfg: ModelConfig, api_key: str | None, completion: Completion | None = None) -> None:
        self._cfg = cfg
        self._api_key = api_key
        self._completion = completion or _litellm_completion
        self.name = cfg.name

    async def classify(self, jpeg: bytes) -> ClassifyResult:
        start = time.monotonic()
        try:
            response = await asyncio.wait_for(self._completion(**self._request(jpeg)), timeout=self._cfg.timeout_s)
            position = parse_position(response.choices[0].message.content)
        except Exception as exc:  # any failure means "unavailable", never "back"
            return ClassifyResult(self.name, None, time.monotonic() - start, f"{type(exc).__name__}: {exc}")
        return ClassifyResult(self.name, position, time.monotonic() - start)

    def _request(self, jpeg: bytes) -> dict[str, Any]:
        image_url = "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()
        kwargs: dict[str, Any] = {
            "model": self._cfg.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": PROMPT},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                }
            ],
            "response_format": RESPONSE_FORMAT,
            "temperature": 0,
            "max_tokens": 300,
            "timeout": self._cfg.timeout_s,
            "drop_params": True,
            # OpenAI-compatible local servers still require a non-empty key.
            "api_key": self._api_key or "none",
        }
        if self._cfg.api_base:
            kwargs["api_base"] = self._cfg.api_base
        if self._cfg.extra_body:
            kwargs["extra_body"] = self._cfg.extra_body
        return kwargs


def build_classifiers(cfg: ClassifierConfig, env: Mapping[str, str]) -> list[Classifier]:
    return [Classifier(m, env.get(m.api_key_env) if m.api_key_env else None) for m in cfg.models]


async def classify_all(classifiers: Sequence[Any], jpeg: bytes) -> list[ClassifyResult]:
    return list(await asyncio.gather(*(c.classify(jpeg) for c in classifiers)))
