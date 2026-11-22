"""One chat interface for every model provider used with RoboLiveMem.

    from robolivemem.models import Chat
    chat = Chat("openai:gpt-4.1-mini")              # or anthropic:..., gemini:..., bedrock:..., paper alias, mock:...
    text, meta = chat(system_prompt, [("user", ["What do you see?", rgb_image])])
    meta["latency_s"]                                # wall-clock seconds of the request: what the live clock charges

A turn is (role, parts) with role "user" or "assistant". A part is a string, an RGB image (numpy array, HxWx3 uint8),
or a dict {"text": ...}, {"png": <base64>} or {"jpeg": <base64>}.

Keys are read from environment variables only (see .env.example); nothing is ever written to disk or logged:
    openai:<model>        OPENAI_API_KEY   (OPENAI_BASE_URL for any OpenAI-compatible server: vLLM, Ollama, OpenRouter, ...)
    anthropic:<model>     ANTHROPIC_API_KEY
    gemini:<model>        GEMINI_API_KEY   (or GOOGLE_API_KEY)
    bedrock:<model-id>    AWS_BEARER_TOKEN_BEDROCK and AWS_REGION (Amazon Bedrock API key)
    mock:<reply>          no key; always answers <reply> (for tests and dry runs)
The short names used in the paper (opus-5.5, gpt-6-astra, ...) are listed in PAPER_MODELS.
"""
from __future__ import annotations

import base64
import io
import json
import os
import re
import time
from typing import Any, Iterable
from urllib.parse import quote

__all__ = ["Chat", "BedrockChat", "PAPER_MODELS", "MODELS", "parse_json_reply", "available_providers"]

# Short names used in the paper -> (provider spec, reasoning effort). All of them ran on Amazon Bedrock except Gemini.
PAPER_MODELS: dict[str, tuple[str, str | None]] = {
    "opus-5.5": ("bedrock:us.anthropic.claude-opus-5-5", None),
    "opus-5.5-think": ("bedrock:us.anthropic.claude-opus-5-5", "think"),
    "opus-5.5-think-high": ("bedrock:us.anthropic.claude-opus-5-5", "think-high"),
    "sonnet-5.5": ("bedrock:us.anthropic.claude-sonnet-5-5", None),
    "haiku-5.5": ("bedrock:us.anthropic.claude-haiku-5-5", None),
    "fable-5.1": ("bedrock:us.anthropic.claude-fable-5-1", None),
    "gpt-6-astra": ("bedrock:us.openai.gpt-6-astra", "low"),
    "gpt-6-astra-med": ("bedrock:us.openai.gpt-6-astra", "medium"),
    "gpt-6.1-sol": ("bedrock:us.openai.gpt-6.1-sol", "low"),
    "gpt-6.1-terra": ("bedrock:us.openai.gpt-6.1-terra", "low"),
    "gpt-6-luna": ("bedrock:us.openai.gpt-6-luna", "low"),
    "grok-4.7": ("bedrock:us.xai.grok-4.7", None),
    "kimi-k3": ("bedrock:us.moonshotai.kimi-k3", "low"),
    "kimi-k2.5": ("bedrock:moonshotai.kimi-k2.5", None),
    "qwen3-vl-235b": ("bedrock:qwen.qwen3-vl-235b-a22b", None),
    "llama4-maverick": ("bedrock:us.meta.llama4-maverick-17b-instruct-v1:0", None),
    "mistral-large-3": ("bedrock:mistral.mistral-large-3-675b-instruct", None),
    "ministral-3-14b": ("bedrock:mistral.ministral-3-14b-instruct", None),
    "gemma-3-27b": ("bedrock:google.gemma-3-27b-it", None),
    "glm-5.3": ("bedrock:us.zai.glm-5.3", "low"),
    "nova-2-lite": ("bedrock:us.amazon.nova-2-lite-v1:0", None),
    "gemini-4-argon": ("gemini:gemini-4-argon", None),
    "gemini-4-carbon": ("gemini:gemini-4-carbon", None),
}
MODELS = PAPER_MODELS  # backwards-compatible name used by the tier code

PROVIDERS = ("openai", "anthropic", "gemini", "bedrock", "mock")
_KEYS = {"openai": ("OPENAI_API_KEY",), "anthropic": ("ANTHROPIC_API_KEY",),
         "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"), "bedrock": ("AWS_BEARER_TOKEN_BEDROCK",), "mock": ()}


def available_providers() -> list[str]:
    """Providers whose key is set in the environment (mock is always available)."""
    return [p for p, names in _KEYS.items() if not names or any(os.environ.get(n) for n in names)]


def _key(provider: str) -> str:
    for name in _KEYS[provider]:
        v = os.environ.get(name, "").strip()
        if v:
            return v
    raise RuntimeError(f"{provider}: set {' or '.join(_KEYS[provider])} in your environment (see .env.example)")


def _png_b64(img) -> str:
    import numpy as np
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(np.asarray(img, dtype=np.uint8)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _norm_part(p: Any) -> dict:
    """-> {"text": str} or {"image": base64, "mime": "image/png" | "image/jpeg"}"""
    if isinstance(p, str):
        return {"text": p}
    if isinstance(p, dict):
        if "text" in p:
            return {"text": str(p["text"])}
        if "jpeg" in p:
            return {"image": p["jpeg"], "mime": "image/jpeg"}
        if "png" in p:
            return {"image": p["png"], "mime": "image/png"}
        raise ValueError(f"unknown part keys {sorted(p)}")
    return {"image": _png_b64(p), "mime": "image/png"}


def _turns(turns: Iterable) -> list[tuple[str, list[dict]]]:
    out = []
    for role, parts in turns:
        if isinstance(parts, (str, dict)) or hasattr(parts, "shape"):
            parts = [parts]
        out.append(("assistant" if role in ("assistant", "model") else "user", [_norm_part(p) for p in parts]))
    return out


class Chat:
    """A multi-turn chat model. Calling it returns (text, meta); it never raises on API errors (meta["error"])."""

    def __init__(self, spec: str, max_tokens: int = 1024, timeout_s: float = 240.0, effort: str | None = None,
                 extra: dict | None = None):
        if spec in PAPER_MODELS:
            self.short = spec
            spec, eff = PAPER_MODELS[spec]
            effort = effort or eff
        else:
            self.short = spec
        if ":" not in spec:
            raise ValueError(f"model spec must be <provider>:<model> or a paper alias, got {spec!r}; "
                             f"providers: {', '.join(PROVIDERS)}; aliases: {', '.join(PAPER_MODELS)}")
        self.provider, self.model_id = spec.split(":", 1)
        if self.provider not in PROVIDERS:
            raise ValueError(f"unknown provider {self.provider!r}; use one of {', '.join(PROVIDERS)}")
        self.spec, self.effort, self.max_tokens, self.timeout_s = spec, effort, max_tokens, timeout_s
        self.extra = dict(extra or {})  # merged into the request body (provider-specific options)
        self.route = "responses" if (self.provider == "bedrock" and ".openai." in self.model_id) else self.provider

    def __repr__(self):
        return f"Chat({self.spec!r}, effort={self.effort!r})"

    # ------------------------------------------------------------------ public
    def __call__(self, system: str, turns: Iterable) -> tuple[str, dict]:
        turns = _turns(turns)
        meta = {"model": self.short, "model_id": self.model_id, "provider": self.provider, "effort": self.effort}
        if self.provider == "mock":
            meta.update(latency_s=0.0, in_tokens=0, out_tokens=0)
            return self.model_id, meta
        try:
            url, headers, body = getattr(self, f"_req_{self.route}")(system, turns)
        except RuntimeError as exc:  # missing key
            meta.update(latency_s=0.0, error=str(exc))
            return "", meta
        body.update(self.extra)
        return self._post(url, headers, body, meta)

    # ------------------------------------------------------------------ payloads
    def _req_openai(self, system, turns):
        base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        msgs = [{"role": "system", "content": system}] if system else []
        for role, parts in turns:
            content = [{"type": "text", "text": p["text"]} if "text" in p else
                       {"type": "image_url", "image_url": {"url": f"data:{p['mime']};base64,{p['image']}"}}
                       for p in parts]
            if role == "assistant":
                content = "".join(p["text"] for p in parts if "text" in p)
            msgs.append({"role": role, "content": content})
        body = {"model": self.model_id, "messages": msgs, "max_completion_tokens": self.max_tokens}
        if self.effort:
            body["reasoning_effort"] = self.effort
        return f"{base}/chat/completions", {"Authorization": f"Bearer {_key('openai')}"}, body

    def _req_anthropic(self, system, turns):
        msgs = []
        for role, parts in turns:
            content = [{"type": "text", "text": p["text"]} if "text" in p else
                       {"type": "image", "source": {"type": "base64", "media_type": p["mime"], "data": p["image"]}}
                       for p in parts]
            msgs.append({"role": role, "content": content})
        body = {"model": self.model_id, "max_tokens": self.max_tokens, "messages": msgs}
        if system:
            body["system"] = system
        return ("https://api.anthropic.com/v1/messages",
                {"x-api-key": _key("anthropic"), "anthropic-version": "2023-06-01"}, body)

    def _req_gemini(self, system, turns):
        contents = []
        for role, parts in turns:
            ps = [{"text": p["text"]} if "text" in p else {"inline_data": {"mime_type": p["mime"], "data": p["image"]}}
                  for p in parts]
            contents.append({"role": "model" if role == "assistant" else "user", "parts": ps})
        body = {"contents": contents, "generationConfig": {"maxOutputTokens": self.max_tokens}}
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_id}:generateContent"
        return url, {"x-goog-api-key": _key("gemini")}, body

    def _bedrock_runtime(self):
        return f"https://bedrock-runtime.{os.environ.get('AWS_REGION', 'us-west-2')}.amazonaws.com"

    def _req_bedrock(self, system, turns):
        """Bedrock Converse API (Claude, Grok, Kimi, Qwen, Gemma, Mistral, Nova, ...)."""
        msgs = []
        for role, parts in turns:
            content = []
            for p in parts:
                if "text" in p:
                    content.append({"text": p["text"]})
                else:
                    content.append({"image": {"format": p["mime"].split("/")[1], "source": {"bytes": p["image"]}}})
            msgs.append({"role": role, "content": content})
        body = {"messages": msgs, "inferenceConfig": {"maxTokens": self.max_tokens}}
        if system:
            body["system"] = [{"text": system}]
        if self.effort and self.effort.startswith("think") and ".anthropic." in self.model_id:
            # adaptive thinking; "think" = medium effort, "think-high" = high effort
            level = self.effort.split("-", 1)[1] if "-" in self.effort else "medium"
            body["inferenceConfig"]["maxTokens"] = max(self.max_tokens, 16000)
            body["additionalModelRequestFields"] = {"thinking": {"type": "adaptive"}, "output_config": {"effort": level}}
        elif self.effort:
            body["additionalModelRequestFields"] = {"reasoning_effort": self.effort}
        url = f"{self._bedrock_runtime()}/model/{quote(self.model_id, safe='')}/converse"
        return url, {"Authorization": f"Bearer {_key('bedrock')}"}, body

    def _req_responses(self, system, turns):
        """OpenAI models on Bedrock (OpenAI-compatible Responses API)."""
        inp = []
        for role, parts in turns:
            content = []
            for p in parts:
                if "text" in p:
                    content.append({"type": "input_text" if role == "user" else "output_text", "text": p["text"]})
                else:
                    content.append({"type": "input_image", "image_url": f"data:{p['mime']};base64,{p['image']}"})
            inp.append({"role": role, "content": content})
        body = {"model": self.model_id, "input": inp, "max_output_tokens": self.max_tokens, "store": False}
        if system:
            body["instructions"] = system
        if self.effort:
            body["reasoning"] = {"effort": self.effort}
        return f"{self._bedrock_runtime()}/openai/v1/responses", {"Authorization": f"Bearer {_key('bedrock')}"}, body

    # ------------------------------------------------------------------ transport
    def _post(self, url, headers, body, meta):
        import requests
        headers = dict(headers, **{"Content-Type": "application/json"})
        secret = next((v for k, v in headers.items() if k.lower() in ("authorization", "x-api-key", "x-goog-api-key")), "")
        t0 = time.monotonic()
        try:
            r = requests.post(url, headers=headers, json=body, timeout=(10, self.timeout_s), allow_redirects=False)
        except requests.RequestException as exc:
            meta.update(latency_s=round(time.monotonic() - t0, 3), error=type(exc).__name__)
            return "", meta
        meta["latency_s"] = round(time.monotonic() - t0, 3)
        meta["http_status"] = r.status_code
        try:
            data = r.json()
        except ValueError:
            meta["error"] = f"non-JSON HTTP {r.status_code}"
            return "", meta
        if not r.ok:
            err = data.get("error") if isinstance(data, dict) else None
            msg = (err.get("message") if isinstance(err, dict) else err) or data.get("message") or "request failed"
            meta["error"] = _redact(str(msg), secret)[:500]
            return "", meta
        text = getattr(self, f"_parse_{self.route}")(data, meta)
        return text, meta

    @staticmethod
    def _parse_openai(data, meta):
        ch = (data.get("choices") or [{}])[0]
        msg = ch.get("message") or {}
        content = msg.get("content")
        text = content if isinstance(content, str) else "".join(
            c.get("text", "") for c in content or [] if isinstance(c, dict))
        u = data.get("usage") or {}
        meta.update(stop=ch.get("finish_reason"), in_tokens=u.get("prompt_tokens"), out_tokens=u.get("completion_tokens"),
                    reasoning_tokens=(u.get("completion_tokens_details") or {}).get("reasoning_tokens"))
        return text or ""

    @staticmethod
    def _parse_anthropic(data, meta):
        text = "".join(c.get("text", "") for c in data.get("content", []) if c.get("type") == "text")
        u = data.get("usage") or {}
        meta.update(stop=data.get("stop_reason"), in_tokens=u.get("input_tokens"), out_tokens=u.get("output_tokens"))
        return text

    @staticmethod
    def _parse_gemini(data, meta):
        cands = data.get("candidates") or [{}]
        parts = (cands[0].get("content") or {}).get("parts") or []
        text = "".join(p.get("text", "") for p in parts if isinstance(p, dict) and not p.get("thought"))
        u = data.get("usageMetadata") or {}
        meta.update(stop=cands[0].get("finishReason"), in_tokens=u.get("promptTokenCount"),
                    out_tokens=u.get("candidatesTokenCount"), reasoning_tokens=u.get("thoughtsTokenCount"))
        return text

    @staticmethod
    def _parse_bedrock(data, meta):
        content = data.get("output", {}).get("message", {}).get("content", [])
        text = "".join(p.get("text", "") for p in content if isinstance(p, dict) and "text" in p)
        u = data.get("usage") or {}
        meta.update(stop=data.get("stopReason"), in_tokens=u.get("inputTokens"), out_tokens=u.get("outputTokens"))
        return text

    @staticmethod
    def _parse_responses(data, meta):
        text = "".join(p.get("text", "") for it in data.get("output", []) for p in it.get("content", []) or []
                       if p.get("type") == "output_text")
        u = data.get("usage") or {}
        meta.update(stop=data.get("status"), in_tokens=u.get("input_tokens"), out_tokens=u.get("output_tokens"),
                    reasoning_tokens=(u.get("output_tokens_details") or {}).get("reasoning_tokens"))
        return text


BedrockChat = Chat  # name used by the tier code


def _redact(msg: str, secret: str) -> str:
    for s in (secret, secret.replace("Bearer ", "")):
        if s and len(s) > 8:
            msg = msg.replace(s, "[REDACTED]")
    return re.sub(r"\b\d{12}\b", "[account]", msg)


def parse_json_reply(text: str) -> dict | None:
    """The last top-level JSON object in a model reply (nested objects allowed), or None."""
    if not text:
        return None
    try:
        v = json.loads(text.strip())
        if isinstance(v, dict):
            return v
    except Exception:
        pass
    dec, found, i = json.JSONDecoder(), [], 0
    while True:
        j = text.find("{", i)
        if j < 0:
            break
        try:
            v, end = dec.raw_decode(text, j)
        except Exception:
            i = j + 1
            continue
        if isinstance(v, dict):
            found.append(v)
        i = end
    if found:
        return found[-1]
    for c in reversed(re.findall(r"\{[^{}]*\}", text, flags=re.S)):
        try:
            return json.loads(c)
        except Exception:
            continue
    return None
