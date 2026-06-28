"""LLM client. One chat() interface over two backends, selected by config: an
OpenAI-compatible endpoint (Ollama / vLLM / TGI) and an in-process transformers
model. Env vars LLM_PROVIDER / LLM_MODEL / LLM_BASE_URL pick the backend.
"""
from __future__ import annotations

import json
import re
from typing import Any


def extract_json(text: str) -> Any:
    """Pull a JSON value out of a (possibly messy) model response.

    Handles Qwen-style <think> blocks, ```json fences, and leading/trailing
    prose by scanning for the first balanced [...] or {...}.
    """
    if not text:
        raise ValueError("empty model response")
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"```(?:json)?", "", text)

    start = next((i for i, ch in enumerate(text) if ch in "[{"), None)
    if start is None:
        raise ValueError("no JSON found in response")
    open_ch = text[start]
    close_ch = "]" if open_ch == "[" else "}"

    depth = 0
    in_str = False
    esc = False
    for j in range(start, len(text)):
        ch = text[j]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return json.loads(text[start : j + 1])
    raise ValueError("unbalanced JSON in response")


def _salvage_objects(text: str) -> list:
    """Recover every complete top-level {...} object from a (possibly truncated)
    response. Lets us keep the items a model produced even if the array was cut
    off mid-way."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"```(?:json)?", "", text)
    objs, depth, start = [], 0, None
    in_str = esc = False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    objs.append(json.loads(text[start : i + 1]))
                except ValueError:
                    pass
                start = None
    return objs


class LLMClient:
    def __init__(self, cfg: dict[str, Any]) -> None:
        llm = cfg["llm"]
        self.provider = (llm.get("provider") or "ollama").lower()
        self.model = llm["model"]
        self.base_url = llm.get("base_url")
        self.api_key = llm.get("api_key", "ollama")
        self.temperature = llm.get("temperature", 0.2)
        self.max_tokens = llm.get("max_tokens", 2048)
        self.no_think = llm.get("no_think", True)
        self._client = None
        self._local = None  # (model, tokenizer) for provider="local"

    # -- remote (OpenAI-compatible: Ollama / vLLM / TGI) -----------------
    def _obj(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        return self._client

    def _remote_chat(self, system: str, user: str) -> str:
        is_qwen = "qwen" in (self.model or "").lower()
        if self.no_think and is_qwen:
            user = f"{user}\n\n/no_think"
        kwargs = dict(
            model=self.model,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        client = self._obj()
        if self.no_think and is_qwen:
            try:
                resp = client.chat.completions.create(extra_body={"think": False}, **kwargs)
            except Exception:
                resp = client.chat.completions.create(**kwargs)
        else:
            resp = client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content or ""

    # -- local (in-process transformers, runs on the GPU) ---------------
    def _ensure_local(self):
        if self._local is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            tok = AutoTokenizer.from_pretrained(self.model)
            model = AutoModelForCausalLM.from_pretrained(
                self.model, torch_dtype="auto", device_map="auto"
            )
            self._local = (model, tok)
        return self._local

    def _local_chat(self, system: str, user: str) -> str:
        model, tok = self._ensure_local()
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": user}]
        tmpl_kwargs = {}
        if self.no_think and "qwen" in (self.model or "").lower():
            tmpl_kwargs["enable_thinking"] = False
        try:
            prompt = tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, **tmpl_kwargs)
        except TypeError:  # tokenizer without enable_thinking kwarg
            prompt = tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        gen_kwargs = dict(max_new_tokens=self.max_tokens, pad_token_id=tok.eos_token_id)
        if self.temperature and self.temperature > 0:
            gen_kwargs.update(do_sample=True, temperature=self.temperature)
        else:
            gen_kwargs.update(do_sample=False)
        out = model.generate(**inputs, **gen_kwargs)
        return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    def chat(self, system: str, user: str) -> str:
        if self.provider == "local":
            return self._local_chat(system, user)
        return self._remote_chat(system, user)

    def json_chat(self, system: str, user: str) -> Any:
        raw = self.chat(system, user)
        try:
            return extract_json(raw)
        except ValueError:
            salvaged = _salvage_objects(raw)  # recover from truncated arrays
            if salvaged:
                return salvaged
            raise
