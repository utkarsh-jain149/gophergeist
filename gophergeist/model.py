"""Stdlib clients for local open-weight model servers. No SDKs, no API keys, no cloud.

Two backends cover nearly every way people run open models on their own machine:

* Ollama's native API (``/api/chat``).
* The OpenAI-compatible ``/v1/chat/completions`` endpoint that llama.cpp's
  ``llama-server``, llamafile, LM Studio, vLLM and Ollama itself all speak.
"""

import json
import os
import urllib.error
import urllib.request

DEFAULT_OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_OPENAI_URL = "http://localhost:8080/v1"
# The fine-tuned Veronica published to the Ollama registry, then a stock model if she can't be pulled.
VERONICA_MODEL = "YOUR_OLLAMA_USER/llama3.2-veronica:1b"
FALLBACK_MODEL = "llama3.2:1b"
DEFAULT_MODEL = os.environ.get("GOPHERGEIST_MODEL", VERONICA_MODEL)


class ModelError(RuntimeError):
    pass


def _url(base, path):
    if not base.startswith("http"):
        base = "http://" + base
    return base.rstrip("/") + path


def _request(url, payload=None, timeout=180):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        raise ModelError(f"{url} returned {e.code}: {e.read().decode(errors='replace')[:200]}") from e
    except (urllib.error.URLError, OSError) as e:
        raise ModelError(f"Could not reach {url}: {e}") from e
    except json.JSONDecodeError as e:
        raise ModelError(f"{url} did not return JSON") from e


def _stream_lines(url, payload, timeout):
    """POST and yield the response body line by line as it arrives (NDJSON or server-sent events)."""
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for raw in resp:
                line = raw.decode("utf-8", errors="replace").strip()
                if line:
                    yield line
    except urllib.error.HTTPError as e:
        raise ModelError(f"{url} returned {e.code}: {e.read().decode(errors='replace')[:200]}") from e
    except (urllib.error.URLError, OSError) as e:
        raise ModelError(f"Could not reach {url}: {e}") from e


def _messages(system, prompt):
    return [{"role": "system", "content": system}, {"role": "user", "content": prompt}]


class Ollama:
    # Keep the weights in memory between questions. Loading a model from disk is often the slowest
    # part of a CPU answer, so a gopherhole that sits idle shouldn't pay it on every visit.
    def __init__(self, model=DEFAULT_MODEL, host=DEFAULT_OLLAMA_HOST, timeout=180, keep_alive="30m"):
        self.model, self.host, self.timeout, self.keep_alive = model, host, timeout, keep_alive

    def __str__(self):
        return f"{self.model} via Ollama"

    def has(self, model):
        data = _request(_url(self.host, "/api/tags"), timeout=5)
        names = [m.get("name", "") for m in data.get("models", [])]
        return model in names or (":" not in model and model + ":latest" in names)

    def check(self):
        if not self.has(self.model):
            raise ModelError(f"Ollama is running but {self.model} isn't pulled. Try: ollama pull {self.model}")

    def pull(self, model, progress=None):
        """Download a model through Ollama. progress(status, completed, total) is called as it goes."""
        url = _url(self.host, "/api/pull")
        for line in _stream_lines(url, {"model": model, "stream": True}, timeout=3600):
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if data.get("error"):
                raise ModelError(f"Couldn't pull {model}: {data['error']}")
            if progress:
                progress(data.get("status", ""), data.get("completed"), data.get("total"))

    def ensure(self, fallbacks=(), progress=None):
        """Make sure a usable model is present, pulling it on first run.
        Tries self.model, then each fallback; switches self.model to whichever worked."""
        errors = []
        for model in (self.model, *fallbacks):
            try:
                if not self.has(model):
                    self.pull(model, progress)
                self.model = model
                return model
            except ModelError as e:
                if "Could not reach" in str(e):
                    raise  # Ollama itself isn't running; no point trying other models
                errors.append(str(e))
        raise ModelError("; ".join(errors))

    def warm(self):
        """Load the model into memory now, so the first visitor doesn't wait for it."""
        _request(_url(self.host, "/api/generate"), {"model": self.model, "keep_alive": self.keep_alive}, self.timeout)

    def _payload(self, system, prompt, max_tokens, stream):
        return {
            "model": self.model,
            "messages": _messages(system, prompt),
            "stream": stream,
            "keep_alive": self.keep_alive,
            "options": {"temperature": 0.3, "num_predict": max_tokens},
        }

    def generate(self, system, prompt, max_tokens=300):
        data = _request(_url(self.host, "/api/chat"), self._payload(system, prompt, max_tokens, False), self.timeout)
        return data.get("message", {}).get("content", "").strip()

    def stream(self, system, prompt, max_tokens=300):
        """Yield pieces of the reply as the model writes them."""
        url = _url(self.host, "/api/chat")
        for line in _stream_lines(url, self._payload(system, prompt, max_tokens, True), self.timeout):
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if data.get("error"):
                raise ModelError(f"{url}: {data['error']}")
            piece = data.get("message", {}).get("content", "")
            if piece:
                yield piece
            if data.get("done"):
                return


class OpenAICompatible:
    def __init__(self, model="local", base_url=DEFAULT_OPENAI_URL, timeout=180):
        self.model, self.base_url, self.timeout = model, base_url, timeout

    def __str__(self):
        return f"{self.model} via {self.base_url}"

    def check(self):
        _request(_url(self.base_url, "/models"), timeout=5)

    def warm(self):
        self.generate("Reply with one word.", "boo", max_tokens=1)

    def _payload(self, system, prompt, max_tokens, stream):
        return {
            "model": self.model,
            "messages": _messages(system, prompt),
            "temperature": 0.3,
            "max_tokens": max_tokens,
            "stream": stream,
        }

    def generate(self, system, prompt, max_tokens=300):
        data = _request(_url(self.base_url, "/chat/completions"), self._payload(system, prompt, max_tokens, False),
                        self.timeout)
        try:
            return data["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, TypeError) as e:
            raise ModelError(f"Unexpected response from {self.base_url}: {str(data)[:200]}") from e

    def stream(self, system, prompt, max_tokens=300):
        url = _url(self.base_url, "/chat/completions")
        for line in _stream_lines(url, self._payload(system, prompt, max_tokens, True), self.timeout):
            if not line.startswith("data:"):
                continue
            body = line[5:].strip()
            if body == "[DONE]":
                return
            try:
                piece = json.loads(body)["choices"][0].get("delta", {}).get("content") or ""
            except (json.JSONDecodeError, KeyError, IndexError, TypeError):
                continue
            if piece:
                yield piece


def make_backend(kind, model=None, host=None):
    """Build a backend from CLI flags. kind is 'ollama', 'openai' or 'none'."""
    if kind == "none":
        return None
    if kind == "ollama":
        return Ollama(model or DEFAULT_MODEL, host or DEFAULT_OLLAMA_HOST)
    if kind == "openai":
        return OpenAICompatible(model or "local", host or DEFAULT_OPENAI_URL)
    raise ValueError(f"Unknown backend {kind!r}")
