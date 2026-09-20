from __future__ import annotations

import hashlib
import math
import os

# Multilingual model suitable for the Vietnamese corpora used in this Lab.
# The local backend remains optional; required checkpoints use MockEmbedder.
LOCAL_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
GEMINI_EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_PROVIDER_ENV = "EMBEDDING_PROVIDER"


class MockEmbedder:
    """Deterministic embedding backend used by tests and default classroom runs."""

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim
        self._backend_name = "mock embeddings fallback"

    def __call__(self, text: str) -> list[float]:
        digest = hashlib.md5(text.encode()).hexdigest()
        seed = int(digest, 16)
        vector = []
        for _ in range(self.dim):
            seed = (seed * 1664525 + 1013904223) & 0xFFFFFFFF
            vector.append((seed / 0xFFFFFFFF) * 2 - 1)
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]


class LocalEmbedder:
    """Sentence Transformers-backed local embedder."""

    def __init__(self, model_name: str = LOCAL_EMBEDDING_MODEL) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self._backend_name = model_name
        self.model = SentenceTransformer(model_name)

    def __call__(self, text: str) -> list[float]:
        embedding = self.model.encode(text, normalize_embeddings=True)
        if hasattr(embedding, "tolist"):
            return embedding.tolist()
        return [float(value) for value in embedding]


class OpenAIEmbedder:
    """OpenAI embeddings API-backed embedder."""

    def __init__(self, model_name: str = OPENAI_EMBEDDING_MODEL) -> None:
        from openai import OpenAI

        self.model_name = model_name
        self._backend_name = model_name
        self.client = OpenAI()

    def __call__(self, text: str) -> list[float]:
        response = self.client.embeddings.create(model=self.model_name, input=text)
        return [float(value) for value in response.data[0].embedding]


class GeminiEmbedder:
    """Google Gemini embeddings API-backed embedder.

    Supports google-genai SDK if installed, with seamless fallback to
    the official Gemini REST API (no extra dependencies required).
    """

    def __init__(self, model_name: str = GEMINI_EMBEDDING_MODEL) -> None:
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY (or GOOGLE_API_KEY) is required for GeminiEmbedder")
        self.api_key = api_key
        self.model_name = model_name
        self._backend_name = f"gemini ({model_name})"
        self._cache: dict[str, list[float]] = {}
        self.client = None

        try:
            from google import genai
            self.client = genai.Client(api_key=api_key)
        except Exception:
            self.client = None

    def __call__(self, text: str) -> list[float]:
        if text in self._cache:
            return self._cache[text]

        if self.client is not None:
            try:
                response = self.client.models.embed_content(model=self.model_name, contents=text)
                vector = [float(value) for value in response.embeddings[0].values]
                self._cache[text] = vector
                return vector
            except Exception:
                pass  # Fall back to REST API

        import json
        import urllib.request

        model_name = self.model_name
        clean_name = model_name if model_name.startswith("models/") else f"models/{model_name}"
        url = f"https://generativelanguage.googleapis.com/v1beta/{clean_name}:embedContent?key={self.api_key}"
        payload = json.dumps({
            "model": clean_name,
            "content": {"parts": [{"text": text}]},
        }).encode("utf-8")

        req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            vector = [float(v) for v in data["embedding"]["values"]]
            self._cache[text] = vector
            return vector


_mock_embed = MockEmbedder()
