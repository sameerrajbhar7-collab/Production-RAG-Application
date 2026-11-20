import hashlib
from typing import List, Optional
import openai
from openai import OpenAI

from config import Config


class EmbeddingService:
    """
    Manages vector embeddings using OpenAI text-embedding-3-small
    with an integrated embedding cache to eliminate duplicate embedding API costs.
    """

    def __init__(self, cache_service=None):
        self.model = Config.OPENAI_EMBEDDING_MODEL
        self.cache_service = cache_service
        self._tokens_used = 0
        self._estimated_cost_usd = 0.0

    def _get_api_key(self) -> str:
        key = Config.OPENAI_API_KEY
        if not key:
            raise openai.OpenAIError(
                "OPENAI_API_KEY is not set. Please set OPENAI_API_KEY in your .env file."
            )
        return key

    def get_embedding(self, text: str) -> List[float]:
        """Generates embedding for a single text, checking the cache first."""
        if not text or not text.strip():
            # Return dummy 1536-dim zero vector for empty strings
            return [0.0] * 1536

        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()

        # Check cache if available
        if self.cache_service:
            cached = self.cache_service.get_cached_embedding(text_hash)
            if cached:
                return cached

        # Call OpenAI Embeddings API
        api_key = self._get_api_key()
        client = OpenAI(api_key=api_key)
        response = client.embeddings.create(
            input=[text],
            model=self.model,
        )

        embedding = response.data[0].embedding

        # Track token usage (text-embedding-3-small is $0.02 / 1M tokens)
        tokens = getattr(response.usage, "total_tokens", len(text) // 4)
        self._tokens_used += tokens
        self._estimated_cost_usd += (tokens / 1_000_000.0) * Config.PRICING["text-embedding-3-small"]["input"]

        # Cache the generated embedding
        if self.cache_service:
            self.cache_service.set_cached_embedding(text_hash, embedding)

        return embedding

    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generates embeddings for a batch of texts.
        Pulls cached embeddings where available and only calls OpenAI for missing texts.
        """
        if not texts:
            return []

        results: List[Optional[List[float]]] = [None] * len(texts)
        missing_indices: List[int] = []
        missing_texts: List[str] = []

        # 1. Check cache for each text
        for i, text in enumerate(texts):
            text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
            cached = self.cache_service.get_cached_embedding(text_hash) if self.cache_service else None
            if cached:
                results[i] = cached
            else:
                missing_indices.append(i)
                missing_texts.append(text)

        # 2. If all were cached, return immediately (zero API calls!)
        if not missing_texts:
            return [r for r in results if r is not None]

        # 3. Fetch missing embeddings from OpenAI in a single batch
        api_key = self._get_api_key()
        client = OpenAI(api_key=api_key)
        response = client.embeddings.create(
            input=missing_texts,
            model=self.model,
        )

        tokens = getattr(response.usage, "total_tokens", sum(len(t) // 4 for t in missing_texts))
        self._tokens_used += tokens
        self._estimated_cost_usd += (tokens / 1_000_000.0) * Config.PRICING["text-embedding-3-small"]["input"]

        for idx, item in enumerate(response.data):
            orig_idx = missing_indices[idx]
            emb = item.embedding
            results[orig_idx] = emb

            # Cache the new embedding
            if self.cache_service:
                text_hash = hashlib.sha256(missing_texts[idx].encode("utf-8")).hexdigest()
                self.cache_service.set_cached_embedding(text_hash, emb)

        return [r for r in results if r is not None]

    def get_metrics(self) -> dict:
        return {
            "embedding_tokens_used": self._tokens_used,
            "embedding_cost_usd": round(self._estimated_cost_usd, 6),
            "model": self.model,
        }

    # ChromaDB Custom Embedding Function Interface
    def name(self) -> str:
        return "openai_embeddings"

    def __call__(self, input: List[str]) -> List[List[float]]:
        return self.get_embeddings_batch(input)
