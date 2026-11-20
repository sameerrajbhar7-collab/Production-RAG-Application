import json
import hashlib
import time
from typing import Dict, Any, Optional, List, Tuple
import numpy as np

from config import Config


class CacheService:
    """
    Enterprise Redis caching service with in-memory fallback for cost optimization.
    Features:
    - Exact Question Caching
    - Semantic Similarity Caching (configurable threshold)
    - Embedding Caching
    - Department ACL & Knowledge Base Version Isolation
    - Cache Invalidation on Document Changes
    """

    def __init__(self, redis_url: Optional[str] = None):
        self.redis_url = redis_url or Config.REDIS_URL
        self.enabled = Config.ENABLE_CACHE
        self.ttl = Config.CACHE_TTL_SECONDS
        self.semantic_threshold = Config.SEMANTIC_CACHE_THRESHOLD
        self.redis_client = None
        self._is_redis_available = False

        # In-memory fallback stores
        self._memory_exact_cache: Dict[str, Dict[str, Any]] = {}
        self._memory_semantic_index: List[Dict[str, Any]] = []
        self._memory_embedding_cache: Dict[str, List[float]] = {}
        self._memory_kb_versions: Dict[str, int] = {}

        # Telemetry metrics
        self.metrics = {
            "exact_hits": 0,
            "semantic_hits": 0,
            "misses": 0,
            "tokens_saved": 0,
            "cost_saved_usd": 0.0,
        }

        self._init_redis()

    def _init_redis(self):
        """Attempts to connect to Redis. Falls back to in-memory store if unavailable."""
        if not self.enabled:
            return

        try:
            import redis
            client = redis.from_url(
                self.redis_url, 
                socket_timeout=1.5, 
                socket_connect_timeout=1.5,
                decode_responses=True
            )
            client.ping()
            self.redis_client = client
            self._is_redis_available = True
        except Exception:
            # Redis is unavailable; graceful in-memory fallback active
            self.redis_client = None
            self._is_redis_available = False

    def is_redis_online(self) -> bool:
        return self._is_redis_available

    # -------------------------------------------------------------
    # Knowledge Base Versioning & Cache Invalidation
    # -------------------------------------------------------------
    def get_kb_version(self, department: str = "ALL") -> int:
        """Gets current version number for a department knowledge base."""
        key = f"rag:kb_version:{department.upper()}"
        if self._is_redis_available and self.redis_client:
            try:
                v = self.redis_client.get(key)
                return int(v) if v else 1
            except Exception:
                pass
        return self._memory_kb_versions.get(key, 1)

    def invalidate_cache(self, department: Optional[str] = None):
        """
        Invalidates cached answers when documents are uploaded, modified, or deleted.
        Increments knowledge base version number so old cache keys are instantly invalidated.
        """
        dept = department.upper() if department else "ALL"
        key = f"rag:kb_version:{dept}"
        global_key = "rag:kb_version:GLOBAL"

        if self._is_redis_available and self.redis_client:
            try:
                self.redis_client.incr(key)
                self.redis_client.incr(global_key)
            except Exception:
                pass

        self._memory_kb_versions[key] = self._memory_kb_versions.get(key, 1) + 1
        self._memory_kb_versions[global_key] = self._memory_kb_versions.get(global_key, 1) + 1

        # Also purge any in-memory semantic items for this department
        if department:
            self._memory_semantic_index = [
                item for item in self._memory_semantic_index
                if department.upper() not in item.get("allowed_departments", [])
            ]
        else:
            self._memory_semantic_index.clear()
            self._memory_exact_cache.clear()

    # -------------------------------------------------------------
    # Cache Key Generation with Security Isolation
    # -------------------------------------------------------------
    def _normalize_text(self, text: str) -> str:
        return " ".join(text.lower().strip().split())

    def _build_exact_cache_key(
        self,
        question: str,
        allowed_departments: List[str],
        model: str,
        prompt_version: str,
    ) -> str:
        """
        Creates a deterministic cache key with strict department ACL, model,
        and knowledge-base version isolation.
        """
        norm_q = self._normalize_text(question)
        dept_signature = ":".join(sorted(d.upper() for d in allowed_departments))
        kb_v = self.get_kb_version("GLOBAL")

        raw_key = f"{norm_q}|{dept_signature}|{model}|{prompt_version}|kb_v{kb_v}"
        key_hash = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()
        return f"rag:exact:{key_hash}"

    # -------------------------------------------------------------
    # Answer Caching: Lookup & Storage
    # -------------------------------------------------------------
    def get_cached_answer(
        self,
        question: str,
        allowed_departments: List[str],
        query_embedding: Optional[List[float]] = None,
        model: Optional[str] = None,
        prompt_version: Optional[str] = None,
    ) -> Tuple[Optional[str], Optional[Dict[str, Any]], str]:
        """
        Looks up cached answer:
        1. Checks Exact Question Cache (O(1))
        2. Checks Semantic Similarity Cache (Cosine similarity >= threshold)
        Returns: (answer_text, metadata, hit_type: "EXACT" | "SEMANTIC" | "MISS")
        """
        if not self.enabled:
            return None, None, "DISABLED"

        target_model = model or Config.OPENAI_CHAT_MODEL
        target_prompt_ver = prompt_version or Config.PROMPT_VERSION
        exact_key = self._build_exact_cache_key(
            question, allowed_departments, target_model, target_prompt_ver
        )

        # 1. Try Exact Cache
        cached_entry = None
        if self._is_redis_available and self.redis_client:
            try:
                raw = self.redis_client.get(exact_key)
                if raw:
                    cached_entry = json.loads(raw)
            except Exception:
                pass
        else:
            entry = self._memory_exact_cache.get(exact_key)
            if entry and (time.time() - entry.get("timestamp", 0) < self.ttl):
                cached_entry = entry

        if cached_entry:
            # Verify security permissions match exactly
            cached_depts = set(cached_entry.get("allowed_departments", []))
            user_depts = set(allowed_departments)
            if cached_depts == user_depts:
                self.metrics["exact_hits"] += 1
                saved_tokens = cached_entry.get("tokens", 500)
                self.metrics["tokens_saved"] += saved_tokens
                self.metrics["cost_saved_usd"] += (saved_tokens / 1_000_000.0) * Config.PRICING["gpt-4o-mini"]["output"]
                return cached_entry["answer"], cached_entry.get("metadata", {}), "EXACT"

        # 2. Try Semantic Cache (if query embedding is provided)
        if query_embedding:
            semantic_hit = self._find_semantic_match(
                query_embedding, allowed_departments, target_model, target_prompt_ver
            )
            if semantic_hit:
                self.metrics["semantic_hits"] += 1
                saved_tokens = semantic_hit.get("tokens", 500)
                self.metrics["tokens_saved"] += saved_tokens
                self.metrics["cost_saved_usd"] += (saved_tokens / 1_000_000.0) * Config.PRICING["gpt-4o-mini"]["output"]
                return semantic_hit["answer"], semantic_hit.get("metadata", {}), "SEMANTIC"

        self.metrics["misses"] += 1
        return None, None, "MISS"

    def set_cached_answer(
        self,
        question: str,
        answer: str,
        allowed_departments: List[str],
        query_embedding: Optional[List[float]] = None,
        model: Optional[str] = None,
        prompt_version: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tokens_used: int = 500,
    ) -> None:
        """Stores a generated RAG answer in both exact and semantic caches."""
        if not self.enabled or not answer or not answer.strip():
            return

        target_model = model or Config.OPENAI_CHAT_MODEL
        target_prompt_ver = prompt_version or Config.PROMPT_VERSION
        exact_key = self._build_exact_cache_key(
            question, allowed_departments, target_model, target_prompt_ver
        )

        entry = {
            "question": question,
            "answer": answer,
            "allowed_departments": sorted(allowed_departments),
            "model": target_model,
            "prompt_version": target_prompt_ver,
            "kb_version": self.get_kb_version("GLOBAL"),
            "metadata": metadata or {},
            "tokens": tokens_used,
            "timestamp": time.time(),
        }

        # Save to Exact Cache
        if self._is_redis_available and self.redis_client:
            try:
                self.redis_client.setex(exact_key, self.ttl, json.dumps(entry))
            except Exception:
                self._memory_exact_cache[exact_key] = entry
        else:
            self._memory_exact_cache[exact_key] = entry

        # Save to Semantic Cache if query embedding is available
        if query_embedding:
            semantic_item = {
                "question": question,
                "answer": answer,
                "embedding": query_embedding,
                "allowed_departments": sorted(allowed_departments),
                "model": target_model,
                "prompt_version": target_prompt_ver,
                "kb_version": self.get_kb_version("GLOBAL"),
                "metadata": metadata or {},
                "tokens": tokens_used,
                "timestamp": time.time(),
            }

            if self._is_redis_available and self.redis_client:
                try:
                    # Append to department-specific semantic list
                    sem_key = f"rag:semantic:{':'.join(sorted(allowed_departments))}"
                    self.redis_client.rpush(sem_key, json.dumps(semantic_item))
                    # Keep list bounded to last 200 queries per ACL slice
                    self.redis_client.ltrim(sem_key, -200, -1)
                    self.redis_client.expire(sem_key, self.ttl)
                except Exception:
                    self._memory_semantic_index.append(semantic_item)
            else:
                self._memory_semantic_index.append(semantic_item)
                # Cap memory cache
                if len(self._memory_semantic_index) > 200:
                    self._memory_semantic_index.pop(0)

    def _find_semantic_match(
        self,
        query_embedding: List[float],
        allowed_departments: List[str],
        model: str,
        prompt_version: str,
    ) -> Optional[Dict[str, Any]]:
        """Compares query embedding with cached query embeddings using cosine similarity."""
        candidates = []
        user_acl = set(allowed_departments)

        if self._is_redis_available and self.redis_client:
            try:
                sem_key = f"rag:semantic:{':'.join(sorted(allowed_departments))}"
                raw_list = self.redis_client.lrange(sem_key, 0, -1)
                for item_str in raw_list:
                    candidates.append(json.loads(item_str))
            except Exception:
                candidates = self._memory_semantic_index
        else:
            candidates = self._memory_semantic_index

        if not candidates:
            return None

        q_vec = np.array(query_embedding)
        norm_q = np.linalg.norm(q_vec)
        if norm_q == 0:
            return None

        best_score = -1.0
        best_candidate = None
        current_kb_v = self.get_kb_version("GLOBAL")

        for item in reversed(candidates):
            # Verify security ACL matches exactly
            if set(item.get("allowed_departments", [])) != user_acl:
                continue

            # Verify knowledge base and prompt versions match
            if item.get("kb_version") != current_kb_v:
                continue
            if item.get("model") != model or item.get("prompt_version") != prompt_version:
                continue

            item_vec = np.array(item["embedding"])
            norm_item = np.linalg.norm(item_vec)
            if norm_item == 0:
                continue

            # Cosine similarity
            cos_sim = float(np.dot(q_vec, item_vec) / (norm_q * norm_item))
            if cos_sim > best_score:
                best_score = cos_sim
                best_candidate = item

        if best_candidate and best_score >= self.semantic_threshold:
            match_meta = dict(best_candidate.get("metadata", {}))
            match_meta["semantic_score"] = round(best_score, 4)
            best_candidate["metadata"] = match_meta
            return best_candidate

        return None

    # -------------------------------------------------------------
    # Embedding Caching
    # -------------------------------------------------------------
    def get_cached_embedding(self, text_hash: str) -> Optional[List[float]]:
        """Retrieves cached embedding by text hash."""
        key = f"rag:emb:{text_hash}"
        if self._is_redis_available and self.redis_client:
            try:
                raw = self.redis_client.get(key)
                if raw:
                    return json.loads(raw)
            except Exception:
                pass
        return self._memory_embedding_cache.get(key)

    def set_cached_embedding(self, text_hash: str, embedding: List[float]) -> None:
        """Saves embedding to cache."""
        key = f"rag:emb:{text_hash}"
        if self._is_redis_available and self.redis_client:
            try:
                self.redis_client.setex(key, Config.EMBEDDING_CACHE_TTL, json.dumps(embedding))
            except Exception:
                self._memory_embedding_cache[key] = embedding
        else:
            self._memory_embedding_cache[key] = embedding

    def get_metrics(self) -> Dict[str, Any]:
        """Returns cache telemetry stats."""
        total = self.metrics["exact_hits"] + self.metrics["semantic_hits"] + self.metrics["misses"]
        hit_rate = (
            ((self.metrics["exact_hits"] + self.metrics["semantic_hits"]) / total * 100)
            if total > 0
            else 0.0
        )
        return {
            "exact_hits": self.metrics["exact_hits"],
            "semantic_hits": self.metrics["semantic_hits"],
            "misses": self.metrics["misses"],
            "total_queries": total,
            "hit_rate_pct": round(hit_rate, 2),
            "tokens_saved": self.metrics["tokens_saved"],
            "cost_saved_usd": round(self.metrics["cost_saved_usd"], 5),
            "redis_connected": self._is_redis_available,
        }
