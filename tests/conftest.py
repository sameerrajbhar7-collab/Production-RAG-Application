import os
import shutil
import tempfile
import hashlib
import pytest
from unittest.mock import MagicMock, patch
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings

from config import Config
from services import (
    DocumentService,
    EmbeddingService,
    RetrievalService,
    CacheService,
    ConversationMemoryService,
    LLMService,
)


class MockChromaEmbeddingFunction(EmbeddingFunction):
    """Compliant ChromaDB embedding function returning deterministic vectors."""
    def __init__(self, fn):
        self._fn = fn

    def name(self) -> str:
        return "openai_embeddings"

    def __call__(self, input: Documents) -> Embeddings:
        return self._fn(input)

    def get_embedding(self, text: str):
        return self._fn([text])[0]

    def get_embeddings_batch(self, texts):
        return self._fn(texts)


@pytest.fixture(scope="session")
def test_temp_dir():
    temp_dir = tempfile.mkdtemp(prefix="rag_test_")
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def mock_embedding_fn():
    """Mock embedding generator producing deterministic distinct 1536-dim vectors without API calls."""
    def _embed(texts):
        res = []
        for t in texts:
            seed = hashlib.sha256(t.encode("utf-8")).digest()
            raw = [(float(seed[i % len(seed)]) - 128.0) / 128.0 for i in range(1536)]
            norm = (sum(x**2 for x in raw)) ** 0.5
            res.append([x / (norm or 1.0) for x in raw])
        return res
    return _embed


@pytest.fixture
def test_cache_service():
    """In-memory CacheService instance for testing."""
    service = CacheService()
    # Force in-memory mode for reliable local tests
    service._is_redis_available = False
    service.redis_client = None
    return service


@pytest.fixture
def test_doc_service(test_temp_dir):
    uploads_dir = os.path.join(test_temp_dir, "test_uploads")
    os.makedirs(uploads_dir, exist_ok=True)
    return DocumentService(upload_folder=uploads_dir)


@pytest.fixture
def test_retrieval_service(test_temp_dir, mock_embedding_fn):
    chroma_dir = os.path.join(test_temp_dir, "test_chroma")
    os.makedirs(chroma_dir, exist_ok=True)

    emb_svc = MockChromaEmbeddingFunction(mock_embedding_fn)
    return RetrievalService(chroma_db_dir=chroma_dir, embedding_service=emb_svc)


@pytest.fixture
def test_memory_service():
    return ConversationMemoryService(redis_client=None)


@pytest.fixture
def test_flask_app():
    from app import app
    app.config["TESTING"] = True
    app.config["WTF_CSRF_ENABLED"] = False
    return app
