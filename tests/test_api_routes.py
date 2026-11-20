import io
import json
import pytest
from unittest.mock import patch, MagicMock

from config import Config


def test_health_endpoint(test_flask_app):
    client = test_flask_app.test_client()
    res = client.get("/health")
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "healthy"
    assert data["company"] == Config.COMPANY_NAME


def test_metrics_endpoint(test_flask_app):
    client = test_flask_app.test_client()
    res = client.get("/metrics")
    assert res.status_code == 200
    data = res.get_json()
    assert "cache" in data
    assert "llm" in data
    assert "embeddings" in data
    assert "hit_rate_pct" in data["cache"]


def test_index_page_loads(test_flask_app):
    client = test_flask_app.test_client()
    res = client.get("/")
    assert res.status_code == 200
    html = res.data.decode("utf-8")
    assert "Production RAG Application" in html
    assert "Documents Management" in html


def test_user_switch_endpoint(test_flask_app):
    client = test_flask_app.test_client()
    res = client.post("/switch_user", data={"username": "hr_lead"}, follow_redirects=True)
    assert res.status_code == 200
    html = res.data.decode("utf-8")
    assert "HR Manager" in html


def test_ask_endpoint_with_mocked_openai(test_flask_app):
    client = test_flask_app.test_client()
    import app as app_module

    fake_embedding = [0.05] * 1536
    mock_llm_result = {
        "answer": "According to our company policy, employee health coverage begins on day 1.",
        "tokens_used": 63,
        "prompt_tokens": 45,
        "completion_tokens": 18,
        "cost_usd": 0.0001,
        "latency_ms": 120.0,
        "model": "gpt-4o-mini",
    }

    with patch.object(Config, "OPENAI_API_KEY", "sk-test-mock-key"), \
         patch.object(app_module.embedding_service, "get_embedding", return_value=fake_embedding), \
         patch.object(app_module.retrieval_service, "search_relevant_chunks", return_value=(
             [{"filename": "Benefits.pdf", "department": "HR", "text": "Health coverage begins on day 1.", "similarity": 0.95, "chunk_index": 0}],
             "Health coverage begins on day 1."
         )), \
         patch.object(app_module.llm_service, "generate_answer", return_value=mock_llm_result):

        res = client.post("/ask", data={"question": "When does health insurance start?"}, follow_redirects=True)
        assert res.status_code == 200
        html = res.data.decode("utf-8")
        assert "health coverage begins on day 1" in html


def test_reset_conversation_endpoint(test_flask_app):
    client = test_flask_app.test_client()
    res = client.post("/reset_conversation", follow_redirects=True)
    assert res.status_code == 200
    html = res.data.decode("utf-8")
    assert "Conversation context has been reset" in html
