import pytest
from services.memory_service import ConversationMemoryService


def test_conversation_memory_add_and_retrieve(test_memory_service):
    session_id = "sess_test_123"
    user_id = "employee_jane"

    test_memory_service.add_message(session_id, user_id, "user", "What is our 401k match?")
    test_memory_service.add_message(
        session_id, user_id, "assistant", "The company matches 100% up to 4% of salary."
    )

    history = test_memory_service.get_history(session_id, user_id)
    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[0]["content"] == "What is our 401k match?"
    assert history[1]["role"] == "assistant"
    assert "4% of salary" in history[1]["content"]

    turns = test_memory_service.get_turn_count(session_id, user_id)
    assert turns == 1


def test_conversation_user_and_session_isolation(test_memory_service):
    """Different users or sessions must never see each other's conversation history."""
    test_memory_service.add_message("sess_1", "user_alice", "user", "Alice confidential question.")
    test_memory_service.add_message("sess_2", "user_bob", "user", "Bob confidential question.")

    alice_history = test_memory_service.get_history("sess_1", "user_alice")
    bob_history = test_memory_service.get_history("sess_2", "user_bob")

    assert len(alice_history) == 1
    assert "Alice" in alice_history[0]["content"]
    assert "Bob" not in alice_history[0]["content"]

    assert len(bob_history) == 1
    assert "Bob" in bob_history[0]["content"]
    assert "Alice" not in bob_history[0]["content"]


def test_conversation_memory_clear(test_memory_service):
    session_id = "sess_clear_test"
    user_id = "test_user"

    test_memory_service.add_message(session_id, user_id, "user", "Hello")
    test_memory_service.add_message(session_id, user_id, "assistant", "Hi there!")

    assert len(test_memory_service.get_history(session_id, user_id)) == 2

    test_memory_service.clear_history(session_id, user_id)
    assert len(test_memory_service.get_history(session_id, user_id)) == 0
    assert test_memory_service.get_turn_count(session_id, user_id) == 0
