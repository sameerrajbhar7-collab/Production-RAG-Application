import json
import time
from typing import List, Dict, Any, Optional

from config import Config


class ConversationMemoryService:
    """
    Manages multi-turn conversation memory using Redis with an in-memory fallback.
    Ensures user and session isolation while preserving context for follow-up inquiries.
    """

    def __init__(self, redis_client=None):
        self.redis_client = redis_client
        self.max_turns = Config.MAX_MEMORY_TURNS
        self.ttl = Config.MEMORY_TTL_SECONDS
        self._memory_store: Dict[str, List[Dict[str, Any]]] = {}

    def _get_key(self, session_id: str, user_id: str) -> str:
        safe_user = user_id.replace(":", "_") if user_id else "anon"
        safe_sess = session_id.replace(":", "_") if session_id else "default"
        return f"rag:mem:{safe_user}:{safe_sess}"

    def add_message(
        self, session_id: str, user_id: str, role: str, content: str
    ) -> None:
        """Appends a message (role: 'user' | 'assistant') to conversation memory."""
        if not content or not content.strip():
            return

        key = self._get_key(session_id, user_id)
        msg_obj = {
            "role": role,
            "content": content.strip(),
            "timestamp": time.time(),
        }

        # Try Redis
        if self.redis_client:
            try:
                self.redis_client.rpush(key, json.dumps(msg_obj))
                # Keep list trimmed to the last 2 * max_turns messages (user + assistant pairs)
                self.redis_client.ltrim(key, -(self.max_turns * 2), -1)
                self.redis_client.expire(key, self.ttl)
                return
            except Exception:
                pass

        # In-memory fallback
        if key not in self._memory_store:
            self._memory_store[key] = []
        self._memory_store[key].append(msg_obj)
        if len(self._memory_store[key]) > (self.max_turns * 2):
            self._memory_store[key] = self._memory_store[key][-(self.max_turns * 2):]

    def get_history(
        self, session_id: str, user_id: str, max_turns: Optional[int] = None
    ) -> List[Dict[str, str]]:
        """
        Retrieves the last N conversation turns for the user/session.
        Returns a list of dicts: [{'role': 'user', 'content': '...'}, ...]
        """
        key = self._get_key(session_id, user_id)
        limit = (max_turns or self.max_turns) * 2

        raw_messages: List[Dict[str, Any]] = []

        # Try Redis
        if self.redis_client:
            try:
                items = self.redis_client.lrange(key, -limit, -1)
                for it in items:
                    raw_messages.append(json.loads(it))
            except Exception:
                raw_messages = self._memory_store.get(key, [])[-limit:]
        else:
            raw_messages = self._memory_store.get(key, [])[-limit:]

        return [
            {"role": m["role"], "content": m["content"]}
            for m in raw_messages
            if "role" in m and "content" in m
        ]

    def clear_history(self, session_id: str, user_id: str) -> None:
        """Clears conversation history for the current session."""
        key = self._get_key(session_id, user_id)
        if self.redis_client:
            try:
                self.redis_client.delete(key)
            except Exception:
                pass
        self._memory_store.pop(key, None)

    def get_turn_count(self, session_id: str, user_id: str) -> int:
        """Returns the number of user question turns in the current conversation."""
        history = self.get_history(session_id, user_id)
        return sum(1 for m in history if m.get("role") == "user")
