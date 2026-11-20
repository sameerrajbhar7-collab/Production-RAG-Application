import os
from datetime import timedelta
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class Config:
    """Production RAG Application Configuration."""

    # 1. Application & Company Identity
    APP_NAME = os.getenv("APP_NAME", "Company RAG Assistant")
    COMPANY_NAME = os.getenv("COMPANY_NAME", "Enterprise Global Corp")
    COMPANY_ROLE = os.getenv(
        "COMPANY_ROLE", 
        "Internal AI Knowledge Assistant for Company Policies and Documentation"
    )
    COMPANY_DEPARTMENTS = ["ALL", "HR", "IT", "FIN", "SALES"]
    RESPONSE_STYLE = os.getenv(
        "RESPONSE_STYLE", 
        "professional, accurate, structured, and strictly grounded in company documentation"
    )

    # 2. Flask Settings
    SECRET_KEY = os.getenv("SECRET_KEY", "production_rag_super_secret_key_change_in_prod")
    PERMANENT_SESSION_LIFETIME = timedelta(days=7)
    MAX_CONTENT_LENGTH = int(os.getenv("MAX_FILE_SIZE_MB", "16")) * 1024 * 1024  # 16 MB default
    ALLOWED_EXTENSIONS = {"pdf", "docx", "txt"}

    # 3. Storage Paths
    UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", os.path.join(BASE_DIR, "uploads"))
    CHROMA_DB_DIR = os.getenv("CHROMA_DB_DIR", os.path.join(BASE_DIR, "chroma_db"))

    # 4. OpenAI Models & Parameters
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
    OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
    OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))
    MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1500"))
    PROMPT_VERSION = os.getenv("PROMPT_VERSION", "v1.2")

    # 5. ChromaDB & Retrieval Settings
    TOP_K_RETRIEVAL = int(os.getenv("TOP_K_RETRIEVAL", "4"))
    MIN_CONFIDENCE_SCORE = float(os.getenv("MIN_CONFIDENCE_SCORE", "0.35"))
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

    # 6. Redis & Caching Settings (Cost Optimization)
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    ENABLE_CACHE = os.getenv("ENABLE_CACHE", "true").lower() in ("true", "1", "yes")
    CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "86400"))  # 24 hours
    SEMANTIC_CACHE_THRESHOLD = float(os.getenv("SEMANTIC_CACHE_THRESHOLD", "0.92"))
    EMBEDDING_CACHE_TTL = int(os.getenv("EMBEDDING_CACHE_TTL", "604800"))  # 7 days

    # 7. Conversation Memory Settings
    MAX_MEMORY_TURNS = int(os.getenv("MAX_MEMORY_TURNS", "8"))
    MEMORY_TTL_SECONDS = int(os.getenv("MEMORY_TTL_SECONDS", "604800"))  # 7 days

    # 8. Pricing per 1M tokens (for cost estimation)
    PRICING = {
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "text-embedding-3-small": {"input": 0.02, "output": 0.00},
    }

    # 9. RBAC Users (Default accounts: username -> password + permissions)
    # In production, replace with corporate SSO / LDAP / Database
    USERS = {
        "admin": {
            "password_hash": "pbkdf2:sha256:600000$admin$b244799047971775a2f5f190bc1f3d89004e03f0b2f153f3e74288019a16f6b5",  # "admin123"
            "role": "admin",
            "full_name": "System Administrator",
            "department": "ALL",
            "allowed_departments": ["ALL", "HR", "IT", "FIN", "SALES"],
        },
        "hr_lead": {
            "password_hash": "pbkdf2:sha256:600000$hr$602058e57827e7d69046c827cf63a033e08f51ec73167b5e40798be1284a1443",  # "hr123"
            "role": "manager",
            "full_name": "HR Manager",
            "department": "HR",
            "allowed_departments": ["ALL", "HR"],
        },
        "it_specialist": {
            "password_hash": "pbkdf2:sha256:600000$it$d10bb073f1ff346bb6f51be0e68e4c8bc1e4d3aa54fcfa8295a704e67848604d",  # "it123"
            "role": "engineer",
            "full_name": "IT Lead Engineer",
            "department": "IT",
            "allowed_departments": ["ALL", "IT"],
        },
        "finance_analyst": {
            "password_hash": "pbkdf2:sha256:600000$fin$bb4c9f75f9dbe4ea1c71360bf0c8fe4ea993f4439c016f49fcbf1fe3b15ad36c",  # "fin123"
            "role": "analyst",
            "full_name": "Finance Analyst",
            "department": "FIN",
            "allowed_departments": ["ALL", "FIN"],
        },
        "employee": {
            "password_hash": "pbkdf2:sha256:600000$emp$d59f71c3569a7444c9b13978fa8169121896790938ff52eb9fa38981f4a5f36e",  # "emp123"
            "role": "employee",
            "full_name": "General Employee",
            "department": "ALL",
            "allowed_departments": ["ALL"],
        },
    }


# Ensure directories exist
os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
os.makedirs(Config.CHROMA_DB_DIR, exist_ok=True)
