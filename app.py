import os
import uuid
import time
import logging
from typing import Dict, Any, Optional

from flask import (
    Flask,
    render_template,
    request,
    flash,
    redirect,
    url_for,
    session,
    jsonify,
)
from werkzeug.security import check_password_hash

from config import Config
from services import (
    DocumentService,
    EmbeddingService,
    RetrievalService,
    CacheService,
    ConversationMemoryService,
    LLMService,
)

# -----------------------------------------------------
# 1. APPLICATION & SERVICE INITIALIZATION
# -----------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s in %(module)s: %(message)s"
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config["SECRET_KEY"] = Config.SECRET_KEY
app.config["MAX_CONTENT_LENGTH"] = Config.MAX_CONTENT_LENGTH
app.config["UPLOAD_FOLDER"] = Config.UPLOAD_FOLDER

# Core Services
cache_service = CacheService()
embedding_service = EmbeddingService(cache_service=cache_service)
retrieval_service = RetrievalService(
    chroma_db_dir=Config.CHROMA_DB_DIR,
    embedding_service=embedding_service,
)
document_service = DocumentService(upload_folder=Config.UPLOAD_FOLDER)
memory_service = ConversationMemoryService(redis_client=cache_service.redis_client)
llm_service = LLMService()

# Make Config accessible to Jinja templates
app.jinja_env.globals["config"] = Config


# -----------------------------------------------------
# 2. AUTHENTICATION & SESSION HELPERS
# -----------------------------------------------------
def get_current_user() -> Dict[str, Any]:
    """
    Retrieves the authenticated user from the session.
    Defaults to the 'admin' account if no session exists, ensuring zero-friction local testing.
    """
    username = session.get("username")
    if username and username in Config.USERS:
        user_info = dict(Config.USERS[username])
        user_info["username"] = username
        return user_info

    # Default demo user (full access)
    default_user = dict(Config.USERS["admin"])
    default_user["username"] = "admin"
    session["username"] = "admin"
    return default_user


def get_or_create_session_id() -> str:
    """Ensures each conversation has a unique, isolated session ID."""
    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4())
    return session["session_id"]


# -----------------------------------------------------
# 3. AUTHENTICATION ROUTES
# -----------------------------------------------------
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip().lower()
        password = request.form.get("password", "").strip()

        user_record = Config.USERS.get(username)
        if user_record and (
            check_password_hash(user_record["password_hash"], password)
            or password in (f"{username}123", "admin123", "password")  # Demo convenience
        ):
            session["username"] = username
            session["session_id"] = str(uuid.uuid4())  # Fresh conversation per login
            flash(f"Welcome back, {user_record['full_name']}!", "success")
            return redirect(url_for("index"))

        return render_template("login.html", error="Invalid username or password.")

    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    flash("You have been signed out.", "success")
    return redirect(url_for("login"))


@app.route("/switch_user", methods=["POST"])
def switch_user():
    """Allows rapid role-switching in the UI for department ACL verification."""
    target_username = request.form.get("username", "").strip().lower()
    if target_username in Config.USERS:
        session["username"] = target_username
        session["session_id"] = str(uuid.uuid4())
        user_name = Config.USERS[target_username]["full_name"]
        flash(f"Switched active role to: {user_name} ({', '.join(Config.USERS[target_username]['allowed_departments'])})", "success")
    return redirect(url_for("index"))


# -----------------------------------------------------
# 4. MAIN DASHBOARD & Q&A ROUTES
# -----------------------------------------------------
@app.route("/")
def index():
    tab = request.args.get("tab", "chat")
    current_user = get_current_user()
    session_id = get_or_create_session_id()

    docs = document_service.get_document_list()
    turn_count = memory_service.get_turn_count(session_id, current_user["username"])
    cache_metrics = cache_service.get_metrics()

    return render_template(
        "index.html",
        active_tab=tab,
        documents=docs,
        current_user=current_user,
        memory_turns=turn_count,
        cache_metrics=cache_metrics,
        available_users=Config.USERS,
    )


@app.route("/ask", methods=["POST"])
def ask():
    question = request.form.get("question", "").strip()
    current_user = get_current_user()
    session_id = get_or_create_session_id()
    allowed_departments = current_user.get("allowed_departments", ["ALL"])

    docs = document_service.get_document_list()
    turn_count = memory_service.get_turn_count(session_id, current_user["username"])
    cache_metrics = cache_service.get_metrics()

    if not question:
        return render_template(
            "index.html",
            error="Please enter a question.",
            active_tab="chat",
            documents=docs,
            current_user=current_user,
            memory_turns=turn_count,
            cache_metrics=cache_metrics,
            available_users=Config.USERS,
        )

    # Validate OpenAI API key
    if not Config.OPENAI_API_KEY:
        return render_template(
            "index.html",
            question=question,
            error="OPENAI_API_KEY is not configured. Please set OPENAI_API_KEY in your .env file.",
            active_tab="chat",
            documents=docs,
            current_user=current_user,
            memory_turns=turn_count,
            cache_metrics=cache_metrics,
            available_users=Config.USERS,
        )

    start_total_time = time.time()

    try:
        # STEP 1: Generate/Check Query Embedding (Cost Optimized via Embedding Cache)
        query_embedding = embedding_service.get_embedding(question)

        # STEP 2: Cost Optimization - Check Exact & Semantic Answer Cache
        # Never calls OpenAI Chat Completion API if a cached answer satisfies the security ACL
        cached_answer, cache_meta, hit_type = cache_service.get_cached_answer(
            question=question,
            allowed_departments=allowed_departments,
            query_embedding=query_embedding,
            model=Config.OPENAI_CHAT_MODEL,
            prompt_version=Config.PROMPT_VERSION,
        )

        if cached_answer and hit_type in ("EXACT", "SEMANTIC"):
            logger.info("Cache %s HIT for query: '%s'", hit_type, question[:40])

            # Update conversation memory with cached interaction
            memory_service.add_message(session_id, current_user["username"], "user", question)
            memory_service.add_message(session_id, current_user["username"], "assistant", cached_answer)

            updated_cache_metrics = cache_service.get_metrics()
            updated_turn_count = memory_service.get_turn_count(session_id, current_user["username"])

            return render_template(
                "index.html",
                question=question,
                answer=cached_answer,
                active_tab="chat",
                documents=docs,
                current_user=current_user,
                memory_turns=updated_turn_count,
                cache_metrics=updated_cache_metrics,
                available_users=Config.USERS,
                cache_hit=hit_type,
                cache_semantic_score=cache_meta.get("semantic_score", 1.0),
                citations=cache_meta.get("citations", []),
                model_used=Config.OPENAI_CHAT_MODEL,
                tokens_used=0,
                cost_usd=0.0,
                latency_ms=round((time.time() - start_total_time) * 1000.0, 1),
            )

        logger.info("Cache MISS for query: '%s'. Executing RAG pipeline...", question[:40])

        # STEP 3: ChromaDB Native Vector Retrieval with Server-Side Department Filtering
        retrieved_items, context_str = retrieval_service.search_relevant_chunks(
            query=question,
            allowed_departments=allowed_departments,
            query_embedding=query_embedding,
            top_k=Config.TOP_K_RETRIEVAL,
            min_confidence=Config.MIN_CONFIDENCE_SCORE,
        )

        # STEP 4: Fetch Isolated Conversation History
        history = memory_service.get_history(session_id, current_user["username"])

        # STEP 5: Call Grounded Company LLM Service
        llm_result = llm_service.generate_answer(
            question=question,
            context=context_str,
            allowed_departments=allowed_departments,
            conversation_history=history,
            company_name=Config.COMPANY_NAME,
            company_role=Config.COMPANY_ROLE,
        )

        generated_answer = llm_result["answer"]

        # STEP 6: Save Answer to Cache for Future Exact & Semantic Requests
        cache_service.set_cached_answer(
            question=question,
            answer=generated_answer,
            allowed_departments=allowed_departments,
            query_embedding=query_embedding,
            model=Config.OPENAI_CHAT_MODEL,
            prompt_version=Config.PROMPT_VERSION,
            metadata={
                "citations": retrieved_items,
                "retrieved_count": len(retrieved_items),
            },
            tokens_used=llm_result["tokens_used"],
        )

        # STEP 7: Save to Conversation Memory
        memory_service.add_message(session_id, current_user["username"], "user", question)
        memory_service.add_message(session_id, current_user["username"], "assistant", generated_answer)

        updated_cache_metrics = cache_service.get_metrics()
        updated_turn_count = memory_service.get_turn_count(session_id, current_user["username"])

        return render_template(
            "index.html",
            question=question,
            answer=generated_answer,
            active_tab="chat",
            documents=docs,
            current_user=current_user,
            memory_turns=updated_turn_count,
            cache_metrics=updated_cache_metrics,
            available_users=Config.USERS,
            cache_hit="MISS",
            citations=retrieved_items,
            model_used=llm_result["model"],
            tokens_used=llm_result["tokens_used"],
            cost_usd=llm_result["cost_usd"],
            latency_ms=llm_result["latency_ms"],
        )

    except Exception as e:
        logger.error("Error processing ask request: %s", str(e), exc_info=True)
        return render_template(
            "index.html",
            question=question,
            error=f"Error generating answer: {str(e)}",
            active_tab="chat",
            documents=docs,
            current_user=current_user,
            memory_turns=turn_count,
            cache_metrics=cache_metrics,
            available_users=Config.USERS,
        )


# -----------------------------------------------------
# 5. DOCUMENT MANAGEMENT ROUTES
# -----------------------------------------------------
@app.route("/upload", methods=["POST"])
def upload_file():
    current_user = get_current_user()
    file = request.files.get("file")
    department = request.form.get("department", "ALL").strip().upper()

    # Enforce department authorization: non-admin users cannot upload to unauthorized departments
    if current_user["role"] != "admin" and department not in current_user.get("allowed_departments", ["ALL"]):
        flash(f"You do not have permission to upload documents for department '{department}'.", "error")
        return redirect(url_for("index", tab="documents"))

    success, message, result = document_service.process_and_save_upload(
        file=file, department=department
    )

    if not success:
        flash(message, "error")
        return redirect(url_for("index", tab="documents"))

    try:
        # Index document chunks into persistent ChromaDB
        chunk_count = retrieval_service.index_document_chunks(
            chunks=result["chunks"],
            chunk_ids=result["chunk_ids"],
            metadatas=result["metadatas"],
            filename=result["filename"],
        )

        # Invalidate cached answers for this department and global cache
        cache_service.invalidate_cache(department=department)

        flash(
            f"Successfully uploaded and indexed '{result['filename']}' ({chunk_count} vector chunks) for department '{department}'.",
            "success",
        )
    except Exception as e:
        logger.error("Error indexing document chunks: %s", str(e), exc_info=True)
        document_service.delete_physical_file(result["filename"])
        flash(f"Failed to vectorize and store document: {str(e)}", "error")

    return redirect(url_for("index", tab="documents"))


@app.route("/delete_document/<filename>", methods=["POST"])
def delete_document(filename):
    current_user = get_current_user()

    # Delete from ChromaDB
    retrieval_service.delete_document(filename)

    # Delete physical file from uploads folder
    document_service.delete_physical_file(filename)

    # Invalidate caches
    cache_service.invalidate_cache()

    flash(f"Document '{filename}' was permanently deleted and vectors removed.", "success")
    return redirect(url_for("index", tab="documents"))


@app.route("/remove_documents", methods=["POST"])
def remove_documents():
    current_user = get_current_user()
    if current_user["role"] != "admin":
        flash("Only administrators can reset the entire company knowledge base.", "error")
        return redirect(url_for("index", tab="documents"))

    # Remove all physical files
    if os.path.exists(Config.UPLOAD_FOLDER):
        for fname in os.listdir(Config.UPLOAD_FOLDER):
            fpath = os.path.join(Config.UPLOAD_FOLDER, fname)
            if os.path.isfile(fpath):
                try:
                    os.remove(fpath)
                except OSError:
                    pass

    # Clear ChromaDB
    retrieval_service.clear_all_documents()

    # Invalidate all caches
    cache_service.invalidate_cache()

    flash("All documents removed and vector database cleared successfully.", "success")
    return redirect(url_for("index", tab="documents"))


@app.route("/reset_conversation", methods=["POST"])
def reset_conversation():
    current_user = get_current_user()
    session_id = get_or_create_session_id()
    memory_service.clear_history(session_id, current_user["username"])
    # Issue a new session id for a pristine conversation context
    session["session_id"] = str(uuid.uuid4())
    flash("Conversation context has been reset.", "success")
    return redirect(url_for("index"))


# -----------------------------------------------------
# 6. METRICS & HEALTHCHECK APIS
# -----------------------------------------------------
@app.route("/metrics", methods=["GET"])
def metrics():
    """Returns real-time telemetry metrics for monitoring."""
    c_metrics = cache_service.get_metrics()
    l_metrics = llm_service.get_metrics()
    e_metrics = embedding_service.get_metrics()

    return jsonify({
        "status": "healthy",
        "company": Config.COMPANY_NAME,
        "cache": c_metrics,
        "llm": l_metrics,
        "embeddings": e_metrics,
        "total_estimated_savings_usd": c_metrics["cost_saved_usd"],
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "service": Config.APP_NAME,
        "company": Config.COMPANY_NAME,
        "redis_connected": cache_service.is_redis_online(),
    })


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
