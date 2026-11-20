from typing import List, Dict, Any, Optional, Tuple
import chromadb

from config import Config


class RetrievalService:
    """
    Manages persistent ChromaDB vector storage and native indexed vector retrieval
    with strict department-level access control.
    """

    def __init__(self, chroma_db_dir: Optional[str] = None, embedding_service=None):
        self.chroma_db_dir = chroma_db_dir or Config.CHROMA_DB_DIR
        self.embedding_service = embedding_service
        self.client = chromadb.PersistentClient(path=self.chroma_db_dir)

    def _get_collection(self):
        """Fetches or creates the persistent ChromaDB collection."""
        kwargs: Dict[str, Any] = {
            "name": "documents",
            "metadata": {"hnsw:space": "cosine"},
        }
        if self.embedding_service:
            kwargs["embedding_function"] = self.embedding_service

        return self.client.get_or_create_collection(**kwargs)

    def index_document_chunks(
        self,
        chunks: List[str],
        chunk_ids: List[str],
        metadatas: List[Dict[str, Any]],
        filename: str,
    ) -> int:
        """
        Indexes document chunks into ChromaDB.
        Automatically removes any existing chunks for the same filename to prevent duplicates.
        """
        if not chunks:
            return 0

        collection = self._get_collection()

        # 1. Remove previous chunks for this filename to ensure zero duplicates
        try:
            existing = collection.get(where={"filename": filename})
            if existing and existing.get("ids"):
                collection.delete(ids=existing["ids"])
        except Exception:
            try:
                collection.delete(where={"filename": filename})
            except Exception:
                pass

        # 2. Upsert new chunks with deterministic IDs
        collection.upsert(
            documents=chunks,
            ids=chunk_ids,
            metadatas=metadatas,
        )

        return len(chunks)

    def delete_document(self, filename: str) -> bool:
        """Deletes all chunks belonging to a document from ChromaDB."""
        try:
            collection = self._get_collection()
            collection.delete(where={"filename": filename})
            return True
        except Exception:
            return False

    def clear_all_documents(self) -> bool:
        """Resets the vector database."""
        try:
            self.client.delete_collection(name="documents")
            return True
        except Exception:
            return False

    def search_relevant_chunks(
        self,
        query: str,
        allowed_departments: List[str],
        query_embedding: Optional[List[float]] = None,
        top_k: Optional[int] = None,
        min_confidence: Optional[float] = None,
    ) -> Tuple[List[Dict[str, Any]], str]:
        """
        Executes ChromaDB native indexed vector search with server-side department ACL filtering.
        Returns: (retrieved_chunks, formatted_context_string)
        """
        if not query or not query.strip():
            return [], ""

        k = top_k or Config.TOP_K_RETRIEVAL
        min_conf = min_confidence if min_confidence is not None else Config.MIN_CONFIDENCE_SCORE
        collection = self._get_collection()

        # Build server-side department filter
        # e.g., {"department": {"$in": ["ALL", "HR"]}} or {"department": "ALL"}
        where_clause: Optional[Dict[str, Any]] = None
        if allowed_departments:
            clean_depts = [d.upper() for d in allowed_departments]
            if len(clean_depts) == 1:
                where_clause = {"department": clean_depts[0]}
            else:
                where_clause = {"department": {"$in": clean_depts}}

        query_params: Dict[str, Any] = {
            "n_results": k,
            "include": ["documents", "metadatas", "distances"],
        }
        if where_clause:
            query_params["where"] = where_clause

        # Use native ChromaDB query with either precomputed embedding or raw text
        if query_embedding:
            query_params["query_embeddings"] = [query_embedding]
        else:
            query_params["query_texts"] = [query]

        try:
            results = collection.query(**query_params)
        except Exception as e:
            # Collection may be empty or error
            return [], ""

        retrieved_items: List[Dict[str, Any]] = []
        context_parts: List[str] = []

        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for idx, doc_text in enumerate(docs):
            dist = distances[idx] if idx < len(distances) else 1.0
            meta = metas[idx] if idx < len(metas) else {}

            # In cosine space: distance = 1 - cosine_similarity
            # Therefore similarity = 1 - distance
            similarity = max(0.0, min(1.0, 1.0 - dist))

            # Filter out chunks below the confidence threshold
            if similarity < min_conf:
                continue

            filename = meta.get("filename", "Unknown Document")
            dept = meta.get("department", "ALL")
            chunk_idx = meta.get("chunk_index", 0)

            item = {
                "text": doc_text,
                "filename": filename,
                "department": dept,
                "chunk_index": chunk_idx,
                "similarity": round(similarity, 4),
                "metadata": meta,
            }
            retrieved_items.append(item)

            context_parts.append(
                f"[Source: {filename} | Dept: {dept} | Chunk {chunk_idx + 1} | Match: {similarity * 100:.1f}%]\n{doc_text}\n"
            )

        formatted_context = "\n---\n".join(context_parts)
        return retrieved_items, formatted_context
