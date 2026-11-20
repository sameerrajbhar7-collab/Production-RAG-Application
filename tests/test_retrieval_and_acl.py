import pytest


def test_chromadb_indexing_and_native_retrieval(test_retrieval_service):
    chunks = [
        "All IT equipment requests must be approved by the IT Lead via Jira.",
        "IT VPN requires multi-factor authentication using Okta.",
    ]
    chunk_ids = ["chunk_it_1", "chunk_it_2"]
    metadatas = [
        {"filename": "IT_Policy.pdf", "department": "IT", "chunk_index": 0},
        {"filename": "IT_Policy.pdf", "department": "IT", "chunk_index": 1},
    ]

    indexed_count = test_retrieval_service.index_document_chunks(
        chunks=chunks,
        chunk_ids=chunk_ids,
        metadatas=metadatas,
        filename="IT_Policy.pdf",
    )
    assert indexed_count == 2

    # Query with IT permissions
    items, context = test_retrieval_service.search_relevant_chunks(
        query="How to request IT equipment and connect to VPN?",
        allowed_departments=["ALL", "IT"],
        top_k=2,
        min_confidence=0.0,
    )
    assert len(items) > 0
    assert "IT_Policy.pdf" in context


def test_department_level_access_control(test_retrieval_service):
    """
    CRITICAL SECURITY & RBAC TEST:
    Verifies that a user authorized ONLY for IT cannot retrieve HR or Finance documents.
    """
    # 1. Index HR Document
    test_retrieval_service.index_document_chunks(
        chunks=["Confidential HR severance formula and performance termination criteria."],
        chunk_ids=["chunk_hr_1"],
        metadatas=[{"filename": "HR_Confidential.pdf", "department": "HR", "chunk_index": 0}],
        filename="HR_Confidential.pdf",
    )

    # 2. Index IT Document
    test_retrieval_service.index_document_chunks(
        chunks=["General IT password reset instructions via portal."],
        chunk_ids=["chunk_it_pwd"],
        metadatas=[{"filename": "IT_Guide.pdf", "department": "IT", "chunk_index": 0}],
        filename="IT_Guide.pdf",
    )

    # 3. IT employee searches for severance info
    it_items, it_context = test_retrieval_service.search_relevant_chunks(
        query="What is the confidential severance formula?",
        allowed_departments=["ALL", "IT"],  # No HR access!
        top_k=5,
        min_confidence=0.0,
    )

    # Must NOT contain HR document
    for item in it_items:
        assert item["department"] != "HR"
        assert item["filename"] != "HR_Confidential.pdf"
    assert "HR_Confidential.pdf" not in it_context

    # 4. HR employee searches with HR access
    hr_items, hr_context = test_retrieval_service.search_relevant_chunks(
        query="What is the confidential severance formula?",
        allowed_departments=["ALL", "HR"],
        top_k=5,
        min_confidence=0.0,
    )
    hr_filenames = [it["filename"] for it in hr_items]
    assert "HR_Confidential.pdf" in hr_filenames


def test_duplicate_document_upload_handling(test_retrieval_service):
    """Re-uploading a document must replace its chunks, never creating duplicates."""
    filename = "Employee_Handbook_2026.pdf"

    # Version 1: 3 chunks
    chunks_v1 = ["Handbook section 1", "Handbook section 2", "Handbook section 3"]
    ids_v1 = ["hb_1", "hb_2", "hb_3"]
    meta_v1 = [{"filename": filename, "department": "ALL", "chunk_index": i} for i in range(3)]

    test_retrieval_service.index_document_chunks(chunks_v1, ids_v1, meta_v1, filename)

    # Version 2: Re-upload with 1 revised chunk
    chunks_v2 = ["Handbook updated single unified section"]
    ids_v2 = ["hb_v2_1"]
    meta_v2 = [{"filename": filename, "department": "ALL", "chunk_index": 0}]

    test_retrieval_service.index_document_chunks(chunks_v2, ids_v2, meta_v2, filename)

    # Query
    items, _ = test_retrieval_service.search_relevant_chunks(
        query="Handbook",
        allowed_departments=["ALL"],
        top_k=10,
        min_confidence=0.0,
    )
    matching_chunks = [it for it in items if it["filename"] == filename]

    # There should only be 1 chunk, NOT 4 chunks (no duplicates)
    assert len(matching_chunks) == 1
    assert matching_chunks[0]["text"] == "Handbook updated single unified section"
