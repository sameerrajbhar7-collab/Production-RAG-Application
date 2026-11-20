import io
import os
import pytest
from werkzeug.datastructures import FileStorage

from services.document_service import DocumentService


def test_allowed_file_extensions(test_doc_service):
    assert test_doc_service.is_allowed_file("policy.pdf") is True
    assert test_doc_service.is_allowed_file("handbook.docx") is True
    assert test_doc_service.is_allowed_file("notes.txt") is True
    assert test_doc_service.is_allowed_file("script.py") is False
    assert test_doc_service.is_allowed_file("malicious.exe") is False
    assert test_doc_service.is_allowed_file("no_extension") is False


def test_document_upload_and_chunking(test_doc_service):
    content = b"This is employee travel policy. Daily allowance is $50. Receipts must be submitted within 30 days."
    file = FileStorage(stream=io.BytesIO(content), filename="Travel_Policy.txt")

    success, message, result = test_doc_service.process_and_save_upload(file, department="FIN")

    assert success is True
    assert result["filename"] == "Travel_Policy.txt"
    assert result["department"] == "FIN"
    assert len(result["chunks"]) >= 1
    assert "doc_Travel_Policy_" in result["doc_id"]
    assert len(result["chunk_ids"]) == len(result["chunks"])
    assert result["metadatas"][0]["department"] == "FIN"


def test_document_deduplication_hash(test_doc_service):
    text1 = "Company remote work policy: Employees may work from home twice a week."
    text2 = "Company remote work policy: Employees may work from home twice a week."
    text3 = "Company remote work policy: Employees may work from home three times a week."

    h1 = test_doc_service.compute_hash(text1)
    h2 = test_doc_service.compute_hash(text2)
    h3 = test_doc_service.compute_hash(text3)

    assert h1 == h2
    assert h1 != h3


def test_delete_physical_file(test_doc_service):
    content = b"Sample text for deletion test."
    file = FileStorage(stream=io.BytesIO(content), filename="temp_delete_test.txt")

    success, _, result = test_doc_service.process_and_save_upload(file, department="IT")
    assert success is True
    assert os.path.exists(result["file_path"]) is True

    deleted = test_doc_service.delete_physical_file("temp_delete_test.txt")
    assert deleted is True
    assert os.path.exists(result["file_path"]) is False
