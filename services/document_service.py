import os
import hashlib
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
from werkzeug.utils import secure_filename
from werkzeug.datastructures import FileStorage
import PyPDF2
import docx
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import Config


class DocumentService:
    """Service for validating, parsing, chunking, and managing company documents."""

    def __init__(self, upload_folder: Optional[str] = None):
        self.upload_folder = upload_folder or Config.UPLOAD_FOLDER
        os.makedirs(self.upload_folder, exist_ok=True)
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def is_allowed_file(self, filename: str) -> bool:
        """Checks if file extension is allowed."""
        if "." not in filename:
            return False
        ext = filename.rsplit(".", 1)[1].lower()
        return ext in Config.ALLOWED_EXTENSIONS

    def extract_text_from_file(self, file_path: str, filename: str) -> str:
        """Extracts text content from PDF, DOCX, or TXT file."""
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        text = ""

        if ext == "pdf":
            with open(file_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for page_num, page in enumerate(reader.pages):
                    page_text = page.extract_text()
                    if page_text:
                        text += f"\n[Page {page_num + 1}]\n" + page_text.strip() + "\n"
        elif ext == "docx":
            doc = docx.Document(file_path)
            for para in doc.paragraphs:
                if para.text.strip():
                    text += para.text.strip() + "\n"
        elif ext == "txt":
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()

        return text.strip()

    def compute_hash(self, content: str) -> str:
        """Computes SHA-256 hash of extracted text for document deduplication and stable IDs."""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def process_and_save_upload(
        self, file: FileStorage, department: str = "ALL"
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Validates, saves, extracts, and chunks an uploaded document.
        Returns: (success, message, result_dict)
        """
        if not file or not file.filename:
            return False, "No file selected.", {}

        if not self.is_allowed_file(file.filename):
            allowed = ", ".join(Config.ALLOWED_EXTENSIONS).upper()
            return False, f"Unsupported file type. Please upload {allowed} files.", {}

        safe_name = secure_filename(file.filename)
        if not safe_name:
            return False, "Invalid filename.", {}

        # Save file to upload directory
        file_path = os.path.join(self.upload_folder, safe_name)
        file.save(file_path)

        # Check file size limit
        file_size = os.path.getsize(file_path)
        if file_size > Config.MAX_CONTENT_LENGTH:
            os.remove(file_path)
            max_mb = Config.MAX_CONTENT_LENGTH // (1024 * 1024)
            return False, f"File size exceeds maximum allowed limit of {max_mb} MB.", {}

        # Extract text content
        try:
            raw_text = self.extract_text_from_file(file_path, safe_name)
        except Exception as e:
            if os.path.exists(file_path):
                os.remove(file_path)
            return False, f"Failed to extract text from document: {str(e)}", {}

        if not raw_text:
            if os.path.exists(file_path):
                os.remove(file_path)
            return False, "Uploaded file contains no readable text.", {}

        # Calculate stable hash and ID
        doc_hash = self.compute_hash(raw_text)
        short_hash = doc_hash[:10]
        base_name = safe_name.rsplit(".", 1)[0]
        doc_id = f"doc_{base_name}_{short_hash}"

        # Normalize department
        dept_clean = department.upper().strip() if department else "ALL"
        if dept_clean not in Config.COMPANY_DEPARTMENTS:
            dept_clean = "ALL"

        # Split text into chunks
        chunks = self.splitter.split_text(raw_text)
        if not chunks:
            chunks = [raw_text]

        chunk_ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
        now_iso = datetime.now().strftime("%Y-%m-%d %H:%M")

        metadatas = [
            {
                "doc_id": doc_id,
                "filename": safe_name,
                "department": dept_clean,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "doc_hash": doc_hash,
                "added_date": now_iso,
                "file_size": file_size,
            }
            for i in range(len(chunks))
        ]

        result = {
            "doc_id": doc_id,
            "filename": safe_name,
            "department": dept_clean,
            "chunks": chunks,
            "chunk_ids": chunk_ids,
            "metadatas": metadatas,
            "doc_hash": doc_hash,
            "file_size": file_size,
            "file_path": file_path,
        }

        return True, "Document processed successfully.", result

    def get_document_list(self) -> List[Dict[str, Any]]:
        """Returns list of active company documents in the upload directory."""
        docs = []
        if os.path.exists(self.upload_folder):
            for fname in sorted(os.listdir(self.upload_folder)):
                fpath = os.path.join(self.upload_folder, fname)
                if os.path.isfile(fpath) and self.is_allowed_file(fname):
                    size_bytes = os.path.getsize(fpath)
                    size_str = (
                        f"{size_bytes / 1024:.1f} KB"
                        if size_bytes < 1024 * 1024
                        else f"{size_bytes / (1024 * 1024):.1f} MB"
                    )
                    ext = fname.rsplit(".", 1)[-1].upper() if "." in fname else "PDF"
                    mtime = os.path.getmtime(fpath)
                    date_str = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")

                    # Infer default department from filename if not tagged
                    name_upper = fname.upper()
                    if "FIN" in name_upper or "TRAVEL" in name_upper or "BUDGET" in name_upper:
                        dept = "FIN"
                    elif "IT" in name_upper or "SECURITY" in name_upper or "STACK" in name_upper:
                        dept = "IT"
                    elif "HR" in name_upper or "LEAVE" in name_upper or "EMPLOYEE" in name_upper:
                        dept = "HR"
                    elif "SALES" in name_upper:
                        dept = "SALES"
                    else:
                        dept = "ALL"

                    docs.append({
                        "filename": fname,
                        "size": size_str,
                        "ext": ext,
                        "department": dept,
                        "pages": "1+ pgs",
                        "vectors": "indexed",
                        "status": "Indexed",
                        "added_date": date_str,
                        "is_sample": False,
                    })

        # Provide sample template company documents if uploads folder is empty
        if not docs:
            docs = [
                {
                    "filename": "Finance_Travel_Policy.pdf",
                    "size": "0.2 KB",
                    "ext": "PDF",
                    "department": "FIN",
                    "pages": "4 pgs",
                    "vectors": "2 vectors",
                    "status": "Indexed",
                    "added_date": "2026-11-09 13:21",
                    "is_sample": True,
                },
                {
                    "filename": "IT_Security_Policy.pdf",
                    "size": "0.2 KB",
                    "ext": "PDF",
                    "department": "IT",
                    "pages": "5 pgs",
                    "vectors": "3 vectors",
                    "status": "Indexed",
                    "added_date": "2026-11-09 13:21",
                    "is_sample": True,
                },
                {
                    "filename": "Employee_Handbook.pdf",
                    "size": "0.2 KB",
                    "ext": "PDF",
                    "department": "ALL",
                    "pages": "16 pgs",
                    "vectors": "8 vectors",
                    "status": "Indexed",
                    "added_date": "2026-11-09 13:21",
                    "is_sample": True,
                },
                {
                    "filename": "HR_Leave_Policy.pdf",
                    "size": "0.4 KB",
                    "ext": "PDF",
                    "department": "HR",
                    "pages": "3 pgs",
                    "vectors": "4 vectors",
                    "status": "Indexed",
                    "added_date": "2026-11-09 13:21",
                    "is_sample": True,
                },
            ]
        return docs

    def delete_physical_file(self, filename: str) -> bool:
        """Deletes a physical file from the uploads directory."""
        safe_name = secure_filename(filename)
        fpath = os.path.join(self.upload_folder, safe_name)
        if os.path.exists(fpath):
            try:
                os.remove(fpath)
                return True
            except OSError:
                return False
        return False
