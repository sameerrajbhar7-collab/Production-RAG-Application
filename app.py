import os
import uuid     
import openai
import PyPDF2
import docx
import chromadb
import numpy as np
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from flask import Flask, render_template, request, flash, redirect, url_for
from werkzeug.utils import secure_filename

# -----------------------------------------------------
# 1. SETUP & API KEY
# -----------------------------------------------------
load_dotenv()
openai.api_key = os.environ.get("OPENAI_API_KEY", "")

app = Flask(__name__)
app.secret_key = "super_secret_key"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
CHROMA_DB_DIR = os.path.join(BASE_DIR, "chroma_db")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# -----------------------------------------------------
# 2. CHROMADB & EMBEDDING SETUP
# -----------------------------------------------------
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_DIR)

# Create a custom embedding function for ChromaDB to use OpenAI
class OpenAIEmbeddingFunction:
    def name(self) -> str:
        return "openai_embeddings"

    def __call__(self, input: list[str]) -> list[list[float]]:
        client = OpenAI()
        response = client.embeddings.create(
            input=input,
            model="text-embedding-3-small"
        )
        return [data.embedding for data in response.data]

embedding_fn = OpenAIEmbeddingFunction()

def get_collection():
    """Always fetch a fresh collection reference to avoid stale UUID issues."""
    return chroma_client.get_or_create_collection(
        name="documents", 
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"}
    )

# -----------------------------------------------------
# 3. TEXT EXTRACTION FUNCTION
# -----------------------------------------------------
def extract_text(file_path, filename):
    ext = filename.rsplit('.', 1)[-1].lower()
    text = ""
    
    if ext == 'pdf':
        with open(file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
    elif ext == 'docx':
        doc = docx.Document(file_path)
        for para in doc.paragraphs:
            text += para.text + "\n"
    elif ext == 'txt':
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
            
    return text

def get_uploaded_documents():
    docs = []
    if os.path.exists(UPLOAD_FOLDER):
        for fname in sorted(os.listdir(UPLOAD_FOLDER)):
            fpath = os.path.join(UPLOAD_FOLDER, fname)
            if os.path.isfile(fpath):
                size_bytes = os.path.getsize(fpath)
                size_str = f"{size_bytes / 1024:.1f} KB" if size_bytes < 1024 * 1024 else f"{size_bytes / (1024*1024):.1f} MB"
                ext = fname.rsplit(".", 1)[-1].upper() if "." in fname else "PDF"
                mtime = os.path.getmtime(fpath)
                date_str = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
                
                name_upper = fname.upper()
                dept = "FIN" if "FIN" in name_upper or "TRAVEL" in name_upper else ("IT" if "IT" in name_upper or "SECURITY" in name_upper else ("HR" if "HR" in name_upper or "LEAVE" in name_upper else "All"))
                
                docs.append({
                    "filename": fname,
                    "size": size_str,
                    "ext": ext,
                    "department": dept,
                    "pages": "4 pgs",
                    "vectors": "2 vectors",
                    "status": "Indexed",
                    "added_date": date_str,
                    "is_sample": False
                })
    if not docs:
        docs = [
            {"filename": "Finance_Travel_Policy.pdf", "size": "0.2 KB", "ext": "PDF", "department": "FIN", "pages": "4 pgs", "vectors": "2 vectors", "status": "Indexed", "added_date": "2026-11-09 13:21", "is_sample": True},
            {"filename": "IT_Security_Policy.pdf", "size": "0.2 KB", "ext": "PDF", "department": "IT", "pages": "5 pgs", "vectors": "3 vectors", "status": "Indexed", "added_date": "2026-11-09 13:21", "is_sample": True},
            {"filename": "Employee_Handbook.pdf", "size": "0.2 KB", "ext": "PDF", "department": "All", "pages": "16 pgs", "vectors": "8 vectors", "status": "Indexed", "added_date": "2026-11-09 13:21", "is_sample": True},
            {"filename": "HR_Leave_Policy.pdf", "size": "0.4 KB", "ext": "PDF", "department": "All", "pages": "3 pgs", "vectors": "4 vectors", "status": "Indexed", "added_date": "2026-11-09 13:21", "is_sample": True},
        ]
    return docs

# -----------------------------------------------------
# 4. FLASK ROUTES
# -----------------------------------------------------
@app.route("/")
def index():
    tab = request.args.get("tab", "chat")
    return render_template("index.html", active_tab=tab, documents=get_uploaded_documents())

@app.route("/upload", methods=["POST"])
def upload_file():
    file = request.files.get('file')
    department = request.form.get("department", "All")
    if not file or file.filename == '':
        flash("No file selected", "error")
        return redirect(url_for("index", tab="documents"))
        
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)
    
    # 1. Extract text
    text = extract_text(file_path, filename)
    if not text.strip():
        flash("No text could be extracted.", "error")
        return redirect(url_for("index", tab="documents"))
        
    # 2. Split into chunks using RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(text)
    
    # 3. Save to ChromaDB (upsert to handle re-uploads of the same file)
    ids = [str(uuid.uuid4()) for _ in range(len(chunks))]
    metadatas = [{"filename": filename, "department": department} for _ in chunks]
    
    collection = get_collection()
    collection.upsert(documents=chunks, metadatas=metadatas, ids=ids)
    
    flash("Document uploaded and processed successfully!", "success")
    return redirect(url_for("index", tab="documents"))

@app.route("/delete_document/<filename>", methods=["POST"])
def delete_document(filename):
    filename = secure_filename(filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if os.path.exists(file_path):
        os.remove(file_path)
    try:
        collection = get_collection()
        collection.delete(where={"filename": filename})
    except Exception:
        pass
    flash(f"Document '{filename}' removed.", "success")
    return redirect(url_for("index", tab="documents"))

@app.route("/ask", methods=["POST"])
def ask():
    question = request.form.get("question")
    docs = get_uploaded_documents()
    if not question:
        return render_template("index.html", error="Please enter a question.", active_tab="chat", documents=docs)

    # 1. Search Database for Context (Manual Cosine Similarity)
    collection = get_collection()
    all_data = collection.get(include=["embeddings", "documents"])
    all_embeddings = all_data.get("embeddings", [])
    all_docs = all_data.get("documents", [])
    
    if all_embeddings is None or len(all_embeddings) == 0:
        context = "No relevant context found."
    else:
        # Generate embedding for the user's question using OpenAI
        question_embedding = embedding_fn([question])[0]
        
        # Convert to numpy arrays for sklearn
        q_arr = np.array(question_embedding).reshape(1, -1)
        docs_arr = np.array(all_embeddings)
        
        # Calculate cosine similarity manually using sklearn
        similarities = cosine_similarity(q_arr, docs_arr)[0]
        
        # Get indices of the top 5 most similar chunks (sorted highest to lowest)
        top_indices = np.argsort(similarities)[-5:][::-1]
        
        # Retrieve the corresponding text chunks
        documents = [all_docs[i] for i in top_indices]
        context = "\n\n".join(documents)
        
    # 2. Build the Prompt for OpenAI
    prompt = f"""Use the following pieces of context to answer the user's question.
If you don't know the answer based on the context, just say that you don't know.

Context:
{context}

Question: {question}

Answer:"""

    # 3. Send to OpenAI
    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful and intelligent AI assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        answer = response.choices[0].message.content
        return render_template("index.html", question=question, answer=answer, active_tab="chat", documents=docs)
    except Exception as e:
        return render_template("index.html", question=question, error=str(e), active_tab="chat", documents=docs)

@app.route("/remove_documents", methods=["POST"])
def remove_documents():
    # 1. Delete physical files
    for filename in os.listdir(UPLOAD_FOLDER):
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
            
    # 2. Clear Database Collection
    try:
        chroma_client.delete_collection(name="documents")
    except:
        pass
    # get_collection() will recreate it on the next request
    
    flash("All documents removed and database cleared.", "success")
    return redirect(url_for("index", tab="documents"))

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
