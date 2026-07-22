import os
import uuid
import openai
import PyPDF2
import docx
import chromadb
from sentence_transformers import SentenceTransformer
import numpy as np
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
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

# Load SentenceTransformer directly
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Create a custom embedding function for ChromaDB to use
class CustomEmbeddingFunction:
    def name(self) -> str:
        return "sentence_transformer"

    def __call__(self, input: list[str]) -> list[list[float]]:
        return embedding_model.encode(input).tolist()

embedding_fn = CustomEmbeddingFunction()

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

# -----------------------------------------------------
# 4. FLASK ROUTES
# -----------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_file():
    file = request.files.get('file')
    if not file or file.filename == '':
        flash("No file selected", "error")
        return redirect(url_for("index"))
        
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)
    
    # 1. Extract text
    text = extract_text(file_path, filename)
    if not text.strip():
        flash("No text could be extracted.", "error")
        return redirect(url_for("index"))
        
    # 2. Split into chunks using RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(text)
    
    # 3. Save to ChromaDB (upsert to handle re-uploads of the same file)
    ids = [str(uuid.uuid4()) for _ in range(len(chunks))]
    metadatas = [{"filename": filename} for _ in chunks]
    
    collection = get_collection()
    collection.upsert(documents=chunks, metadatas=metadatas, ids=ids)
    
    flash("Document uploaded and processed successfully!", "success")
    return redirect(url_for("index"))

@app.route("/ask", methods=["POST"])
def ask():
    question = request.form.get("question")
    if not question:
        return render_template("index.html", error="Please enter a question.")

    # 1. Search Database for Context (Manual Cosine Similarity)
    collection = get_collection()
    all_data = collection.get(include=["embeddings", "documents"])
    all_embeddings = all_data.get("embeddings", [])
    all_docs = all_data.get("documents", [])
    
    if all_embeddings is None or len(all_embeddings) == 0:
        context = "No relevant context found."
    else:
        # Generate embedding for the user's question directly using the model
        question_embedding = embedding_model.encode([question])[0]
        
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
        return render_template("index.html", question=question, answer=answer)
    except Exception as e:
        return render_template("index.html", question=question, error=str(e))

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
    return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
