# Production-RAG-Application
An end-to-end RAG (Retrieval-Augmented Generation) application built with Flask, ChromaDB, Sentence Transformers, and OpenAI GPT-4o Mini. Upload PDF, DOCX, or TXT files, retrieve relevant context using vector search, and get accurate AI-powered answers through semantic search and LLMs.

* Features

Upload PDF, DOCX, and TXT documents
Automatic text extraction and preprocessing
Recursive text chunking for efficient retrieval
Semantic embeddings using all-MiniLM-L6-v2
Persistent vector database with ChromaDB
Manual cosine similarity search for relevant document retrieval
Context-aware question answering using GPT-4o Mini
Remove all uploaded documents and reset the vector database
Simple and responsive Flask web interface

* Tech Stack

Python
Flask
OpenAI API
ChromaDB
Sentence Transformers
Scikit-learn
NumPy
PyPDF2
python-docx
LangChain Text Splitters
HTML, CSS, JavaScript

* How It Works

Upload one or more documents.
Extract text from the uploaded files.
Split the text into manageable chunks.
Generate vector embeddings using Sentence Transformers.
Store embeddings in ChromaDB.
When a user asks a question, retrieve the most relevant document chunks using cosine similarity.
Provide the retrieved context to GPT-4o Mini to generate an accurate answer.

* Use Cases
  
Company Policy Assistant
Legal Document Search
Research Paper Q&A
Knowledge Base Chatbot
Educational Document Assistant
Internal Documentation Search
