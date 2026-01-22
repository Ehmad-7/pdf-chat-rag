# 📄 PDF Chat with RAG (Retrieval-Augmented Generation)

Chat with your PDF documents using embeddings, semantic search, and an LLM.

This project implements a complete RAG pipeline:
PDF → Chunking → Embeddings → Retrieval → LLM Answer → Chat Memory

---

## 🚀 Features

- Load and read PDF documents
- Clean and chunk text into meaningful pieces
- Generate embeddings using SentenceTransformers
- Retrieve relevant chunks using cosine similarity
- Answer questions using an LLM
- Maintain chat history for follow-up questions

---

## 🧠 Tech Stack

- Python
- SentenceTransformers
- HuggingFace Transformers
- Scikit-learn
- PyPDF

---

## 📁 Project Structure

pdf-chat-rag/
│
├── app.py
├── requirements.txt
├── README.md
│
└── data/
└── sample.pdf

