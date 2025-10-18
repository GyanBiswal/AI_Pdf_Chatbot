# PDF Chatbot (Streamlit + Chroma + Sentence-Transformers)

## Overview
A simple Retrieval-Augmented Generation (RAG) PDF chatbot:
- Upload PDFs through a Streamlit UI
- Extract text using `pypdf`
- Convert text chunks into embeddings with `sentence-transformers`
- Store and search embeddings in `chromadb`
- Answer user queries using a QA model (`deepset/roberta-base-squad2`)

## Project Flow
1. **Upload PDF** – User uploads a PDF via Streamlit UI.  
2. **Load & Split** – `pdf_loader.py` extracts text and splits it into overlapping chunks.  
3. **Embeddings** – `vector_store.py` converts chunks into numeric vectors using MiniLM.  
4. **Vector Store** – ChromaDB stores embeddings for fast similarity search.  
5. **User Query** – User types a question.  
6. **Retrieve** – Top-k similar chunks are fetched from ChromaDB using embeddings.  
7. **RAG / QA** – `rag_pipeline.py` combines retrieved chunks as context and runs the QA model to generate an answer.  
8. **Display Answer** – Streamlit shows the answer along with relevant context.  

## Tech Stack
- **Frontend**: Streamlit  
- **PDF Processing**: pypdf  
- **Embeddings**: sentence-transformers (MiniLM)  
- **Vector Store**: ChromaDB  
- **RAG / QA**: Hugging Face Transformers pipeline  
- **Language**: Python 3.12  

## Project Structure
   pdf_chatbot/
   ├── app.py
   ├── requirements.txt
   └── backend/
   ├── init.py
   ├── pdf_loader.py
   ├── vector_store.py
   └── rag_pipeline.py


## Setup
```bash
# Clone repo
git clone <repo_url>
cd pdf_chatbot

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Run app
streamlit run app.py
