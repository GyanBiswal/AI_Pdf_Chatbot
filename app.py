import streamlit as st
from transformers import pipeline
from backend.pdf_loader import load_and_split_pdf
from backend.vector_store import VectorStore

st.title("📄 PDF Chatbot (Python, GenAI Demo)")

if "vs" not in st.session_state:
    st.session_state.vs = VectorStore("pdf_collection")

qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file:
    with open("uploaded.pdf", "wb") as f:
        f.write(uploaded_file.read())

    with st.spinner("Processing and chunking PDF..."):
        docs = load_and_split_pdf("uploaded.pdf")
        st.session_state.vs.add_documents(docs)

    st.success("✅ PDF processed and stored!")

    query = st.text_input("Ask a question about the document:")

    if query:
        with st.spinner("Thinking..."):
            # Retrieve top 3 similar chunks
            results = st.session_state.vs.similarity_search(query, k=3)
            combined_context = " ".join([r["document"] for r in results])

            # Run QA model on combined context
            answer = qa_pipeline({
                "context": combined_context,
                "question": query
            })

            st.subheader("Answer:")
            st.write(answer["answer"])

            with st.expander("🔍 Retrieved Context"):
                st.write(combined_context)
