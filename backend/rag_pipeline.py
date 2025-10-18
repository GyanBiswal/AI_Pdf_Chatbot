from transformers import pipeline
from typing import List, Dict

class RAGPipeline:
    def __init__(self):
        # Load a free, offline QA model
        self.qa_pipeline = pipeline(
            "question-answering",  # Tells Hugging Face: “I want to answer questions from a text.”
            model="deepset/roberta-base-squad2",  # A small, free model trained for QA.
            tokenizer="deepset/roberta-base-squad2",
            device=-1  # Use CPU; 0 would use GPU if available.
        )

    # Combine top-k retrieved chunks into a single context string
    def combine_contexts(self, retrieved_docs: List[Dict], top_k: int = 3) -> str:
        """
        Take the top-k chunks from VectorStore.
        No filtering by keywords — works for any PDF content.
        """
        contexts = [doc["document"] for doc in retrieved_docs[:top_k]]
        return "\n\n".join(contexts)  # Combine into one string for the QA model

    def answer_question(self, query: str, retrieved_docs: List[Dict]) -> Dict[str, str]:
        """
        Generate an answer using top retrieved chunks.
        Works for any type of query without assuming keywords.
        """
        if not retrieved_docs:
            return {"answer": "No relevant context found.", "context": ""}

        # Combine top chunks for richer context
        context = self.combine_contexts(retrieved_docs, top_k=3)

        # Run QA model
        answer = self.qa_pipeline(question=query, context=context)
        final_answer = answer["answer"]

        return {
            "answer": final_answer,
            "context": context
        }

# 🔹 Notes:
# 1. A pipeline in Transformers wraps tokenization, model execution, and decoding in one function.
# 2. This RAGPipeline now works for any PDF, any query — no keyword assumptions.
# 3. VectorStore retrieves the most relevant chunks; QA model generates precise answers.
