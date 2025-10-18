# from sentence_transformers import SentenceTransformer # makes “embeddings” (meaningful numeric versions of text)
# import chromadb # stores and searches those embeddings
# from typing import List, Optional # just to tell Python what types we’re using (for clarity)

# class VectorStore:
#     def __init__(self, collection_name: str = "default_collection"):
#         self.model = SentenceTransformer("all-MiniLM-L6-v2") # Loads a MiniLM model (a small free AI model for embeddings)
#         self.client = chromadb.Client() # Connects to a ChromaDB client (in-memory database)
#         try:
#             self.collection = self.client.get_collection(collection_name) # Tries to open a memory folder (called a collection) inside Chroma
#         except Exception:
#             self.collection = self.client.create_collection(name=collection_name) # If it doesn’t exist, it creates one

#     # So now you have a memory folder named "default_collection" ready to store knowledge.

#     # Converting Text to Numbers
#     def embed_texts(self, texts: List[str]):
#         return self.model.encode(texts).tolist() # [0.134, -0.256, 0.984, …], Each sentence becomes a vector

#     def add_documents(self, docs: List[str], metadatas: Optional[List[dict]] = None):
#         ids = [f"doc_{i}" for i in range(len(docs))]
#         embeddings = self.embed_texts(docs)
        
#         # Ensure each metadata dict has at least one key-value pair
#         safe_metadatas = metadatas or [{"source": "pdf"} for _ in docs]
#         for m in safe_metadatas:
#             if not m:  # if it's empty
#                 m["source"] = "pdf"
        
#         self.collection.add(
#             documents=docs,
#             metadatas=safe_metadatas,
#             ids=ids,
#             embeddings=embeddings
#         )
#         # At this point → your chatbot’s “knowledge” is stored in the database.

#     # ChromaDB finds the top k chunks that are most similar to your question.
#     def similarity_search(self, query: str, k: int = 4):
#         query_embedding = self.embed_texts([query])[0]
#         # I have only one text (the user’s question),but I still need to pass it inside a list, because the function expects a list of strings. We just want that first embedding, not the outer list.
#         results = self.collection.query(
#             query_embeddings=[query_embedding],
#             n_results=k,
#             include=["documents", "metadatas", "distances"]
#         )
#         docs = results["documents"][0]
#         metadatas = results["metadatas"][0]
#         distances = results["distances"][0]

#         out = []
#         # zip() is a Python built-in function that lets you loop through multiple lists together, matching their elements by position
#         for d, m, dist in zip(docs, metadatas, distances): 
#             out.append({"document": d, "metadata": m, "distance": dist})
#         return out



from sentence_transformers import SentenceTransformer # makes “embeddings” (meaningful numeric versions of text)
import chromadb # stores and searches those embeddings
from typing import List, Optional, Union, Dict # just to tell Python what types we’re using (for clarity)

class VectorStore:
    def __init__(self, collection_name: str = "default_collection"):
        self.model = SentenceTransformer("all-MiniLM-L6-v2") # Loads a MiniLM model (a small free AI model for embeddings)
        self.client = chromadb.Client() # Connects to a ChromaDB client (in-memory database)
        try:
            self.collection = self.client.get_collection(collection_name) # Tries to open a memory folder (called a collection) inside Chroma
        except Exception:
            self.collection = self.client.create_collection(name=collection_name) # If it doesn’t exist, it creates one

    # So now you have a memory folder named "default_collection" ready to store knowledge.

    # Converting Text to Numbers
    def embed_texts(self, texts: List[str]):
        return self.model.encode(texts).tolist() # [0.134, -0.256, 0.984, …], Each sentence becomes a vector

    def add_documents(self, docs: List[Union[str, Dict]], metadatas: Optional[List[dict]] = None):
        """
        Add documents (list of strings or dicts) to the Chroma collection as embeddings.
        Handles both:
        - list of strings: ["text1", "text2", ...]
        - list of dicts: [{"text": "text1", "metadata": {...}}, ...]
        """
        # Extract texts and metadata safely
        texts = []
        safe_metadatas = []
        for i, doc in enumerate(docs):
            if isinstance(doc, dict):
                text = doc.get("text", "")
                meta = doc.get("metadata", {"source": "pdf"})
            else:  # assume string
                text = doc
                meta = {"source": "pdf"}

            texts.append(text)
            safe_metadatas.append(meta)

        ids = [f"doc_{i}" for i in range(len(texts))]
        embeddings = self.embed_texts(texts)

        # Add to Chroma
        self.collection.add(
            documents=texts,
            metadatas=safe_metadatas,
            ids=ids,
            embeddings=embeddings
        )
        # At this point → your chatbot’s “knowledge” is stored in the database.

    # ChromaDB finds the top k chunks that are most similar to your question.
    def similarity_search(self, query: str, k: int = 4):
        query_embedding = self.embed_texts([query])[0]
        # I have only one text (the user’s question), but I still need to pass it inside a list,
        # because the function expects a list of strings. We just want that first embedding, not the outer list.
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"]
        )
        docs = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        out = []
        # zip() is a Python built-in function that lets you loop through multiple lists together,
        # matching their elements by position
        for d, m, dist in zip(docs, metadatas, distances):
            out.append({"document": d, "metadata": m, "distance": dist})
        return out
