import os
from typing import List

import chromadb

from app.retrieval.document import Document
from app.retrieval.embeddings import EmbeddingGenerator


class VectorStore:
    """Wrapper around ChromaDB for persisting and querying document chunks."""

    def __init__(
        self,
        persist_directory: str = "./data/chroma_db",
        collection_name: str = "ai_knowledge",
    ):
        """Initialize ChromaDB persistent client and Embedding Generator."""
        os.makedirs(persist_directory, exist_ok=True)

        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(name=collection_name)
        self.embedder = EmbeddingGenerator()

    def add_documents(self, documents: List[Document]):
        """Generates embeddings and adds documents to ChromaDB."""
        if not documents:
            return

        texts = [doc.content for doc in documents]
        metadatas = [doc.metadata for doc in documents]

        # Generate unique IDs for ChromaDB
        ids = []
        for i, doc in enumerate(documents):
            source = doc.metadata.get("source", "unknown")
            chunk_idx = doc.metadata.get("chunk_start_token", i)
            # Replace slashes and problematic chars for ID safety
            safe_source = source.replace("/", "_").replace("\\", "_")
            ids.append(f"{safe_source}_chunk_{chunk_idx}_{i}")

        embeddings = self.embedder.generate(texts)

        self.collection.upsert(
            ids=ids, embeddings=embeddings, metadatas=metadatas, documents=texts
        )

    def search(self, query: str, n_results: int = 3) -> List[Document]:
        """Searches for the most similar documents to the query."""
        query_embedding = self.embedder.generate([query])[0]

        results = self.collection.query(
            query_embeddings=[query_embedding], n_results=n_results
        )

        retrieved_docs = []
        if results and results["documents"] and results["documents"][0]:
            docs = results["documents"][0]
            metas = results["metadatas"][0]

            for i in range(len(docs)):
                retrieved_docs.append(
                    Document(content=docs[i], metadata=metas[i] if metas else {})
                )

        return retrieved_docs
