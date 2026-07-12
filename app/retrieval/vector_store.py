import os
from typing import List
from pinecone import Pinecone, ServerlessSpec
from app.retrieval.document import Document
from app.retrieval.embeddings import EmbeddingGenerator
from app.core.config import settings

class VectorStore:
    """Wrapper around Pinecone Cloud Vector DB for persisting and querying document chunks."""
    
    def __init__(self, index_name: str = "ai-knowledge"):
        """Initialize Pinecone client and Embedding Generator."""
        self.api_key = settings.PINECONE_API_KEY
        if not self.api_key:
            print("Warning: PINECONE_API_KEY is not set. Vector search will be disabled.")
            self.index = None
            return
            
        self.pc = Pinecone(api_key=self.api_key)
        self.index_name = index_name
        self.embedder = EmbeddingGenerator()
        
        # Ensure the index exists for our 384-dimensional HuggingFace embeddings
        existing_indexes = [idx.name for idx in self.pc.list_indexes()]
        if index_name not in existing_indexes:
            self.pc.create_index(
                name=index_name,
                dimension=384, # Dimension for all-MiniLM-L6-v2
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            
        self.index = self.pc.Index(index_name)
        
    def add_documents(self, documents: List[Document]):
        """Generates local embeddings and uploads them to Pinecone Cloud."""
        if not documents or not self.index:
            return
            
        texts = [doc.content for doc in documents]
        
        vectors_to_upsert = []
        embeddings = self.embedder.generate(texts)
        
        for i, doc in enumerate(documents):
            source = doc.metadata.get("source", "unknown")
            chunk_idx = doc.metadata.get("chunk_start_token", i)
            safe_source = source.replace("/", "_").replace("\\", "_")
            doc_id = f"{safe_source}_chunk_{chunk_idx}_{i}"
            
            # Pinecone allows us to store the raw text inside the metadata
            meta = doc.metadata.copy()
            meta["text_content"] = doc.content
            
            vectors_to_upsert.append({
                "id": doc_id,
                "values": embeddings[i],
                "metadata": meta
            })
            
        # Upload to Cloud
        self.index.upsert(vectors=vectors_to_upsert)
        
    def search(self, query: str, n_results: int = 3) -> List[Document]:
        """Searches Pinecone for the most similar documents to the query."""
        if not self.index:
            return []
            
        # Generate the query vector locally on the Mac
        query_embedding = self.embedder.generate([query])[0]
        
        # Search the Pinecone cloud
        results = self.index.query(
            vector=query_embedding,
            top_k=n_results,
            include_metadata=True
        )
        
        retrieved_docs = []
        for match in results.matches:
            meta = match.metadata or {}
            content = meta.pop("text_content", "")
            retrieved_docs.append(Document(
                content=content,
                metadata=meta
            ))
                
        return retrieved_docs
