import asyncio
import os
from pathlib import Path
from app.retrieval.loaders import MarkdownLoader
from app.retrieval.chunker import TokenChunker
from app.retrieval.vector_store import VectorStore
from app.core.config import settings

def ingest_rag_documents():
    """Reads large Markdown files and ingests them into the Pinecone Cloud Database."""
    
    input_dir = Path("data/raw/rag")
    
    if not input_dir.exists():
        print(f"Directory {input_dir} not found. Please create it and add markdown files.")
        return
        
    md_files = list(input_dir.glob("*.md"))
    if not md_files:
        print(f"No markdown files found in {input_dir}. Please add some large documentation files for RAG.")
        return
        
    print(f"Found {len(md_files)} markdown files for RAG. Initializing Cloud Vector Store...")
    
    if not settings.PINECONE_API_KEY:
        print("ERROR: PINECONE_API_KEY is not set in the .env file. Cannot upload to cloud.")
        return
        
    vector_store = VectorStore()
    chunker = TokenChunker(chunk_size=500, chunk_overlap=50)
    
    total_chunks_uploaded = 0
    
    for file_path in md_files:
        print(f"Processing {file_path.name}...")
        
        # 1. Load Document
        loader = MarkdownLoader(str(file_path))
        docs = loader.load()
        
        if not docs:
            print(f"Warning: Could not read {file_path.name}")
            continue
            
        # 2. Chunk Document
        chunks = chunker.chunk_documents(docs)
        print(f"  -> Created {len(chunks)} chunks.")
        
        # 3. Upload to Pinecone (Embeddings happen locally inside VectorStore)
        print(f"  -> Generating local embeddings and uploading to Pinecone...")
        vector_store.add_documents(chunks)
        
        total_chunks_uploaded += len(chunks)
        print(f"  -> Successfully uploaded {file_path.name}.")
        
    print("=" * 50)
    print(f"✅ RAG Ingestion Complete! Uploaded {total_chunks_uploaded} chunks to Pinecone.")

if __name__ == "__main__":
    ingest_rag_documents()
