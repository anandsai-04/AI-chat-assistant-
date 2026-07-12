import os
import pytest
from app.retrieval.document import Document
from app.retrieval.vector_store import VectorStore

@pytest.fixture
def vector_store(tmp_path):
    """Fixture to provide a clean VectorStore pointing to a temporary directory."""
    db_path = str(tmp_path / "chroma_test")
    return VectorStore(persist_directory=db_path, collection_name="test_collection")

def test_vector_store_add_and_search(vector_store):
    # Create test documents
    docs = [
        Document(content="To build a fitness expert, you need medical data.", metadata={"source": "fitness.txt"}),
        Document(content="FastAPI is a modern web framework for building APIs.", metadata={"source": "coding.txt"})
    ]
    
    # Add to store
    vector_store.add_documents(docs)
    
    # Search for a coding related query
    results = vector_store.search("How do I build an API?", n_results=1)
    
    assert len(results) == 1
    assert "FastAPI" in results[0].content
    assert results[0].metadata["source"] == "coding.txt"
