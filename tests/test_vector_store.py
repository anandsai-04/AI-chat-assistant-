import os
import pytest
from unittest.mock import MagicMock, patch
from app.retrieval.document import Document
from app.retrieval.vector_store import VectorStore

@pytest.fixture
def mock_pinecone():
    with patch("app.retrieval.vector_store.Pinecone") as MockPinecone:
        mock_pc = MagicMock()
        mock_index = MagicMock()
        mock_pc.Index.return_value = mock_index
        MockPinecone.return_value = mock_pc
        yield mock_pc, mock_index

@pytest.fixture
def vector_store(mock_pinecone, monkeypatch):
    """Fixture to provide a VectorStore with mocked Pinecone client."""
    monkeypatch.setattr("app.retrieval.vector_store.settings.PINECONE_API_KEY", "fake_key")
    return VectorStore(index_name="test-index")

def test_vector_store_add_documents(vector_store, mock_pinecone):
    _, mock_index = mock_pinecone
    docs = [
        Document(content="To build a fitness expert, you need medical data.", metadata={"source": "fitness.txt"})
    ]
    
    # Add to store
    vector_store.add_documents(docs)
    
    # Verify upsert was called on the index
    mock_index.upsert.assert_called_once()
    
def test_vector_store_search(vector_store, mock_pinecone):
    _, mock_index = mock_pinecone
    
    # Mock the return value of pinecone search
    mock_response = MagicMock()
    mock_match = MagicMock()
    mock_match.metadata = {"source": "fitness.txt", "text_content": "Mocked content"}
    mock_response.matches = [mock_match]
    mock_index.query.return_value = mock_response
    
    results = vector_store.search("How do I build an API?", n_results=1)
    
    assert len(results) == 1
    assert results[0].content == "Mocked content"
    assert results[0].metadata["source"] == "fitness.txt"
