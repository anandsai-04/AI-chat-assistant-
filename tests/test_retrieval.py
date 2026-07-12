import os
from app.retrieval.document import Document
from app.retrieval.loaders import DocumentLoader
from app.retrieval.chunker import TokenChunker

def test_document_model():
    doc = Document(content="test content", metadata={"source": "test.txt"})
    assert doc.content == "test content"
    assert doc.metadata["source"] == "test.txt"

def test_chunker():
    doc = Document(content="This is a simple test document for chunking.", metadata={})
    chunker = TokenChunker(chunk_size=4, chunk_overlap=1)
    chunks = chunker.split_document(doc)
    
    assert len(chunks) > 0
    # The exact decoded token contents depend on cl100k_base
    assert chunks[0].metadata["chunk_start_token"] == 0
    assert chunks[0].metadata["chunk_end_token"] == 4

def test_markdown_loader(tmp_path):
    test_file = tmp_path / "test.md"
    test_file.write_text("# Test\nThis is a test file.")
    
    doc = DocumentLoader.load_markdown(str(test_file))
    assert doc.content == "# Test\nThis is a test file."
    assert doc.metadata["type"] == "markdown"
