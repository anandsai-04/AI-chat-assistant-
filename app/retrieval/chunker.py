import tiktoken
from typing import List
from app.retrieval.document import Document

class TokenChunker:
    """Utility to split large text into smaller overlapping chunks using tokens."""
    
    def __init__(
        self, 
        model_name: str = "cl100k_base", 
        chunk_size: int = 500, 
        chunk_overlap: int = 50
    ):
        """
        Initialize the token-based chunker.
        :param model_name: The tiktoken encoding to use.
        :param chunk_size: Maximum tokens per chunk.
        :param chunk_overlap: Number of overlapping tokens between chunks.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.encoding = tiktoken.get_encoding(model_name)
        
    def split_document(self, document: Document) -> List[Document]:
        """Splits a single Document into multiple token-chunked Documents."""
        text = document.content
        chunks = []
        
        if not text:
            return chunks
            
        if self.chunk_overlap >= self.chunk_size:
            self.chunk_overlap = self.chunk_size - 1
            
        tokens = self.encoding.encode(text)
        
        start = 0
        while start < len(tokens):
            end = start + self.chunk_size
            chunk_tokens = tokens[start:end]
            
            chunk_text = self.encoding.decode(chunk_tokens)
            
            chunk_metadata = document.metadata.copy()
            chunk_metadata["chunk_start_token"] = start
            chunk_metadata["chunk_end_token"] = end
            
            chunks.append(
                Document(
                    content=chunk_text,
                    metadata=chunk_metadata
                )
            )
            
            start += self.chunk_size - self.chunk_overlap
                
        return chunks
