import os
from pypdf import PdfReader

from app.core.exceptions import AIException
from app.retrieval.document import Document


class DocumentLoader:
    """Utility to load files into Document objects."""
    
    @staticmethod
    def load_markdown(file_path: str) -> Document:
        """Loads a markdown file."""
        if not os.path.exists(file_path):
            raise AIException(f"File not found: {file_path}", status_code=404)
            
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            
        return Document(
            content=content,
            metadata={"source": file_path, "type": "markdown"}
        )
        
    @staticmethod
    def load_pdf(file_path: str) -> Document:
        """Loads a PDF file using PyPDF."""
        if not os.path.exists(file_path):
            raise AIException(f"File not found: {file_path}", status_code=404)
            
        text_content = []
        try:
            reader = PdfReader(file_path)
            for page in reader.pages:
                text_content.append(page.extract_text() or "")
        except Exception as e:
            raise AIException(f"Failed to read PDF: {str(e)}", status_code=500)
            
        return Document(
            content="\n".join(text_content),
            metadata={"source": file_path, "type": "pdf", "pages": len(text_content)}
        )
