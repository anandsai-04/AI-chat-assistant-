import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from typing import List

class EmbeddingGenerator:
    """Generates dense vector embeddings using HuggingFace Transformers."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initializes the model and tokenizer for local embedding generation.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        
        # Use Apple Silicon GPU (MPS) if available, else CPU
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
            
        self.model.to(self.device)
        self.model.eval()

    def generate(self, texts: List[str]) -> List[List[float]]:
        """
        Takes a list of strings and returns a list of embedding vectors.
        """
        if not texts:
            return []
            
        # Tokenize sentences
        encoded_input = self.tokenizer(
            texts, padding=True, truncation=True, return_tensors='pt'
        ).to(self.device)

        # Compute token embeddings
        with torch.no_grad():
            model_output = self.model(**encoded_input)

        # Perform mean pooling
        embeddings = self._mean_pooling(model_output, encoded_input['attention_mask'])
        
        # Normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings.cpu().numpy().tolist()

    def _mean_pooling(self, model_output, attention_mask):
        """
        Mean Pooling - Take attention mask into account for correct averaging.
        """
        token_embeddings = model_output[0] # First element contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
        return sum_embeddings / sum_mask
