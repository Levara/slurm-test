from typing import List
from transformers import AutoModel
import torch
from langchain_core.embeddings import Embeddings
import gc

class JinaEmbeddings(Embeddings):
    def __init__(self, model_name: str = "jinaai/jina-embeddings-v3", preload: bool = True):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        if preload:
            self.load_model()
        self.embedded_count = 0
        
    def load_model(self):
        """Load model if not already loaded."""
        if self.model is None:
            self.model = AutoModel.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                attn_implementation='eager',
                torch_dtype=torch.float16,
                device_map=self.device
            )
            print(f"Model loaded on {self.device}")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents using Jina."""
        if not texts:
            return []

        self.embedded_count += 1
            
        self.load_model()  # Idempotent - only loads if not already loaded
        task = 'retrieval.passage'
        print(f"{self.embedded_count}. Embedding {len(texts)} documents")
        embeddings = self.model.encode(texts, task=task)
        return embeddings.tolist()
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a query using Jina."""
        self.load_model()  # Idempotent - only loads if not already loaded
        task = 'retrieval.query'
        embedding = self.model.encode([text], task=task)[0]
        return embedding.tolist()

