# faiss_indexing.py
import faiss
import numpy as np

class FaissIndexing:
    def __init__(self, dim=768, M=32, ef_construction=200):
        self.index = faiss.IndexHNSWFlat(dim, M)
        self.index.hnsw.efConstruction = ef_construction
        self.index.hnsw.efSearch = 50
        self.metadata = []

    def add_to_faiss_index(self, embeddings, metadatas):
        embeddings_np = np.array(embeddings).astype(np.float32)
        self.index.add(embeddings_np)
        self.metadata.extend(metadatas)

    def search_faiss_index(self, query_embedding, k=5):
        distances, indices = self.index.search(np.array([query_embedding]).astype(np.float32), k)
        return distances, indices
