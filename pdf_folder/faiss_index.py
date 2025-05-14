import json
import faiss
import numpy as np

import re
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
from text_utils import clean_text, chunk_text
from embedding import SciBERTEmbeddings


# -------- FAISS Document Processor -------- #
class DocumentProcessor:
    def __init__(self, dataset, output_dir, index_name="research_papers_index"):
        self.dataset = dataset
        self.output_dir = Path(output_dir)
        self.embedder = SciBERTEmbeddings()
        self.index_name = index_name
        self.index = None
        self.metadata = []
        self.output_dir.mkdir(exist_ok=True)

    def create_faiss_index(self, dim=768, M=32, ef_construction=200):
        self.index = faiss.IndexHNSWFlat(dim, M)
        self.index.hnsw.efConstruction = ef_construction
        self.index.hnsw.efSearch = 50

    def add_to_faiss_index(self, embeddings, metadatas):
        if self.index is None:
            raise ValueError("FAISS index not created.")
        embeddings_np = np.array(embeddings).astype(np.float32)
        self.index.add(embeddings_np)
        self.metadata.extend(metadatas)

    def search_faiss_index(self, query_embedding, k=5):
        if self.index is None:
            raise ValueError("FAISS index not created.")
        distances, indices = self.index.search(np.array([query_embedding]).astype(np.float32), k)
        return distances, indices

    def process_dataset(self):
        self.create_faiss_index(dim=768)
        all_documents = []

        for doc_id, doc in tqdm(enumerate(self.dataset), desc="Processing dataset"):
            try:
                text = clean_text(doc["article"])
                chunks = chunk_text(text, chunk_size=300, overlap=50)

                metadata_base = {
                    "source": doc.get("title", f"document_{doc_id}"),
                    "section_titles": doc.get("section_names", [])
                }

                embeddings = self.embedder.embed_documents(chunks)

                for i, (chunk_content, embedding) in enumerate(zip(chunks, embeddings)):
                    metadata = metadata_base.copy()
                    metadata["chunk_id"] = i
                    metadata["start_index"] = i * 250
                    self.add_to_faiss_index([embedding], [metadata])
                    self._save_chunk(f"document_{doc_id}_chunk{i+1}", chunk_content, metadata)

                    all_documents.append(Document(page_content=chunk_content, metadata=metadata))

            except Exception as e:
                print(f"Error processing document {doc_id}: {e}")

        return all_documents

    def _save_chunk(self, name, text, metadata):
        path = self.output_dir / f"{name}.txt"
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"Source: {metadata.get('source', 'unknown')}\n")
            f.write(f"Section Titles: {metadata.get('section_titles', [])}\n")
            f.write(f"Start Index: {metadata.get('start_index', 0)}\n")
            f.write("-" * 50 + "\n")
            f.write(text)

# -------- JSON Loader and Converter -------- #
def load_and_convert_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)

    dataset = []
    for paper_id, paper in json_data.items():
        full_text = " ".join([sec["content"] for sec in paper["sections"]])
        section_names = [sec["section"] for sec in paper["sections"]]
        dataset.append({
            "title": paper["title"],
            "article": full_text,
            "section_names": section_names
        })

    return dataset

# -------- Example Usage -------- #
if __name__ == "__main__":
    json_path = "processed_papers.json"  # <-- Update this with your file
    dataset = load_and_convert_json(json_path)

    processor = DocumentProcessor(
        dataset=dataset,
        output_dir="processed_chunks_json"
    )
    processed_documents = processor.process_dataset()
