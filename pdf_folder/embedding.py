# embedding.py

import torch
from transformers import AutoTokenizer, AutoModel
from langchain_core.embeddings import Embeddings

class SciBERTEmbeddings(Embeddings):
    def __init__(self, model_name="allenai/scibert_scivocab_uncased"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)

    def _embed(self, texts):
        inputs = self.tokenizer(
            texts, padding=True, truncation=True,
            return_tensors="pt", max_length=512
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            return outputs.last_hidden_state.mean(dim=1).cpu().numpy().tolist()

    def embed_documents(self, texts):
        return self._embed(texts)

    def embed_query(self, text):
        return self._embed([text])[0]