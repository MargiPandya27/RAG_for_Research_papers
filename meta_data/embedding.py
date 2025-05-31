from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

def embedding(documents):
    # Load the sentence-transformer model
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Extract abstracts and encode them
    corpus = [doc.get('abstract', '') for doc in documents]
    corpus_embeddings = model.encode(corpus, show_progress_bar=True)

    # Create FAISS index
    dimension = corpus_embeddings[0].shape[0]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(corpus_embeddings))

    print("FAISS index created and embeddings added.")
    return index
