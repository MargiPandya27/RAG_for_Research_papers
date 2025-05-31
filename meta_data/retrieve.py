import numpy as np

def retrieve(query, index, documents, model, top_k=5):
    query_embedding = model.encode([query])
    D, I = index.search(np.array(query_embedding), top_k)  # D = distances, I = indices
    results = [documents[i] for i in I[0]]
    return results
