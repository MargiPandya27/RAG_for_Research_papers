# text_processing.py
import re

# Text cleaning function
def clean_text(text):
    text = text.replace("\\n", " ")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\$.*?\$", "", text)
    text = re.sub(r"\[[^\]]*\]", "", text)
    return text.strip()

# Chunking function
def chunk_text(text, chunk_size=300, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i+chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks
